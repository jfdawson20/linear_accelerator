"""Calibration against measured hardware.

Designed to be useful before any prototype exists: with no measurement files
present every correction factor is 1.0 and the model behaves exactly as it does
uncalibrated. Nothing in the physics modules branches on whether data exists --
they take scale factors as constructor arguments and never read files.

Measurement files are YAML, and every field is optional:

    coil_id: stage0
    measured:
      L_air:         58.2e-6     # H, LCR meter, no slug
      L_slug_in:     410e-6      # H, LCR meter, slug centred
      R_dc:          0.671       # ohm, four-wire
      peak_current:  262         # A, from a scope shot
      exit_velocity: 41.3        # m/s, chronograph
    conditions:
      V0: 200
      C:  0.006
      ambient_C: 22
    traces:
      current: traces/stage0_i.csv    # two columns: t,i

The highest-value first measurement is L_air together with L_slug_in. Those two
numbers pin mu_eff directly, which is the largest remaining uncertainty in the
model: the demagnetising factor is computed for a prolate spheroid, and a real
cylinder differs.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class Measurement:
    """One coil's bench data. Every field may be absent."""

    coil_id: str
    l_air: float | None = None
    l_slug_in: float | None = None
    r_dc: float | None = None
    peak_current: float | None = None
    exit_velocity: float | None = None
    conditions: Mapping[str, Any] = field(default_factory=dict)
    current_trace: str | None = None

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], base_dir: str = "") -> "Measurement":
        m = d.get("measured", {}) or {}
        traces = d.get("traces", {}) or {}
        trace = traces.get("current")
        if trace and base_dir:
            trace = os.path.join(base_dir, trace)
        return cls(
            coil_id=str(d.get("coil_id", "unknown")),
            l_air=_opt_float(m.get("L_air")),
            l_slug_in=_opt_float(m.get("L_slug_in")),
            r_dc=_opt_float(m.get("R_dc")),
            peak_current=_opt_float(m.get("peak_current")),
            exit_velocity=_opt_float(m.get("exit_velocity")),
            conditions=d.get("conditions", {}) or {},
            current_trace=trace,
        )

    def load_current_trace(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Read a two-column t,i CSV. Returns None if there is no trace."""
        if not self.current_trace or not os.path.exists(self.current_trace):
            return None
        data = np.loadtxt(self.current_trace, delimiter=",", skiprows=1)
        return data[:, 0], data[:, 1]


def _opt_float(value: Any) -> float | None:
    return None if value is None else float(value)


@dataclass(frozen=True)
class Corrections:
    """Multiplicative corrections applied to modelled quantities.

    All default to 1.0 / None, which reproduces uncalibrated behaviour exactly.
    """

    l_air_scale: float = 1.0
    r_scale: float = 1.0
    mu_eff_override: float | None = None

    @property
    def is_identity(self) -> bool:
        return (
            self.l_air_scale == 1.0
            and self.r_scale == 1.0
            and self.mu_eff_override is None
        )


def derive_corrections(
    measurement: Measurement, predicted_l_air: float, predicted_r: float
) -> Corrections:
    """Turn a measurement into correction factors.

    mu_eff is derived from the pair (L_air, L_slug_in) rather than from the
    spheroid approximation, using the same relation the forward model uses:

        L_slug_in = L_air * (1 + (mu_eff - 1) * max_fill)

    so mu_eff is only recoverable if both inductances were measured. When only
    L_air is available it corrects the Wheeler estimate and leaves mu_eff alone.
    """
    l_scale = 1.0
    if measurement.l_air is not None and predicted_l_air > 0:
        l_scale = measurement.l_air / predicted_l_air

    r_scale = 1.0
    if measurement.r_dc is not None and predicted_r > 0:
        r_scale = measurement.r_dc / predicted_r

    return Corrections(l_air_scale=l_scale, r_scale=r_scale)


def mu_eff_from_measurements(
    l_air: float, l_slug_in: float, max_fill: float
) -> float:
    """Recover mu_eff from a pair of LCR readings.

    This is the measurement worth taking first: five minutes with an LCR meter
    settles the largest open question in the model.
    """
    if max_fill <= 0:
        raise ValueError("max_fill must be positive")
    if l_air <= 0:
        raise ValueError("L_air must be positive")
    return 1.0 + (l_slug_in / l_air - 1.0) / max_fill


def load_measurements(directory: str = "measurements") -> dict[str, Measurement]:
    """Load every YAML file in `directory`, keyed by coil_id.

    A missing directory is not an error -- it is the normal state before a
    prototype exists.
    """
    if not os.path.isdir(directory):
        return {}
    import yaml

    out: dict[str, Measurement] = {}
    for path in sorted(glob.glob(os.path.join(directory, "*.y*ml"))):
        with open(path) as fh:
            data = yaml.safe_load(fh)
        if not data:
            continue
        m = Measurement.from_dict(data, base_dir=directory)
        out[m.coil_id] = m
    return out


# -- predicted vs measured ---------------------------------------------


@dataclass
class Comparison:
    quantity: str
    predicted: float
    measured: float
    unit: str

    @property
    def error_pct(self) -> float:
        return (
            100.0 * (self.predicted - self.measured) / self.measured
            if self.measured
            else float("nan")
        )


def compare(sim, result, measurements: Mapping[str, Measurement]) -> list[Comparison]:
    """Predicted vs measured for whatever was actually measured.

    Runs on any subset of fields; absent measurements are simply skipped.
    """
    rows: list[Comparison] = []
    for k, circ in enumerate(sim.circuits):
        m = measurements.get(f"stage{k}")
        if m is None:
            continue
        mag = circ.magnetics
        if m.l_air is not None:
            rows.append(Comparison(f"stage{k} L_air", mag.l_air, m.l_air, "H"))
        if m.l_slug_in is not None:
            x_centre = circ.config.position + (
                circ.config.coil.length + sim.config.projectile.length
            ) / 2.0
            rows.append(
                Comparison(
                    f"stage{k} L_slug_in",
                    float(mag.inductance(x_centre)),
                    m.l_slug_in,
                    "H",
                )
            )
        if m.r_dc is not None:
            rows.append(Comparison(f"stage{k} R_dc", circ.r20, m.r_dc, "ohm"))
        if m.peak_current is not None and result is not None:
            rows.append(
                Comparison(
                    f"stage{k} peak I",
                    float(result.peak_current[k]),
                    m.peak_current,
                    "A",
                )
            )
    exit_m = next(
        (mm.exit_velocity for mm in measurements.values() if mm.exit_velocity),
        None,
    )
    if exit_m is not None and result is not None:
        rows.append(
            Comparison("exit velocity", result.exit_velocity, exit_m, "m/s")
        )
    return rows


def comparison_table(rows: list[Comparison]):
    from prettytable import PrettyTable

    t = PrettyTable()
    t.field_names = ["Quantity", "Predicted", "Measured", "Unit", "Error %"]
    t.align = "r"
    t.align["Quantity"] = "l"
    for r in rows:
        t.add_row(
            [
                r.quantity,
                f"{r.predicted:.6g}",
                f"{r.measured:.6g}",
                r.unit,
                f"{r.error_pct:+.1f}",
            ]
        )
    return t
