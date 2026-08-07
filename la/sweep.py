"""Parameter sweeps.

v1's `Optimize` crashed on the first improvement it found (a missing `self.`
and a list indexed by a string), swept a profile library whose ten members were
identical, and mutated the library in place as it went. This replaces it.

Three things make sweeps affordable:

  - `record=False`, so the per-step trace is never allocated; the running
    summary carries everything a sweep needs
  - a coarser timestep. 10 us costs 0.19% on exit velocity against 1 us and
    runs 12x faster; `check_timestep()` still guards against going too far
  - multiprocessing across grid points, which is embarrassingly parallel

Every configuration is built from scratch by `build_config`, so there is no
shared mutable state between points -- the failure mode that made v1's
optimiser unusable.
"""

from __future__ import annotations

import itertools
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .config import (
    CapacitorBank,
    ControlConfig,
    SimConfig,
    SwitchSpec,
    ThermalConfig,
    uniform_stages,
)
from .engine import Simulation
from .geometry import CoilGeometry, ProjectileSpec
from .wire import WireSpec

# Defaults mirror the CLI so a sweep and a single run agree when the swept
# parameter happens to sit at its default.
DEFAULTS: dict[str, Any] = {
    "coil_length": 0.035,
    "turns": 150,
    "gauge": 26,
    "build": "single",
    "bore": 0.0098,
    "ratio": 2.0,  # coil length / projectile length
    "proj_len": None,  # set explicitly, or derived from ratio
    "proj_dia": 0.006,
    "density": 7850.0,
    "mu_r": 100.0,
    "b_sat": 1.6,
    "stages": 4,
    "spacing": 0.0188,
    "capacitance": 0.006,
    "voltage": 200.0,
    "diode_vf": 1.5,
    "topology": "diode",      # "diode" | "ahb" (asymmetric half bridge)
    "device_drop": 0.0,       # V per conducting device
    "prefire": True,
    "prefire_scale": 1.0,
    "turn_off_fraction": None,  # None = tail clears sensor; else fraction of (Lc+Lp)/2
    "sensor_latency": 0.0,
    "switch_latency": 0.0,
    "saturation": True,
    "coupling": "bore",
    "flux_return": 0.0,
    "l_shell_factor": 1.0,
    "dt": 1e-5,
    "ambient": 25.0,
    "max_temp": 60.0,
}


def build_config(**overrides: Any) -> SimConfig:
    """Build a SimConfig from named parameters, filling in DEFAULTS.

    Projectile length follows `ratio` unless given explicitly, so sweeping the
    coil-to-projectile ratio is a first-class operation -- it is the design
    premise, and v1 could not vary it at all.
    """
    p = dict(DEFAULTS)
    unknown = set(overrides) - set(p)
    if unknown:
        raise ValueError(f"unknown sweep parameters: {sorted(unknown)}")
    p.update(overrides)

    coil = CoilGeometry(
        length=p["coil_length"],
        bore_diameter=p["bore"],
        turns=int(p["turns"]),
        wire=WireSpec(gauge=int(p["gauge"]), build=p["build"]),
    )
    proj_len = p["proj_len"] if p["proj_len"] else p["coil_length"] / p["ratio"]
    projectile = ProjectileSpec(
        length=proj_len,
        diameter=p["proj_dia"],
        density=p["density"],
        mu_r=p["mu_r"],
        b_sat=p["b_sat"],
    )
    switch = SwitchSpec(
        diode_vf=p["diode_vf"],
        turn_on_latency=p["switch_latency"],
        turn_off_latency=p["switch_latency"],
        topology=p["topology"],
        device_drop=p["device_drop"],
    )
    stages = uniform_stages(
        coil=coil,
        bank=CapacitorBank(capacitance=p["capacitance"], voltage=p["voltage"]),
        count=int(p["stages"]),
        spacing=p["spacing"],
        switch=switch,
    )
    return SimConfig(
        projectile=projectile,
        stages=stages,
        control=ControlConfig(
            prefire=p["prefire"],
            sensor_latency=p["sensor_latency"],
            turn_off_fraction=p["turn_off_fraction"],
            prefire_scale=p["prefire_scale"],
        ),
        thermal=ThermalConfig(ambient_c=p["ambient"], max_c=p["max_temp"]),
        dt=p["dt"],
        saturation=p["saturation"],
        coupling=p["coupling"],
        flux_return=p["flux_return"],
        l_shell_factor=p["l_shell_factor"],
        record=False,
    )


@dataclass
class SweepResult:
    """Outcome of one grid point. Scalars only, so it pickles cheaply."""

    params: dict[str, Any]
    exit_velocity: float = 0.0
    muzzle_energy: float = 0.0
    efficiency: float = 0.0
    stored_energy: float = 0.0
    peak_current: float = 0.0
    saturation_ratio: float = 0.0
    peak_temperature: float = 0.0
    suck_back_pct: float = 0.0
    closure_error: float = 0.0
    terminated: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def within_thermal_limit(self) -> bool:
        return self.peak_temperature <= self.params.get("max_temp", 60.0)

    @property
    def velocity_per_joule(self) -> float:
        return self.exit_velocity / self.stored_energy if self.stored_energy else 0.0


def evaluate(params: Mapping[str, Any]) -> SweepResult:
    """Run one configuration. Module level so it survives pickling."""
    params = dict(params)
    try:
        config = build_config(**params)
        sim = Simulation(config)
        result = sim.run()
    except Exception as exc:  # a grid point may be geometrically impossible
        return SweepResult(params=params, error=f"{type(exc).__name__}: {exc}")

    fwd = float(result.forward_impulse().sum())
    back = float(result.suck_back_impulse().sum())
    peak = float(result.peak_current.max())
    i_sat = min(c.magnetics.i_sat for c in sim.circuits)
    return SweepResult(
        params=params,
        exit_velocity=result.exit_velocity,
        muzzle_energy=result.energy.kinetic,
        efficiency=result.energy.efficiency,
        stored_energy=result.energy.initial,
        peak_current=peak,
        saturation_ratio=peak / i_sat if i_sat else 0.0,
        peak_temperature=float(result.peak_temperature.max()),
        suck_back_pct=100.0 * abs(back) / fwd if fwd > 0 else 0.0,
        closure_error=result.energy.closure_error,
        terminated=result.terminated,
    )


@dataclass
class ParameterSpace:
    """A grid of parameter values. Anything not named keeps its default."""

    axes: dict[str, Sequence[Any]] = field(default_factory=dict)

    def __init__(self, **axes: Sequence[Any]) -> None:
        unknown = set(axes) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown sweep parameters: {sorted(unknown)}")
        self.axes = {k: list(v) for k, v in axes.items()}

    def __len__(self) -> int:
        n = 1
        for values in self.axes.values():
            n *= len(values)
        return n

    def points(self) -> Iterable[dict[str, Any]]:
        if not self.axes:
            yield {}
            return
        names = list(self.axes)
        for combo in itertools.product(*(self.axes[n] for n in names)):
            yield dict(zip(names, combo))


def sweep(
    space: ParameterSpace,
    fixed: Mapping[str, Any] | None = None,
    workers: int | None = None,
    progress: bool = True,
) -> list[SweepResult]:
    """Evaluate every point in `space`, in parallel.

    `fixed` applies to every point (e.g. stages=4, dt=1e-5).
    """
    fixed = dict(fixed or {})
    points = [{**fixed, **p} for p in space.points()]
    if workers is None:
        workers = max(1, (os.cpu_count() or 2))

    results: list[SweepResult] = []
    if workers == 1:
        for n, p in enumerate(points, 1):
            results.append(evaluate(p))
            if progress:
                _tick(n, len(points))
        return results

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for n, r in enumerate(pool.map(evaluate, points, chunksize=1), 1):
            results.append(r)
            if progress:
                _tick(n, len(points))
    if progress:
        print()
    return results


def _tick(n: int, total: int) -> None:
    print(f"\r  {n}/{total} ({100 * n / total:.0f}%)", end="", flush=True)


# -- analysis -----------------------------------------------------------

OBJECTIVES: dict[str, Callable[[SweepResult], float]] = {
    "velocity": lambda r: r.exit_velocity,
    "efficiency": lambda r: r.efficiency,
    "velocity_per_joule": lambda r: r.velocity_per_joule,
    "muzzle_energy": lambda r: r.muzzle_energy,
}


def rank(
    results: Sequence[SweepResult],
    objective: str = "velocity",
    thermal_limit: bool = True,
    max_suck_back_pct: float | None = None,
    max_current: float | None = None,
) -> list[SweepResult]:
    """Sort viable results best-first.

    Constraints matter as much as the objective: a configuration that cooks its
    windings or exceeds the switch rating is not a design, and v1 had no way to
    express either.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}; {sorted(OBJECTIVES)}")
    key = OBJECTIVES[objective]
    viable = [r for r in results if r.ok]
    if thermal_limit:
        viable = [r for r in viable if r.within_thermal_limit]
    if max_current is not None:
        viable = [r for r in viable if r.peak_current <= max_current]
    if max_suck_back_pct is not None:
        viable = [r for r in viable if r.suck_back_pct <= max_suck_back_pct]
    return sorted(viable, key=key, reverse=True)


def results_table(
    results: Sequence[SweepResult], axes: Sequence[str], limit: int = 20
):
    from prettytable import PrettyTable

    t = PrettyTable()
    t.field_names = [
        *axes,
        "exit v",
        "eff %",
        "J stored",
        "peak I",
        "I/Isat",
        "maxT",
        "back %",
    ]
    t.align = "r"
    for r in results[:limit]:
        t.add_row(
            [
                *[_fmt(r.params.get(a)) for a in axes],
                f"{r.exit_velocity:.2f}",
                f"{r.efficiency * 100:.2f}",
                f"{r.stored_energy:.0f}",
                f"{r.peak_current:.0f}",
                f"{r.saturation_ratio:.1f}",
                f"{r.peak_temperature:.0f}",
                f"{r.suck_back_pct:.2f}",
            ]
        )
    return t


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def sensitivity(
    results: Sequence[SweepResult], axis: str, objective: str = "velocity"
) -> list[tuple[Any, float, float]]:
    """(value, best, mean) of the objective for each value of one axis.

    Shows which parameters actually move the outcome, which is more useful than
    a single winning configuration.
    """
    key = OBJECTIVES[objective]
    buckets: dict[Any, list[float]] = {}
    for r in results:
        if r.ok:
            buckets.setdefault(r.params.get(axis), []).append(key(r))
    return [
        (v, max(vals), float(np.mean(vals)))
        for v, vals in sorted(buckets.items(), key=lambda kv: str(kv[0]))
    ]
