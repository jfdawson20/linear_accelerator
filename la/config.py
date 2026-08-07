"""Configuration objects for a simulation run.

v1 passed configuration around as string-keyed dicts, and built its stage list
with `tmp = cfg` -- an alias, not a copy -- so every stage shared one dict and
the caller's profile was mutated in place. Frozen dataclasses remove that whole
class of bug: sharing is safe because nothing can be mutated.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

from .geometry import CoilGeometry, ProjectileSpec
from .wire import WireSpec


@dataclass(frozen=True)
class CapacitorBank:
    capacitance: float  # F
    voltage: float  # V, initial charge

    def __post_init__(self) -> None:
        if self.capacitance <= 0:
            raise ValueError("capacitance must be positive")

    @property
    def stored_energy(self) -> float:
        """0.5*C*V^2 (J)."""
        return 0.5 * self.capacitance * self.voltage**2


@dataclass(frozen=True)
class SwitchSpec:
    """Switching topology and its losses.

    `diode_vf` and the latencies are what make coil turn-off a physical event
    rather than an instantaneous one. v1 modelled turn-off by resetting the
    coil's clock to zero, which forced current to zero in a single timestep.

    Two topologies:

    "diode" -- a series switch with a freewheel diode across the coil. On
        turn-off the coil sees only -diode_vf, so the current decays on L/R and
        the stored field energy is dissipated in the winding.

    "ahb" -- an asymmetric half bridge: a switch above and below the coil, both
        gated together, with diodes returning current to the bank. On turn-off
        the coil sees -(V_cap + 2*diode_vf), which is two orders of magnitude
        more reverse volts, and the field energy is recovered into the
        capacitor rather than burnt. The cost is two device drops in the
        conduction path instead of one, plus isolated high-side gate drive.
    """

    on_resistance: float = 0.00014  # ohm
    diode_vf: float = 1.5  # V, diode forward drop
    turn_on_latency: float = 0.0  # s, gate command -> conduction
    turn_off_latency: float = 0.0  # s
    off_current_threshold: float = 0.5  # A, below this the return path opens
    topology: str = "diode"  # "diode" | "ahb"
    device_drop: float = 0.0  # V, per-device saturation drop in conduction

    def __post_init__(self) -> None:
        if self.topology not in ("diode", "ahb"):
            raise ValueError(
                f"unknown topology {self.topology!r}; use 'diode' or 'ahb'"
            )

    @property
    def conduction_devices(self) -> int:
        """Devices in series with the coil while conducting."""
        return 2 if self.topology == "ahb" else 1

    @property
    def recovers_energy(self) -> bool:
        return self.topology == "ahb"


@dataclass(frozen=True)
class ThermalConfig:
    ambient_c: float = 25.0
    max_c: float = 60.0
    max_current: float = 450.0  # A, design limit used for on-time budgeting


@dataclass(frozen=True)
class ControlConfig:
    """Trigger logic for a stage.

    The design under test: an ingress sensor at the coil mouth fires the coil,
    and the coil shuts off once the projectile has travelled one projectile
    length past it -- which, with a coil twice the projectile length, is short
    of coil centre.
    """

    prefire: bool = True
    sensor_latency: float = 0.0  # s, detection -> gate command
    sensor_offset: float = 0.0  # m, sensor position relative to coil mouth
    # Turn-off as a fraction of the distance to force reversal, (Lc+Lp)/2.
    # None keeps the original rule: turn off once the tail clears the sensor.
    turn_off_fraction: float | None = None


@dataclass(frozen=True)
class StageConfig:
    coil: CoilGeometry
    position: float  # m, x of the coil mouth
    bank: CapacitorBank
    switch: SwitchSpec = SwitchSpec()

    @property
    def centre(self) -> float:
        return self.position + self.coil.length / 2.0

    @property
    def end(self) -> float:
        return self.position + self.coil.length


@dataclass(frozen=True)
class SimConfig:
    projectile: ProjectileSpec
    stages: tuple[StageConfig, ...]
    control: ControlConfig = ControlConfig()
    thermal: ThermalConfig = ThermalConfig()
    dt: float = 1e-6  # s
    saturation: bool = True
    record: bool = True  # False -> summary statistics only, no per-step trace

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("at least one stage is required")
        if self.dt <= 0:
            raise ValueError("dt must be positive")

    @property
    def num_stages(self) -> int:
        return len(self.stages)

    @property
    def barrel_length(self) -> float:
        """x at which the last coil ends."""
        return self.stages[-1].end

    @property
    def total_stored_energy(self) -> float:
        return sum(s.bank.stored_energy for s in self.stages)

    @property
    def coil_to_projectile_ratio(self) -> float:
        """The design premise, as a number. Intended to be 2.0."""
        return self.stages[0].coil.length / self.projectile.length


def uniform_stages(
    coil: CoilGeometry,
    bank: CapacitorBank,
    count: int,
    spacing: float,
    switch: SwitchSpec | None = None,
    first_position: float = 0.0,
) -> tuple[StageConfig, ...]:
    """Build `count` identical stages, `spacing` metres of gap between them.

    Positions follow v1's convention: stage i sits at i*(spacing + coil_length),
    so `spacing` is the gap from one coil's end to the next coil's mouth.
    """
    if count < 1:
        raise ValueError("count must be >= 1")
    sw = switch if switch is not None else SwitchSpec()
    pitch = spacing + coil.length
    return tuple(
        StageConfig(
            coil=coil,
            position=first_position + i * pitch,
            bank=bank,
            switch=sw,
        )
        for i in range(count)
    )


def with_coil(stage: StageConfig, coil: CoilGeometry) -> StageConfig:
    """Return a copy of `stage` with a different coil. Used by sweeps."""
    return replace(stage, coil=coil)


# -- serialisation ------------------------------------------------------


def coil_from_dict(d: Mapping[str, Any]) -> CoilGeometry:
    return CoilGeometry(
        length=float(d["length"]),
        bore_diameter=float(d["bore_diameter"]),
        turns=int(d["turns"]),
        wire=WireSpec(gauge=int(d["gauge"]), build=str(d.get("build", "single"))),
    )


def projectile_from_dict(d: Mapping[str, Any]) -> ProjectileSpec:
    return ProjectileSpec(
        length=float(d["length"]),
        diameter=float(d["diameter"]),
        density=float(d.get("density", 7850.0)),
        mu_r=float(d.get("mu_r", 100.0)),
        b_sat=float(d.get("b_sat", 1.6)),
    )


def from_dict(d: Mapping[str, Any]) -> SimConfig:
    """Build a SimConfig from a plain mapping (YAML/JSON friendly).

    Expects a uniform stage layout; heterogeneous stages are constructed
    directly in Python for now.
    """
    coil = coil_from_dict(d["coil"])
    bank = CapacitorBank(
        capacitance=float(d["bank"]["capacitance"]),
        voltage=float(d["bank"]["voltage"]),
    )
    switch = SwitchSpec(**d.get("switch", {}))
    stages = uniform_stages(
        coil=coil,
        bank=bank,
        count=int(d["stages"]["count"]),
        spacing=float(d["stages"]["spacing"]),
        switch=switch,
    )
    return SimConfig(
        projectile=projectile_from_dict(d["projectile"]),
        stages=stages,
        control=ControlConfig(**d.get("control", {})),
        thermal=ThermalConfig(**d.get("thermal", {})),
        dt=float(d.get("dt", 1e-6)),
        saturation=bool(d.get("saturation", True)),
    )


def load(path: str) -> SimConfig:
    """Load a SimConfig from a YAML or JSON file."""
    import json

    with open(path) as fh:
        text = fh.read()
    if path.endswith((".yaml", ".yml")):
        import yaml

        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    return from_dict(data)
