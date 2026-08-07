"""Multi-stage linear accelerator (coilgun) design tool.

Ballpark simulator for chained solenoid accelerator stages. See PLAN.md for the
physics model and the rationale behind it.
"""

from .config import (
    CapacitorBank,
    ControlConfig,
    SimConfig,
    StageConfig,
    SwitchSpec,
    ThermalConfig,
    uniform_stages,
)
from .geometry import CoilGeometry, ProjectileSpec
from .wire import WireSpec

__all__ = [
    "CapacitorBank",
    "CoilGeometry",
    "ControlConfig",
    "ProjectileSpec",
    "SimConfig",
    "StageConfig",
    "SwitchSpec",
    "ThermalConfig",
    "WireSpec",
    "uniform_stages",
]
