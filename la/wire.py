"""Magnet wire properties.

Two diameters matter and v1 conflated them:

  - the *bare conductor* diameter sets the cross-sectional area, and therefore
    resistance and the Onderdonk fusing/heating current
  - the *insulated* outer diameter sets how tightly turns pack, and therefore
    turns-per-layer, layer count, winding depth, and inductance

Using bare diameter for packing (as v1 did) overstates turns per layer and
understates winding depth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Annealed copper at 20 C.
RHO_CU_20C = 1.724e-8  # ohm.m
ALPHA_CU = 0.00393  # 1/C, temperature coefficient of resistance
T_REF_C = 20.0

# m^2 -> circular mils. 1 m^2 = 1.5500e9 sq mil; circular mils = sq mil * 4/pi.
M2_TO_CIRCULAR_MILS = 1.9735252e9

# Insulation build, as the *increase* in diameter over bare, in mm. Nominal
# values from the NEMA MW 1000 tables; real wire varies by a percent or two
# between suppliers. These feed packing geometry only, never conductor area.
#
# Heavy build is roughly twice the film thickness of single build, as expected.
#
# "single" (Grade 1) is the default because it reproduces the values noted in
# the original sim.py (26 AWG -> 0.43 mm, 20 AWG -> 0.85 mm).
_BUILD_INCREASE_MM = {
    #    single  heavy
    18: (0.036, 0.073),
    20: (0.036, 0.067),
    22: (0.031, 0.057),
    24: (0.030, 0.049),
    26: (0.022, 0.042),
    28: (0.022, 0.037),
    30: (0.019, 0.032),
}

BUILDS = ("bare", "single", "heavy")
_BUILD_INDEX = {"single": 0, "heavy": 1}


def awg_bare_diameter(gauge: int) -> float:
    """Exact AWG definition, in metres. d = 0.127 mm * 92^((36-n)/39).

    Computed rather than tabulated: v1's hand-entered table was off by ~0.3% at
    22, 26 and 30 AWG, which is ~0.6% on resistance since R scales as 1/d^2.
    """
    return 0.127e-3 * math.pow(92.0, (36 - gauge) / 39.0)


@dataclass(frozen=True)
class WireSpec:
    """A gauge of magnet wire at a given insulation build."""

    gauge: int
    build: str = "single"

    def __post_init__(self) -> None:
        if self.gauge not in _BUILD_INCREASE_MM:
            raise ValueError(
                f"unknown gauge {self.gauge} AWG; "
                f"known: {sorted(_BUILD_INCREASE_MM)}"
            )
        if self.build not in BUILDS:
            raise ValueError(f"unknown build {self.build!r}; known: {BUILDS}")

    @property
    def bare_diameter(self) -> float:
        """Conductor diameter (m). Sets resistance."""
        return awg_bare_diameter(self.gauge)

    @property
    def outer_diameter(self) -> float:
        """Insulated diameter (m). Sets packing.

        'bare' is available for reproducing v1's geometry, not for design work.
        """
        if self.build == "bare":
            return self.bare_diameter
        increase = _BUILD_INCREASE_MM[self.gauge][_BUILD_INDEX[self.build]]
        return self.bare_diameter + increase * 1e-3

    @property
    def area(self) -> float:
        """Conductor cross-section (m^2)."""
        return math.pi * (self.bare_diameter / 2.0) ** 2

    @property
    def area_circular_mils(self) -> float:
        """Conductor cross-section in circular mils, for Onderdonk."""
        return self.area * M2_TO_CIRCULAR_MILS

    def resistance(self, length: float, temperature_c: float = T_REF_C) -> float:
        """DC resistance (ohm) of `length` metres at `temperature_c`.

        v1 held resistivity at 20 C while predicting 60 C coils -- roughly 16%
        of error in the quantity that sets circuit damping.
        """
        r20 = RHO_CU_20C * length / self.area
        return r20 * (1.0 + ALPHA_CU * (temperature_c - T_REF_C))

    def __str__(self) -> str:
        return f"{self.gauge}AWG/{self.build}"


def available_gauges() -> tuple[int, ...]:
    return tuple(sorted(_BUILD_INCREASE_MM))
