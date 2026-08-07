"""Physical geometry of the coil and projectile, and the quantities derived
directly from it.

This module is deliberately free of any dynamic behaviour: everything here is a
static consequence of the dimensions and materials. Magnetic behaviour lives in
`magnetics`, circuit behaviour in `circuit`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .wire import WireSpec, T_REF_C

MU_0 = 4.0e-7 * math.pi  # H/m


@dataclass(frozen=True)
class ProjectileSpec:
    """A cylindrical ferromagnetic slug.

    v1 treated the projectile as a dimensionless point with a hardcoded mass,
    which made the coil-length-to-projectile-length ratio -- the entire design
    premise -- impossible to vary. Here length is a first-class parameter and
    mass is derived from it.
    """

    length: float  # m
    diameter: float  # m
    density: float = 7850.0  # kg/m^3, mild/carbon steel
    mu_r: float = 100.0  # bulk relative permeability
    b_sat: float = 1.6  # T, saturation flux density

    def __post_init__(self) -> None:
        for name in ("length", "diameter", "density", "mu_r", "b_sat"):
            if getattr(self, name) <= 0:
                raise ValueError(f"projectile {name} must be positive")
        if self.mu_r < 1.0:
            raise ValueError("mu_r must be >= 1")

    @property
    def area(self) -> float:
        """Cross-sectional area (m^2)."""
        return math.pi * (self.diameter / 2.0) ** 2

    @property
    def volume(self) -> float:
        return self.area * self.length

    @property
    def mass(self) -> float:
        """Derived, not hardcoded (v1 pinned 0.00569 kg in four places)."""
        return self.volume * self.density

    @property
    def aspect_ratio(self) -> float:
        """Length / diameter. Drives the demagnetising factor."""
        return self.length / self.diameter


@dataclass(frozen=True)
class CoilGeometry:
    """A multilayer solenoid wound on a cylindrical former.

    `bore_diameter` is the outer diameter of the barrel/former the wire is wound
    onto -- i.e. the inner diameter of the winding, not the projectile clearance.
    """

    length: float  # m, axial
    bore_diameter: float  # m, winding inner diameter
    turns: int
    wire: WireSpec

    def __post_init__(self) -> None:
        if self.length <= 0 or self.bore_diameter <= 0:
            raise ValueError("coil length and bore diameter must be positive")
        if self.turns < 1:
            raise ValueError("coil must have at least one turn")
        if self.turns_per_layer < 1:
            raise ValueError(
                f"coil length {self.length * 1e3:.2f} mm cannot fit even one turn "
                f"of {self.wire} ({self.wire.outer_diameter * 1e3:.3f} mm)"
            )

    # -- winding layout -------------------------------------------------

    @property
    def turns_per_layer(self) -> int:
        """Close-packed turns per layer, using the *insulated* diameter."""
        return int(self.length // self.wire.outer_diameter)

    @property
    def layers(self) -> int:
        return math.ceil(self.turns / self.turns_per_layer)

    @property
    def winding_depth(self) -> float:
        """Radial thickness of the winding (m)."""
        return self.layers * self.wire.outer_diameter

    @property
    def inner_radius(self) -> float:
        return self.bore_diameter / 2.0

    @property
    def outer_radius(self) -> float:
        return self.inner_radius + self.winding_depth

    @property
    def mean_radius(self) -> float:
        """Mean winding radius -- the `a` term in Wheeler's formula."""
        return self.inner_radius + self.winding_depth / 2.0

    @property
    def bore_area(self) -> float:
        return math.pi * self.inner_radius**2

    @property
    def wire_length(self) -> float:
        """Total conductor length (m).

        Each turn sits at the *centre* of its layer, so layer k is at radius
        r_inner + (k + 1/2) * d. v1 placed layer k at r_inner + k*d, i.e. on the
        layer's inner surface, losing half a wire diameter per layer.
        """
        per_layer = self.turns_per_layer
        total = 0.0
        remaining = self.turns
        layer = 0
        while remaining > 0:
            n = min(per_layer, remaining)
            radius = self.inner_radius + (layer + 0.5) * self.wire.outer_diameter
            total += n * 2.0 * math.pi * radius
            remaining -= n
            layer += 1
        return total

    # -- electrical -----------------------------------------------------

    def resistance(self, temperature_c: float = T_REF_C) -> float:
        """Coil DC resistance (ohm) at temperature."""
        return self.wire.resistance(self.wire_length, temperature_c)

    @property
    def inductance_air(self) -> float:
        """Air-core inductance (H), Wheeler's multilayer approximation.

            L[uH] = 31.6 * N^2 * a^2 / (6a + 9b + 10c)

        with a = mean radius, b = coil length, c = winding depth, all in metres.
        Accurate to a few percent for a/b ratios typical of coilgun coils.
        """
        a = self.mean_radius
        b = self.length
        c = self.winding_depth
        l_uh = 31.6 * self.turns**2 * a**2 / (6.0 * a + 9.0 * b + 10.0 * c)
        return l_uh * 1e-6

    def clearance(self, projectile: ProjectileSpec) -> float:
        """Radial gap between bore and projectile (m).

        Reported for build sanity only. It is deliberately *not* used in any
        force calculation -- v1's central error was substituting this radial
        clearance into an axial-gap reluctance formula.
        """
        return self.inner_radius - projectile.diameter / 2.0
