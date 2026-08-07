"""Magnetic model: flux linkage, incremental inductance, and force.

Because saturation is modelled, the magnetic system is nonlinear and is
described by flux linkage lambda(x, i) rather than by an inductance L(x). Three
distinct quantities follow, and using the wrong one in the wrong place is the
classic error in this kind of model:

    coefficient on di/dt   ->  incremental inductance  L_inc = d(lambda)/di
    back-EMF               ->  d(lambda)/dx * v
    force                  ->  co-energy gradient      dW'/dx

In the unsaturated limit all three collapse to L(x) and 0.5*i^2*dL/dx. That
identity is asserted in the tests.

Model
-----
    fill(x)     fraction of the coil's magnetic path occupied by iron
    lambda(x,i) = L_air*i + (mu_eff-1)*L_air*fill(x)*i_sat*tanh(i/i_sat)

The air-core term stays linear (air does not saturate); only the slug's
contribution rolls off. Every partial is closed-form, so nothing is
differentiated numerically inside the integrator's inner loop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .geometry import MU_0, CoilGeometry, ProjectileSpec


def demagnetising_factor(aspect_ratio: float) -> float:
    """Axial demagnetising factor N_d for a prolate spheroid of the given
    length/diameter ratio.

    A slug in an air-cored coil sits in an open magnetic circuit, so its own
    poles oppose the applied field. This is why bulk mu_r badly overstates the
    inductance change on insertion -- v1 used mu_r = 100 directly and thereby
    made L swing by 100x.

    The commonly quoted asymptotic form (1/m^2)*(ln(2m) - 1) is 20% low at the
    aspect ratios of interest here (and 81% low at m = 1.5), so the exact
    spheroid expression is used. A real cylinder differs somewhat from a
    spheroid of the same aspect ratio; measured L_slug_in should override this
    once bench data exists.
    """
    m = float(aspect_ratio)
    if m <= 0:
        raise ValueError("aspect ratio must be positive")
    if abs(m - 1.0) < 1e-3:
        return 1.0 / 3.0  # sphere
    if m > 1.0:
        s = math.sqrt(m * m - 1.0)
        return (1.0 / (m * m - 1.0)) * ((m / s) * math.log(m + s) - 1.0)
    # oblate (disc-like); not expected for a projectile, supported for completeness
    e = math.sqrt(1.0 - m * m)
    return (1.0 / (1.0 - m * m)) * (1.0 - (m / e) * math.asin(e))


def effective_permeability(mu_r: float, aspect_ratio: float) -> float:
    """Apparent permeability of a rod in an open magnetic circuit.

        mu_eff = mu_r / (1 + N_d*(mu_r - 1))

    For a 17.5 x 6 mm carbon steel slug (mu_r = 100) this gives ~8.2, not 100.
    """
    nd = demagnetising_factor(aspect_ratio)
    return mu_r / (1.0 + nd * (mu_r - 1.0))


def _softplus(v: np.ndarray | float, w: float) -> np.ndarray | float:
    """Smooth ramp max(0, v), rounded over a width w. Numerically stable."""
    return w * np.logaddexp(0.0, np.asarray(v, dtype=float) / w)


def _sigmoid(v: np.ndarray | float, w: float) -> np.ndarray | float:
    """d/dv of _softplus."""
    return 1.0 / (1.0 + np.exp(-np.asarray(v, dtype=float) / w))


@dataclass
class MagneticModel:
    """Flux linkage and force for one coil / projectile pair.

    Position `x` is the projectile's *leading edge* (nose), in absolute
    simulation coordinates; `coil_position` is the x of the coil mouth. Nose
    coordinates are used because that is what an ingress sensor detects.

    Calibration scale factors default to 1.0 and multiply the modelled values,
    so an empty measurements/ directory reproduces uncalibrated behaviour
    exactly.
    """

    coil: CoilGeometry
    projectile: ProjectileSpec
    coil_position: float = 0.0
    saturation: bool = True
    #  Fringing width. The transition is not physically sharp: the field
    #  extends beyond the coil mouth over a scale set by the bore radius, so
    #  the slug begins to feel force before it arrives. Smoothing the corners
    #  is therefore physical as well as convenient for the integrator.
    fringe_width: float | None = None
    l_air_scale: float = 1.0
    mu_eff_scale: float = 1.0
    mu_eff_override: float | None = None
    #  How much of the coil's flux the slug actually intercepts.
    #  "bore" assumes every turn couples to the slug as well as the innermost
    #  one, which over-rewards deep multilayer windings: an outer layer at
    #  40 mm diameter has flux paths that largely bypass a 6 mm slug.
    #  "mean" references the mean winding radius instead, penalising depth.
    #  The truth lies between; running both bounds the answer.
    coupling: str = "bore"

    def __post_init__(self) -> None:
        self.l_air = self.coil.inductance_air * self.l_air_scale
        if self.mu_eff_override is not None:
            self.mu_eff = self.mu_eff_override
        else:
            self.mu_eff = (
                effective_permeability(
                    self.projectile.mu_r, self.projectile.aspect_ratio
                )
                * self.mu_eff_scale
            )
        if self.fringe_width is None:
            self.fringe_width = 0.5 * self.coil.inner_radius
        if self.coupling not in ("bore", "mean"):
            raise ValueError(f"unknown coupling {self.coupling!r}")
        # Fraction of the coil's cross-section the slug occupies.
        if self.coupling == "bore":
            reference_area = self.coil.bore_area
        else:
            reference_area = math.pi * self.coil.mean_radius**2
        self.area_ratio = self.projectile.area / reference_area
        if self.projectile.area > self.coil.bore_area:
            raise ValueError("projectile diameter exceeds coil bore")
        self.i_sat = self._saturation_current()

    def _saturation_current(self) -> float:
        """Coil current at which the slug reaches B_sat.

            H = N*i/l,   B = mu_0*mu_eff*H  =>  i_sat = B_sat*l/(mu_0*mu_eff*N)
        """
        return (
            self.projectile.b_sat
            * self.coil.length
            / (MU_0 * self.mu_eff * self.coil.turns)
        )

    # -- geometry -------------------------------------------------------

    def fill(self, x):
        """Fraction of the coil's magnetic path filled by iron, in [0, 1].

        The slug spans [x - Lp, x]; the coil spans [xc, xc + Lc]. Their overlap,
        written as a difference of four ramps so that it is smooth:

            overlap(u) = R(u) - R(u-Lp) - R(u-Lc) + R(u-Lp-Lc),  u = x - xc
        """
        u = np.asarray(x, dtype=float) - self.coil_position
        lp = self.projectile.length
        lc = self.coil.length
        w = self.fringe_width
        overlap = (
            _softplus(u, w)
            - _softplus(u - lp, w)
            - _softplus(u - lc, w)
            + _softplus(u - lp - lc, w)
        )
        return (overlap / lc) * self.area_ratio

    def dfill_dx(self, x):
        """d(fill)/dx. Positive on entry, ~zero when fully enclosed, negative on
        exit -- which is what produces suck-back without any special-casing."""
        u = np.asarray(x, dtype=float) - self.coil_position
        lp = self.projectile.length
        lc = self.coil.length
        w = self.fringe_width
        d_overlap = (
            _sigmoid(u, w)
            - _sigmoid(u - lp, w)
            - _sigmoid(u - lc, w)
            + _sigmoid(u - lp - lc, w)
        )
        return (d_overlap / lc) * self.area_ratio

    # -- magnetics ------------------------------------------------------

    def inductance(self, x):
        """Apparent (small-signal, unsaturated) inductance L(x).

        Reporting and the linear model only; the integrator uses `l_incremental`.
        """
        return self.l_air * (1.0 + (self.mu_eff - 1.0) * self.fill(x))

    def _tanh_terms(self, i):
        """(tanh(u), u) with u = i/i_sat, or the linear equivalents."""
        u = np.asarray(i, dtype=float) / self.i_sat
        return np.tanh(u), u

    def flux_linkage(self, x, i):
        """lambda(x, i), in Wb-turns."""
        i = np.asarray(i, dtype=float)
        if not self.saturation:
            return self.inductance(x) * i
        t, _ = self._tanh_terms(i)
        return self.l_air * i + (self.mu_eff - 1.0) * self.l_air * self.fill(
            x
        ) * self.i_sat * t

    def l_incremental(self, x, i):
        """L_inc = d(lambda)/di -- the coefficient on di/dt in the circuit.

        Falls to l_air deep in saturation: the slug stops contributing to
        incremental flux once its iron is fully used up.
        """
        if not self.saturation:
            return self.inductance(x)
        t, _ = self._tanh_terms(i)
        sech2 = 1.0 - t * t  # stable; sech^2 = 1 - tanh^2
        return self.l_air * (1.0 + (self.mu_eff - 1.0) * self.fill(x) * sech2)

    def dlambda_dx(self, x, i):
        """d(lambda)/dx -- the motional back-EMF coefficient.

        This term is the mechanical work extraction. v1 omitted it entirely,
        which is why its energy books never balanced.
        """
        i = np.asarray(i, dtype=float)
        if not self.saturation:
            return self.l_air * (self.mu_eff - 1.0) * self.dfill_dx(x) * i
        t, _ = self._tanh_terms(i)
        return (
            (self.mu_eff - 1.0)
            * self.l_air
            * self.dfill_dx(x)
            * self.i_sat
            * t
        )

    def force(self, x, i):
        """Axial force on the projectile (N), from the co-energy gradient.

            W'(x,i) = 0.5*L_air*i^2
                    + (mu_eff-1)*L_air*fill(x)*i_sat^2*ln(cosh(i/i_sat))
            F       = dW'/dx

        Positive is forward. Negative values past coil centre are suck-back and
        are physical -- v1 hard-zeroed them, which is why it could not test the
        design premise.
        """
        i = np.asarray(i, dtype=float)
        if not self.saturation:
            return 0.5 * i * i * self.l_air * (self.mu_eff - 1.0) * self.dfill_dx(x)
        _, u = self._tanh_terms(i)
        # ln(cosh u) computed stably: |u| - ln2 + log1p(exp(-2|u|))
        au = np.abs(u)
        ln_cosh = au - math.log(2.0) + np.log1p(np.exp(-2.0 * au))
        return (
            (self.mu_eff - 1.0)
            * self.l_air
            * self.dfill_dx(x)
            * self.i_sat**2
            * ln_cosh
        )

    # -- diagnostics ----------------------------------------------------

    def saturation_ratio(self, i):
        """|i| / i_sat. Above ~1 the slug is saturated and extra current buys
        progressively less force."""
        return np.abs(np.asarray(i, dtype=float)) / self.i_sat

    def summary(self) -> dict[str, float]:
        return {
            "L_air_H": self.l_air,
            "mu_eff": self.mu_eff,
            "N_d": demagnetising_factor(self.projectile.aspect_ratio),
            "i_sat_A": self.i_sat,
            "area_ratio": self.area_ratio,
            "fringe_width_m": self.fringe_width,
            "max_fill": float(
                self.fill(self.coil_position + self.coil.length / 2.0)
            ),
        }
