"""Tests for the magnetic model.

The important ones here are the invariants: the linear limit, the numerical
consistency of the analytic partials, and the sign of the force on exit. Those
are what catch a sign or factor-of-two error in the co-energy derivation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from la.geometry import CoilGeometry, ProjectileSpec
from la.magnetics import (
    MagneticModel,
    demagnetising_factor,
    effective_permeability,
)
from la.wire import WireSpec


def make_model(**kw) -> MagneticModel:
    coil = CoilGeometry(
        length=0.035, bore_diameter=0.0098, turns=150, wire=WireSpec(26, "single")
    )
    proj = ProjectileSpec(length=0.0175, diameter=0.006)
    return MagneticModel(coil=coil, projectile=proj, **kw)


# -- demagnetising factor -----------------------------------------------


def test_sphere_limit_is_one_third():
    assert demagnetising_factor(1.0) == pytest.approx(1 / 3)
    assert demagnetising_factor(1.0001) == pytest.approx(1 / 3, rel=1e-3)


def test_demag_factor_decreases_with_elongation():
    values = [demagnetising_factor(m) for m in (1.0, 2.0, 4.0, 10.0, 50.0)]
    assert values == sorted(values, reverse=True)
    assert values[-1] < 0.01  # long thin rod barely demagnetises


def test_demag_factor_matches_asymptotic_form_when_long():
    """The (1/m^2)(ln 2m - 1) approximation is only good for large m."""
    for m in (20.0, 50.0):
        approx = (1 / m**2) * (math.log(2 * m) - 1)
        assert demagnetising_factor(m) == pytest.approx(approx, rel=0.02)


def test_effective_permeability_is_far_below_bulk():
    """The headline correction to v1: mu_eff ~ 8, not mu_r = 100."""
    assert effective_permeability(100.0, 2.9167) == pytest.approx(8.24, rel=0.01)
    # and it saturates against mu_r: a huge mu_r cannot help an open circuit
    assert effective_permeability(10000.0, 2.9167) < 10.0


def test_oblate_branch_is_continuous_with_sphere():
    assert demagnetising_factor(0.999) == pytest.approx(1 / 3, rel=1e-2)


# -- fill geometry ------------------------------------------------------


def test_fill_is_zero_far_outside_and_peaks_inside():
    m = make_model()
    assert m.fill(-0.05) == pytest.approx(0.0, abs=1e-6)
    assert m.fill(0.20) == pytest.approx(0.0, abs=1e-6)
    assert m.fill(0.0175) > 0.1


def test_fill_is_symmetric_about_coil_centre():
    """The slug's midpoint passing the coil's midpoint is the axis of symmetry.

    Nose position x means the slug centre is at x - Lp/2, so symmetry is about
    a nose position of coil_centre + Lp/2.
    """
    m = make_model()
    lc, lp = m.coil.length, m.projectile.length
    axis = lc / 2 + lp / 2
    for d in (0.002, 0.006, 0.012, 0.02):
        assert m.fill(axis - d) == pytest.approx(m.fill(axis + d), rel=1e-6)


def test_max_fill_matches_the_two_to_one_geometry():
    """With Lp = Lc/2 the slug can occupy at most half the coil's length, and it
    fills (6/9.8)^2 of the bore area."""
    m = make_model()
    expected = 0.5 * (0.006**2 / 0.0098**2)
    assert m.fill(0.0175 + 0.0088) == pytest.approx(expected, rel=0.05)


def test_dfill_dx_matches_numerical_derivative():
    m = make_model()
    xs = np.linspace(-0.02, 0.08, 400)
    h = 1e-7
    numeric = (m.fill(xs + h) - m.fill(xs - h)) / (2 * h)
    assert np.allclose(m.dfill_dx(xs), numeric, atol=1e-4)


def test_dfill_dx_changes_sign_on_exit():
    """This is suck-back. v1 could not represent it at all."""
    m = make_model()
    assert m.dfill_dx(0.005) > 0  # entering
    assert m.dfill_dx(0.045) < 0  # leaving


def test_fully_enclosed_slug_feels_almost_no_gradient():
    """A slug wholly inside a solenoid sits in a uniform field, so there is no
    axial gradient to push on it."""
    m = make_model()
    lc, lp = m.coil.length, m.projectile.length
    centre_nose = lc / 2 + lp / 2
    assert abs(m.dfill_dx(centre_nose)) < 0.1 * abs(m.dfill_dx(0.005))


# -- analytic partials --------------------------------------------------


@pytest.mark.parametrize("saturation", [True, False])
def test_l_incremental_matches_d_lambda_di(saturation):
    m = make_model(saturation=saturation)
    x = 0.012
    for i in (1.0, 20.0, 100.0, 400.0):
        h = i * 1e-6
        numeric = (m.flux_linkage(x, i + h) - m.flux_linkage(x, i - h)) / (2 * h)
        assert m.l_incremental(x, i) == pytest.approx(numeric, rel=1e-4)


@pytest.mark.parametrize("saturation", [True, False])
def test_dlambda_dx_matches_numerical_derivative(saturation):
    m = make_model(saturation=saturation)
    i = 150.0
    for x in (0.004, 0.012, 0.030, 0.044):
        h = 1e-7
        numeric = (m.flux_linkage(x + h, i) - m.flux_linkage(x - h, i)) / (2 * h)
        assert m.dlambda_dx(x, i) == pytest.approx(numeric, rel=1e-3)


@pytest.mark.parametrize("saturation", [True, False])
def test_force_matches_coenergy_gradient(saturation):
    """F = dW'/dx with W' obtained by numerically integrating lambda di.

    This is the strongest check on the force expression: it goes back to the
    definition rather than to the closed form.
    """
    m = make_model(saturation=saturation)
    i = 200.0

    def coenergy(x):
        grid = np.linspace(0.0, i, 4001)
        return float(np.trapezoid(m.flux_linkage(x, grid), grid))

    for x in (0.004, 0.012, 0.030, 0.044):
        h = 1e-6
        numeric = (coenergy(x + h) - coenergy(x - h)) / (2 * h)
        assert m.force(x, i) == pytest.approx(numeric, rel=1e-3)


# -- the linear limit ---------------------------------------------------


def test_saturating_model_reduces_to_linear_at_low_current():
    """As i << i_sat, ln(cosh u) -> u^2/2 and F -> 0.5*i^2*dL/dx.

    This identity is why --no-saturation is a special case of the model rather
    than a separate code path.
    """
    sat = make_model(saturation=True)
    lin = make_model(saturation=False)
    i = 0.01 * sat.i_sat
    for x in (0.004, 0.012, 0.030, 0.044):
        assert sat.force(x, i) == pytest.approx(lin.force(x, i), rel=1e-3)
        assert sat.flux_linkage(x, i) == pytest.approx(
            lin.flux_linkage(x, i), rel=1e-3
        )
        assert sat.l_incremental(x, i) == pytest.approx(
            lin.l_incremental(x, i), rel=1e-3
        )


def test_linear_force_equals_half_i_squared_dL_dx():
    """Explicit check against the textbook expression."""
    lin = make_model(saturation=False)
    i = 100.0
    x = 0.008
    h = 1e-7
    dl_dx = (lin.inductance(x + h) - lin.inductance(x - h)) / (2 * h)
    assert lin.force(x, i) == pytest.approx(0.5 * i * i * dl_dx, rel=1e-4)


# -- saturation behaviour -----------------------------------------------


def test_force_is_quadratic_below_saturation_and_linear_above():
    """Two regimes, and the crossover is the whole point of modelling saturation.

    Below i_sat the slug's magnetisation tracks the applied field, so both the
    moment and the gradient scale with i and F ~ i^2. Above it the moment is
    pinned at M_sat, so F = m.grad(B) scales only with the gradient: F ~ i.

    v1 had F ~ i^2 without bound, so more current always looked better.
    """
    m = make_model()
    x = 0.008

    lo = m.force(x, 0.1 * m.i_sat)
    mid = m.force(x, 0.2 * m.i_sat)
    assert mid / lo == pytest.approx(4.0, rel=0.05)  # quadratic

    hi = m.force(x, 10 * m.i_sat)
    vhi = m.force(x, 20 * m.i_sat)
    assert vhi / hi == pytest.approx(2.0, rel=0.05)  # linear


def test_doubling_current_buys_far_less_when_saturated():
    """The design consequence, stated directly."""
    m = make_model()
    x = 0.008
    below = m.force(x, 2 * m.i_sat) / m.force(x, m.i_sat)
    above = m.force(x, 16 * m.i_sat) / m.force(x, 8 * m.i_sat)
    assert below > above
    assert above < 2.2


def test_incremental_inductance_falls_to_air_core_when_saturated():
    m = make_model()
    x = 0.012
    assert m.l_incremental(x, 0.01 * m.i_sat) > 1.5 * m.l_air
    assert m.l_incremental(x, 50 * m.i_sat) == pytest.approx(m.l_air, rel=1e-3)


def test_saturation_current_is_low_for_this_design():
    """At 150 turns and mu_eff ~ 8 the slug saturates around 35 A, so v1's
    280 A operating point was roughly an order of magnitude into saturation."""
    m = make_model()
    assert 20.0 < m.i_sat < 60.0
    assert m.saturation_ratio(280.0) > 5.0


def test_force_is_finite_at_extreme_current():
    m = make_model()
    assert np.isfinite(m.force(0.008, 1e6))
    assert np.isfinite(m.flux_linkage(0.008, 1e6))
    assert np.isfinite(m.l_incremental(0.008, 1e6))


def test_force_sign_follows_fill_gradient():
    m = make_model()
    assert m.force(0.005, 200.0) > 0
    assert m.force(0.045, 200.0) < 0
    # force is even in current: reversing polarity still pulls the slug in
    assert m.force(0.005, -200.0) == pytest.approx(m.force(0.005, 200.0))


def test_measured_mu_eff_overrides_the_model():
    """Calibration path: an LCR measurement should win over the spheroid
    approximation."""
    m = make_model(mu_eff_override=6.0)
    assert m.mu_eff == 6.0
    assert m.summary()["mu_eff"] == 6.0


def test_projectile_wider_than_bore_is_rejected():
    coil = CoilGeometry(0.035, 0.0098, 150, WireSpec(26, "single"))
    proj = ProjectileSpec(length=0.0175, diameter=0.02)
    with pytest.raises(ValueError, match="exceeds coil bore"):
        MagneticModel(coil=coil, projectile=proj)
