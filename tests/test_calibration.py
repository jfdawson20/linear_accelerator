"""Calibration plumbing, and a cross-check of the thermal model.

The thermal test matters because v1 used Onderdonk's equation directly and this
rewrite replaced it with straightforward adiabatic heating against a
temperature-dependent resistance. Those should be the same physics, and this
asserts that they are.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

from la.calibration import (
    Corrections,
    Measurement,
    comparison_table,
    compare,
    derive_corrections,
    load_measurements,
    mu_eff_from_measurements,
)
from la.circuit import CU_DENSITY, CU_SPECIFIC_HEAT, StageCircuit
from la.config import CapacitorBank, SimConfig, uniform_stages
from la.engine import Simulation
from la.geometry import CoilGeometry, ProjectileSpec
from la.wire import ALPHA_CU, RHO_CU_20C, M2_TO_CIRCULAR_MILS, WireSpec


def make_config(stages: int = 1) -> SimConfig:
    coil = CoilGeometry(0.035, 0.0098, 150, WireSpec(26, "single"))
    proj = ProjectileSpec(length=0.0175, diameter=0.006)
    return SimConfig(
        projectile=proj,
        stages=uniform_stages(coil, CapacitorBank(0.006, 200.0), stages, 0.0188),
        dt=2e-6,
    )


# -- thermal model vs Onderdonk ----------------------------------------


def test_adiabatic_heating_reproduces_onderdonk():
    """Onderdonk's equation is the adiabatic solution for a conductor whose
    resistance rises with temperature. Integrating

        dT/dt = i^2 * R(T) / (m * c)

    analytically gives ln((234+Tf)/(234+Ti)) = i^2 * rho * alpha * t / (A^2 d c),
    which must match Onderdonk's

        log10((234+Tf)/(234+Ti)) = I^2 * t / (0.0297 * A_cmil^2)

    Agreement to ~1% is expected; the residual is rounding in the 0.0297
    constant.
    """
    mine = RHO_CU_20C * ALPHA_CU / (CU_DENSITY * CU_SPECIFIC_HEAT)
    onderdonk = math.log(10.0) / (0.0297 * M2_TO_CIRCULAR_MILS**2)
    assert mine == pytest.approx(onderdonk, rel=0.02)


def test_simulated_temperature_rise_matches_the_closed_form():
    """Drive a stage at constant current and compare against the analytic
    adiabatic solution."""
    cfg = make_config(1)
    circ = StageCircuit(config=cfg.stages[0], projectile=cfg.projectile)

    i, dt, steps = 200.0, 1e-6, 20000
    temp = 25.0
    for _ in range(steps):
        temp += dt * (i * i * circ.coil_resistance(temp)) / circ.thermal_mass

    t_total = dt * steps
    k = ALPHA_CU * i * i * circ.r20 / circ.thermal_mass
    u0 = 1.0 + ALPHA_CU * (25.0 - 20.0)
    expected = 20.0 + (u0 * math.exp(k * t_total) - 1.0) / ALPHA_CU
    assert temp == pytest.approx(expected, rel=1e-3)


def test_hotter_windings_dissipate_more():
    """Resistance rises with temperature, so heating accelerates. v1 held
    resistivity at 20 C throughout."""
    cfg = make_config(1)
    circ = StageCircuit(config=cfg.stages[0], projectile=cfg.projectile)
    assert circ.coil_resistance(80.0) > circ.coil_resistance(20.0)


# -- corrections --------------------------------------------------------


def test_absent_measurements_are_the_identity():
    """The tool must work correctly before any prototype exists."""
    c = Corrections()
    assert c.is_identity
    assert c.l_air_scale == 1.0 and c.r_scale == 1.0


def test_no_measurements_directory_is_not_an_error():
    assert load_measurements("definitely/not/a/real/path") == {}


def test_corrections_scale_towards_the_measurement():
    m = Measurement(coil_id="stage0", l_air=60e-6, r_dc=0.70)
    c = derive_corrections(m, predicted_l_air=56.75e-6, predicted_r=0.67)
    assert c.l_air_scale == pytest.approx(60e-6 / 56.75e-6)
    assert c.r_scale == pytest.approx(0.70 / 0.67)
    assert not c.is_identity


def test_partial_measurements_only_correct_what_was_measured():
    m = Measurement(coil_id="stage0", r_dc=0.70)  # no inductance measured
    c = derive_corrections(m, predicted_l_air=56.75e-6, predicted_r=0.67)
    assert c.l_air_scale == 1.0
    assert c.r_scale != 1.0


def test_corrections_applied_to_a_circuit_move_it_to_the_measurement():
    cfg = make_config(1)
    base = StageCircuit(config=cfg.stages[0], projectile=cfg.projectile)
    target_l, target_r = 62e-6, 0.72
    c = derive_corrections(
        Measurement("stage0", l_air=target_l, r_dc=target_r),
        base.magnetics.l_air,
        base.r20,
    )
    tuned = StageCircuit(
        config=cfg.stages[0],
        projectile=cfg.projectile,
        l_air_scale=c.l_air_scale,
        r_scale=c.r_scale,
    )
    assert tuned.magnetics.l_air == pytest.approx(target_l)
    assert tuned.r20 == pytest.approx(target_r)


# -- recovering mu_eff from the bench ----------------------------------


def test_mu_eff_recovered_from_two_inductance_readings():
    """Round-trip: predict L_slug_in from a known mu_eff, then recover it.

    This is the measurement worth taking first -- it settles the demagnetising
    factor, which is the largest remaining uncertainty in the model.
    """
    sim = Simulation(make_config(1))
    mag = sim.circuits[0].magnetics
    max_fill = mag.summary()["max_fill"]
    l_air = mag.l_air
    true_mu = mag.mu_eff
    l_slug_in = l_air * (1.0 + (true_mu - 1.0) * max_fill)
    assert mu_eff_from_measurements(l_air, l_slug_in, max_fill) == pytest.approx(
        true_mu, rel=1e-9
    )


def test_recovered_mu_eff_flags_a_disagreeing_bench():
    """A measured L_slug_in below prediction implies a lower mu_eff."""
    sim = Simulation(make_config(1))
    mag = sim.circuits[0].magnetics
    max_fill = mag.summary()["max_fill"]
    modelled = float(mag.inductance(0.02625))
    recovered = mu_eff_from_measurements(mag.l_air, modelled * 0.8, max_fill)
    assert recovered < mag.mu_eff


# -- files and comparison ----------------------------------------------


def test_loads_yaml_measurements(tmp_path):
    (tmp_path / "stage0.yaml").write_text(
        "coil_id: stage0\n"
        "measured:\n"
        "  L_air: 58.2e-6\n"
        "  R_dc: 0.671\n"
        "  exit_velocity: 41.3\n"
        "conditions:\n"
        "  V0: 200\n"
    )
    loaded = load_measurements(str(tmp_path))
    assert set(loaded) == {"stage0"}
    m = loaded["stage0"]
    assert m.l_air == pytest.approx(58.2e-6)
    assert m.r_dc == pytest.approx(0.671)
    assert m.peak_current is None
    assert m.conditions["V0"] == 200


def test_empty_measurement_file_is_skipped(tmp_path):
    (tmp_path / "blank.yaml").write_text("")
    assert load_measurements(str(tmp_path)) == {}


def test_comparison_reports_signed_error(tmp_path):
    sim = Simulation(make_config(1))
    result = sim.run()
    measurements = {
        "stage0": Measurement("stage0", l_air=sim.circuits[0].magnetics.l_air * 1.1)
    }
    rows = compare(sim, result, measurements)
    assert len(rows) == 1
    assert rows[0].error_pct == pytest.approx(-100 * 0.1 / 1.1, rel=1e-6)
    assert comparison_table(rows) is not None


def test_comparison_is_empty_without_measurements():
    sim = Simulation(make_config(1))
    assert compare(sim, sim.run(), {}) == []


def test_current_trace_round_trip(tmp_path):
    path = tmp_path / "trace.csv"
    t = np.linspace(0, 1e-3, 50)
    i = 200 * np.sin(np.pi * t / 1e-3)
    np.savetxt(path, np.column_stack([t, i]), delimiter=",", header="t,i")
    m = Measurement("stage0", current_trace=str(path))
    loaded = m.load_current_trace()
    assert loaded is not None
    assert np.allclose(loaded[1], i, atol=1e-9)


def test_missing_trace_returns_none():
    assert Measurement("stage0", current_trace="nope.csv").load_current_trace() is None
