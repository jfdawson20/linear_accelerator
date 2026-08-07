"""The vectorised kernel must agree with the scalar reference implementation.

`la/magnetics.py` and `la/circuit.py` are the readable statement of the physics;
`la/kernel.py` is a performance rewrite of the same equations. An optimisation
that changes answers is a bug, so this pins them together.
"""

from __future__ import annotations

import numpy as np
import pytest

from la.circuit import StageCircuit, SwitchState
from la.config import CapacitorBank, ControlConfig, SimConfig, uniform_stages
from la.engine import Simulation
from la.geometry import CoilGeometry, ProjectileSpec
from la.kernel import FREEWHEEL, OFF, ON, StageKernel, _ln_cosh, _sigmoid
from la.wire import WireSpec


def make_sim(stages: int = 3, **kw) -> Simulation:
    coil = CoilGeometry(0.035, 0.0098, 150, WireSpec(26, "single"))
    proj = ProjectileSpec(length=0.0175, diameter=0.006)
    params = dict(
        projectile=proj,
        stages=uniform_stages(coil, CapacitorBank(0.006, 200.0), stages, 0.0188),
        control=ControlConfig(prefire=True),
        dt=1e-5,
    )
    params.update(kw)
    return Simulation(SimConfig(**params))


# -- helpers ------------------------------------------------------------


def test_sigmoid_helper_is_the_logistic_function():
    z = np.linspace(-40, 40, 401)
    expected = 1.0 / (1.0 + np.exp(-np.clip(z, -700, 700)))
    assert np.allclose(_sigmoid(z), expected, atol=1e-12)


def test_ln_cosh_is_stable_at_extremes():
    assert _ln_cosh(np.array([0.0]))[0] == pytest.approx(0.0, abs=1e-12)
    big = _ln_cosh(np.array([800.0]))[0]
    assert np.isfinite(big)
    assert big == pytest.approx(800.0 - np.log(2.0), rel=1e-9)


# -- geometry and magnetics vs the reference ---------------------------


@pytest.mark.parametrize("saturation", [True, False])
def test_kernel_geometry_matches_magnetics_module(saturation):
    sim = make_sim(saturation=saturation)
    for x in (-0.01, 0.0, 0.008, 0.0175, 0.0263, 0.05, 0.12):
        fill, dfill = sim.kernel.geometry(x)
        for k, circ in enumerate(sim.circuits):
            assert fill[k] == pytest.approx(float(circ.magnetics.fill(x)), rel=1e-12)
            assert dfill[k] == pytest.approx(
                float(circ.magnetics.dfill_dx(x)), rel=1e-12
            )


@pytest.mark.parametrize("saturation", [True, False])
def test_kernel_magnetics_match_reference(saturation):
    sim = make_sim(saturation=saturation)
    for x in (0.004, 0.0175, 0.03, 0.06):
        for current in (0.5, 40.0, 270.0, 2000.0):
            i = np.full(sim.n, current)
            fill, dfill = sim.kernel.geometry(x)
            l_inc, dlam, force = sim.kernel.magnetics(fill, dfill, i)
            for k, circ in enumerate(sim.circuits):
                m = circ.magnetics
                assert l_inc[k] == pytest.approx(
                    float(m.l_incremental(x, current)), rel=1e-11
                )
                assert dlam[k] == pytest.approx(
                    float(m.dlambda_dx(x, current)), rel=1e-11
                )
                assert force[k] == pytest.approx(
                    float(m.force(x, current)), rel=1e-11
                )


@pytest.mark.parametrize("saturation", [True, False])
def test_kernel_stored_energy_matches_reference(saturation):
    sim = make_sim(saturation=saturation)
    x, current = 0.012, 180.0
    got = sim.kernel.stored_magnetic_energy(x, np.full(sim.n, current))
    for k, circ in enumerate(sim.circuits):
        assert got[k] == pytest.approx(
            circ.stored_magnetic_energy(x, current), rel=1e-11
        )


# -- circuit derivatives vs the reference -------------------------------


@pytest.mark.parametrize(
    "codes,states",
    [
        ([ON, ON, ON], [SwitchState.ON] * 3),
        ([OFF, OFF, OFF], [SwitchState.OFF] * 3),
        ([FREEWHEEL] * 3, [SwitchState.FREEWHEEL] * 3),
        ([ON, FREEWHEEL, OFF], [SwitchState.ON, SwitchState.FREEWHEEL, SwitchState.OFF]),
    ],
)
def test_kernel_derivatives_match_reference_in_every_switch_state(codes, states):
    sim = make_sim(3)
    x, v = 0.010, 25.0
    i = np.array([210.0, -30.0, 5.0])
    vc = np.array([180.0, 60.0, 200.0])
    temp = np.array([30.0, 45.0, 25.0])

    di, dvc, dtemp, force, ext = sim.kernel.derivatives(
        x, v, i, vc, temp, np.array(codes)
    )

    exp_force = 0.0
    exp_ext = 0.0
    for k, circ in enumerate(sim.circuits):
        r_di, r_dvc, r_dtemp, r_ext = circ.derivatives(
            i[k], vc[k], temp[k], x, v, states[k]
        )
        assert di[k] == pytest.approx(r_di, rel=1e-11)
        assert dvc[k] == pytest.approx(r_dvc, rel=1e-11)
        assert dtemp[k] == pytest.approx(r_dtemp, rel=1e-11)
        exp_ext += r_ext
        exp_force += float(circ.magnetics.force(x, i[k]))

    assert ext == pytest.approx(exp_ext, rel=1e-11)
    assert force == pytest.approx(exp_force, rel=1e-11)


def test_off_stage_contributes_nothing():
    sim = make_sim(3)
    di, dvc, dtemp, _, ext = sim.kernel.derivatives(
        0.01, 10.0, np.zeros(3), np.full(3, 200.0), np.full(3, 25.0),
        np.array([OFF, OFF, OFF]),
    )
    assert np.allclose(di, 0.0)
    assert np.allclose(dvc, 0.0)
    assert np.allclose(dtemp, 0.0)
    assert ext == 0.0


# -- end to end ---------------------------------------------------------


def test_record_false_gives_the_same_answer_as_record_true():
    """The summary is maintained during the run, so dropping the trace must
    not change any reported number."""
    full = make_sim(4, record=True).run()
    lean = make_sim(4, record=False).run()
    assert lean.exit_velocity == pytest.approx(full.exit_velocity, rel=1e-12)
    assert lean.energy.efficiency == pytest.approx(full.energy.efficiency, rel=1e-12)
    assert np.allclose(lean.peak_current, full.peak_current, rtol=1e-12)
    assert np.allclose(lean.peak_temperature, full.peak_temperature, rtol=1e-12)
    assert np.allclose(
        lean.suck_back_impulse(), full.suck_back_impulse(), rtol=1e-12, atol=1e-15
    )
    assert np.allclose(lean.forward_impulse(), full.forward_impulse(), rtol=1e-12)


def test_summary_impulse_matches_the_recorded_trace():
    """The running summary and a trapezoidal sum over the trace should agree."""
    r = make_sim(3, record=True).run()
    dt = r.config.dt
    from_trace = np.where(r.force > 0, r.force, 0.0).sum(axis=0) * dt
    assert np.allclose(r.forward_impulse(), from_trace, rtol=1e-12)


def test_recorded_switch_states_distinguish_freewheel():
    """The trace records the actual switch code, so freewheel is visible rather
    than collapsed into 'on'."""
    r = make_sim(2, record=True).run()
    seen = set(np.unique(r.switch_state))
    assert {OFF, ON} <= seen
    assert FREEWHEEL in seen
