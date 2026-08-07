"""Integration-level tests: energy closure, convergence, and switch behaviour.

Energy closure is the load-bearing one. It is a property no individual formula
guarantees -- it only holds if the circuit, the back-EMF, the force and the
thermal model are mutually consistent. v1 had no such check and reported 26%
efficiency without flagging it.
"""

from __future__ import annotations

import numpy as np
import pytest

from la.circuit import StageCircuit, SwitchState
from la.config import (
    CapacitorBank,
    ControlConfig,
    SimConfig,
    SwitchSpec,
    uniform_stages,
)
from la.engine import Simulation
from la.geometry import CoilGeometry, ProjectileSpec
from la.wire import WireSpec


def make_config(stages: int = 2, **kw) -> SimConfig:
    coil = CoilGeometry(0.035, 0.0098, 150, WireSpec(26, "single"))
    proj = ProjectileSpec(length=0.0175, diameter=0.006)
    bank = CapacitorBank(capacitance=0.006, voltage=200.0)
    params = dict(
        projectile=proj,
        stages=uniform_stages(coil, bank, stages, 0.0188),
        control=ControlConfig(prefire=True),
        dt=1e-6,
    )
    params.update(kw)
    return SimConfig(**params)


# -- energy -------------------------------------------------------------


@pytest.mark.parametrize("saturation", [True, False])
def test_energy_closes(saturation):
    """Every joule out of the capacitors must land somewhere accountable."""
    result = Simulation(make_config(2, saturation=saturation, dt=2e-6)).run()
    assert result.energy.closure_error < 1e-3


def test_energy_closes_across_many_stages():
    result = Simulation(make_config(8, dt=2e-6)).run()
    assert result.energy.closure_error < 1e-3


def test_efficiency_is_physically_plausible():
    """Real coilguns manage a few percent. v1 reported 26%."""
    result = Simulation(make_config(8, dt=2e-6)).run()
    assert 0.0 < result.energy.efficiency < 0.10


def test_kinetic_energy_matches_velocity():
    result = Simulation(make_config(2)).run()
    m = result.config.projectile.mass
    assert result.energy.kinetic == pytest.approx(
        0.5 * m * result.exit_velocity**2, rel=1e-6
    )


def test_impulse_matches_momentum():
    """Sum of impulses over all stages must equal final momentum."""
    result = Simulation(make_config(4, dt=2e-6)).run()
    impulse = (result.forward_impulse() + result.suck_back_impulse()).sum()
    momentum = result.config.projectile.mass * result.exit_velocity
    assert impulse == pytest.approx(momentum, rel=0.02)


# -- convergence --------------------------------------------------------

def test_halving_dt_barely_changes_the_answer():
    coarse = Simulation(make_config(2, dt=2e-6)).run()
    fine = Simulation(make_config(2, dt=1e-6)).run()
    rel = abs(fine.exit_velocity - coarse.exit_velocity) / coarse.exit_velocity
    assert rel < 0.005


def test_coarse_timestep_is_flagged():
    sim = Simulation(make_config(1, dt=5e-5))
    assert sim.check_timestep()


# -- saturation ---------------------------------------------------------


def test_saturation_reduces_predicted_performance():
    """The whole reason saturation is in scope: without it the tool flatters
    the design. Here it is worth ~40% of exit velocity."""
    sat = Simulation(make_config(4, saturation=True, dt=2e-6)).run()
    lin = Simulation(make_config(4, saturation=False, dt=2e-6)).run()
    assert sat.exit_velocity < lin.exit_velocity
    assert sat.exit_velocity / lin.exit_velocity < 0.8


# -- switching ----------------------------------------------------------


def test_current_decays_rather_than_stopping_instantly():
    """v1 forced current to zero in one timestep on turn-off. Real current
    decays through the freewheel diode on L/R, and that decay is the interval
    the 2:1 geometry exists to accommodate."""
    result = Simulation(make_config(1)).run()
    i = result.current[:, 0]
    peak = np.argmax(np.abs(i))
    tail = np.abs(i[peak:])
    # never drops by more than a few percent of peak in a single step
    steps = np.abs(np.diff(tail))
    assert steps.max() < 0.05 * np.abs(i).max()


def test_stage_conducts_then_freewheels_then_stops():
    """The trace records the actual switch state (0 off, 1 on, 2 freewheel), so
    the whole sequence is visible rather than collapsed into on/off."""
    from la.kernel import FREEWHEEL, OFF, ON

    result = Simulation(make_config(1)).run()
    states = list(result.switch_state[:, 0])
    assert ON in states
    assert FREEWHEEL in states
    assert states[-1] == OFF
    # and the order is on -> freewheel -> off, never freewheel before on
    assert states.index(ON) < states.index(FREEWHEEL)


def test_gate_turns_off_when_projectile_is_fully_inside():
    """The design premise: shutoff when the tail clears the ingress sensor."""
    sim = Simulation(make_config(1))
    sim.run()
    ctrl = sim.controllers[0]
    assert ctrl.release_position == pytest.approx(0.0175)
    assert ctrl.off_time is not None and ctrl.off_time > ctrl.fire_time


def test_late_turn_off_produces_suck_back_and_costs_velocity():
    """Falsifiability check: the model must be able to show the failure mode
    the design is built to avoid. v1 could not -- it hard-zeroed force past
    coil centre."""
    good = Simulation(make_config(1))
    good_result = good.run()

    bad = Simulation(make_config(1))
    bad.controllers[0].release_travel *= 3.0  # hold the coil on far too long
    bad_result = bad.run()

    assert bad_result.suck_back_impulse()[0] < -1e-3
    assert bad_result.exit_velocity < good_result.exit_velocity


def test_force_reverses_at_slug_centre_not_coil_centre():
    """With an extended slug the axis of symmetry is where the slug's centre
    meets the coil's centre, i.e. nose = (Lc + Lp)/2 -- not the coil midpoint.

    This matters: it is the real deadline the field has to collapse before.
    """
    sim = Simulation(make_config(1))
    mag = sim.circuits[0].magnetics
    lc, lp = 0.035, 0.0175
    axis = (lc + lp) / 2
    assert float(mag.force(axis, 200.0)) == pytest.approx(0.0, abs=1e-9)
    assert float(mag.force(axis - 0.003, 200.0)) > 0
    assert float(mag.force(axis + 0.003, 200.0)) < 0


# -- thermal ------------------------------------------------------------


def test_winding_heat_matches_temperature_rise():
    """The audit's winding term and the temperature state must agree."""
    sim = Simulation(make_config(2))
    result = sim.run()
    expected = sum(
        c.thermal_mass * (result.temperature[-1, k] - result.config.thermal.ambient_c)
        for k, c in enumerate(sim.circuits)
    )
    assert result.energy.winding_heat == pytest.approx(expected, rel=1e-9)


def test_temperature_only_rises():
    """v1 integrated heating on a clock that reset to zero at every switch
    event, which drove the temperature toward -234 C."""
    result = Simulation(make_config(2)).run()
    for k in range(result.config.num_stages):
        assert np.all(np.diff(result.temperature[:, k]) >= -1e-12)
        assert result.temperature[0, k] == pytest.approx(25.0)


# -- geometry-driven termination ---------------------------------------


def test_run_length_follows_the_machine_not_a_constant():
    """v1 ran to a hardcoded 1 m regardless of how long the machine was."""
    short = Simulation(make_config(1)).run()
    long = Simulation(make_config(6, dt=2e-6)).run()
    assert short.position[-1] < long.position[-1]
    assert short.terminated == "cleared_barrel"


def test_zero_voltage_run_terminates_and_does_not_hang():
    """No force anywhere must not spin forever."""
    coil = CoilGeometry(0.035, 0.0098, 150, WireSpec(26, "single"))
    proj = ProjectileSpec(length=0.0175, diameter=0.006)
    cfg = SimConfig(
        projectile=proj,
        stages=uniform_stages(coil, CapacitorBank(0.006, 0.0), 2, 0.0188),
        dt=1e-6,
    )
    result = Simulation(cfg).run(max_time=0.05)
    assert result.terminated == "stalled"
    assert result.exit_velocity == pytest.approx(0.0)


# -- circuit details ----------------------------------------------------


def _scan_peak_time(circ, cap) -> float:
    """Brute-force time of peak current for a constant-L series RLC, picking
    the branch that actually applies."""
    L, C = circ.magnetics.l_air, cap
    R = circ.r20 + circ.config.switch.on_resistance
    alpha, w0 = R / (2 * L), 1 / np.sqrt(L * C)
    ts = np.linspace(0, 5e-3, 500001)
    if alpha < w0:
        wd = np.sqrt(w0**2 - alpha**2)
        i = np.exp(-alpha * ts) * np.sin(wd * ts)
    else:
        beta = np.sqrt(alpha**2 - w0**2)
        i = np.exp((-alpha + beta) * ts) - np.exp((-alpha - beta) * ts)
    return float(ts[np.argmax(i)])


def test_time_to_peak_current_matches_a_direct_scan_overdamped():
    """The default design is overdamped: 6 mF against 57 uH gives
    alpha/w0 = 3.4, so the current rises and falls slowly rather than ringing.
    """
    cfg = make_config(1)
    circ = StageCircuit(config=cfg.stages[0], projectile=cfg.projectile)
    L, C = circ.magnetics.l_air, cfg.stages[0].bank.capacitance
    R = circ.r20 + cfg.stages[0].switch.on_resistance
    assert R / (2 * L) > 1 / np.sqrt(L * C)  # confirm the regime
    assert circ.time_to_peak_current() == pytest.approx(
        _scan_peak_time(circ, C), rel=1e-3
    )


def test_time_to_peak_current_matches_a_direct_scan_underdamped():
    """A smaller capacitor puts the same coil into the underdamped branch."""
    cfg = make_config(1)
    stage = cfg.stages[0]
    small = CapacitorBank(capacitance=200e-6, voltage=200.0)
    circ = StageCircuit(
        config=type(stage)(
            coil=stage.coil, position=0.0, bank=small, switch=stage.switch
        ),
        projectile=cfg.projectile,
    )
    L = circ.magnetics.l_air
    R = circ.r20 + stage.switch.on_resistance
    assert R / (2 * L) < 1 / np.sqrt(L * small.capacitance)
    assert circ.time_to_peak_current() == pytest.approx(
        _scan_peak_time(circ, small.capacitance), rel=1e-3
    )


def test_thyristor_hands_current_to_the_diode_on_gate_off():
    cfg = make_config(1)
    circ = StageCircuit(config=cfg.stages[0], projectile=cfg.projectile)
    nxt = circ.next_switch_state(SwitchState.ON, gate_on=False, i=100.0,
                                 conducted=True)
    assert nxt is SwitchState.FREEWHEEL


def test_freewheel_opens_only_once_current_has_decayed():
    cfg = make_config(1)
    circ = StageCircuit(config=cfg.stages[0], projectile=cfg.projectile)
    assert (
        circ.next_switch_state(SwitchState.FREEWHEEL, False, 50.0, True)
        is SwitchState.FREEWHEEL
    )
    assert (
        circ.next_switch_state(SwitchState.FREEWHEEL, False, 0.01, True)
        is SwitchState.OFF
    )


def test_off_switch_carries_no_current():
    cfg = make_config(1)
    circ = StageCircuit(config=cfg.stages[0], projectile=cfg.projectile)
    d = circ.derivatives(0.0, 200.0, 25.0, 0.0, 0.0, SwitchState.OFF)
    assert d == (0.0, 0.0, 0.0, 0.0)


def test_back_emf_opposes_motion():
    """Moving into a rising-inductance region must oppose the current that
    creates the force. This term is the mechanical work extraction; v1 omitted
    it entirely."""
    cfg = make_config(1)
    circ = StageCircuit(config=cfg.stages[0], projectile=cfg.projectile)
    x, i = 0.008, 150.0
    still = circ.derivatives(i, 200.0, 25.0, x, 0.0, SwitchState.ON)[0]
    moving = circ.derivatives(i, 200.0, 25.0, x, 40.0, SwitchState.ON)[0]
    assert moving < still
