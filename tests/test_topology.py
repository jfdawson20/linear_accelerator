"""Asymmetric half bridge vs series switch + freewheel diode.

The load-bearing test is energy closure under regeneration. With the half
bridge, capacitor energy goes *up* during turn-off as the field collapses back
into the bank, so the audit exercises a sign it never sees with a plain
freewheel diode. If the recovery term were wrong, closure would break.
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
from la.kernel import FREEWHEEL, ON
from la.wire import WireSpec


def make_config(topology="diode", stages=1, device_drop=0.0, **kw) -> SimConfig:
    coil = CoilGeometry(0.0508, 0.009, 400, WireSpec(22, "single"))
    proj = ProjectileSpec(length=0.0254, diameter=0.00635)
    switch = SwitchSpec(topology=topology, device_drop=device_drop)
    params = dict(
        projectile=proj,
        stages=uniform_stages(
            coil, CapacitorBank(0.004, 300.0), stages, 0.015, switch=switch
        ),
        control=ControlConfig(prefire=True),
        dt=5e-6,
    )
    params.update(kw)
    return SimConfig(**params)


# -- validity -----------------------------------------------------------


def test_unknown_topology_rejected():
    with pytest.raises(ValueError, match="unknown topology"):
        SwitchSpec(topology="h-bridge")


def test_topology_reports_its_device_count_and_recovery():
    assert SwitchSpec(topology="diode").conduction_devices == 1
    assert SwitchSpec(topology="ahb").conduction_devices == 2
    assert not SwitchSpec(topology="diode").recovers_energy
    assert SwitchSpec(topology="ahb").recovers_energy


# -- energy -------------------------------------------------------------


@pytest.mark.parametrize("topology", ["diode", "ahb"])
def test_energy_closes_for_both_topologies(topology):
    r = Simulation(make_config(topology, stages=2)).run()
    assert r.energy.closure_error < 1e-3


def test_regeneration_returns_energy_to_the_bank():
    """The whole point of the half bridge: field energy goes back into the
    capacitor instead of being dissipated in the winding."""
    diode = Simulation(make_config("diode", stages=2)).run()
    ahb = Simulation(make_config("ahb", stages=2)).run()
    assert ahb.energy.capacitor > diode.energy.capacitor
    assert ahb.energy.winding_heat < diode.energy.winding_heat


def test_capacitor_voltage_recovers_during_turn_off():
    """Voltage must rise again while the field collapses -- a sign the plain
    freewheel path never produces."""
    r = Simulation(make_config("ahb", stages=1, record=True)).run()
    vc = r.voltage[:, 0]
    assert np.diff(vc).max() > 0.0
    diode = Simulation(make_config("diode", stages=1, record=True)).run()
    assert np.diff(diode.voltage[:, 0]).max() <= 1e-9


# -- the actual question ------------------------------------------------


def _decay_time(sim, res, fraction: float) -> float:
    """Time from the gate-off command until current falls below `fraction` of
    its value at that instant. Measured from gate-off, not from the current
    peak, so it isolates the turn-off transient from the driven period."""
    i = np.abs(res.current[:, 0])
    k_off = int(np.searchsorted(res.time, sim.controllers[0].off_time))
    after = i[k_off:]
    below = np.where(after < fraction * after[0])[0]
    return float(below[0] * res.config.dt) if len(below) else np.inf


def test_half_bridge_advantage_compounds_as_current_falls():
    """The gain is not uniform, and the naive intuition overstates the early
    part while understating the tail.

    At turn-off, i*R is ~193 V for this coil, so the plain freewheel decay is
    already dominated by resistance and the extra 1.5 V of diode drop barely
    matters -- the half bridge is only ~2x faster there. As current falls, i*R
    collapses and the diode path has just 1.5 V left driving it while the
    bridge still has the full bank voltage, so the advantage grows.
    """
    d_sim = Simulation(make_config("diode", stages=1, record=True))
    d = d_sim.run()
    a_sim = Simulation(make_config("ahb", stages=1, record=True))
    a = a_sim.run()

    half = _decay_time(d_sim, d, 0.5) / _decay_time(a_sim, a, 0.5)
    tenth = _decay_time(d_sim, d, 0.1) / _decay_time(a_sim, a, 0.1)
    assert 1.5 < half < 3.5
    assert tenth > 3.0
    assert tenth > half  # the advantage compounds


def test_half_bridge_reduces_suck_back():
    """A faster collapse means less current left when the force reverses."""
    diode = Simulation(make_config("diode", stages=2)).run()
    ahb = Simulation(make_config("ahb", stages=2)).run()
    d_back = abs(diode.suck_back_impulse().sum()) / diode.forward_impulse().sum()
    a_back = abs(ahb.suck_back_impulse().sum()) / ahb.forward_impulse().sum()
    assert a_back < d_back


# -- conduction cost ----------------------------------------------------


def test_device_drop_costs_energy_and_velocity():
    """Two devices in series is the honest cost of the topology; without
    charging for it the comparison would be rigged."""
    free = Simulation(make_config("ahb", stages=2, device_drop=0.0)).run()
    real = Simulation(make_config("ahb", stages=2, device_drop=2.0)).run()
    assert real.energy.external_loss > free.energy.external_loss
    assert real.exit_velocity < free.exit_velocity
    assert real.energy.closure_error < 1e-3


def test_device_drop_counts_both_devices():
    cfg = make_config("ahb", device_drop=2.0)
    circ = StageCircuit(config=cfg.stages[0], projectile=cfg.projectile)
    i = 100.0
    di_ahb = circ.derivatives(i, 300.0, 25.0, 0.01, 0.0, SwitchState.ON)[0]

    cfg1 = make_config("diode", device_drop=2.0)
    circ1 = StageCircuit(config=cfg1.stages[0], projectile=cfg1.projectile)
    di_diode = circ1.derivatives(i, 300.0, 25.0, 0.01, 0.0, SwitchState.ON)[0]
    # two drops instead of one leaves less voltage across the coil
    assert di_ahb < di_diode


# -- kernel agreement ---------------------------------------------------


@pytest.mark.parametrize("topology", ["diode", "ahb"])
@pytest.mark.parametrize("code,state", [(ON, SwitchState.ON),
                                        (FREEWHEEL, SwitchState.FREEWHEEL)])
def test_kernel_matches_reference_for_each_topology(topology, code, state):
    sim = Simulation(make_config(topology, stages=3, device_drop=1.8))
    x, v = 0.02, 30.0
    i = np.array([180.0, 90.0, 20.0])
    vc = np.array([280.0, 150.0, 300.0])
    temp = np.array([30.0, 40.0, 25.0])

    di, dvc, dtemp, _, ext = sim.kernel.derivatives(
        x, v, i, vc, temp, np.full(3, code)
    )
    exp_ext = 0.0
    for k, circ in enumerate(sim.circuits):
        r_di, r_dvc, r_dtemp, r_ext = circ.derivatives(
            i[k], vc[k], temp[k], x, v, state
        )
        assert di[k] == pytest.approx(r_di, rel=1e-11)
        assert dvc[k] == pytest.approx(r_dvc, rel=1e-11)
        assert dtemp[k] == pytest.approx(r_dtemp, rel=1e-11)
        exp_ext += r_ext
    assert ext == pytest.approx(exp_ext, rel=1e-11)
