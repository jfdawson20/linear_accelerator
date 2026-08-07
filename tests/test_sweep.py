"""Parameter sweep tests.

v1's optimiser crashed on its first improvement, swept ten identical profiles,
and mutated the profile library in place. The isolation tests below pin the
last of those: every grid point must build its configuration from scratch.
"""

from __future__ import annotations

import pytest

from la.sweep import (
    DEFAULTS,
    ParameterSpace,
    build_config,
    evaluate,
    rank,
    results_table,
    sensitivity,
    sweep,
)


# -- configuration building --------------------------------------------


def test_defaults_build_a_valid_config():
    cfg = build_config()
    assert cfg.num_stages == DEFAULTS["stages"]
    assert cfg.record is False  # sweeps never keep traces


def test_ratio_drives_projectile_length():
    """The design premise as a swept parameter -- v1 could not vary it."""
    two_to_one = build_config(coil_length=0.035, ratio=2.0)
    assert two_to_one.projectile.length == pytest.approx(0.0175)
    assert two_to_one.coil_to_projectile_ratio == pytest.approx(2.0)

    flat = build_config(coil_length=0.035, ratio=1.0)
    assert flat.projectile.length == pytest.approx(0.035)
    assert flat.projectile.mass > two_to_one.projectile.mass


def test_explicit_projectile_length_overrides_ratio():
    cfg = build_config(coil_length=0.035, ratio=2.0, proj_len=0.010)
    assert cfg.projectile.length == pytest.approx(0.010)


def test_unknown_parameter_is_rejected():
    with pytest.raises(ValueError, match="unknown sweep parameters"):
        build_config(nonexistent=1)
    with pytest.raises(ValueError, match="unknown sweep parameters"):
        ParameterSpace(nonsense=[1, 2])


def test_each_point_gets_an_independent_config():
    """No shared mutable state between grid points."""
    a = build_config(turns=100)
    b = build_config(turns=200)
    assert a.stages[0].coil.turns == 100
    assert b.stages[0].coil.turns == 200
    assert a.stages[0] is not b.stages[0]


def test_stages_are_distinct_objects_at_distinct_positions():
    cfg = build_config(stages=4, spacing=0.02, coil_length=0.035)
    positions = [s.position for s in cfg.stages]
    assert positions == sorted(positions)
    assert len(set(positions)) == 4


# -- the grid -----------------------------------------------------------


def test_space_enumerates_the_cartesian_product():
    space = ParameterSpace(turns=[100, 200], voltage=[200.0, 300.0, 400.0])
    points = list(space.points())
    assert len(space) == 6 == len(points)
    assert {tuple(sorted(p.items())) for p in points}.__len__() == 6


def test_empty_space_yields_one_default_point():
    assert list(ParameterSpace().points()) == [{}]


# -- evaluation ---------------------------------------------------------


def test_evaluate_returns_a_scalar_summary():
    r = evaluate(dict(stages=1, dt=2e-5))
    assert r.ok
    assert r.exit_velocity > 0
    assert 0.0 < r.efficiency < 0.1
    assert r.closure_error < 1e-3


def test_impossible_geometry_is_captured_not_raised():
    """A grid point that cannot be built must not abort the whole sweep."""
    r = evaluate(dict(coil_length=1e-5, stages=1))
    assert not r.ok
    assert r.error


def test_sweep_runs_every_point():
    space = ParameterSpace(voltage=[200.0, 300.0])
    results = sweep(space, fixed=dict(stages=1, dt=2e-5), workers=1, progress=False)
    assert len(results) == 2
    assert all(r.ok for r in results)
    assert results[1].exit_velocity > results[0].exit_velocity


def test_parallel_and_serial_agree():
    """Multiprocessing must not change any number."""
    space = ParameterSpace(voltage=[200.0, 300.0, 400.0])
    fixed = dict(stages=1, dt=2e-5)
    serial = sweep(space, fixed=fixed, workers=1, progress=False)
    parallel = sweep(space, fixed=fixed, workers=2, progress=False)
    for a, b in zip(serial, parallel):
        assert a.params == b.params
        assert a.exit_velocity == pytest.approx(b.exit_velocity, rel=1e-12)


# -- ranking and constraints -------------------------------------------


def test_rank_orders_best_first():
    space = ParameterSpace(voltage=[200.0, 300.0, 400.0])
    results = sweep(space, fixed=dict(stages=1, dt=2e-5), workers=1, progress=False)
    ordered = rank(results, "velocity", thermal_limit=False)
    velocities = [r.exit_velocity for r in ordered]
    assert velocities == sorted(velocities, reverse=True)


def test_thermal_limit_excludes_overheating_points():
    """A configuration that cooks its windings is not a design. v1 had no way
    to express that constraint."""
    space = ParameterSpace(voltage=[200.0, 900.0])
    results = sweep(
        space, fixed=dict(stages=1, dt=2e-5, max_temp=60.0), workers=1,
        progress=False,
    )
    assert len(rank(results, "velocity", thermal_limit=False)) == 2
    kept = rank(results, "velocity", thermal_limit=True)
    assert all(r.peak_temperature <= 60.0 for r in kept)
    assert len(kept) < 2


def test_suck_back_constraint_filters():
    space = ParameterSpace(voltage=[200.0, 400.0])
    results = sweep(space, fixed=dict(stages=1, dt=2e-5), workers=1, progress=False)
    strict = rank(results, "velocity", thermal_limit=False, max_suck_back_pct=0.0)
    assert all(r.suck_back_pct <= 0.0 for r in strict)


def test_unknown_objective_is_rejected():
    with pytest.raises(ValueError, match="unknown objective"):
        rank([], "sharpness")


def test_velocity_per_joule_prefers_the_cheaper_shot():
    space = ParameterSpace(voltage=[200.0, 400.0])
    results = sweep(space, fixed=dict(stages=1, dt=2e-5), workers=1, progress=False)
    by_v = rank(results, "velocity", thermal_limit=False)
    by_vpj = rank(results, "velocity_per_joule", thermal_limit=False)
    assert by_v[0].exit_velocity >= by_vpj[0].exit_velocity
    assert by_vpj[0].velocity_per_joule >= by_v[0].velocity_per_joule


# -- reporting ----------------------------------------------------------


def test_sensitivity_buckets_by_axis_value():
    space = ParameterSpace(voltage=[200.0, 300.0], turns=[100, 200])
    results = sweep(space, fixed=dict(stages=1, dt=2e-5), workers=1, progress=False)
    rows = sensitivity(results, "voltage", "velocity")
    assert len(rows) == 2
    for value, best, mean in rows:
        assert best >= mean


def test_results_table_renders():
    space = ParameterSpace(voltage=[200.0, 300.0])
    results = sweep(space, fixed=dict(stages=1, dt=2e-5), workers=1, progress=False)
    assert results_table(results, ["voltage"]) is not None
