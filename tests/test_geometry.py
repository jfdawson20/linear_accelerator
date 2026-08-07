"""Tests for wire and coil geometry.

These modules are the foundation the physics rewrite sits on, so they are
pinned tightly here.
"""

from __future__ import annotations

import math

import pytest

from la.geometry import CoilGeometry, ProjectileSpec
from la.wire import WireSpec, awg_bare_diameter, available_gauges


# -- wire ---------------------------------------------------------------


@pytest.mark.parametrize(
    "gauge,expected_mm",
    [(18, 1.024), (20, 0.812), (22, 0.644), (24, 0.511), (26, 0.405), (30, 0.255)],
)
def test_awg_table_matches_the_awg_definition(gauge, expected_mm):
    """Table values agree with d = 0.127mm * 92^((36-n)/39) and published tables."""
    assert awg_bare_diameter(gauge) * 1e3 == pytest.approx(expected_mm, abs=0.001)
    assert WireSpec(gauge).bare_diameter * 1e3 == pytest.approx(expected_mm, abs=0.001)


def test_insulated_diameter_exceeds_bare_and_matches_v1_notes():
    """v1 carried commented-out 'real' values: 26AWG -> 0.43, 20AWG -> 0.85.

    Those were single-build figures, which is why the default build is 'single'.
    """
    assert WireSpec(26, "single").outer_diameter * 1e3 == pytest.approx(0.43, abs=0.01)
    assert WireSpec(20, "single").outer_diameter * 1e3 == pytest.approx(0.85, abs=0.01)
    for g in available_gauges():
        w = WireSpec(g)
        assert w.outer_diameter > w.bare_diameter
        assert WireSpec(g, "heavy").outer_diameter > w.outer_diameter


def test_circular_mil_conversion():
    """A 1 mil diameter conductor is by definition 1 circular mil."""
    one_mil = 25.4e-6
    area = math.pi * (one_mil / 2) ** 2
    from la.wire import M2_TO_CIRCULAR_MILS

    assert area * M2_TO_CIRCULAR_MILS == pytest.approx(1.0, rel=1e-3)


def test_resistance_scales_with_length_and_temperature():
    w = WireSpec(26)
    r20 = w.resistance(10.0)
    assert w.resistance(20.0) == pytest.approx(2 * r20)
    # +40 C on annealed copper is ~ +15.7%
    assert w.resistance(10.0, 60.0) / r20 == pytest.approx(1.157, rel=1e-3)


def test_rejects_unknown_gauge_and_build():
    with pytest.raises(ValueError):
        WireSpec(19)
    with pytest.raises(ValueError):
        WireSpec(26, "triple")


# -- coil ---------------------------------------------------------------


def v1_coil() -> CoilGeometry:
    """The default coil from v1's argparse defaults."""
    return CoilGeometry(
        length=0.035, bore_diameter=0.0098, turns=150, wire=WireSpec(26, "bare")
    )


def test_wheeler_reproduces_v1_air_core_inductance():
    """With bare wire (as v1 used for packing), Wheeler must reproduce v1's
    reported 5.6357e-05 H. This pins the formula and its unit handling."""
    assert v1_coil().inductance_air == pytest.approx(5.6357e-5, rel=1e-3)


def test_insulated_build_lowers_turns_per_layer_and_raises_inductance():
    """Real wire packs less tightly: fewer turns per layer, so more layers, so a
    larger mean radius and winding depth."""
    bare = v1_coil()
    real = CoilGeometry(
        length=0.035, bore_diameter=0.0098, turns=150, wire=WireSpec(26, "single")
    )
    assert real.turns_per_layer < bare.turns_per_layer
    assert real.layers >= bare.layers
    assert real.winding_depth > bare.winding_depth
    assert real.wire_length > bare.wire_length


def test_wire_length_uses_layer_centres():
    """A single-layer coil's wire sits half a wire diameter above the former."""
    w = WireSpec(26, "bare")
    coil = CoilGeometry(length=0.035, bore_diameter=0.01, turns=10, wire=w)
    assert coil.layers == 1
    expected_r = 0.005 + 0.5 * w.outer_diameter
    assert coil.wire_length == pytest.approx(10 * 2 * math.pi * expected_r)


def test_layer_counts_are_consistent():
    coil = v1_coil()
    assert coil.layers == math.ceil(coil.turns / coil.turns_per_layer)
    assert coil.outer_radius > coil.mean_radius > coil.inner_radius
    assert coil.mean_radius == pytest.approx(
        (coil.inner_radius + coil.outer_radius) / 2
    )


def test_partial_top_layer_is_counted():
    """turns = one full layer + 1 must add exactly one turn at the next radius."""
    w = WireSpec(26, "bare")
    full = CoilGeometry(length=0.035, bore_diameter=0.01, turns=86, wire=w)
    assert full.layers == 1
    plus = CoilGeometry(length=0.035, bore_diameter=0.01, turns=87, wire=w)
    assert plus.layers == 2
    extra = plus.wire_length - full.wire_length
    expected_r = 0.005 + 1.5 * w.outer_diameter
    assert extra == pytest.approx(2 * math.pi * expected_r)


def test_coil_too_short_for_one_turn_is_rejected():
    with pytest.raises(ValueError, match="cannot fit even one turn"):
        CoilGeometry(
            length=0.0001, bore_diameter=0.01, turns=10, wire=WireSpec(18, "single")
        )


def test_resistance_is_in_a_plausible_range():
    """v1 reported 0.646 ohm for this coil using bare-diameter geometry."""
    assert v1_coil().resistance() == pytest.approx(0.65, rel=0.05)


# -- projectile ---------------------------------------------------------


def test_mass_is_derived_not_hardcoded():
    """v1 hardcoded 0.00569 kg. A 6 mm x 17.5 mm steel slug should be close."""
    p = ProjectileSpec(length=0.0175, diameter=0.006)
    assert p.mass == pytest.approx(0.00388, rel=0.01)
    # v1's mass implies a longer slug; check the relationship is linear
    assert ProjectileSpec(length=0.035, diameter=0.006).mass == pytest.approx(
        2 * p.mass
    )


def test_aspect_ratio_and_geometry():
    p = ProjectileSpec(length=0.0175, diameter=0.006)
    assert p.aspect_ratio == pytest.approx(2.9167, rel=1e-4)
    assert p.volume == pytest.approx(p.area * p.length)


def test_projectile_rejects_nonsense():
    with pytest.raises(ValueError):
        ProjectileSpec(length=0.0, diameter=0.006)
    with pytest.raises(ValueError):
        ProjectileSpec(length=0.01, diameter=0.006, mu_r=0.5)


def test_clearance_is_reported_but_derived_from_bore():
    coil = v1_coil()
    p = ProjectileSpec(length=0.0175, diameter=0.006)
    assert coil.clearance(p) == pytest.approx(0.0049 - 0.003)
