#!/usr/bin/env python3
"""Emit DESIGN-12MM.md: alternate proposal optimised for muzzle energy.

Regenerate with:  python design_12mm.py
"""

from __future__ import annotations

from la.engine import Simulation
from la.geometry import CoilGeometry, ProjectileSpec
from la.sweep import build_config
from la.wire import WireSpec

DIA = 0.012
BORE = DIA + 2.65e-3
OUT = "DESIGN-12MM.md"

COMMON = dict(
    stages=8, proj_dia=DIA, bore=BORE, capacitance=8e-3, voltage=300.0,
    spacing=0.015, topology="ahb", device_drop=1.8, turn_off_fraction=0.667,
    prefire_scale=1.0, flux_return=0.7, l_shell_factor=2.0, dt=5e-6,
    max_temp=60.0,
)

VARIANTS = {
    "A -- moderate (L/d 6)": dict(proj_len=6 * DIA, coil_length=1.5 * 6 * DIA,
                                  turns=500, gauge=18),
    "B -- compact (L/d 5)": dict(proj_len=5 * DIA, coil_length=1.25 * 5 * DIA,
                                 turns=400, gauge=20),
}


def run(spec, coupling):
    sim = Simulation(build_config(**COMMON, **spec, coupling=coupling))
    return sim, sim.run()


def main() -> None:
    L = []
    w = L.append
    w("# Design Summary -- 12 mm Projectile (alternate proposal)")
    w("")
    w("Optimised for **maximum muzzle energy** with a 12 mm steel projectile, "
      "carrying forward every optimisation established for the 0.25 in design: "
      "asymmetric half bridge, prefire at scale 1.0, and a ferromagnetic shell "
      "with end caps.")
    w("")
    w("| constraint | value |")
    w("|---|---|")
    w("| projectile diameter | 12 mm |")
    w("| capacitor voltage | <= 300 V |")
    w("| peak current | <= 300 A |")
    w("| stages | 8 |")
    w("| objective | maximum muzzle energy |")
    w("")
    w("Regenerate with `python design_12mm.py`. See `DESIGN.md` for the "
      "0.25 in design this is an alternative to.")
    w("")
    w("## Read this first")
    w("")
    w("**No optimum was found in projectile aspect ratio.** Muzzle energy was "
      "still rising at L/d = 8, the edge of the swept range:")
    w("")
    w("| L/d | length | mass | barrel | KE (bore) | KE (mean) | best winding |")
    w("|---|---|---|---|---|---|---|")
    for a, m, bl, kb, km, cfg in [
        (4, 42.6, 0.70, 182, 122, "400t 20 AWG, c/p 1.5-1.75"),
        (5, 53.3, 0.72, 234, 147, "400-500t, 18-20 AWG"),
        (6, 63.9, 0.98, 276, 176, "500t 18 AWG, c/p 1.5"),
        (7, 74.6, 0.96, 313, 190, "500t 18 AWG, c/p 1.25"),
        (8, 85.2, 1.08, 336, 201, "600t 18 AWG, c/p 1.25"),
    ]:
        w(f"| {a} | {a*12} mm | {m} g | {bl} m | {kb} J | {km} J | {cfg} |")
    w("")
    w("`KE ~ m*v^2` with `m ~ L` and `v ~ 1/sqrt(L)` alone gives constant "
      "energy, but mu_eff keeps rising as the slug elongates and beats that "
      "scaling. It must plateau eventually -- mu_eff is capped at mu_r = 100 as "
      "the demagnetising factor tends to zero -- but not within the swept range. "
      "18 AWG was also at the edge of the swept gauges.")
    w("")
    w("**So the size below is a practical choice, not a physics optimum.** If a "
      "longer projectile and barrel are acceptable, energy keeps rising.")
    w("")
    w("## The winding answer")
    w("")
    w("At L/d >= 6 **both coupling models agree exactly** on the winding: "
      "**18 AWG, 500-600 turns, coil-to-projectile ratio 1.25-1.5**. That is a "
      "stronger consensus than the 0.25 in design produced.")
    w("")
    w("The shift from 22 AWG / 400 turns is direct: a 12 mm bore means longer "
      "turns, so thicker wire is needed to hold resistance down, and a longer "
      "coil fits more turns per layer.")
    w("")

    for name, spec in VARIANTS.items():
        coil = CoilGeometry(spec["coil_length"], BORE, spec["turns"],
                            WireSpec(spec["gauge"], "single"))
        proj = ProjectileSpec(length=spec["proj_len"], diameter=DIA)
        pitch = spec["coil_length"] + COMMON["spacing"]
        rev = (spec["coil_length"] + proj.length) / 2
        runs = {cm: run(spec, cm) for cm in ("bore", "mean")}
        sim_b = runs["bore"][0]
        lead = sim_b.circuits[0].time_to_peak_current()
        rel = sim_b.controllers[0].release_travel

        w(f"## Variant {name}")
        w("")
        w("| | |")
        w("|---|---|")
        w(f"| Projectile | {proj.length*1e3:.0f} x {DIA*1e3:.0f} mm steel, "
          f"**{proj.mass*1e3:.1f} g** |")
        w(f"| Coil length | {spec['coil_length']*1e3:.0f} mm "
          f"(c/p {spec['coil_length']/proj.length:.2f}) |")
        w(f"| Winding ID (bore) | {BORE*1e3:.2f} mm "
          f"(barrel wall {(BORE-DIA)/2*1e3-0.25:.3f} mm at 0.25 mm clearance) |")
        w(f"| Winding OD | {coil.outer_radius*2e3:.1f} mm "
          f"({coil.layers} layers, {coil.winding_depth*1e3:.2f} mm deep) |")
        w(f"| Wire | **{spec['gauge']} AWG** single-build, "
          f"{coil.wire.outer_diameter*1e3:.3f} mm OD |")
        w(f"| Turns | **{spec['turns']}** per coil, "
          f"{coil.turns_per_layer} per layer |")
        w(f"| Wire length | {coil.wire_length:.1f} m per coil, "
          f"**{coil.wire_length*8:.0f} m total** |")
        w(f"| Resistance | {coil.resistance():.3f} ohm at 20 C, "
          f"{coil.resistance(60):.3f} ohm at 60 C |")
        w(f"| Stage pitch | {pitch*1e3:.0f} mm |")
        w(f"| Barrel length | **{8*pitch*1e3:.0f} mm** |")
        w("")
        w("### Inductance")
        w("")
        w("| | bore coupling | mean coupling |")
        w("|---|---|---|")
        v = {}
        for cm in ("bore", "mean"):
            m = runs[cm][0].circuits[0].magnetics
            v[cm] = (float(m.inductance(-1.0)), float(m.inductance(rev)),
                     float(m.l_incremental(rev, 250.0)), m.mu_eff, m.i_sat)
        w(f"| Air core, slug out | {v['bore'][0]*1e6:.0f} uH | "
          f"{v['mean'][0]*1e6:.0f} uH |")
        w(f"| Slug centred, small signal | **{v['bore'][1]*1e6:.0f} uH** | "
          f"**{v['mean'][1]*1e6:.0f} uH** |")
        w(f"| Ratio | {v['bore'][1]/v['bore'][0]:.2f} x | "
          f"{v['mean'][1]/v['mean'][0]:.2f} x |")
        w(f"| Incremental at 250 A | {v['bore'][2]*1e6:.0f} uH | "
          f"{v['mean'][2]*1e6:.0f} uH |")
        w(f"| Effective permeability | {v['bore'][3]:.1f} | {v['mean'][3]:.1f} |")
        w(f"| Saturation current | {v['bore'][4]:.1f} A | {v['mean'][4]:.1f} A |")
        w("")
        w("Note the air-core figure already includes the assumed shell "
          "(`l_shell_factor` 2.0); a bare coil would be half of it.")
        w("")
        w("### Control")
        w("")
        w("| | |")
        w("|---|---|")
        w(f"| Fire | {lead*1e6:.0f} us of travel ahead of the coil mouth |")
        w(f"| Turn off | after {rel*1e3:.1f} mm of nose travel past the sensor |")
        w(f"| Force reverses at | {rev*1e3:.1f} mm |")
        w(f"| Collapse margin | {(rev-rel)*1e3:.1f} mm |")
        w("")
        w("### Predicted performance")
        w("")
        w("| | bore (optimistic) | mean (pessimistic) |")
        w("|---|---|---|")

        def row(lbl, fn):
            w(f"| {lbl} | {fn(*runs['bore'])} | {fn(*runs['mean'])} |")

        row("**Muzzle energy**", lambda s, r: f"**{r.energy.kinetic:.1f} J**")
        row("Exit velocity", lambda s, r: f"{r.exit_velocity:.1f} m/s")
        row("Peak current", lambda s, r: f"{r.peak_current.max():.0f} A")
        row("Peak winding temperature", lambda s, r: f"{r.peak_temperature.max():.0f} C")
        row("Suck-back", lambda s, r: f"{100*abs(r.suck_back_impulse().sum())/r.forward_impulse().sum():.2f} %")
        row("Energy drawn", lambda s, r: f"{r.energy.initial-r.energy.capacitor:.0f} J")
        row("Conversion efficiency", lambda s, r: f"{r.energy.kinetic/(r.energy.initial-r.energy.capacitor)*100:.2f} %")
        row("Shot duration", lambda s, r: f"{r.summary.duration*1e3:.2f} ms")
        row("Energy closure error", lambda s, r: f"{r.energy.closure_error*100:.4f} %")
        w("")

    w("## Against the 0.25 in design")
    w("")
    w("| | 0.25 in (DESIGN.md) | 12 mm variant A |")
    w("|---|---|---|")
    a = run(VARIANTS["A -- moderate (L/d 6)"], "bore")[1]
    am = run(VARIANTS["A -- moderate (L/d 6)"], "mean")[1]
    w(f"| Muzzle energy | 14.6 - 35.5 J | **{am.energy.kinetic:.0f} - "
      f"{a.energy.kinetic:.0f} J** |")
    w("| Projectile | 6.3 g | 63.9 g |")
    w("| Barrel | 0.43 m | 0.98 m |")
    w("| Wire total | 139 m | see above |")
    w("")
    w("Roughly **5-12x the muzzle energy**, at the cost of a heavier projectile, "
      "a barrel more than twice as long, and considerably more copper.")
    w("")
    w("## Caveats specific to this proposal")
    w("")
    w("1. **The shell assumption is load-bearing.** `flux_return` 0.7 and "
      "`l_shell_factor` 2.0 are assumed, not measured, and the inductance factor "
      "does most of the work -- raising mu_eff alone buys only ~6-8% because "
      "i_sat falls in proportion. Without the shell these energies would be "
      "materially lower.")
    w("2. **Layer coupling** remains the dominant uncertainty, as in DESIGN.md, "
      "and is why every figure is quoted as a range.")
    w("3. **No aspect-ratio optimum was found.** Sizing is a practical choice.")
    w("4. **These are heavy, slow projectiles** -- 50-85 g at 60-90 m/s. If "
      "there is a minimum velocity requirement it constrains the design "
      "separately from energy.")
    w("5. **Not modelled**: eddy losses, shell saturation, total bus current "
      "across simultaneously firing stages, barrel friction.")
    w("")
    w("## Next step")
    w("")
    w("Same as DESIGN.md, plus one: measure L with and without the **shell** "
      "fitted. That ratio is `l_shell_factor` directly, and it is currently a "
      "guess doing a lot of work.")
    w("")

    with open(OUT, "w") as fh:
        fh.write("\n".join(L))
    print(f"wrote {OUT} ({len(L)} lines)")


if __name__ == "__main__":
    main()
