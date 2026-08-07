#!/usr/bin/env python3
"""Emit DESIGN.md: the current best configuration under the build constraints.

Regenerate with:  python design_summary.py

Constraints encoded here:
    projectile diameter   0.25 in, fixed
    projectile length     <= 1 in
    capacitor voltage     <= 300 V
    peak current          <= 300 A
    stages                8
    objective             maximum muzzle energy
"""

from __future__ import annotations

import numpy as np

from la.engine import Simulation
from la.geometry import CoilGeometry, ProjectileSpec
from la.sweep import build_config
from la.wire import WireSpec

INCH = 0.0254
OUT = "DESIGN.md"

P = dict(
    stages=8,
    proj_len=1.0 * INCH,
    proj_dia=0.25 * INCH,
    coil_length=1.5 * INCH,
    bore=0.009,
    gauge=22,
    turns=400,
    capacitance=8e-3,
    voltage=300.0,
    spacing=0.015,
    topology="ahb",
    device_drop=1.8,
    turn_off_fraction=0.5,
    prefire_scale=1.0,
    dt=2e-6,
)


def main() -> None:
    coil = CoilGeometry(P["coil_length"], P["bore"], P["turns"], WireSpec(22, "single"))
    proj = ProjectileSpec(length=P["proj_len"], diameter=P["proj_dia"])
    pitch = P["coil_length"] + P["spacing"]
    reversal = (P["coil_length"] + proj.length) / 2

    runs = {}
    for cm in ("bore", "mean"):
        sim = Simulation(build_config(**P, coupling=cm))
        runs[cm] = (sim, sim.run())
    sim_b = runs["bore"][0]
    lead = sim_b.circuits[0].time_to_peak_current()
    release = sim_b.controllers[0].release_travel

    L = []
    w = L.append
    w("# Design Summary")
    w("")
    w("Best configuration found under the build constraints, optimised for "
      "**maximum muzzle energy**.")
    w("")
    w("| constraint | value |")
    w("|---|---|")
    w("| projectile diameter | 0.25 in (fixed) |")
    w("| projectile length | <= 1 in |")
    w("| capacitor voltage | <= 300 V |")
    w("| peak current | <= 300 A |")
    w("| stages | 8 |")
    w("")
    w("Regenerate this file with `python design_summary.py`.")
    w("")
    w("## Mechanical")
    w("")
    w("| | |")
    w("|---|---|")
    w(f"| Projectile | {proj.length*1e3:.1f} x {proj.diameter*1e3:.2f} mm steel, "
      f"**{proj.mass*1e3:.2f} g** |")
    w(f"| Coil length | {P['coil_length']*1e3:.1f} mm (1.5 : 1 coil-to-projectile) |")
    w(f"| Winding ID (bore) | {P['bore']*1e3:.2f} mm |")
    w(f"| Barrel wall | {(P['bore']-proj.diameter)/2*1e3-0.25:.3f} mm at 0.25 mm "
      f"radial clearance |")
    w(f"| Winding OD | {coil.outer_radius*2e3:.1f} mm ({coil.layers} layers, "
      f"{coil.winding_depth*1e3:.2f} mm deep) |")
    w(f"| Turns per layer | {coil.turns_per_layer} |")
    w(f"| Stage pitch | {pitch*1e3:.1f} mm ({P['spacing']*1e3:.0f} mm gap) |")
    w(f"| Barrel length | **{P['stages']*pitch*1e3:.0f} mm** over {P['stages']} stages |")
    w("")
    w("## Winding")
    w("")
    w("| | |")
    w("|---|---|")
    w(f"| Wire | 22 AWG single-build ({coil.wire.bare_diameter*1e3:.3f} mm bare, "
      f"{coil.wire.outer_diameter*1e3:.3f} mm OD) |")
    w(f"| Turns | {P['turns']} per coil |")
    w(f"| Wire length | {coil.wire_length:.1f} m per coil, "
      f"**{coil.wire_length*P['stages']:.0f} m total** |")
    w(f"| Resistance | {coil.resistance():.3f} ohm at 20 C, "
      f"{coil.resistance(60):.3f} ohm at 60 C |")
    w("")
    w("## Inductance")
    w("")
    w("| | bore coupling | mean coupling |")
    w("|---|---|---|")
    vals = {}
    for cm in ("bore", "mean"):
        m = runs[cm][0].circuits[0].magnetics
        vals[cm] = dict(
            far=float(m.inductance(-0.5)),
            ctr=float(m.inductance(reversal)),
            inc=float(m.l_incremental(reversal, 265.0)),
            mu=m.mu_eff, isat=m.i_sat, fill=m.summary()["max_fill"],
        )
    b, n = vals["bore"], vals["mean"]
    w(f"| Air core, slug out | {b['far']*1e6:.1f} uH | {n['far']*1e6:.1f} uH |")
    w(f"| Slug centred, small signal | **{b['ctr']*1e6:.0f} uH** | "
      f"**{n['ctr']*1e6:.0f} uH** |")
    w(f"| Ratio | {b['ctr']/b['far']:.2f} x | {n['ctr']/n['far']:.2f} x |")
    w(f"| Incremental at 265 A | {b['inc']*1e6:.1f} uH | {n['inc']*1e6:.1f} uH |")
    w(f"| Effective permeability | {b['mu']:.2f} | {n['mu']:.2f} |")
    w(f"| Saturation current | {b['isat']:.1f} A | {n['isat']:.1f} A |")
    w(f"| Max fill fraction | {b['fill']:.3f} | {n['fill']:.3f} |")
    w("")
    w("Three things to know before measuring:")
    w("")
    w(f"- The two coupling models disagree by {b['ctr']/n['ctr']:.1f}x on slug-in "
      f"inductance. **This is the measurement that resolves the output "
      f"uncertainty** -- an LCR reading on the finished coil says which is right.")
    w("- What an LCR meter reads is not what the circuit sees. At 265 A the slug "
      "is ~26x saturated, so incremental inductance collapses back to the "
      f"air-core {b['far']*1e6:.0f} uH. Scope traces will imply ~600 uH, not 2700.")
    w(f"- L/R = {coil.inductance_air/coil.resistance()*1e6:.0f} us (air core); "
      f"time to peak current {lead*1e6:.0f} us, heavily overdamped.")
    w("")
    w("## Electrical, per stage")
    w("")
    w("| | |")
    w("|---|---|")
    w(f"| Capacitor bank | {P['capacitance']*1e3:.0f} mF at {P['voltage']:.0f} V "
      f"= {0.5*P['capacitance']*P['voltage']**2:.0f} J |")
    w(f"| Total stored | {P['stages']*0.5*P['capacitance']*P['voltage']**2/1000:.2f} kJ |")
    w("| Topology | Asymmetric half bridge: 2 switching devices + 2 diodes |")
    w("| Device drop assumed | 1.8 V per conducting device |")
    w(f"| Peak current | {runs['bore'][1].peak_current.max():.0f} A "
      f"(limit {300:.0f} A) |")
    w("")
    w("The bank is deliberately oversized to shorten recharge between shots, so "
      "the ~78% left on the capacitors is reserve, not loss. The meaningful "
      "efficiency figure is conversion of the energy actually drawn.")
    w("")
    w("## Control")
    w("")
    w("| | |")
    w("|---|---|")
    w("| Sensor | ingress detector at each coil mouth |")
    w(f"| Fire | when the projectile is {lead*1e6:.0f} us of travel away "
      f"(prefire scale 1.0) |")
    w(f"| Turn off | after {release*1e3:.1f} mm of nose travel past the sensor |")
    w(f"| Force reverses at | {reversal*1e3:.1f} mm of nose travel |")
    w(f"| Collapse margin | {(reversal-release)*1e3:.1f} mm |")
    w("")
    w(f"At ~100 m/s the {lead*1e6:.0f} us lead is roughly {lead*100*1e3:.0f} mm, "
      f"about {lead*100/pitch:.1f} stage pitches upstream. **Several stages are "
      "energised simultaneously**, so the control logic cannot be a simple "
      "per-stage state machine, and total bus current is a multiple of the "
      "per-stage figure. Bus current is not modelled.")
    w("")
    w("Prefire is load-bearing, not a refinement: without it output falls by "
      "roughly 45-59%, because the bank takes 1789 us to reach peak current "
      "and the projectile would be past before the current arrived.")
    w("")
    w("## Predicted performance")
    w("")
    w("| | bore (optimistic) | mean (pessimistic) |")
    w("|---|---|---|")

    def row(label, fn):
        w(f"| {label} | {fn(*runs['bore'])} | {fn(*runs['mean'])} |")

    row("**Muzzle energy**", lambda s, r: f"**{r.energy.kinetic:.2f} J**")
    row("Exit velocity", lambda s, r: f"{r.exit_velocity:.2f} m/s")
    row("Peak current", lambda s, r: f"{r.peak_current.max():.0f} A")
    row("Peak winding temperature", lambda s, r: f"{r.peak_temperature.max():.0f} C")
    row("Suck-back", lambda s, r: f"{100*abs(r.suck_back_impulse().sum())/r.forward_impulse().sum():.2f} %")
    row("Energy drawn from bank", lambda s, r: f"{r.energy.initial-r.energy.capacitor:.0f} J")
    row("Conversion efficiency", lambda s, r: f"{r.energy.kinetic/(r.energy.initial-r.energy.capacitor)*100:.2f} %")
    row("Bank remaining", lambda s, r: f"{r.energy.capacitor/r.energy.initial*100:.0f} %")
    row("Shot duration", lambda s, r: f"{r.summary.duration*1e3:.2f} ms")
    row("Energy closure error", lambda s, r: f"{r.energy.closure_error*100:.4f} %")
    w("")
    w(f"**Headline uncertainty: {runs['mean'][1].energy.kinetic:.1f} - "
      f"{runs['bore'][1].energy.kinetic:.1f} J.** Every mechanical, electrical and "
      "control parameter above is pinned and robust across both coupling "
      "assumptions. Only the layer-coupling question is open, and it is worth "
      f"{runs['bore'][1].energy.kinetic/runs['mean'][1].energy.kinetic:.1f}x on output.")
    w("")
    w("## How each choice was pinned")
    w("")
    w("| parameter | rationale | confidence |")
    w("|---|---|---|")
    w("| 400 turns | interior optimum under both couplings; beyond ~500 suck-back returns | high |")
    w("| 22 AWG | 20 AWG only wins where the model over-rewards deep windings | medium |")
    w("| 1.5 : 1 coil ratio | both couplings agree; 3:1 costs 17-30% | high |")
    w("| 9 mm bore | thinning to a 0.325 mm wall buys only 3-13% | high |")
    w("| Half bridge | +17% over a well-timed freewheel design, plus energy recovery and wide timing tolerance | high |")
    w("| Turn-off at 0.5 | flat optimum; half bridge tolerant across 0.5-1.0 | high |")
    w("| Prefire scale 1.0 | peak-at-mouth is optimal under both couplings; flat from 0.75 to 1.5 | high |")
    w("| 8 mF / 300 V | voltage at the ceiling; current binds at 265 A | high |")
    w("| 8 stages | chosen; energy scales near-linearly with stage count if extended | high |")
    w("")
    w("## Known model limitations")
    w("")
    w("1. **Layer coupling.** `fill()` scales the slug's contribution by "
      "`proj_area/bore_area`, independent of winding depth, so outer layers are "
      "credited with coupling they do not have. At 8 layers this is the dominant "
      "uncertainty. Bracketed by the two coupling models, not resolved.")
    w("2. **Demagnetising factor** uses a prolate spheroid for a cylinder.")
    w("3. **No flux return modelled by default.** A shell plus end caps is "
      "parameterised (`flux_return`, `l_shell_factor`) but uncalibrated. Raising "
      "mu_eff alone buys only ~6-8%, because i_sat falls in proportion; the real "
      "gain is the inductance rise, worth ~19% if L roughly doubles.")
    w("4. **Not modelled**: eddy losses, shell saturation, total bus current "
      "across simultaneously firing stages, barrel friction, projectile eddy "
      "currents.")
    w("")
    w("## Next step")
    w("")
    w("Wind one coil and measure L with the slug out and centred. That single "
      "reading discriminates the two coupling models and collapses the output "
      "range. `la.calibration.mu_eff_from_measurements()` and the `coupling` "
      "switch are wired and tested for it; drop the numbers into "
      "`measurements/stage0.yaml`.")
    w("")

    with open(OUT, "w") as fh:
        fh.write("\n".join(L))
    print(f"wrote {OUT} ({len(L)} lines)")


if __name__ == "__main__":
    main()
