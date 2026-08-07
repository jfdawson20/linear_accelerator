"""Tables, plots and the energy audit.

Reporting is a separate concern here. In v1, `Coil.__init__` printed a 25-row
table as a side effect of construction, so building a library of 525 candidate
coils dumped 525 tables to stdout before any work began.
"""

from __future__ import annotations

import numpy as np
from prettytable import PrettyTable

from .engine import RunResult, Simulation


def _t(fields) -> PrettyTable:
    t = PrettyTable()
    t.field_names = fields
    t.align = "r"
    t.align[fields[0]] = "l"
    return t


def coil_table(sim: Simulation) -> PrettyTable:
    """Static design parameters, one column per stage."""
    cfg = sim.config
    coil = cfg.stages[0].coil
    proj = cfg.projectile
    mag = sim.circuits[0].magnetics

    t = _t(["Parameter", "Value", "Unit", "Note"])
    rows = [
        ("Stages", cfg.num_stages, "", ""),
        ("Coil length", coil.length * 1e3, "mm", ""),
        ("Projectile length", proj.length * 1e3, "mm", ""),
        (
            "Coil : projectile",
            cfg.coil_to_projectile_ratio,
            "",
            "design premise = 2.0",
        ),
        ("Projectile mass", proj.mass * 1e3, "g", "derived from geometry"),
        ("Bore diameter", coil.bore_diameter * 1e3, "mm", ""),
        ("Projectile diameter", proj.diameter * 1e3, "mm", ""),
        ("Radial clearance", coil.clearance(proj) * 1e3, "mm", "not used in force"),
        ("Turns", coil.turns, "", ""),
        ("Wire", str(coil.wire), "", ""),
        ("Turns per layer", coil.turns_per_layer, "", "insulated diameter"),
        ("Layers", coil.layers, "", ""),
        ("Winding depth", coil.winding_depth * 1e3, "mm", ""),
        ("Wire length", coil.wire_length, "m", ""),
        ("Resistance (20C)", coil.resistance(), "ohm", ""),
        ("Resistance (max T)", coil.resistance(cfg.thermal.max_c), "ohm", ""),
        ("Air-core inductance", mag.l_air * 1e6, "uH", "Wheeler multilayer"),
        ("Inductance, slug in", float(mag.inductance(mag.coil_position + 0.0263)) * 1e6, "uH", ""),
        ("Demagnetising factor", mag.summary()["N_d"], "", "prolate spheroid"),
        ("Effective permeability", mag.mu_eff, "", f"bulk mu_r = {proj.mu_r:g}"),
        ("Max fill fraction", mag.summary()["max_fill"], "", "of magnetic path"),
        ("Saturation current", mag.i_sat, "A", "slug reaches B_sat"),
        ("Capacitance", cfg.stages[0].bank.capacitance * 1e6, "uF", ""),
        ("Initial voltage", cfg.stages[0].bank.voltage, "V", ""),
        ("Stored energy / stage", cfg.stages[0].bank.stored_energy, "J", ""),
        ("Total stored energy", cfg.total_stored_energy, "J", ""),
        ("Timestep", cfg.dt * 1e6, "us", ""),
        ("Saturation modelled", cfg.saturation, "", ""),
    ]
    for name, value, unit, note in rows:
        shown = f"{value:.4g}" if isinstance(value, (int, float)) else str(value)
        t.add_row([name, shown, unit, note])
    return t


def stage_table(result: RunResult, sim: Simulation) -> PrettyTable:
    """Per-stage outcome."""
    cfg = result.config
    t = _t(
        [
            "Stage",
            "Fire (ms)",
            "On (us)",
            "Peak I (A)",
            "I/Isat",
            "Peak T (C)",
            "Vcap end (V)",
            "Fwd (N.s)",
            "Back (N.s)",
            "Loss %",
        ]
    )
    fwd = result.forward_impulse()
    back = result.suck_back_impulse()
    on_time = result.stage_conduction_time()
    peak_i = result.peak_current
    for k in range(cfg.num_stages):
        ctrl = result.controllers[k]
        isat = sim.circuits[k].magnetics.i_sat
        loss = 100.0 * abs(back[k]) / fwd[k] if fwd[k] > 0 else 0.0
        t.add_row(
            [
                k,
                f"{ctrl.fire_time * 1e3:.2f}" if ctrl.fire_time is not None else "-",
                f"{on_time[k] * 1e6:.0f}",
                f"{peak_i[k]:.1f}",
                f"{peak_i[k] / isat:.1f}",
                f"{result.temperature[:, k].max():.1f}",
                f"{result.voltage[-1, k]:.1f}",
                f"{fwd[k]:.4f}",
                f"{back[k]:.4f}",
                f"{loss:.2f}",
            ]
        )
    return t


def energy_table(result: RunResult) -> PrettyTable:
    """Where the energy went. The closure error is the model's self-check."""
    e = result.energy
    t = _t(["Term", "Energy (J)", "% of stored"])
    for label, value in (
        ("Initially stored", e.initial),
        ("Remaining in capacitors", e.capacitor),
        ("Remaining as field", e.magnetic),
        ("Kinetic (projectile)", e.kinetic),
        ("Winding heat (I2R)", e.winding_heat),
        ("Switch + diode loss", e.external_loss),
        ("Discarded at cutoff", e.discarded),
        ("Unaccounted", e.residual),
    ):
        pct = 100.0 * value / e.initial if e.initial else 0.0
        t.add_row([label, f"{value:.4f}", f"{pct:.3f}"])
    return t


def summary_lines(result: RunResult, sim: Simulation) -> list[str]:
    e = result.energy
    cfg = result.config
    fwd = result.forward_impulse().sum()
    back = result.suck_back_impulse().sum()
    lines = [
        f"Exit velocity      : {result.exit_velocity:.2f} m/s",
        f"Muzzle energy      : {e.kinetic:.2f} J",
        f"Efficiency         : {e.efficiency * 100:.2f} %",
        f"Energy closure err : {e.closure_error * 100:.4f} %",
        f"Suck-back          : {back:.5f} N.s "
        f"({100 * abs(back) / fwd if fwd else 0:.2f} % of forward impulse)",
        f"Run                : {len(result.time)} steps, "
        f"{result.time[-1] * 1e3:.2f} ms, ended '{result.terminated}'",
    ]
    peak_ratio = max(
        result.peak_current[k] / sim.circuits[k].magnetics.i_sat
        for k in range(cfg.num_stages)
    )
    if peak_ratio > 2.0:
        lines.append(
            f"NOTE: peak current is {peak_ratio:.1f}x the saturation current; "
            f"above I_sat force grows only linearly with current."
        )
    if e.closure_error > 0.01:
        lines.append(
            f"WARNING: energy closure error {e.closure_error * 100:.2f}% "
            f"exceeds 1%. Reduce dt."
        )
    over = [
        k
        for k in range(cfg.num_stages)
        if result.temperature[:, k].max() > cfg.thermal.max_c
    ]
    if over:
        lines.append(
            f"WARNING: stages {over} exceed the {cfg.thermal.max_c:g} C "
            f"winding limit."
        )
    for w in result.warnings:
        lines.append(f"WARNING: {w}")
    return lines


def print_report(result: RunResult, sim: Simulation, verbose: bool = False) -> None:
    if verbose:
        print("\nDESIGN")
        print(coil_table(sim))
    print("\nPER-STAGE RESULTS")
    print(stage_table(result, sim))
    print("\nENERGY AUDIT")
    print(energy_table(result))
    print("\nSUMMARY")
    for line in summary_lines(result, sim):
        print("  " + line)
    print()


def plot(result: RunResult, path: str | None = None) -> None:
    """Six panels: current, force, position, velocity, temperature, energy.

    Every stage is plotted against the single shared time base, so unlike v1
    there is no mismatch between the x and y arrays.
    """
    import matplotlib

    if path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t_ms = result.time * 1e3
    fig, axs = plt.subplots(3, 2, figsize=(13, 9), sharex=True)

    for k in range(result.config.num_stages):
        label = f"stage {k}"
        axs[0, 0].plot(t_ms, result.current[:, k], lw=0.9, label=label)
        axs[0, 1].plot(t_ms, result.force[:, k], lw=0.9, label=label)
        axs[2, 0].plot(t_ms, result.temperature[:, k], lw=0.9, label=label)
        axs[2, 1].plot(t_ms, result.voltage[:, k], lw=0.9, label=label)

    axs[0, 1].axhline(0.0, color="k", lw=0.6, ls=":")
    axs[1, 0].plot(t_ms, result.position * 1e3, color="C0")
    axs[1, 1].plot(t_ms, result.velocity, color="C1")

    for stage in result.config.stages:
        axs[1, 0].axhline(stage.position * 1e3, color="grey", lw=0.4, alpha=0.5)

    titles = [
        (0, 0, "Coil current", "A"),
        (0, 1, "Force on projectile (negative = suck-back)", "N"),
        (1, 0, "Projectile position", "mm"),
        (1, 1, "Projectile velocity", "m/s"),
        (2, 0, "Winding temperature", "C"),
        (2, 1, "Capacitor voltage", "V"),
    ]
    for r, c, title, unit in titles:
        axs[r, c].set_title(title, fontsize=10)
        axs[r, c].set_ylabel(unit)
        axs[r, c].grid(alpha=0.25)
    for c in (0, 1):
        axs[2, c].set_xlabel("time (ms)")
    if result.config.num_stages <= 8:
        axs[0, 0].legend(fontsize=7, ncol=2)

    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120)
        print(f"wrote {path}")
    else:
        plt.show()


def force_profile_table(sim: Simulation, current: float = 200.0) -> PrettyTable:
    """Force against nose position for a single stage at fixed current.

    Shows where force reverses, which is the quantity the coil-to-projectile
    ratio is chosen to stay ahead of.
    """
    mag = sim.circuits[0].magnetics
    cfg = sim.config
    lc, lp = cfg.stages[0].coil.length, cfg.projectile.length
    t = _t(["Nose (mm)", "Fill", "dFill/dx", f"Force @ {current:g}A (N)"])
    for x in np.linspace(-lp, lc + lp + lp / 2, 22):
        t.add_row(
            [
                f"{x * 1e3:.1f}",
                f"{float(mag.fill(x)):.4f}",
                f"{float(mag.dfill_dx(x)):+.2f}",
                f"{float(mag.force(x, current)):+.2f}",
            ]
        )
    return t
