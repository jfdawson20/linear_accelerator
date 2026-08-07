#!/usr/bin/env python3
"""Re-optimise the winding for a 12 mm projectile, maximising muzzle energy.

Carries forward every optimisation established so far: asymmetric half bridge,
prefire at scale 1.0, and a ferromagnetic shell with end caps.

The shell numbers (flux_return 0.7, l_shell_factor 2.0) are ASSUMED, not
measured. flux_return alone is worth little because i_sat falls as mu_eff
rises; the l_shell_factor inductance rise does the work. Both need calibrating
against a real coil.

Constraints: <= 300 V, <= 300 A, 8 stages, 12 mm steel projectile.
"""
from __future__ import annotations
import json, time
from concurrent.futures import ProcessPoolExecutor
from la.sweep import evaluate

DIA = 0.012
WALLCLR = 2.65e-3          # 1.075 mm wall + 0.25 mm clearance, per side
BORE = DIA + WALLCLR
MAX_I, MAX_T = 300.0, 60.0

ASPECTS = [2, 3, 4, 5]          # projectile length / diameter
COIL_RATIOS = [1.0, 1.5, 2.0]   # coil length / projectile length
TURNS = [300, 400, 500]
GAUGES = [20, 22]
FRACS = [0.5, 0.667]

def points():
    for a in ASPECTS:
        plen = a * DIA
        for cr in COIL_RATIOS:
            for t in TURNS:
                for g in GAUGES:
                    for f in FRACS:
                        for cm in ("bore", "mean"):
                            yield dict(stages=8, proj_dia=DIA, proj_len=plen,
                                coil_length=cr*plen, bore=BORE, gauge=g, turns=t,
                                capacitance=8e-3, voltage=300.0, spacing=0.015,
                                topology="ahb", device_drop=1.8,
                                turn_off_fraction=f, prefire_scale=1.0,
                                flux_return=0.7, l_shell_factor=2.0,
                                dt=1e-5, max_temp=MAX_T, coupling=cm)

def main() -> None:
    pts = list(points())
    print(f"12 mm study: {len(pts)} configurations")
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(evaluate, pts, chunksize=4))
    print(f"done in {time.perf_counter()-t0:.0f}s")
    json.dump([{**r.params, "muzzle_energy": r.muzzle_energy,
                "exit_velocity": r.exit_velocity, "efficiency": r.efficiency,
                "peak_current": r.peak_current, "peak_temperature": r.peak_temperature,
                "suck_back_pct": r.suck_back_pct, "saturation_ratio": r.saturation_ratio,
                "stored_energy": r.stored_energy, "closure_error": r.closure_error,
                "terminated": r.terminated, "error": r.error} for r in res],
              open("study_12mm.json", "w"))
    bad = [r for r in res if r.ok and r.closure_error > 1e-3]
    late = [r for r in res if r.ok and r.terminated == "max_time"]
    print(f"{len(bad)} closure failures, {len(late)} hit the time limit")
    ok = [r for r in res if r.ok and r.peak_current <= MAX_I
          and r.peak_temperature <= MAX_T and r.terminated != "max_time"
          and r.closure_error < 1e-3]
    print(f"{len(ok)}/{len(res)} viable\n")
    for cm in ("bore", "mean"):
        sub = sorted([r for r in ok if r.params["coupling"] == cm],
                     key=lambda r: -r.muzzle_energy)[:8]
        print(f"=== coupling={cm}: top 8 by muzzle energy ===")
        print(f"{'L/d':>4} {'plen':>6} {'coil':>6} {'c/p':>5} {'turns':>6} {'awg':>4} "
              f"{'frac':>6} | {'KE J':>7} {'v m/s':>7} {'I':>5} {'I/Is':>6} {'T':>4} {'back%':>7}")
        for r in sub:
            p = r.params
            print(f"{p['proj_len']/DIA:4.0f} {p['proj_len']*1e3:6.1f} "
                  f"{p['coil_length']*1e3:6.1f} {p['coil_length']/p['proj_len']:5.1f} "
                  f"{p['turns']:6d} {p['gauge']:4d} {p['turn_off_fraction']:6.3f} | "
                  f"{r.muzzle_energy:7.2f} {r.exit_velocity:7.2f} {r.peak_current:5.0f} "
                  f"{r.saturation_ratio:6.1f} {r.peak_temperature:4.0f} {r.suck_back_pct:6.2f}%")
        print()

if __name__ == "__main__":
    main()
