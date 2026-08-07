#!/usr/bin/env python3
"""How early should the coil turn off?

turn_off_fraction is the nose travel at which turn-off is commanded, as a
fraction of the distance to force reversal (Lc + Lp)/2. 1.0 commands turn-off
exactly as force reverses -- far too late, since the field then still has to
collapse. The original design rule (turn off when the tail clears the sensor)
corresponds to Lp/((Lc+Lp)/2), which is 0.667 at a 2:1 coil ratio.

Turning off earlier trades peak force for a cleaner exit. The half bridge
collapses the field faster, so it should tolerate a later turn-off than the
freewheel design -- that is the hypothesis being tested.
"""
from __future__ import annotations
import json, time
from la.sweep import ParameterSpace, rank, results_table, sweep

INCH = 0.0254
SPACE = ParameterSpace(
    topology=["diode", "ahb"],
    turn_off_fraction=[0.2, 0.3, 0.4, 0.5, 0.667, 0.8, 0.9, 1.0],
    turns=[300, 400, 500],
)
FIXED = dict(stages=8, ratio=2.0, proj_dia=0.25 * INCH, bore=0.009,
             coil_length=2.0 * INCH, gauge=22, capacitance=8e-3,
             spacing=0.015, voltage=300.0, dt=1e-5, max_temp=60.0,
             device_drop=1.8)
MAX_I = 300.0

def main() -> None:
    print(f"timing study: {len(SPACE)} configurations")
    t0 = time.perf_counter()
    res = sweep(SPACE, fixed=FIXED, workers=8)
    print(f"done in {time.perf_counter()-t0:.0f}s")
    json.dump([{**r.params, "exit_velocity": r.exit_velocity,
                "efficiency": r.efficiency, "peak_current": r.peak_current,
                "peak_temperature": r.peak_temperature,
                "suck_back_pct": r.suck_back_pct,
                "closure_error": r.closure_error, "terminated": r.terminated,
                "error": r.error} for r in res], open("study_timing.json","w"))

    bad = [r for r in res if r.ok and r.closure_error > 1e-3]
    print(f"{len(bad)} points failed energy closure")

    for turns in (300, 400, 500):
        print(f"\n=== {turns} TURNS ===")
        print(f"{'frac':>6} | {'diode v':>8} {'back %':>7} | {'ahb v':>8} {'back %':>7} | {'gain':>7}")
        for f in SPACE.axes["turn_off_fraction"]:
            row = {}
            for t in ("diode", "ahb"):
                m = [r for r in res if r.ok and r.params["topology"] == t
                     and r.params["turn_off_fraction"] == f
                     and r.params["turns"] == turns]
                row[t] = m[0] if m else None
            d, a = row["diode"], row["ahb"]
            if d and a:
                mark = "  <- design rule" if abs(f - 0.667) < 1e-9 else ""
                print(f"{f:6.3f} | {d.exit_velocity:8.2f} {d.suck_back_pct:6.2f}% | "
                      f"{a.exit_velocity:8.2f} {a.suck_back_pct:6.2f}% | "
                      f"{(a.exit_velocity/d.exit_velocity-1)*100:+6.1f}%{mark}")

    print("\nBEST PER TOPOLOGY (<=300 A, <=60 C):")
    for t in ("diode", "ahb"):
        sub = rank([r for r in res if r.params["topology"] == t], "velocity",
                   thermal_limit=True, max_current=MAX_I)
        if sub:
            b = sub[0]
            print(f"  {t:5}: {b.exit_velocity:6.2f} m/s at frac="
                  f"{b.params['turn_off_fraction']}, {b.params['turns']} turns, "
                  f"back {b.suck_back_pct:.2f}%")

if __name__ == "__main__":
    main()
