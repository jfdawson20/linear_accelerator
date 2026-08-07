#!/usr/bin/env python3
"""Is the asymmetric half bridge worth the extra complexity?

Compares a series switch + freewheel diode against a two-IGBT asymmetric half
bridge over the region the coarse study identified, with turns extended past
the boundary the coarse grid hit.

Both topologies are charged a realistic device drop, so the bridge pays for its
two devices in series against the diode design's one.
"""
from __future__ import annotations
import json, time
from la.sweep import ParameterSpace, rank, results_table, sweep

INCH = 0.0254
SPACE = ParameterSpace(
    topology=["diode", "ahb"],
    turns=[300, 400, 500, 600],
    gauge=[20, 22, 24],
    coil_length=[1.5 * INCH, 2.0 * INCH],
    capacitance=[2e-3, 4e-3, 8e-3],
)
FIXED = dict(stages=8, ratio=2.0, proj_dia=0.25 * INCH, bore=0.009,
             spacing=0.015, voltage=300.0, dt=1e-5, max_temp=60.0,
             device_drop=1.8)
MAX_I = 300.0

def main() -> None:
    print(f"topology study: {len(SPACE)} configurations")
    t0 = time.perf_counter()
    res = sweep(SPACE, fixed=FIXED, workers=8)
    print(f"done in {time.perf_counter()-t0:.0f}s")
    json.dump([{**r.params, "exit_velocity": r.exit_velocity,
                "efficiency": r.efficiency, "peak_current": r.peak_current,
                "peak_temperature": r.peak_temperature,
                "suck_back_pct": r.suck_back_pct, "stored_energy": r.stored_energy,
                "closure_error": r.closure_error, "terminated": r.terminated,
                "error": r.error} for r in res], open("study_topology.json", "w"))

    axes = list(SPACE.axes)
    for topo in ("diode", "ahb"):
        sub = [r for r in res if r.params["topology"] == topo]
        good = rank(sub, "velocity", thermal_limit=True, max_current=MAX_I)
        print(f"\n=== {topo.upper()}  ({len(good)}/{len(sub)} viable) ===")
        print(results_table(good, axes, limit=6))
    bad = [r for r in res if r.ok and r.closure_error > 1e-3]
    print(f"\n{len(bad)} points failed energy closure")

if __name__ == "__main__":
    main()
