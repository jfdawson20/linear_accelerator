#!/usr/bin/env python3
"""How far can turn count go now the half bridge removes the suck-back limit?

CAVEAT, stated up front: fill() scales the slug's permeability contribution by
proj_area/bore_area, which does not depend on winding depth. Every layer is
therefore treated as coupling to the slug equally well, when in reality outer
layers have flux paths that largely bypass it. The model over-rewards high turn
counts, and the error grows with layer count. Results are annotated with layers
and outer diameter so the unreliable region is visible.
"""
from __future__ import annotations
import json, time
from la.geometry import CoilGeometry
from la.wire import WireSpec
from la.sweep import ParameterSpace, sweep

INCH = 0.0254
BORE = 0.009
COIL_LEN = 2.0 * INCH

SPACE = ParameterSpace(
    turns=[200, 300, 400, 500, 600, 800, 1000, 1200],
    gauge=[20, 22, 24],
    turn_off_fraction=[0.4, 0.667, 0.8],
    capacitance=[4e-3, 8e-3],
)
COUPLING = __import__("os").environ.get("COUPLING","bore")
FIXED = dict(coupling=COUPLING, stages=8, ratio=2.0, proj_dia=0.25 * INCH, bore=BORE,
             coil_length=COIL_LEN, spacing=0.015, voltage=300.0, dt=1e-5,
             max_temp=60.0, device_drop=1.8, topology="ahb")
MAX_I = 300.0


def geom(turns, gauge):
    c = CoilGeometry(COIL_LEN, BORE, turns, WireSpec(gauge, "single"))
    return c.layers, c.outer_radius * 2 * 1e3, c.resistance(), c.inductance_air * 1e6


def main() -> None:
    print(f"turn-count study: {len(SPACE)} configurations")
    t0 = time.perf_counter()
    res = sweep(SPACE, fixed=FIXED, workers=8)
    print(f"done in {time.perf_counter()-t0:.0f}s")
    json.dump([{**r.params, "exit_velocity": r.exit_velocity,
                "efficiency": r.efficiency, "peak_current": r.peak_current,
                "peak_temperature": r.peak_temperature,
                "suck_back_pct": r.suck_back_pct,
                "closure_error": r.closure_error, "terminated": r.terminated,
                "error": r.error} for r in res], open("study_turns.json", "w"))
    bad = [r for r in res if r.ok and r.closure_error > 1e-3]
    print(f"{len(bad)} failed energy closure\n")

    ok = [r for r in res if r.ok and r.peak_current <= MAX_I
          and r.peak_temperature <= 60 and r.terminated != "max_time"]
    for gauge in (20, 22, 24):
        print(f"=== {gauge} AWG ===")
        print(f"{'turns':>6} {'layers':>7} {'OD mm':>7} {'L uH':>7} "
              f"{'best v':>8} {'frac':>6} {'back %':>7} {'peak I':>7} {'maxT':>5}")
        for t in SPACE.axes["turns"]:
            cand = [r for r in ok if r.params["turns"] == t
                    and r.params["gauge"] == gauge]
            if not cand:
                print(f"{t:6d}   -- no viable point (current or thermal limit) --")
                continue
            b = max(cand, key=lambda r: r.exit_velocity)
            layers, od, R, L = geom(t, gauge)
            flag = "  <-- many layers, model over-rewards" if layers > 8 else ""
            print(f"{t:6d} {layers:7d} {od:7.1f} {L:7.0f} {b.exit_velocity:8.2f} "
                  f"{b.params['turn_off_fraction']:6.3f} {b.suck_back_pct:6.2f}% "
                  f"{b.peak_current:7.0f} {b.peak_temperature:5.0f}{flag}")
        print()


if __name__ == "__main__":
    main()
