#!/usr/bin/env python3
"""Coarse design search under the build constraints.

Constraints (from the build brief):
    projectile diameter  0.25" (6.35 mm), fixed
    projectile length    0.25" - 1" (6.35 - 25.4 mm)
    coil length          2x projectile length (the design premise)
    capacitor voltage    <= 300 V
    peak current         <= 300 A
    stages               <= 8

Objective: maximum exit velocity.

Bore is fixed at 9 mm: a 7 mm ID barrel with a 1 mm wall around a 6.35 mm slug.
It is a real design lever (force scales with the fill fraction, which goes as
(proj_dia/bore)^2) and is swept separately in the follow-up study.
"""

from __future__ import annotations

import json
import time

from la.sweep import ParameterSpace, rank, results_table, sensitivity, sweep

MM = 1e-3
INCH = 25.4 * MM

# Coil length = 2 x projectile length, so these span 0.25" - 1" projectiles.
COIL_LENGTHS = [0.5 * INCH, 0.75 * INCH, 1.0 * INCH, 1.5 * INCH, 2.0 * INCH]

SPACE = ParameterSpace(
    coil_length=[round(v, 6) for v in COIL_LENGTHS],
    turns=[50, 100, 200, 300, 400],
    gauge=[20, 22, 24, 26],
    capacitance=[1e-3, 2e-3, 4e-3, 8e-3],
    voltage=[225.0, 300.0],
)

FIXED = dict(
    stages=8,
    ratio=2.0,
    proj_dia=0.25 * INCH,
    bore=9.0 * MM,
    spacing=15.0 * MM,
    dt=1e-5,
    max_temp=60.0,
)

MAX_CURRENT = 300.0


def main() -> None:
    print(f"coarse search: {len(SPACE)} configurations")
    t0 = time.perf_counter()
    results = sweep(SPACE, fixed=FIXED, workers=8)
    elapsed = time.perf_counter() - t0
    print(f"done in {elapsed:.0f}s ({elapsed / len(SPACE) * 1000:.0f} ms/point)")

    with open("study_coarse.json", "w") as fh:
        json.dump(
            [
                {
                    "params": r.params,
                    "exit_velocity": r.exit_velocity,
                    "efficiency": r.efficiency,
                    "stored_energy": r.stored_energy,
                    "peak_current": r.peak_current,
                    "saturation_ratio": r.saturation_ratio,
                    "peak_temperature": r.peak_temperature,
                    "suck_back_pct": r.suck_back_pct,
                    "closure_error": r.closure_error,
                    "terminated": r.terminated,
                    "error": r.error,
                }
                for r in results
            ],
            fh,
        )

    axes = list(SPACE.axes)
    failed = [r for r in results if not r.ok]
    unfinished = [r for r in results if r.ok and r.terminated == "max_time"]
    bad_closure = [r for r in results if r.ok and r.closure_error > 1e-3]

    print(f"\n{len(failed)} failed to build, {len(unfinished)} hit the time limit, "
          f"{len(bad_closure)} failed energy closure")
    if failed:
        print(f"  first failure: {failed[0].error}")

    viable = rank(results, "velocity", thermal_limit=True, max_current=MAX_CURRENT)
    print(f"\n{len(viable)} viable of {len(results)} "
          f"(within 60 C and {MAX_CURRENT:.0f} A)")

    print("\nTOP 15 BY EXIT VELOCITY (constrained)")
    print(results_table(viable, axes, limit=15))

    print("\nTOP 5 IF THE THERMAL LIMIT IS RELAXED (current limit still applies)")
    hot = rank(results, "velocity", thermal_limit=False, max_current=MAX_CURRENT)
    print(results_table(hot, axes, limit=5))

    print("\nSENSITIVITY (best exit velocity per axis value, constrained set)")
    for axis in axes:
        rows = sensitivity(viable, axis, "velocity")
        if not rows:
            continue
        span = max(r[1] for r in rows) - min(r[1] for r in rows)
        print(f"\n  {axis}  (spans {span:.1f} m/s)")
        for value, best, mean in rows:
            print(f"    {str(value):>10}   best {best:7.2f}   mean {mean:7.2f}")


if __name__ == "__main__":
    main()
