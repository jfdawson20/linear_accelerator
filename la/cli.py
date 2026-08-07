"""Command line entry point.

Notes on the flags, against v1:

  - boolean flags use store_true. v1 declared them with `default=False` and a
    string type, so `bool("False")` was True and `-v False` enabled verbose.
  - projectile length is a real parameter, defaulting to half the coil length
    (the 2:1 design premise). v1 had no such parameter and a hardcoded mass
    that implied a 1.37:1 ratio.
  - turns and gauge are exposed. v1 hardcoded them in the main block, so the
    sweep space it built had exactly one distinct member.
"""

from __future__ import annotations

import argparse
import sys

from .config import (
    CapacitorBank,
    ControlConfig,
    SimConfig,
    ThermalConfig,
    uniform_stages,
)
from .engine import Simulation
from .geometry import CoilGeometry, ProjectileSpec
from .report import (
    coil_table,
    force_profile_table,
    plot,
    print_report,
)
from .wire import WireSpec, available_gauges


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="la",
        description="Multi-stage solenoid linear accelerator design tool",
    )
    p.add_argument(
        "command",
        choices=("run", "design", "profile"),
        help="run: simulate. design: static parameters only. "
        "profile: force vs position for one stage.",
    )

    coil = p.add_argument_group("coil")
    coil.add_argument("-l", "--length", type=float, default=0.035,
                      help="coil length, m (default 0.035)")
    coil.add_argument("-n", "--turns", type=int, default=150,
                      help="turns per coil (default 150)")
    coil.add_argument("-g", "--gauge", type=int, default=26,
                      choices=available_gauges(), help="wire gauge, AWG")
    coil.add_argument("--build", default="single",
                      choices=("bare", "single", "heavy"),
                      help="insulation build (default single)")
    coil.add_argument("--bore", type=float, default=0.0098,
                      help="winding inner diameter, m (default 0.0098)")

    proj = p.add_argument_group("projectile")
    proj.add_argument("--proj-len", type=float, default=None,
                      help="projectile length, m (default: half the coil "
                           "length, i.e. the 2:1 design premise)")
    proj.add_argument("--proj-dia", type=float, default=0.006,
                      help="projectile diameter, m (default 0.006)")
    proj.add_argument("--density", type=float, default=7850.0,
                      help="projectile density, kg/m3 (default 7850, steel)")
    proj.add_argument("--mu-r", type=float, default=100.0,
                      help="bulk relative permeability (default 100)")
    proj.add_argument("--b-sat", type=float, default=1.6,
                      help="saturation flux density, T (default 1.6)")

    elec = p.add_argument_group("electrical")
    elec.add_argument("-s", "--stages", type=int, default=8,
                      help="number of stages (default 8)")
    elec.add_argument("--spacing", type=float, default=0.0188,
                      help="gap between stages, m (default 0.0188)")
    elec.add_argument("-c", "--cap", type=float, default=0.006,
                      help="capacitance per stage, F (default 0.006)")
    elec.add_argument("-V", "--volts", type=float, default=200.0,
                      help="initial capacitor voltage (default 200)")
    elec.add_argument("--diode-vf", type=float, default=1.5,
                      help="freewheel diode forward drop, V (default 1.5)")

    ctrl = p.add_argument_group("control")
    ctrl.add_argument("--no-prefire", action="store_true",
                      help="fire on arrival rather than leading the projectile")
    ctrl.add_argument("--sensor-latency", type=float, default=0.0,
                      help="sensor detect to gate command, s")
    ctrl.add_argument("--switch-latency", type=float, default=0.0,
                      help="gate command to conduction, s")

    model = p.add_argument_group("model")
    model.add_argument("--no-saturation", action="store_true",
                       help="disable saturation (reduces to 0.5*i^2*dL/dx)")
    model.add_argument("--dt", type=float, default=1e-6,
                       help="timestep, s (default 1e-6)")
    model.add_argument("--max-time", type=float, default=0.5,
                       help="simulation time limit, s (default 0.5)")
    model.add_argument("--ambient", type=float, default=25.0,
                       help="ambient temperature, C")
    model.add_argument("--max-temp", type=float, default=60.0,
                       help="winding temperature limit, C")

    out = p.add_argument_group("output")
    out.add_argument("-v", "--verbose", action="store_true",
                     help="include the design table")
    out.add_argument("-p", "--plot", action="store_true", help="show plots")
    out.add_argument("--save-plot", metavar="PATH", help="write plots to a file")
    out.add_argument("--profile-current", type=float, default=200.0,
                     help="current for the force profile, A (default 200)")
    return p


def config_from_args(args: argparse.Namespace) -> SimConfig:
    coil = CoilGeometry(
        length=args.length,
        bore_diameter=args.bore,
        turns=args.turns,
        wire=WireSpec(gauge=args.gauge, build=args.build),
    )
    proj_len = args.proj_len if args.proj_len is not None else args.length / 2.0
    projectile = ProjectileSpec(
        length=proj_len,
        diameter=args.proj_dia,
        density=args.density,
        mu_r=args.mu_r,
        b_sat=args.b_sat,
    )
    from .config import SwitchSpec

    switch = SwitchSpec(
        diode_vf=args.diode_vf,
        turn_on_latency=args.switch_latency,
        turn_off_latency=args.switch_latency,
    )
    stages = uniform_stages(
        coil=coil,
        bank=CapacitorBank(capacitance=args.cap, voltage=args.volts),
        count=args.stages,
        spacing=args.spacing,
        switch=switch,
    )
    return SimConfig(
        projectile=projectile,
        stages=stages,
        control=ControlConfig(
            prefire=not args.no_prefire, sensor_latency=args.sensor_latency
        ),
        thermal=ThermalConfig(ambient_c=args.ambient, max_c=args.max_temp),
        dt=args.dt,
        saturation=not args.no_saturation,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = config_from_args(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    sim = Simulation(config)

    if args.command == "design":
        print(coil_table(sim))
        return 0

    if args.command == "profile":
        print(force_profile_table(sim, args.profile_current))
        return 0

    result = sim.run(max_time=args.max_time)
    print_report(result, sim, verbose=args.verbose)
    if args.plot or args.save_plot:
        plot(result, args.save_plot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
