"""Simulation engine: RK4 integration of the coupled circuit and kinematics.

State vector layout (numpy float64):

    [0]           x, projectile nose position (m)
    [1]           v, projectile velocity (m/s)
    [2]           accumulated loss outside the windings (J) -- switch and diode
    [3 + 3k + 0]  i_k,  stage k coil current (A)
    [3 + 3k + 1]  vc_k, stage k capacitor voltage (V)
    [3 + 3k + 2]  T_k,  stage k winding temperature (C)

Switch states are discrete and are held fixed across an RK4 step, then updated
at the step boundary. Everything else is integrated continuously.

Every stage contributes force at every step, not just the stage the projectile
is currently inside. This is what lets the model see cross-stage effects: a
previous stage still freewheeling pulls backward on a departing projectile. v1
zeroed force outside [coil_start, coil_centre], which is why stage spacing had
no effect on its results.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .circuit import StageCircuit, SwitchState
from .config import SimConfig
from .control import StageController, build_controllers
from .kernel import FREEWHEEL, OFF, ON, StageKernel

_STATE_BASE = 3
_PER_STAGE = 3

_SWITCH_CODE = {
    SwitchState.OFF: OFF,
    SwitchState.ON: ON,
    SwitchState.FREEWHEEL: FREEWHEEL,
}


@dataclass
class RunSummary:
    """Scalar outcomes, maintained during the run so that `record=False` still
    yields everything a parameter sweep needs without keeping the trace."""

    exit_velocity: float
    peak_current: np.ndarray
    peak_temperature: np.ndarray
    forward_impulse: np.ndarray
    suck_back_impulse: np.ndarray
    conduction_time: np.ndarray
    steps: int
    duration: float


@dataclass
class EnergyAudit:
    """Where the capacitor energy went. Should close to within a fraction of a
    percent; if it does not, the model or the timestep is wrong.

    v1 had no such check, and reported 248 J of kinetic energy from a 960 J
    bank (26% efficiency) without flagging it.
    """

    initial: float
    capacitor: float
    magnetic: float
    kinetic: float
    winding_heat: float
    external_loss: float
    discarded: float

    @property
    def accounted(self) -> float:
        return (
            self.capacitor
            + self.magnetic
            + self.kinetic
            + self.winding_heat
            + self.external_loss
            + self.discarded
        )

    @property
    def residual(self) -> float:
        return self.initial - self.accounted

    @property
    def closure_error(self) -> float:
        """Unaccounted fraction of the initial energy."""
        return abs(self.residual) / self.initial if self.initial else 0.0

    @property
    def efficiency(self) -> float:
        return self.kinetic / self.initial if self.initial else 0.0


@dataclass
class RunResult:
    config: SimConfig
    time: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    current: np.ndarray  # (steps, stages)
    voltage: np.ndarray  # (steps, stages)
    temperature: np.ndarray  # (steps, stages)
    force: np.ndarray  # (steps, stages)
    inductance: np.ndarray  # (steps, stages)
    switch_state: np.ndarray  # (steps, stages) as small ints
    energy: EnergyAudit
    controllers: list[StageController]
    terminated: str
    summary: RunSummary
    warnings: list[str] = field(default_factory=list)

    @property
    def recorded(self) -> bool:
        return self.config.record

    @property
    def exit_velocity(self) -> float:
        return self.summary.exit_velocity

    @property
    def peak_current(self) -> np.ndarray:
        return self.summary.peak_current

    @property
    def peak_temperature(self) -> np.ndarray:
        return self.summary.peak_temperature

    def stage_conduction_time(self) -> np.ndarray:
        """Time each stage spent with current flowing (ON or FREEWHEEL)."""
        return self.summary.conduction_time

    def suck_back_impulse(self) -> np.ndarray:
        """Negative (retarding) impulse per stage, in N.s.

        The number the 2:1 design premise exists to keep small. Non-zero values
        mean current was still flowing when the force reversed.
        """
        return self.summary.suck_back_impulse

    def forward_impulse(self) -> np.ndarray:
        return self.summary.forward_impulse


class Simulation:
    def __init__(self, config: SimConfig):
        self.config = config
        self.mass = config.projectile.mass
        self.circuits = [
            StageCircuit(
                config=stage,
                projectile=config.projectile,
                saturation=config.saturation,
                coupling=config.coupling,
            )
            for stage in config.stages
        ]
        self.controllers = build_controllers(
            stages=config.stages,
            projectile_length=config.projectile.length,
            prefire=config.control.prefire,
            sensor_latency=config.control.sensor_latency,
            sensor_offset=config.control.sensor_offset,
            lead_times=[c.time_to_peak_current() * config.control.prefire_scale
                        for c in self.circuits],
            turn_off_fraction=config.control.turn_off_fraction,
        )
        self.n = len(self.circuits)
        self.kernel = StageKernel(
            self.circuits,
            saturation=config.saturation,
            projectile_length=config.projectile.length,
        )

    # -- validation -----------------------------------------------------

    def check_timestep(self) -> list[str]:
        """Warn if dt is too coarse for the fastest dynamics in the model.

        v1 hardcoded 1e-6 in six places and never checked it against anything.
        """
        warnings: list[str] = []
        dt = self.config.dt
        for k, circ in enumerate(self.circuits):
            L = circ.magnetics.l_air
            R = circ.r20 + circ.config.switch.on_resistance
            C = circ.config.bank.capacitance
            tau = L / R
            period = 2 * np.pi * np.sqrt(L * C)
            if dt > tau / 10:
                warnings.append(
                    f"stage {k}: dt={dt:.2e}s is coarse against L/R={tau:.2e}s"
                )
            if dt > period / 50:
                warnings.append(
                    f"stage {k}: dt={dt:.2e}s is coarse against the ring "
                    f"period {period:.2e}s"
                )
        return warnings

    # -- integration ----------------------------------------------------

    def _derivatives(self, y: np.ndarray, switch_codes: np.ndarray) -> np.ndarray:
        """Fused, vectorised derivative evaluation across all stages.

        `circuit.StageCircuit.derivatives` is the readable reference for the
        same equations; tests/test_kernel.py asserts the two agree.
        """
        x, v = y[0], y[1]
        dy = np.empty_like(y)
        dy[0] = v

        i = y[_STATE_BASE + 0 :: _PER_STAGE]
        vc = y[_STATE_BASE + 1 :: _PER_STAGE]
        temp = y[_STATE_BASE + 2 :: _PER_STAGE]

        di, dvc, dtemp, force, external = self.kernel.derivatives(
            x, v, i, vc, temp, switch_codes
        )

        dy[1] = force / self.mass
        dy[2] = external
        dy[_STATE_BASE + 0 :: _PER_STAGE] = di
        dy[_STATE_BASE + 1 :: _PER_STAGE] = dvc
        dy[_STATE_BASE + 2 :: _PER_STAGE] = dtemp
        return dy

    def _rk4(self, y: np.ndarray, switch_codes: np.ndarray, dt: float):
        k1 = self._derivatives(y, switch_codes)
        k2 = self._derivatives(y + 0.5 * dt * k1, switch_codes)
        k3 = self._derivatives(y + 0.5 * dt * k2, switch_codes)
        k4 = self._derivatives(y + dt * k3, switch_codes)
        return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def run(self, max_time: float = 0.5) -> RunResult:
        cfg = self.config
        dt = cfg.dt
        warnings = self.check_timestep()

        max_steps = int(max_time / dt) + 2
        y = np.zeros(_STATE_BASE + _PER_STAGE * self.n)
        for k, circ in enumerate(self.circuits):
            b = _STATE_BASE + _PER_STAGE * k
            y[b + 1] = circ.config.bank.voltage
            y[b + 2] = cfg.thermal.ambient_c

        switches = [SwitchState.OFF] * self.n
        switch_codes = np.zeros(self.n, dtype=np.int64)
        conducted = [False] * self.n
        discarded = 0.0

        # Recording buffers. With record=False only the running summary is
        # kept, which is what makes parameter sweeps affordable.
        cap = max_steps if cfg.record else 1
        t_a = np.zeros(cap)
        x_a = np.zeros(cap)
        v_a = np.zeros(cap)
        i_a = np.zeros((cap, self.n))
        vc_a = np.zeros((cap, self.n))
        tp_a = np.zeros((cap, self.n))
        f_a = np.zeros((cap, self.n))
        l_a = np.zeros((cap, self.n))
        sw_a = np.zeros((cap, self.n), dtype=np.int8)

        # Terminate once the projectile has cleared the last coil, with a
        # little margin so any trailing suck-back is captured. v1 ran to a
        # hardcoded 1 m regardless of the machine's actual length.
        finish_x = cfg.barrel_length + 2 * cfg.projectile.length
        stall_after = 0.0
        terminated = "max_time"
        step = 0

        # Running summary, maintained whether or not the trace is recorded.
        peak_i = np.zeros(self.n)
        peak_t = np.full(self.n, cfg.thermal.ambient_c)
        fwd_impulse = np.zeros(self.n)
        back_impulse = np.zeros(self.n)
        conduction = np.zeros(self.n)

        for step in range(max_steps):
            t = step * dt
            x, v = y[0], y[1]
            i_vec = y[_STATE_BASE + 0 :: _PER_STAGE]

            force_vec = self.kernel.force_only(x, i_vec)
            np.maximum(peak_i, np.abs(i_vec), out=peak_i)
            np.maximum(peak_t, y[_STATE_BASE + 2 :: _PER_STAGE], out=peak_t)
            fwd_impulse += np.where(force_vec > 0, force_vec, 0.0) * dt
            back_impulse += np.where(force_vec < 0, force_vec, 0.0) * dt
            conduction += (switch_codes != 0) * dt

            if cfg.record:
                t_a[step] = t
                x_a[step] = x
                v_a[step] = v
                i_a[step] = i_vec
                vc_a[step] = y[_STATE_BASE + 1 :: _PER_STAGE]
                tp_a[step] = y[_STATE_BASE + 2 :: _PER_STAGE]
                f_a[step] = force_vec
                l_a[step] = self.kernel.inductance_only(x, i_vec)
                sw_a[step] = switch_codes

            if x >= finish_x:
                terminated = "cleared_barrel"
                break

            y = self._rk4(y, switch_codes, dt)
            t_next = t + dt

            # Discrete updates at the step boundary. Only once per step, so
            # this stays on the readable per-stage path.
            quiescent = True
            for k, circ in enumerate(self.circuits):
                b = _STATE_BASE + _PER_STAGE * k
                gate = self.controllers[k].update(t_next, y[0], y[1])
                threshold = circ.config.switch.off_current_threshold
                if abs(y[b]) > threshold:
                    conducted[k] = True
                nxt = circ.next_switch_state(switches[k], gate, y[b], conducted[k])
                if nxt is SwitchState.OFF and switches[k] is not SwitchState.OFF:
                    # Residual magnetic energy is dropped when the diode stops
                    # conducting; book it so the energy audit stays closed.
                    discarded += circ.stored_magnetic_energy(y[0], y[b])
                    y[b] = 0.0
                switches[k] = nxt
                switch_codes[k] = _SWITCH_CODE[nxt]
                if abs(y[b]) > threshold:
                    quiescent = False

            # Stall guard: nothing is moving and no current is flowing
            # anywhere, so nothing can change. Keyed on current rather than on
            # switch state, because a stage can sit gated ON with a flat
            # capacitor and never draw anything.
            if quiescent and abs(y[1]) < 1e-9:
                stall_after += dt
                if stall_after > 5e-3:
                    terminated = "stalled"
                    break
            else:
                stall_after = 0.0
        else:
            step = max_steps - 1

        used = step + 1
        energy = self._audit(y, discarded)
        summary = RunSummary(
            exit_velocity=float(y[1]),
            peak_current=peak_i,
            peak_temperature=peak_t,
            forward_impulse=fwd_impulse,
            suck_back_impulse=back_impulse,
            conduction_time=conduction,
            steps=used,
            duration=used * dt,
        )

        return RunResult(
            config=cfg,
            time=t_a[:used],
            position=x_a[:used],
            velocity=v_a[:used],
            current=i_a[:used],
            voltage=vc_a[:used],
            temperature=tp_a[:used],
            force=f_a[:used],
            inductance=l_a[:used],
            switch_state=sw_a[:used],
            energy=energy,
            controllers=self.controllers,
            terminated=terminated,
            warnings=warnings,
            summary=summary,
        )

    def _audit(self, y: np.ndarray, discarded: float) -> EnergyAudit:
        cfg = self.config
        i = y[_STATE_BASE + 0 :: _PER_STAGE]
        vc = y[_STATE_BASE + 1 :: _PER_STAGE]
        temp = y[_STATE_BASE + 2 :: _PER_STAGE]
        return EnergyAudit(
            initial=cfg.total_stored_energy,
            capacitor=float((0.5 * self.kernel.cap * vc * vc).sum()),
            magnetic=float(self.kernel.stored_magnetic_energy(y[0], i).sum()),
            kinetic=0.5 * self.mass * y[1] * y[1],
            winding_heat=float(
                (self.kernel.thermal_mass * (temp - cfg.thermal.ambient_c)).sum()
            ),
            external_loss=y[2],
            discarded=discarded,
        )


def run(config: SimConfig, max_time: float = 0.5) -> RunResult:
    return Simulation(config).run(max_time=max_time)
