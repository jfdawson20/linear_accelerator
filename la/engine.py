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

_STATE_BASE = 3
_PER_STAGE = 3


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
    warnings: list[str] = field(default_factory=list)

    @property
    def exit_velocity(self) -> float:
        return float(self.velocity[-1])

    @property
    def peak_current(self) -> np.ndarray:
        return np.abs(self.current).max(axis=0)

    @property
    def peak_temperature(self) -> np.ndarray:
        return self.temperature.max(axis=0)

    def stage_conduction_time(self) -> np.ndarray:
        """Time each stage spent with current flowing (ON or FREEWHEEL)."""
        dt = self.config.dt
        return (self.switch_state != 0).sum(axis=0) * dt

    def suck_back_impulse(self) -> np.ndarray:
        """Negative (retarding) impulse per stage, in N.s.

        The number the 2:1 design premise exists to keep small. Non-zero values
        mean the field had not collapsed before the projectile crossed centre.
        """
        dt = self.config.dt
        return np.where(self.force < 0, self.force, 0.0).sum(axis=0) * dt

    def forward_impulse(self) -> np.ndarray:
        dt = self.config.dt
        return np.where(self.force > 0, self.force, 0.0).sum(axis=0) * dt


class Simulation:
    def __init__(self, config: SimConfig):
        self.config = config
        self.mass = config.projectile.mass
        self.circuits = [
            StageCircuit(
                config=stage,
                projectile=config.projectile,
                saturation=config.saturation,
            )
            for stage in config.stages
        ]
        self.controllers = build_controllers(
            stages=config.stages,
            projectile_length=config.projectile.length,
            prefire=config.control.prefire,
            sensor_latency=config.control.sensor_latency,
            sensor_offset=config.control.sensor_offset,
            lead_times=[c.time_to_peak_current() for c in self.circuits],
        )
        self.n = len(self.circuits)

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

    def _derivatives(self, y: np.ndarray, switches: list[SwitchState]) -> np.ndarray:
        x, v = y[0], y[1]
        dy = np.zeros_like(y)
        dy[0] = v

        total_force = 0.0
        external_power = 0.0
        for k, circ in enumerate(self.circuits):
            b = _STATE_BASE + _PER_STAGE * k
            i, vc, temp = y[b], y[b + 1], y[b + 2]
            di, dvc, dtemp, p_ext = circ.derivatives(
                i, vc, temp, x, v, switches[k]
            )
            dy[b], dy[b + 1], dy[b + 2] = di, dvc, dtemp
            external_power += p_ext
            # Every stage pushes or pulls, wherever the projectile is.
            total_force += float(circ.magnetics.force(x, i))

        dy[1] = total_force / self.mass
        dy[2] = external_power
        return dy

    def _rk4(self, y: np.ndarray, switches: list[SwitchState], dt: float):
        k1 = self._derivatives(y, switches)
        k2 = self._derivatives(y + 0.5 * dt * k1, switches)
        k3 = self._derivatives(y + 0.5 * dt * k2, switches)
        k4 = self._derivatives(y + dt * k3, switches)
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
        conducted = [False] * self.n
        discarded = 0.0

        # Recording buffers.
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

        for step in range(max_steps):
            t = step * dt
            x, v = y[0], y[1]

            if cfg.record:
                t_a[step] = t
                x_a[step] = x
                v_a[step] = v
                for k, circ in enumerate(self.circuits):
                    b = _STATE_BASE + _PER_STAGE * k
                    i = y[b]
                    i_a[step, k] = i
                    vc_a[step, k] = y[b + 1]
                    tp_a[step, k] = y[b + 2]
                    f_a[step, k] = float(circ.magnetics.force(x, i))
                    l_a[step, k] = float(circ.magnetics.l_incremental(x, i))
                    sw_a[step, k] = 0 if switches[k] is SwitchState.OFF else 1

            if x >= finish_x:
                terminated = "cleared_barrel"
                break

            y = self._rk4(y, switches, dt)
            t_next = t + dt

            # Discrete updates at the step boundary.
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
        )

    def _audit(self, y: np.ndarray, discarded: float) -> EnergyAudit:
        cfg = self.config
        cap_energy = 0.0
        magnetic = 0.0
        winding = 0.0
        for k, circ in enumerate(self.circuits):
            b = _STATE_BASE + _PER_STAGE * k
            i, vc, temp = y[b], y[b + 1], y[b + 2]
            cap_energy += 0.5 * circ.config.bank.capacitance * vc * vc
            magnetic += circ.stored_magnetic_energy(y[0], i)
            winding += circ.thermal_mass * (temp - cfg.thermal.ambient_c)
        return EnergyAudit(
            initial=cfg.total_stored_energy,
            capacitor=cap_energy,
            magnetic=magnetic,
            kinetic=0.5 * self.mass * y[1] * y[1],
            winding_heat=winding,
            external_loss=y[2],
            discarded=discarded,
        )


def run(config: SimConfig, max_time: float = 0.5) -> RunResult:
    return Simulation(config).run(max_time=max_time)
