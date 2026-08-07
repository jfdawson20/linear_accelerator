"""Stage trigger logic.

Models the real control chain rather than an idealised one:

    ingress sensor at the coil mouth detects the projectile nose
      -> sensor latency
      -> gate command
      -> switch turn-on latency
      -> coil conducts

and symmetrically for turn-off, which is commanded once the projectile has
travelled one projectile length past the sensor -- i.e. when its tail clears
the sensor and it is "completely inside".

At 150 m/s a 20 us latency is 3 mm of travel against a 17.5 mm half-coil, so
these delays are design constraints, not rounding errors. v1 modelled the whole
chain as instantaneous.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


class TriggerPhase(enum.Enum):
    WAITING = "waiting"  # projectile has not reached the trigger point
    COMMANDED_ON = "on"  # gate commanded on, coil conducting or about to
    COMMANDED_OFF = "off"  # gate commanded off, freewheeling or done


@dataclass
class StageController:
    """Trigger state for one stage.

    `lead_time` implements prefire: the gate is commanded early so that current
    peaks as the projectile arrives, rather than starting to rise then. It is
    recomputed against live velocity each step -- v1 compared against a value
    snapshotted at construction and, because it computed the lead distance as
    the full distance to the coil, produced a condition that was never true.
    """

    sensor_position: float  # m, absolute x of the ingress sensor
    projectile_length: float  # m, travel from trigger to "completely inside"
    lead_time: float = 0.0  # s, prefire lead (0 disables prefire)
    sensor_latency: float = 0.0  # s, detection -> gate command
    turn_on_latency: float = 0.0  # s, gate command -> conduction
    turn_off_latency: float = 0.0  # s

    phase: TriggerPhase = field(default=TriggerPhase.WAITING, init=False)
    gate_on: bool = field(default=False, init=False)
    _on_at: float | None = field(default=None, init=False)
    _off_at: float | None = field(default=None, init=False)
    fire_time: float | None = field(default=None, init=False)
    off_time: float | None = field(default=None, init=False)

    @property
    def release_position(self) -> float:
        """Nose position at which the tail has cleared the sensor."""
        return self.sensor_position + self.projectile_length

    def update(self, t: float, x: float, v: float) -> bool:
        """Advance the trigger state. Returns the current gate command.

        Called at step boundaries only; the gate is held fixed across an RK4
        step.
        """
        if self.phase is TriggerPhase.WAITING and self._should_fire(x, v):
            self.phase = TriggerPhase.COMMANDED_ON
            self._on_at = t + self.sensor_latency + self.turn_on_latency
            self.fire_time = t

        if self.phase is TriggerPhase.COMMANDED_ON and x >= self.release_position:
            self.phase = TriggerPhase.COMMANDED_OFF
            self._off_at = t + self.sensor_latency + self.turn_off_latency
            self.off_time = t

        on = self._on_at is not None and t >= self._on_at
        off = self._off_at is not None and t >= self._off_at
        self.gate_on = on and not off
        return self.gate_on

    def _should_fire(self, x: float, v: float) -> bool:
        """Trigger condition.

        Without prefire the coil fires as the nose reaches the sensor. With
        prefire it fires `lead_time` of travel earlier, so that current has
        risen by the time the projectile arrives.
        """
        if x >= self.sensor_position:
            return True
        if self.lead_time > 0.0 and v > 0.0:
            return (self.sensor_position - x) <= v * self.lead_time
        return False


def build_controllers(
    stages,
    projectile_length: float,
    prefire: bool,
    sensor_latency: float,
    sensor_offset: float,
    lead_times,
) -> list[StageController]:
    """One controller per stage.

    `lead_times` comes from each stage's circuit (time from firing to peak
    current), so prefire aims to have current peaking as the projectile arrives.
    """
    controllers = []
    for stage, lead in zip(stages, lead_times):
        controllers.append(
            StageController(
                sensor_position=stage.position + sensor_offset,
                projectile_length=projectile_length,
                lead_time=lead if prefire else 0.0,
                sensor_latency=sensor_latency,
                turn_on_latency=stage.switch.turn_on_latency,
                turn_off_latency=stage.switch.turn_off_latency,
            )
        )
    return controllers
