"""Per-stage electrical model: capacitor, coil, switch, freewheel path.

v1 evaluated a closed-form constant-L RLC solution at each timestep with a
freshly recomputed L. That is not a solution of the ODE with time-varying L: it
discontinuously jumps onto a different analytic waveform every step, omits the
motional back-EMF entirely, and never lets the capacitor voltage feed back into
the current. Here the circuit is integrated as a genuine ODE alongside the
kinematics.

Turn-off is a physical event. v1 modelled it by resetting the coil's clock to
zero, forcing current from ~280 A to 0 in a single 1 us step. Real current
decays through the freewheel diode on L/R, and that decay is precisely the
interval the 2:1 coil-to-projectile geometry exists to accommodate.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np

from .config import StageConfig
from .geometry import ProjectileSpec
from .magnetics import MagneticModel

# Copper, for adiabatic coil heating.
CU_DENSITY = 8960.0  # kg/m^3
CU_SPECIFIC_HEAT = 385.0  # J/(kg.K)


class SwitchState(enum.Enum):
    """Discrete switch states, held fixed across an RK4 step and updated at
    step boundaries."""

    OFF = "off"
    ON = "on"
    FREEWHEEL = "freewheel"


@dataclass
class StageCircuit:
    """Immutable per-stage model. All mutable state lives in the engine's state
    vector so the derivative function stays pure."""

    config: StageConfig
    projectile: ProjectileSpec
    saturation: bool = True
    mu_eff_override: float | None = None
    l_air_scale: float = 1.0
    r_scale: float = 1.0

    def __post_init__(self) -> None:
        self.magnetics = MagneticModel(
            coil=self.config.coil,
            projectile=self.projectile,
            coil_position=self.config.position,
            saturation=self.saturation,
            l_air_scale=self.l_air_scale,
            mu_eff_override=self.mu_eff_override,
        )
        self.r20 = self.config.coil.resistance() * self.r_scale
        # Thermal mass of the winding, for adiabatic heating.
        copper_volume = self.config.coil.wire_length * self.config.coil.wire.area
        self.thermal_mass = copper_volume * CU_DENSITY * CU_SPECIFIC_HEAT  # J/K

    # -- electrical -----------------------------------------------------

    def coil_resistance(self, temperature_c: float) -> float:
        """Winding resistance at temperature. v1 held this at 20 C while
        predicting 60 C coils -- ~16% on the quantity that sets damping."""
        from .wire import ALPHA_CU, T_REF_C

        return self.r20 * (1.0 + ALPHA_CU * (temperature_c - T_REF_C))

    def derivatives(
        self,
        i: float,
        vc: float,
        temperature_c: float,
        x: float,
        v: float,
        switch: SwitchState,
    ) -> tuple[float, float, float, float]:
        """Return (di/dt, dVc/dt, dT/dt, power_dissipated_outside_the_coil).

        Sign convention: i > 0 is capacitor discharging into the coil.
        """
        if switch is SwitchState.OFF:
            return 0.0, 0.0, 0.0, 0.0

        r_coil = self.coil_resistance(temperature_c)
        l_inc = float(self.magnetics.l_incremental(x, i))
        back_emf = float(self.magnetics.dlambda_dx(x, i)) * v

        sw = self.config.switch
        n_dev = sw.conduction_devices

        if switch is not SwitchState.ON and i <= 0.0:
            # The return path is diodes, which block reverse current. Without
            # this the current can be driven straight through zero -- the half
            # bridge puts hundreds of volts across the coil, so di/dt is steep
            # enough to overshoot within one step, and sign(i) then flips
            # mid-substep and pumps energy the wrong way.
            return 0.0, 0.0, 0.0, 0.0

        if switch is SwitchState.ON:
            r_total = r_coil + sw.on_resistance
            drop = n_dev * sw.device_drop * np.sign(i)
            drive = vc - drop
            dvc_dt = -i / self.config.bank.capacitance
            external_power = i * i * sw.on_resistance + abs(i) * n_dev * sw.device_drop
        elif sw.recovers_energy:
            # Asymmetric half bridge, both devices off: the coil is clamped
            # across the bank in reverse through two diodes, and its current
            # charges the capacitor back up.
            r_total = r_coil
            drive = -(abs(vc) + 2.0 * sw.diode_vf) * np.sign(i)
            dvc_dt = abs(i) / self.config.bank.capacitance
            external_power = abs(i) * 2.0 * sw.diode_vf
        else:
            # Series switch with a freewheel diode: the coil sees only the
            # diode drop, so the field decays on L/R and its energy is burnt.
            r_total = r_coil
            drive = -sw.diode_vf * np.sign(i)
            dvc_dt = 0.0
            external_power = abs(i) * sw.diode_vf

        di_dt = (drive - i * r_total - back_emf) / l_inc
        dT_dt = (i * i * r_coil) / self.thermal_mass
        return di_dt, dvc_dt, dT_dt, external_power

    # -- discrete transitions -------------------------------------------

    def next_switch_state(
        self, switch: SwitchState, gate_on: bool, i: float, conducted: bool
    ) -> SwitchState:
        """Switch state at the next step boundary.

        Models the usual coilgun topology: a series thyristor with a freewheel
        diode across the coil. The thyristor cannot block until its current
        reaches zero, and it cannot conduct in reverse -- so commanding the gate
        off hands the current to the diode rather than extinguishing it.
        """
        threshold = self.config.switch.off_current_threshold

        if switch is SwitchState.ON:
            if not gate_on:
                return SwitchState.FREEWHEEL
            if conducted and i <= 0.0:
                # capacitor has rung down and current wants to reverse;
                # the thyristor commutates off and the diode picks up
                return SwitchState.FREEWHEEL
            return SwitchState.ON

        if switch is SwitchState.FREEWHEEL:
            if i <= threshold:
                return SwitchState.OFF
            return SwitchState.FREEWHEEL

        # OFF
        if gate_on:
            return SwitchState.ON
        return SwitchState.OFF

    # -- diagnostics ----------------------------------------------------

    def stored_magnetic_energy(self, x: float, i: float) -> float:
        """W = lambda*i - W', the true stored energy (not the co-energy).

        The two differ once the model saturates; using co-energy here would
        break the energy audit in exactly the regime this design runs in.
        """
        lam = float(self.magnetics.flux_linkage(x, i))
        coenergy = self._coenergy(x, i)
        return lam * i - coenergy

    def _coenergy(self, x: float, i: float) -> float:
        import math

        mag = self.magnetics
        if not mag.saturation:
            return 0.5 * float(mag.inductance(x)) * i * i
        u = abs(i) / mag.i_sat
        ln_cosh = u - math.log(2.0) + math.log1p(math.exp(-2.0 * u))
        return 0.5 * mag.l_air * i * i + (
            (mag.mu_eff - 1.0)
            * mag.l_air
            * float(mag.fill(x))
            * mag.i_sat**2
            * ln_cosh
        )

    def time_to_peak_current(self) -> float:
        """Time from firing to peak current, for the air-core (slug absent)
        case. Used by the prefire controller to pick a lead time.

        v1 computed this once at construction by scanning a closed-form
        solution, then compared it against a stale value forever.
        """
        L = self.magnetics.l_air
        R = self.r20 + self.config.switch.on_resistance
        C = self.config.bank.capacitance
        alpha = R / (2.0 * L)
        w0 = 1.0 / np.sqrt(L * C)
        if alpha < w0:  # underdamped
            wd = np.sqrt(w0 * w0 - alpha * alpha)
            return float(np.arctan2(wd, alpha) / wd)
        if alpha > w0:  # overdamped
            beta = np.sqrt(alpha * alpha - w0 * w0)
            s1, s2 = -alpha + beta, -alpha - beta
            return float(np.log(s2 / s1) / (s1 - s2))
        return float(1.0 / alpha)  # critically damped
