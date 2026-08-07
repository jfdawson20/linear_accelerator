"""Vectorised evaluation of every stage at once.

This is a performance optimisation of the equations in `magnetics` and
`circuit`, not a different model. Those modules remain the readable reference
implementation and are what the physics tests exercise; `tests/test_kernel.py`
asserts the two agree to floating-point tolerance.

Why it exists: profiling the scalar path showed 1.5M calls to a scalar sigmoid
and 3.2M calls to np.asarray for a single 8-stage run. numpy's per-call overhead
dominates completely when the operands are scalars. Two changes fix it:

  - evaluate all stages as one array operation rather than looping in Python
  - compute the four ramp terms once per evaluation and share them between
    fill, dfill/dx, the inductance, the back-EMF and the force, instead of
    recomputing dfill/dx twice per derivative evaluation
"""

from __future__ import annotations

import math

import numpy as np

from .wire import ALPHA_CU, T_REF_C

LN2 = math.log(2.0)

# Switch state codes. Kept as small ints so they can live in a numpy array.
OFF = 0
ON = 1
FREEWHEEL = 2


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Logistic function, via tanh so it is stable for any magnitude and costs
    a single ufunc call."""
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def _ln_cosh(u: np.ndarray) -> np.ndarray:
    """log(cosh(u)), stable for large |u| where cosh overflows."""
    au = np.abs(u)
    return au - LN2 + np.log1p(np.exp(-2.0 * au))


class StageKernel:
    """Per-stage parameters as parallel arrays, plus a fused derivative step."""

    def __init__(self, circuits, saturation: bool, projectile_length: float):
        n = len(circuits)
        self.n = n
        self.saturation = saturation
        self.lp = float(projectile_length)

        f = lambda fn: np.array([fn(c) for c in circuits], dtype=float)
        self.pos = f(lambda c: c.config.position)
        self.lc = f(lambda c: c.config.coil.length)
        self.width = f(lambda c: c.magnetics.fringe_width)
        self.l_air = f(lambda c: c.magnetics.l_air)
        self.mu_m1 = f(lambda c: c.magnetics.mu_eff - 1.0)
        self.i_sat = f(lambda c: c.magnetics.i_sat)
        self.area_ratio = f(lambda c: c.magnetics.area_ratio)
        self.r20 = f(lambda c: c.r20)
        self.r_switch = f(lambda c: c.config.switch.on_resistance)
        self.cap = f(lambda c: c.config.bank.capacitance)
        self.diode_vf = f(lambda c: c.config.switch.diode_vf)
        self.thermal_mass = f(lambda c: c.thermal_mass)
        # Topology: number of devices in the conduction path, their drop, and
        # whether turn-off returns the field energy to the bank.
        self.n_dev = f(lambda c: c.config.switch.conduction_devices)
        self.device_drop = f(lambda c: c.config.switch.device_drop)
        self.regen = f(lambda c: float(c.config.switch.recovers_energy))

        # Ramp offsets, precomputed: fill is a difference of four ramps at
        # u, u-Lp, u-Lc, u-Lp-Lc.
        self._off = np.stack(
            [
                np.zeros(n),
                np.full(n, self.lp),
                self.lc,
                self.lc + self.lp,
            ]
        )
        self._sign = np.array([1.0, -1.0, -1.0, 1.0]).reshape(4, 1)
        self._norm = self.area_ratio / self.lc

    # -- geometry -------------------------------------------------------

    def geometry(self, x: float) -> tuple[np.ndarray, np.ndarray]:
        """(fill, dfill/dx) for every stage at nose position x."""
        z = (x - self.pos - self._off) / self.width  # (4, n)
        fill = (self.width * np.logaddexp(0.0, z) * self._sign).sum(axis=0)
        dfill = (_sigmoid(z) * self._sign).sum(axis=0)
        return fill * self._norm, dfill * self._norm

    # -- magnetics ------------------------------------------------------

    def magnetics(
        self, fill: np.ndarray, dfill: np.ndarray, i: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(L_inc, dlambda/dx, force) for every stage."""
        if not self.saturation:
            l_inc = self.l_air * (1.0 + self.mu_m1 * fill)
            dlam = self.l_air * self.mu_m1 * dfill * i
            force = 0.5 * i * i * self.l_air * self.mu_m1 * dfill
            return l_inc, dlam, force

        u = i / self.i_sat
        t = np.tanh(u)
        l_inc = self.l_air * (1.0 + self.mu_m1 * fill * (1.0 - t * t))
        base = self.mu_m1 * self.l_air * dfill * self.i_sat
        dlam = base * t
        force = base * self.i_sat * _ln_cosh(u)
        return l_inc, dlam, force

    # -- circuit --------------------------------------------------------

    def derivatives(
        self,
        x: float,
        v: float,
        i: np.ndarray,
        vc: np.ndarray,
        temp: np.ndarray,
        switch: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        """(di/dt, dVc/dt, dT/dt, total_force, external_power).

        Force is summed over every stage, not just the one the projectile is
        inside: a stage still freewheeling pulls backward on a departing
        projectile, which is how cross-stage effects reach the kinematics.
        """
        fill, dfill = self.geometry(x)
        l_inc, dlam, force = self.magnetics(fill, dfill, i)

        on = switch == ON
        # The return path is diodes, which block reverse current. Without this
        # the current can be driven straight through zero -- the half bridge
        # puts hundreds of volts across the coil, so di/dt is steep enough to
        # overshoot within one step, and sign(i) then flips mid-substep and
        # pumps energy the wrong way.
        fw = (switch == FREEWHEEL) & (i > 0.0)
        active = on | fw

        sign_i = np.sign(i)
        abs_i = np.abs(i)
        regen = fw & (self.regen > 0.0)
        plain_fw = fw & (self.regen == 0.0)

        r_coil = self.r20 * (1.0 + ALPHA_CU * (temp - T_REF_C))
        r_total = r_coil + np.where(on, self.r_switch, 0.0)

        # ON: bank drives the coil, less the device drops.
        # Regenerative turn-off: coil clamped across the bank in reverse.
        # Plain freewheel: coil sees only the diode drop.
        drive = (
            np.where(on, vc - self.n_dev * self.device_drop * sign_i, 0.0)
            - np.where(regen, (np.abs(vc) + 2.0 * self.diode_vf) * sign_i, 0.0)
            - np.where(plain_fw, self.diode_vf * sign_i, 0.0)
        )

        di = np.where(active, (drive - i * r_total - dlam * v) / l_inc, 0.0)
        # Regeneration charges the bank back up rather than leaving it flat.
        dvc = np.where(on, -i / self.cap, 0.0) + np.where(regen, abs_i / self.cap, 0.0)
        dtemp = np.where(active, i * i * r_coil / self.thermal_mass, 0.0)

        external = float(
            (
                np.where(on, i * i * self.r_switch + abs_i * self.n_dev
                         * self.device_drop, 0.0)
                + np.where(regen, abs_i * 2.0 * self.diode_vf, 0.0)
                + np.where(plain_fw, abs_i * self.diode_vf, 0.0)
            ).sum()
        )
        return di, dvc, dtemp, float(force.sum()), external

    # -- diagnostics ----------------------------------------------------

    def force_only(self, x: float, i: np.ndarray) -> np.ndarray:
        fill, dfill = self.geometry(x)
        return self.magnetics(fill, dfill, i)[2]

    def inductance_only(self, x: float, i: np.ndarray) -> np.ndarray:
        fill, dfill = self.geometry(x)
        return self.magnetics(fill, dfill, i)[0]

    def stored_magnetic_energy(self, x: float, i: np.ndarray) -> np.ndarray:
        """W = lambda*i - W'. Uses stored energy, not co-energy: the two differ
        once saturated, and using the wrong one breaks the energy audit."""
        fill, _ = self.geometry(x)
        if not self.saturation:
            l = self.l_air * (1.0 + self.mu_m1 * fill)
            return 0.5 * l * i * i
        u = i / self.i_sat
        lam = self.l_air * i + self.mu_m1 * self.l_air * fill * self.i_sat * np.tanh(u)
        coenergy = (
            0.5 * self.l_air * i * i
            + self.mu_m1 * self.l_air * fill * self.i_sat**2 * _ln_cosh(u)
        )
        return lam * i - coenergy
