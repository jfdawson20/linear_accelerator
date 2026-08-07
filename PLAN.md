# Multi-Stage Linear Accelerator Simulator — Rebuild Plan

## Purpose

Ballpark design tool for a chained solenoid (coilgun) accelerator. It exists to lock in
physical design parameters before metal gets cut:

- solenoid geometry — turns, wire gauge, coil length, bore
- projectile geometry and mass
- per-stage fire timing, stage spacing, capacitor bank sizing

It is explicitly **not** a high-fidelity FEA model. The target is "right order of
magnitude, right trends, right trade-offs."

## The design premise under test

Each stage's coil is **twice the projectile length**. An ingress sensor at the coil mouth
fires the coil; the coil shuts off once the projectile has fully passed the sensor (i.e.
travelled one projectile length). Because the coil is 2x the projectile, the projectile
has not yet reached coil centre at shutoff — the intent is to give the coil's field time
to collapse before the projectile crosses centre, avoiding suck-back.

**The tool must be able to falsify this.** The v1 simulator could not: it modelled coil
turn-off as instantaneous and hard-zeroed any force past coil centre, so suck-back was
defined out of existence rather than simulated. Validating the 2:1 ratio is the single
most important capability of the rebuild.

---

## What was wrong with v1

Recorded so the rewrite doesn't quietly reintroduce any of it.

| # | Issue | Impact |
|---|---|---|
| 1 | Force used an axial-gap reluctance-actuator formula with a *radial* clearance substituted for the gap | Force independent of position; magnitude wrong by ~100x; `1/g^2` on a meaningless parameter |
| 2 | Force hard-zeroed past coil centre | Suck-back unmodellable |
| 3 | Turn-off modelled as `coil_time = 0`, giving `i = 0` in one 1 us step | Field-collapse time — the entire design question — not modelled |
| 4 | `L` scaled by full bulk `mu_r = 100` on slug insertion | ~10x too high; circuit flips damping regime mid-flight |
| 5 | Closed-form constant-L RLC re-solved each step with a new L | Not a solution of the ODE; motional back-EMF absent; energy not conserved |
| 6 | Capacitor voltage computed post-hoc, never fed back | Every re-fire restarts from a full bank |
| 7 | No saturation | `F ~ i^2` unbounded |
| 8 | Projectile modelled as a point; no `proj_len`; mass hardcoded in 4 places | 2:1 ratio unsweepable |
| 9 | Force zero between coils | Stage spacing has no effect except transit time; optimiser drives it to zero |
| 10 | Thermal integrated on `coil_time`, which resets to 0 | Temperature collapses toward -234 C at every switch event |
| 11 | Two divergent copies of the run loop (`Exec` / `Exec2`) | `Optimize` ran the older, buggier one |
| 12 | `getCurrent` sign error + reversed `dt` in the charge integral | Two bugs cancelling |

Symptom that ties it together: the v1 default config reported **101 m/s from a single
stage** and a capacitor energy drop of 31 J for 28.5 J of kinetic energy — ~92%
efficiency before I^2R. Real single-stage coilguns run 1-5%.

---

## Physics model

### State

Global: `x` (projectile position), `v` (velocity), `t`.
Per stage: `i` (coil current), `Vc` (capacitor voltage), `switch_state`.

Integrated with RK4 at fixed `dt`, with a startup assertion that `dt` is well below
`L_min/R`, `1/omega`, and `x_resolution/v_max`.

### Flux linkage (nonlinear)

Because saturation is in scope, the magnetic system is described by flux linkage
`lambda(x, i)`, not by `L(x)`. Three distinct quantities follow, and using the wrong one
in the wrong place is the classic error here:

| Used for | Quantity |
|---|---|
| coefficient on `di/dt` | incremental inductance `L_inc = d(lambda)/di` |
| back-EMF | `d(lambda)/dx * v` |
| force | co-energy gradient `d/dx of integral(0..i) lambda di'` |

Model:

```
fill(x)      = (axial overlap of slug and coil / coil length) * (A_proj / A_bore)
lambda(x,i)  = L_air*i  +  (mu_eff - 1)*L_air*fill(x)*i_sat*tanh(i / i_sat)
```

The air-core term stays linear (air does not saturate); only the slug's contribution
rolls off. Both partials are closed-form, so no numerical differentiation inside the RK4
inner loop. Corners of `fill(x)` are smoothed so RK4 does not see a discontinuity.

Force, from co-energy:

```
W'(x,i) = 0.5*L_air*i^2 + (mu_eff-1)*L_air*fill(x)*i_sat^2*ln(cosh(i/i_sat))
F       = dW'/dx = (mu_eff-1)*L_air*fill'(x)*i_sat^2*ln(cosh(i/i_sat))
```

**Linear-limit check.** As `i << i_sat`, `ln(cosh u) -> u^2/2`, so

```
F -> 0.5*i^2*L_air*(mu_eff-1)*fill'(x)  ==  0.5*i^2*dL/dx
```

which is exactly the linear result for `L(x) = L_air*(1 + (mu_eff-1)*fill(x))`. The
saturating model is therefore a strict generalisation, and `--no-saturation` recovers the
linear model as a genuine special case rather than a separate code path. This identity is
a unit test (Phase 4).

**Suck-back falls out for free.** `fill'(x) > 0` on entry (force forward), `fill'(x) < 0`
on exit (force backward). No special-casing.

### Effective permeability

Bulk `mu_r` is wrong for a short rod in an open magnetic circuit — the demagnetising
factor dominates. For a prolate spheroid of aspect ratio `m = proj_len / proj_dia`:

```
N_d     = (1/(m^2-1)) * ( (m/sqrt(m^2-1)) * ln(m + sqrt(m^2-1)) - 1 )
mu_eff  = mu_r / (1 + N_d*(mu_r - 1))
```

The exact expression is used, not the commonly quoted asymptotic form
`(1/m^2)*(ln(2m) - 1)`, which is 20% low at `m = 2.9` and 81% low at `m = 1.5`.

At the current design point (17.5 mm x 6 mm slug, carbon steel `mu_r = 100`):
`m = 2.92`, `N_d = 0.1125`, **`mu_eff = 8.24`** — not 100.

A real cylinder is not a spheroid; a measured `L_slug_in` should override this once
bench data exists (see Calibration).

**Consequence for inductance.** The slug fills 37% of the bore area and, at a 2:1 coil
ratio, at most half the coil length, so `max fill = 0.169` and

```
L: 56.8 uH (empty) -> 133.1 uH (slug in)  =  2.35x
```

v1 asserted a **100x** swing. A 2.35x swing keeps the circuit in one damping regime for
the whole shot, which is a qualitatively different simulation.

`mu_eff` now depends on slug geometry, so aspect ratio becomes a sweepable design
variable.

### Saturation — why it dominates here

```
i_sat = B_sat * l_coil / (mu_0 * mu_eff * N)
```

At `B_sat = 1.6 T`, `l = 35 mm`, `N = 150`, `mu_eff = 8.24`: **`i_sat = 36 A`**.

The v1 design point runs at 280 A peak. Linear theory implies:

| current | implied B in slug | vs B_sat |
|---|---|---|
| 36 A | 1.6 T | 1.0x |
| 100 A | 4.4 T | 2.8x |
| 280 A | 12.4 T | **7.8x** |
| 450 A | 20.0 T | **12.5x** |

The current operating point is roughly an order of magnitude into saturation.

**Force has two regimes**, and the crossover is the reason to model this at all:

- below `i_sat` the slug's magnetisation tracks the applied field, so both the moment and
  the gradient scale with `i`, giving `F ~ i^2`
- above it the moment is pinned at `M_sat`, so `F = m.grad(B)` scales with the gradient
  alone, giving `F ~ i`

v1 had `F ~ i^2` without bound, so more current always looked better. Expect the rebuilt
model to predict substantially lower performance and to show sharply diminishing returns
above ~36 A. Getting this right is the difference between a tool that guides the design
and one that flatters it.

### Circuit

```
di/dt = (Vc - i*R(T) - i*(d lambda/dx)*v) / L_inc(x,i)
dVc/dt = -i / C
dv/dt  = F / m
dx/dt  = v
```

Capacitor depletion and motional back-EMF are now inside the loop, so energy balances by
construction rather than by accident.

### Switch model

Three states:

- `ON` — capacitor drives the coil
- `FREEWHEEL` — turn-off commanded; source replaced by `-V_diode`, current decays on `L/R`
- `OFF` — current below threshold

The `FREEWHEEL` state is what makes the design premise testable. At `L ~ 5.6e-5 H` and
`R ~ 0.65 ohm` the decay constant is ~86 us; at 150 m/s the projectile covers 13 mm in a
single tau, against a 17.5 mm half-coil. **This may show that 2:1 is not enough**, which
is precisely the question the tool exists to answer.

### Control

Models the real trigger chain: ingress sensor at coil mouth, shutoff when the nose has
travelled `proj_len` past it, plus configurable `sensor_latency` and `switch_latency`.
At these velocities tens of microseconds of gate-driver delay is millimetres of travel.
Optional prefire lead, computed against a live peak-current estimate rather than a
snapshot taken at construction time.

---

## Architecture

```
la/config.py       dataclasses + YAML/JSON load
la/wire.py         AWG table (bare + insulated OD), R(T)
la/geometry.py     coil/projectile -> layers, wire length, R, L_air (Wheeler)
la/magnetics.py    mu_eff, fill(x), lambda(x,i), L_inc, d lambda/dx, force
la/circuit.py      per-stage state, switch machine
la/control.py      sensors, trigger logic, latencies
la/engine.py       RK4 integrator, run loop, energy accounting
la/calibration.py  measurement load, correction factors, compare/fit
la/report.py       tables, plots, energy audit
la/cli.py          argparse entry point
tests/
```

numpy throughout. Physics modules take calibration scale factors as constructor
arguments — they never read files, and never branch on whether measurements exist.

---

## Calibration against real hardware

No bench data yet. The tool must run correctly with none, and accept it incrementally as
the prototype comes together.

`measurements/*.yaml`, every field optional:

```yaml
coil_id: stage0
measured:
  L_air:         58.2e-6     # H, LCR meter, no slug
  L_slug_in:     410e-6      # H, slug centred
  R_dc:          0.671       # ohm, 4-wire
  peak_current:  262         # A, from scope
  exit_velocity: 41.3        # m/s, chrono
conditions:
  V0: 200
  C:  0.006
  ambient_C: 22
traces:
  current: traces/stage0_i.csv   # t,i
```

Correction factors (`L_scale = L_measured/L_predicted`, `R_scale`, ...) default to `1.0`
when absent, so an empty `measurements/` directory reproduces uncalibrated behaviour
exactly.

Two CLI modes:

- `--compare` — predicted vs measured side by side, % error, per stage
- `--fit` — least-squares fit of `mu_eff` and `R` to a scope trace of `i(t)`

**Highest-value first measurement:** `L_air` and `L_slug_in` with an LCR meter and a
slug. Five minutes of bench time directly settles the `mu_eff` question, which is the
largest remaining uncertainty in the model.

---

## Phases

Each phase leaves the tool runnable. Separate commits, straight to `main`.

### Phase 0 — Safety net
- [x] `requirements.txt` (prettytable, matplotlib, numpy, pyyaml, pytest)
- [x] Capture v1 output across several configs to `baseline/` — wrong numbers, but the
      reference for "did this change what I expected"

### Phase 1 — Structural floor
No physics changes.
- [x] `la/` package skeleton
- [x] Config dataclasses replace string-keyed dicts (kills the `tmp = cfg` aliasing bug)
- [x] Add `proj_len`, `proj_density`, `mu_r`, `B_sat`, `switch_latency`, `diode_vf` —
      defined now, wired up in Phase 2. Mass becomes derived.
- [x] Delete `Exec`, `timeToTargetI`, `Projectile.update`, `CoilCircuit.StepCircuit`
- [x] Reporting out of constructors (`Coil.__init__` currently prints a 25-row table)
- [x] Fix global-`sim`-instead-of-`self`, and `bool("False") == True` in argparse
- [x] `wire.py` + `geometry.py` land in final form (they survive Phase 2 unchanged)

### Phase 2 — Physics core
- [x] 2a RK4 state-space integration; delete `getCurrent`
- [x] 2b Force from co-energy gradient; delete the reluctance formula and `airGap`
- [x] 2c `lambda(x,i)` from slug/coil overlap; delete the `exp(-log(mu_r)/l * x)` ramp
- [x] 2d `mu_eff` via demagnetising factor + `tanh` saturation law
- [x] 2e Switch model with freewheel decay
- [x] 2f Trigger/control with sensor and switch latency

### Phase 3 — Supporting models
- [x] Insulated magnet-wire OD (26 AWG: 0.43 mm, not 0.404)
- [x] Wire length uses layer *centres*, not inner surfaces
- [x] `R(T) = R_20*(1 + 0.00393*(T-20))` — ~16% over a 40 C rise
- [x] Thermal integrated on absolute time, gated on coil-on. Onderdonk math itself is
      correct — leave it alone
- [x] Reporting fixes: `s[st]` indexing samples by stage number; "Min Inductance" reading
      `s[0]`; `plotData` always using `abstimes[0]`
- [x] Run length from geometry, not `while x < 1`; stall guard

### Phase 4 — Validation
- [x] Energy audit printed every run: `E_cap_spent`, `E_kinetic`, `E_resistive`,
      `E_magnetic_residual`, `E_diode`. Assert closure < 1%
- [x] Efficiency as a headline number (sanity: 1-5%)
- [x] pytest:
  - [x] constant-L RLC integration vs closed-form (v1's `getCurrent` math is a valid
        oracle for constant L)
  - [x] **zero net impulse** — a slug passing fully through a DC-energised coil gains
        ~zero net momentum. Hard invariant; strongest test that suck-back is right
  - [x] `lambda(x,i)` symmetric about coil centre
  - [x] **linear limit** — saturating vs non-saturating agree < 0.1% at `i = 0.01*i_sat`
  - [x] Wheeler and Onderdonk against published reference values
  - [x] energy closure over a full multi-stage run
- [x] Convergence: halve `dt`, exit velocity moves < 0.5%

### Phase 5 — Optimiser

Phase 3 was absorbed entirely into Phases 1-2.

- [x] Performance. Profiling the scalar path showed 1.5M scalar sigmoid calls and
      3.2M np.asarray calls per 8-stage run: numpy per-call overhead dominated
      completely. `la/kernel.py` evaluates all stages as one array operation and
      shares the four ramp terms between fill, dfill/dx, inductance, back-EMF and
      force rather than recomputing dfill/dx twice per derivative evaluation.
      `record=False` drops the per-step trace and keeps a running summary.
- [x] Rewrite `Optimize` as `la/sweep.py`. Every grid point builds its config from
      scratch, so there is no shared mutable state -- the failure mode that made
      v1's optimiser unusable.
- [x] Sweeps over turns, gauge, coil length, coil/projectile ratio, stage spacing,
      capacitance, voltage, prefire and latencies, with constraints (thermal
      limit, suck-back ceiling) and per-axis sensitivity.
- [x] `la sweep --vary NAME=V1,V2,...` CLI, parallel across cores.

### Cost of a run

| | before | after |
|---|---|---|
| 8 stages, dt=2us, recorded | 11.2 s | 5.5 s |
| 8 stages, dt=2us, record=False | - | 3.7 s |
| 8 stages, dt=10us, record=False | - | 0.80 s |
| sweep point, wall, 8 cores | - | 0.16 s |

Timestep convergence against dt=1us: 2us costs 0.02%, 5us costs 0.08%, 10us costs
0.19%, 20us costs 0.43%. Sweeps default to 10us; `check_timestep()` still warns.

---

## Results after the rebuild

Default 8-stage design (35 mm coils, 150 turns 26 AWG, 6 mF at 200 V, 2:1 slug).

| | v1 | v2 | |
|---|---|---|---|
| Exit velocity | 295 m/s | 47.5 m/s | |
| Efficiency | 26% | 0.46% | v1 was not physically possible |
| Energy closure | not checked | 0.0000% | new invariant |
| Suck-back | unmodellable | 0.03% of forward impulse | |

Saturation alone accounts for 43% of exit velocity (83.6 m/s linear -> 47.5 m/s
saturating). Peak current runs 7.5x `i_sat`.

### The design premise holds — but not for the stated reason

Force reverses when the **slug's centre** meets the **coil's centre**, at a nose position
of `(Lc + Lp)/2 = 26.25 mm` — not at the coil midpoint (17.5 mm). With an extended slug
there is no axial force at all while it is fully enclosed, because the interior field is
uniform. So the deadline is 26.25 mm, and turn-off is commanded at 17.5 mm: **8.75 mm of
margin, not the ~0 the premise assumed.**

The field does *not* fully collapse in that distance. From stage 1 onward the current is
still flowing when force reverses — margin runs from -3.8 mm at stage 1 to -20 mm at
stage 7, because the L/R decay time is fixed while the projectile gets faster. What saves
the design is that the *residual* current by then is small, and force scales with current
once saturated. Suck-back costs 0.03% of forward impulse.

This is a conditional pass. It depends on the current being well decayed, not on the
geometry alone, and it would degrade at higher velocity or with a faster-rising bank.

### Capacitor sizing is the live design question

4-stage sweep at 200 V:

| C | alpha/w0 | J/stage | exit v | efficiency | winding heat |
|---|---|---|---|---|---|
| 100 uF | 0.44 | 2.0 | 1.5 m/s | 0.05% | 96% |
| 330 uF | 0.81 | 6.6 | 4.4 m/s | 0.14% | 100% |
| 1 mF | 1.41 | 20 | 15.4 m/s | 0.57% | 99% |
| **2 mF** | **1.99** | **40** | **25.4 m/s** | **0.78%** | **86%** |
| 6 mF | 3.44 | 120 | 32.2 m/s | 0.42% | 45% |

Efficiency peaks near critical damping at ~2 mF, which gets **79% of the velocity for 33%
of the stored energy**. The 6 mF bank buys more absolute velocity but at nearly double
the energy cost per joule delivered. Worth a finer sweep once Phase 5 lands.

Stage 0 exceeds the 60 C winding limit at 6 mF.

## Open questions

- `B_sat = 1.6 T` assumes generic carbon steel. Worth pinning to an actual grade once the
  projectile material is chosen.
- The `tanh` saturation law is chosen for smoothness and analytic differentiability, not
  because it is the true B-H curve. If bench data shows it matters, a Froehlich or
  Jiles-Atherton fit is the upgrade path.
- Eddy currents in the projectile are not modelled. At these timescales they are probably
  not negligible; deferred until there is measured data to justify the complexity.
- No barrel friction or drag. Assumed small against the magnetic forces; revisit if
  chrono data disagrees with predictions.
