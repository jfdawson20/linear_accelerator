# Design Summary

Best configuration found under the build constraints, optimised for **maximum muzzle energy**.

| constraint | value |
|---|---|
| projectile diameter | 0.25 in (fixed) |
| projectile length | <= 1 in |
| capacitor voltage | <= 300 V |
| peak current | <= 300 A |
| stages | 8 |

Regenerate this file with `python design_summary.py`.

## Mechanical

| | |
|---|---|
| Projectile | 25.4 x 6.35 mm steel, **6.31 g** |
| Coil length | 38.1 mm (1.5 : 1 coil-to-projectile) |
| Winding ID (bore) | 9.00 mm |
| Barrel wall | 1.075 mm at 0.25 mm radial clearance |
| Winding OD | 19.8 mm (8 layers, 5.40 mm deep) |
| Turns per layer | 56 |
| Stage pitch | 53.1 mm (15 mm gap) |
| Barrel length | **425 mm** over 8 stages |

## Winding

| | |
|---|---|
| Wire | 22 AWG single-build (0.644 mm bare, 0.675 mm OD) |
| Turns | 400 per coil |
| Wire length | 17.4 m per coil, **139 m total** |
| Resistance | 0.920 ohm at 20 C, 1.065 ohm at 60 C |

## Inductance

| | bore coupling | mean coupling |
|---|---|---|
| Air core, slug out | 595.5 uH | 595.5 uH |
| Slug centred, small signal | **2710 uH** | **1422 uH** |
| Ratio | 4.55 x | 2.39 x |
| Incremental at 265 A | 595.5 uH | 595.5 uH |
| Effective permeability | 11.81 | 11.81 |
| Saturation current | 10.3 A | 10.3 A |
| Max fill fraction | 0.247 | 0.097 |

Three things to know before measuring:

- The two coupling models disagree by 1.9x on slug-in inductance. **This is the measurement that resolves the output uncertainty** -- an LCR reading on the finished coil says which is right.
- What an LCR meter reads is not what the circuit sees. At 265 A the slug is ~26x saturated, so incremental inductance collapses back to the air-core 595 uH. Scope traces will imply ~600 uH, not 2700.
- L/R = 647 us (air core); time to peak current 1789 us, heavily overdamped.

## Electrical, per stage

| | |
|---|---|
| Capacitor bank | 8 mF at 300 V = 360 J |
| Total stored | 2.88 kJ |
| Topology | Asymmetric half bridge: 2 switching devices + 2 diodes |
| Device drop assumed | 1.8 V per conducting device |
| Peak current | 265 A (limit 300 A) |

The bank is deliberately oversized to shorten recharge between shots, so the ~78% left on the capacitors is reserve, not loss. The meaningful efficiency figure is conversion of the energy actually drawn.

## Control

| | |
|---|---|
| Sensor | ingress detector at each coil mouth |
| Fire | when the projectile is 1789 us of travel away (prefire scale 1.0) |
| Turn off | after 15.9 mm of nose travel past the sensor |
| Force reverses at | 31.8 mm of nose travel |
| Collapse margin | 15.9 mm |

At ~100 m/s the 1789 us lead is roughly 179 mm, about 3.4 stage pitches upstream. **Several stages are energised simultaneously**, so the control logic cannot be a simple per-stage state machine, and total bus current is a multiple of the per-stage figure. Bus current is not modelled.

Prefire is load-bearing, not a refinement: without it output falls by roughly 45-59%, because the bank takes 1789 us to reach peak current and the projectile would be past before the current arrived.

## Predicted performance

| | bore (optimistic) | mean (pessimistic) |
|---|---|---|
| **Muzzle energy** | **35.50 J** | **14.62 J** |
| Exit velocity | 106.03 m/s | 68.05 m/s |
| Peak current | 265 A | 266 A |
| Peak winding temperature | 29 C | 31 C |
| Suck-back | 3.48 % | 0.29 % |
| Energy drawn from bank | 646 J | 814 J |
| Conversion efficiency | 5.49 % | 1.80 % |
| Bank remaining | 78 % | 72 % |
| Shot duration | 7.91 ms | 12.41 ms |
| Energy closure error | 0.0000 % | 0.0000 % |

**Headline uncertainty: 14.6 - 35.5 J.** Every mechanical, electrical and control parameter above is pinned and robust across both coupling assumptions. Only the layer-coupling question is open, and it is worth 2.4x on output.

## How each choice was pinned

| parameter | rationale | confidence |
|---|---|---|
| 400 turns | interior optimum under both couplings; beyond ~500 suck-back returns | high |
| 22 AWG | 20 AWG only wins where the model over-rewards deep windings | medium |
| 1.5 : 1 coil ratio | both couplings agree; 3:1 costs 17-30% | high |
| 9 mm bore | thinning to a 0.325 mm wall buys only 3-13% | high |
| Half bridge | +17% over a well-timed freewheel design, plus energy recovery and wide timing tolerance | high |
| Turn-off at 0.5 | flat optimum; half bridge tolerant across 0.5-1.0 | high |
| Prefire scale 1.0 | peak-at-mouth is optimal under both couplings; flat from 0.75 to 1.5 | high |
| 8 mF / 300 V | voltage at the ceiling; current binds at 265 A | high |
| 8 stages | chosen; energy scales near-linearly with stage count if extended | high |

## Known model limitations

1. **Layer coupling.** `fill()` scales the slug's contribution by `proj_area/bore_area`, independent of winding depth, so outer layers are credited with coupling they do not have. At 8 layers this is the dominant uncertainty. Bracketed by the two coupling models, not resolved.
2. **Demagnetising factor** uses a prolate spheroid for a cylinder.
3. **No flux return modelled by default.** A shell plus end caps is parameterised (`flux_return`, `l_shell_factor`) but uncalibrated. Raising mu_eff alone buys only ~6-8%, because i_sat falls in proportion; the real gain is the inductance rise, worth ~19% if L roughly doubles.
4. **Not modelled**: eddy losses, shell saturation, total bus current across simultaneously firing stages, barrel friction, projectile eddy currents.

## Next step

Wind one coil and measure L with the slug out and centred. That single reading discriminates the two coupling models and collapses the output range. `la.calibration.mu_eff_from_measurements()` and the `coupling` switch are wired and tested for it; drop the numbers into `measurements/stage0.yaml`.
