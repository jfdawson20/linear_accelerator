# Design Summary -- 12 mm Projectile (alternate proposal)

Optimised for **maximum muzzle energy** with a 12 mm steel projectile, carrying forward every optimisation established for the 0.25 in design: asymmetric half bridge, prefire at scale 1.0, and a ferromagnetic shell with end caps.

| constraint | value |
|---|---|
| projectile diameter | 12 mm |
| capacitor voltage | <= 300 V |
| peak current | <= 300 A |
| stages | 8 |
| objective | maximum muzzle energy |

Regenerate with `python design_12mm.py`. See `DESIGN.md` for the 0.25 in design this is an alternative to.

## Read this first

**No optimum was found in projectile aspect ratio.** Muzzle energy was still rising at L/d = 8, the edge of the swept range:

| L/d | length | mass | barrel | KE (bore) | KE (mean) | best winding |
|---|---|---|---|---|---|---|
| 4 | 48 mm | 42.6 g | 0.7 m | 182 J | 122 J | 400t 20 AWG, c/p 1.5-1.75 |
| 5 | 60 mm | 53.3 g | 0.72 m | 234 J | 147 J | 400-500t, 18-20 AWG |
| 6 | 72 mm | 63.9 g | 0.98 m | 276 J | 176 J | 500t 18 AWG, c/p 1.5 |
| 7 | 84 mm | 74.6 g | 0.96 m | 313 J | 190 J | 500t 18 AWG, c/p 1.25 |
| 8 | 96 mm | 85.2 g | 1.08 m | 336 J | 201 J | 600t 18 AWG, c/p 1.25 |

`KE ~ m*v^2` with `m ~ L` and `v ~ 1/sqrt(L)` alone gives constant energy, but mu_eff keeps rising as the slug elongates and beats that scaling. It must plateau eventually -- mu_eff is capped at mu_r = 100 as the demagnetising factor tends to zero -- but not within the swept range. 18 AWG was also at the edge of the swept gauges.

**So the size below is a practical choice, not a physics optimum.** If a longer projectile and barrel are acceptable, energy keeps rising.

## The winding answer

At L/d >= 6 **both coupling models agree exactly** on the winding: **18 AWG, 500-600 turns, coil-to-projectile ratio 1.25-1.5**. That is a stronger consensus than the 0.25 in design produced.

The shift from 22 AWG / 400 turns is direct: a 12 mm bore means longer turns, so thicker wire is needed to hold resistance down, and a longer coil fits more turns per layer.

## Variant A -- moderate (L/d 6)

| | |
|---|---|
| Projectile | 72 x 12 mm steel, **63.9 g** |
| Coil length | 108 mm (c/p 1.50) |
| Winding ID (bore) | 14.65 mm (barrel wall 1.075 mm at 0.25 mm clearance) |
| Winding OD | 25.2 mm (5 layers, 5.30 mm deep) |
| Wire | **18 AWG** single-build, 1.060 mm OD |
| Turns | **500** per coil, 101 per layer |
| Wire length | 31.3 m per coil, **250 m total** |
| Resistance | 0.655 ohm at 20 C, 0.758 ohm at 60 C |
| Stage pitch | 123 mm |
| Barrel length | **984 mm** |

### Inductance

| | bore coupling | mean coupling |
|---|---|---|
| Air core, slug out | 1449 uH | 1449 uH |
| Slug centred, small signal | **29157 uH** | **16393 uH** |
| Ratio | 20.12 x | 11.31 x |
| Incremental at 250 A | 1449 uH | 1449 uH |
| Effective permeability | 43.8 | 43.8 |
| Saturation current | 6.3 A | 6.3 A |

Note the air-core figure already includes the assumed shell (`l_shell_factor` 2.0); a bare coil would be half of it.

### Control

| | |
|---|---|
| Fire | 3693 us of travel ahead of the coil mouth |
| Turn off | after 60.0 mm of nose travel past the sensor |
| Force reverses at | 90.0 mm |
| Collapse margin | 30.0 mm |

### Predicted performance

| | bore (optimistic) | mean (pessimistic) |
|---|---|---|
| **Muzzle energy** | **276.2 J** | **176.5 J** |
| Exit velocity | 93.0 m/s | 74.3 m/s |
| Peak current | 296 A | 296 A |
| Peak winding temperature | 27 C | 28 C |
| Suck-back | 9.56 % | 4.63 % |
| Energy drawn | 1509 J | 1722 J |
| Conversion efficiency | 18.31 % | 10.25 % |
| Shot duration | 20.42 ms | 26.61 ms |
| Energy closure error | 0.0000 % | 0.0000 % |

## Variant B -- compact (L/d 5)

| | |
|---|---|
| Projectile | 60 x 12 mm steel, **53.3 g** |
| Coil length | 75 mm (c/p 1.25) |
| Winding ID (bore) | 14.65 mm (barrel wall 1.075 mm at 0.25 mm clearance) |
| Winding OD | 23.1 mm (5 layers, 4.24 mm deep) |
| Wire | **20 AWG** single-build, 0.848 mm OD |
| Turns | **400** per coil, 88 per layer |
| Wire length | 23.3 m per coil, **186 m total** |
| Resistance | 0.776 ohm at 20 C, 0.898 ohm at 60 C |
| Stage pitch | 90 mm |
| Barrel length | **720 mm** |

### Inductance

| | bore coupling | mean coupling |
|---|---|---|
| Air core, slug out | 1165 uH | 1165 uH |
| Slug centred, small signal | **23733 uH** | **14740 uH** |
| Ratio | 20.37 x | 12.65 x |
| Incremental at 250 A | 1165 uH | 1165 uH |
| Effective permeability | 37.6 | 37.6 |
| Saturation current | 6.3 A | 6.3 A |

Note the air-core figure already includes the assumed shell (`l_shell_factor` 2.0); a bare coil would be half of it.

### Control

| | |
|---|---|
| Fire | 3036 us of travel ahead of the coil mouth |
| Turn off | after 45.0 mm of nose travel past the sensor |
| Force reverses at | 67.5 mm |
| Collapse margin | 22.5 mm |

### Predicted performance

| | bore (optimistic) | mean (pessimistic) |
|---|---|---|
| **Muzzle energy** | **214.5 J** | **146.8 J** |
| Exit velocity | 89.7 m/s | 74.2 m/s |
| Peak current | 275 A | 275 A |
| Peak winding temperature | 29 C | 30 C |
| Suck-back | 6.61 % | 3.64 % |
| Energy drawn | 1252 J | 1398 J |
| Conversion efficiency | 17.13 % | 10.50 % |
| Shot duration | 16.18 ms | 20.11 ms |
| Energy closure error | 0.0000 % | 0.0000 % |

## Against the 0.25 in design

| | 0.25 in (DESIGN.md) | 12 mm variant A |
|---|---|---|
| Muzzle energy | 14.6 - 35.5 J | **177 - 276 J** |
| Projectile | 6.3 g | 63.9 g |
| Barrel | 0.43 m | 0.98 m |
| Wire total | 139 m | see above |

Roughly **5-12x the muzzle energy**, at the cost of a heavier projectile, a barrel more than twice as long, and considerably more copper.

## Caveats specific to this proposal

1. **The shell assumption is load-bearing.** `flux_return` 0.7 and `l_shell_factor` 2.0 are assumed, not measured, and the inductance factor does most of the work -- raising mu_eff alone buys only ~6-8% because i_sat falls in proportion. Without the shell these energies would be materially lower.
2. **Layer coupling** remains the dominant uncertainty, as in DESIGN.md, and is why every figure is quoted as a range.
3. **No aspect-ratio optimum was found.** Sizing is a practical choice.
4. **These are heavy, slow projectiles** -- 50-85 g at 60-90 m/s. If there is a minimum velocity requirement it constrains the design separately from energy.
5. **Not modelled**: eddy losses, shell saturation, total bus current across simultaneously firing stages, barrel friction.

## Next step

Same as DESIGN.md, plus one: measure L with and without the **shell** fitted. That ratio is `l_shell_factor` directly, and it is currently a guess doing a lot of work.
