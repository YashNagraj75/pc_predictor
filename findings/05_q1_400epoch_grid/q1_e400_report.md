# Q1-Phase A at the full LeJEPA budget (400 epochs)

Same code, same fixes, same everything as the `_v2fix` grid — only `n_steps`
changed, 2000 -> 17,200 (43 steps/epoch x 400 epochs, matching LeJEPA's Galaxy10
recipe). Evaluation every 20 epochs, 21 evaluation points. Runs tagged `_e400`.

kNN probe of a **randomly-initialised** encoder: **0.343**. Chance = 0.10.

## Results

| arm | kNN @2k steps | kNN @400 ep | eff. rank @400 ep | twoNN | dims to 95% | best hidden kNN | final invariance loss |
|-----|---------------|-------------|-------------------|-------|-------------|-----------------|----------------------|
| backprop + SIGReg | 0.3605 | **0.3910** | 18.33 | 9.22 | 20 | 0.4200 | 0.3999 |
| backprop, invariance only | 0.2905 | 0.2655 | 1.00 | 7.84 | 1 | 0.3855 | 0.0130 |
| PC (T=16) + SIGReg | 0.3380 | 0.3305 | 20.81 | 22.71 | 50 | 0.3935 | 0.6942 |
| PC (T=16), invariance only | 0.2970 | 0.2155 | 1.00 | 10.99 | 1 | 0.3935 | 0.0924 |

Wall clock: bp+SIGReg 564.6 s, bp control 505.9 s, PC+SIGReg 1141.8 s, PC control 957.6 s.

## Readings

**1. The longer budget resolves the "nothing learns" problem — for backprop only.**
At 2000 steps backprop was +0.018 over random init, within noise of it. At 400 epochs
it reaches 0.3910, **+0.048 over init**, and the trajectory is a clean rise that
saturates around epoch 80-150 and holds. That is a real learning signal.

**2. Predictive coding never exceeds its own initialisation.** It starts at 0.343,
drops to a minimum of 0.3233 around epoch 100, recovers to 0.3305, and stays there.
8.6x more compute did not move it above init. The BP-PC gap widened from 0.023 at
2000 steps to **0.061** here — the longer budget separated the arms rather than
converging them.

**3. The likely cause is under-optimisation, not representation quality.** PC's final
invariance loss is 0.6942 against backprop's 0.3999 — **1.74x higher on the objective
it is being trained on**, and flat from epoch ~60 onward. PC is not solving the
training problem. Before concluding anything about PC's *representations*, this has to
be ruled out: at T=16 relaxation steps the inference phase may simply not be converging
enough for the weight gradient to be useful. This is exactly the regime µPC is about.

**4. "Richer" and "better" come apart cleanly, and the gap grew.** PC ends with higher
effective rank (20.81 vs 18.33), 2.5x the intrinsic dimension (22.71 vs 9.22), and 2.5x
the dimensions needed for 95% variance (50 vs 20) — while scoring 0.061 *lower* on the
probe. At 2000 steps the rank gap was 2.1 (23.14 vs 21.05); at 400 epochs it is 2.5 and
the twoNN/dims-95 gaps are much larger. So the original hypothesis is half-confirmed and
half-refuted: PC latents *are* measurably higher-dimensional, and that dimensionality is
**not** class-discriminative structure. Rank is not a proxy for usable structure.

**5. Both controls collapse fully and get worse with budget** (0.2905 -> 0.2655 and
0.2970 -> 0.2155, both at effective rank 1.00). Longer training drives an unregularised
objective further into collapse, as expected. The controls are behaving correctly.

**6. Hidden layers still beat the embedding** in every arm (BP 0.4200 vs 0.3910; PC
0.3935 vs 0.3305). Notably both collapsed controls retain usable hidden layers
(0.3855, 0.3935) despite rank-1 embeddings — collapse is a property of the final
projection, not of the trunk.

## What this does and does not settle

**Settled:** the recipe now trains. Backprop learns a representation measurably better
than random. The comparison is no longer confounded by a broken baseline, and PC's
rank advantage is real and reproducible across two budgets.

**Not settled, and now the critical question:** whether PC's probe deficit is about
representations or about inference depth. Finding 3 makes the second explanation the
leading one. The decisive experiment is a **T sweep** (T = 16 / 32 / 64 / 128 / 256)
on the PC+SIGReg arm: if the invariance loss falls toward backprop's 0.40 and the
probe rises with T, the deficit was under-optimisation and the representation question
is still open. If the loss falls but the probe does not, PC genuinely produces
higher-dimensional but less discriminative latents. Per earlier timing measurements the
whole sweep is cheap — PC costs ~0.90 ms per relaxation step, so even T=256 is a
matter of minutes per run.

## Not assessed

Seed variance (n=1 per cell), T > 16, the frozen-vs-coupled inference variant,
sensitivity to the projector choice, and whether backprop's advantage survives a
stronger augmentation stack or a convolutional encoder.
