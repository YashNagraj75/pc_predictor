# Q1-Phase A re-run after the three recipe fixes

## What changed

| # | Fix | File | What it was | What it is now |
|---|-----|------|-------------|----------------|
| 1 | Invariance-loss reduction | `sigreg.py:prediction_loss` | `((mu[None]-Zv)**2).sum(-1).mean()` | `((mu[None]-Zv)**2).mean()` |
| 2 | Normalising projector | `sigreg.py:batchnorm_views`, called in `lejepa_loss` | SIGReg saw the raw encoder output | parameterless BatchNorm over the pooled (V*B) axis, per feature, before **both** loss terms |
| 3 | LR schedule + decay | `train.py:optim` | constant `param_lr`, optax default weight decay | `warmup_cosine_decay_schedule` (warmup 100 steps, peak 1e-3, end 1e-6) + AdamW `weight_decay=5e-4` |

Fix 1 removed a factor of `embed_dim = 128` on the invariance term. Fix 2 removes the
scale direction from the loss landscape, so "shrink everything to zero" — an exact
zero-gradient fixed point of the Epps-Pulley statistic — is no longer reachable.
Fix 3 matches LeJEPA MINIMAL.md L154-161. Config knobs added: `OptCfg.lr_schedule`,
`OptCfg.warmup_steps`, `OptCfg.weight_decay`, `RunCfg.projector`. Originals kept as
`*.pre_fix_bak`. Runs tagged `_v2fix`; the pre-fix runs were not overwritten.

## Results (2000 steps, Galaxy10, width 128 / depth 8 residual MLP, T=16 for PC)

kNN probe of a **randomly-initialised** encoder: **0.343**. Chance = 0.10.

| arm | kNN pre | kNN post | eff. rank pre | eff. rank post | twoNN post | dims to 95% post | best hidden-layer kNN post |
|-----|---------|----------|---------------|----------------|------------|------------------|---------------------------|
| backprop + SIGReg | 0.1335 | **0.3605** | 2.07 | 21.05 | 20.92 | 46 | 0.3985 |
| backprop, invariance only | 0.1280 | 0.2905 | 1.82 | 1.02 | 12.74 | 1 | 0.3680 |
| PC (T=16) + SIGReg | 0.2980 | 0.3380 | 21.54 | 23.14 | 20.73 | 41 | 0.3935 |
| PC (T=16), invariance only | 0.2800 | 0.2970 | 23.07 | 1.36 | 20.66 | 1 | 0.3935 |

Wall clock: bp+SIGReg 147.2 s, bp control 117.6 s, PC+SIGReg 191.6 s, PC control 167.5 s.

## Readings

**1. The backprop arm no longer collapses.** Effective rank goes 2.07 -> 21.05, twoNN
0.11 -> 20.92, and the SIGReg statistic descends (13.26 at step 0 -> ~3.5 by step 200)
instead of freezing at 104.6837. Both BP arms were previously sitting near chance
(0.128-0.134 with 10 classes); they are not any more.

**2. The "PC gives richer latents" result does not survive.** With a working baseline,
backprop is *ahead* on the downstream probe: 0.3605 vs 0.3380. The pre-fix ordering was
an artifact of the BP arm being in a degenerate fixed point, not evidence about PC.

**3. PC's rank advantage is real but does not pay.** PC still ends with higher effective
rank (23.14 vs 21.05) and comparable intrinsic dimension. That extra rank buys no probe
accuracy — the correlation runs the wrong way. Rank is not a proxy for usable structure
here.

**4. The controls now behave like controls.** Before the fix, PC-without-SIGReg had the
*highest* rank in the whole grid (23.07) — nonsense for an arm with no anti-collapse
term. That number came from PC being less effective at minimising the mis-scaled
invariance loss (final `loss_ff` 0.0362 for PC vs 0.0001 for BP: BP collapsed
thoroughly, PC only partially). Under-optimisation was reading as anti-collapse. With
the projector in place both controls collapse (rank 1.02 / 1.36), as they should.

**5. The result that limits all the others: neither arm beats its own initialisation by
much.** Probe accuracy at step 0 is 0.343. After 2000 steps backprop reaches 0.3605
(+0.018) and PC reaches 0.3380 (-0.005, i.e. *below* init). Almost all of the probe
accuracy in this grid is random-feature accuracy. The BP-vs-PC gap of 0.023 is real but
small against a baseline that is doing 95% of the work.

**6. Hidden layers beat the embedding in every arm** (0.3985 vs 0.3605 for BP; 0.3935 vs
0.3380 for PC). Consistent with SIGReg pinning the embedding's marginal and costing some
class structure — which is why the measurement plan reads spectra at hidden layers.

## What this means for the research question

The question was whether PC yields richer latents better suited to patch prediction.
This grid cannot answer it yet, and now for an honest reason rather than a broken one:
the training signal is too weak for either arm to separate from a random encoder. Before
the PC-vs-BP comparison can carry weight, the setup needs to *learn* — longer budget
(LeJEPA uses 400 epochs; 2000 steps here is ~46), a stronger encoder, or a stronger
augmentation stack. Reporting "PC gives richer latents" off either version of this grid
would be unearned.

## Not assessed

Seed variance (n=1 per cell), sensitivity to the projector choice (parameterless BN vs a
learned MLP head), and whether the ordering holds at longer budget.
