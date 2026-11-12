# 46 — Latent analysis of the stage-A ViT predictors, pass 1 (2026-11-12)

Host job 7b2c751c (commit b315c0b), `Q1-PhaseA/latent_vit.py` (rev 2 after 3 reviewers, 34 findings), tests 19/19, GPU gate
d10/d6 pass. 20 models: BP seeds 0-4 (lr 3e-3), BP seed 0 at lr 1e-3 (lr-matched control for the no-multiplier ePC arm),
ePC-ALM median rho, ePC-ALM per-block rho, ePC no multipliers (rho = gain/4, lr 1e-3), untrained seed-0 init; depths 6 and 10.
Standard PC (spc_u1) is added by pass 2 after its training. PC arms: one seed. BP numbers: mean [min, max] over 5 seeds.
Every trained model reproduces its reported test nmse (<= 1e-4 rel) and kNN. The job's output check flagged 3 PC models for an
elementwise instrumented-vs-model difference of 0.4-4e-4 of max|P| (limit 1e-4, float32 graph differences); their reported
nmse is still reproduced to 1e-4, so the results stand; the limit was too strict.

1. Noise. Under sigma 0.2 every trained predictor is WORSE than the untrained one (raw 1.03-1.13 vs 1.00): the prediction moves
   by more than its own size (||dP||^2/||P||^2 1.06-1.17). The calibrated-error increase is almost entirely that movement
   (cross term -0.006..-0.03). With the clean-fitted calibration (alpha ~ 1 for every model) PC's advantage is the full raw
   advantage: d6 median 0.947, per-block 0.959 vs BP 1.121 [1.114, 1.127]; d10 per-block 1.030 vs 1.124. Refitting alpha on the
   noisy predictions (findings/44 'calibrated', alpha ~ 0.43-0.55) shrinks all to 0.83-0.85. PC's predictions move less per unit
   context shift (noise gain x alpha 0.70-0.76 vs 0.79), although their worst-case Jacobian is larger (1.4-4x BP).
   lr control: BP at lr 1e-3 is also less noise-sensitive (1.076 / 1.082) and matches the no-multiplier ePC arm (lr 1e-3, 1.070 /
   1.048). The multiplier arms train at lr 3e-3 with gradient ratio ~1 to BP and still move less, so their noise edge is not a
   smaller effective step.
2. Missing context. BP relies more on nearby context: removing every visible context patch within 1.5 of a target (encoder
   fixed, 18.7 of 31.5 patches) costs BP +0.42-0.46 nmse vs +0.34-0.45 for PC; minus the same number removed at random: BP
   0.29-0.34, PC 0.20-0.24. PC attention is less local (mass-weighted distance / uniform 0.70-0.82 vs 0.65-0.66). BP's lead
   under context dropout is its clean lead scaled (ratio ctx_drop50/clean BP 1.47, per-block d10 1.49), not a reliance on near context.
3. Clean error. PC's extra error is in the fine directions: error / target variance in the lowest-variance band (95-100% of
   variance) d6 0.63-0.69 vs BP 0.54, d10 0.59-0.64 vs 0.53; top band 0.111-0.119 vs 0.107-0.108. PC under-predicts those
   directions (slope 0.35-0.45 vs 0.48-0.49, corr 0.56-0.65 vs 0.68-0.69); prediction effective rank 55-58.5 vs 58.6-59.0.
4. Internal solution (linear CKA on position-residualised states, nodes 1..2L; BP seed-seed 0.98-0.99 at every node; shuffled
   0.00; untrained init falls to ~0.6 by the last node; early nodes are near 1 for every model, uninformative). PC differs from BP
   in the middle and late layers (d10 per-block / no-mult 0.69-0.78 at nodes 13-17, untrained 0.60-0.67) and partly re-converges
   at the last nodes (0.93-0.95). Shared initialisation does not matter (vs BP seed 0 and vs seeds 1-4 differ <= 0.014).
   Linear probes of hidden states (0.60-0.62) do not separate anything: the untrained net (0.61) and the context embedding (0.626)
   score the same, so the predictor adds no linearly decodable class information. kNN of predictions 0.54-0.55 for every model.
5. Depth. From d6 to d10 PC's fine-band gap to BP halves (per-block 0.115 -> 0.058) while the top-band gap shrinks 0.011 -> 0.004;
   middle/late-layer CKA rises (per-block d6 0.44-0.64 at nodes 9-12 vs d10 0.74-0.94 at the last nodes): deeper PC gets closer
   to BP's solution, mostly in the fine detail.
Oddities left open: d6 per-block last node CKA 0.44 (below the untrained floor), d6 median node-1 CKA 0.45, d10 median early-node
ridge 0.49-0.51. Single PC seed throughout.
