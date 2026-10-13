# 20 — Predictors on frozen latents: feedforward PC vs BP, SDPC generative predictor, BP set predictor

Job fdf01095 on the A4000 (2026-10-13), backbone runs/bp_sigreg_etafix_s0 (backprop LeJEPA+SIGReg; control = same
architecture untrained). Task: predict the frozen embedding of an unseen augmented view from k seen views + all transforms.
Protocol pre-registered in pred_sdpc_prereg.json before any test number. Metrics: nmse (standardised latents, lower better),
retrieval@1 among 2000 test images. 3 seeds on the trained backbone (sd <= 0.002 everywhere), 1 on the untrained one.

| method (bp backbone) | k=1 | k=2 | k=3 |
|---|---|---|---|
| copy a seen view            | 0.502 | 0.381 | 0.333 |
| feedforward, backprop       | 0.299 | — | — |
| feedforward, PC (adaptive step) | 0.332 | — | — |
| backprop set predictor (549k params) | 0.294 | 0.252 | 0.236 |
| SDPC, 5 layers, local learning (557k params) | 0.393 | 0.350 | 0.329 |

Robustness (k=2, one seen view corrupted in pixel space, increase in nmse): SDPC +0.10 / +0.36 / +0.43 vs BP set
+0.13 / +0.45 / +0.50 vs copy +0.18 / +0.66 / +0.63 (noise 0.1 / noise 0.2 / 25 % occlusion) -- but SDPC's absolute
corrupted error stays above or level with BP set's (0.453 vs 0.385; 0.707 vs 0.707; 0.778 vs 0.755).
Inference budget (SDPC, k=1): nmse 0.782 / 0.640 / 0.449 / 0.393 / 0.398 / 0.408 at T = 5/10/25/50/100/200 while
retrieval@1 keeps rising 0.001 -> 0.152.  Untrained backbone: SDPC fails (nmse ~1.0-1.07) where BP set reaches 0.31-0.33.
SDPC validation sweep: all 8 (lam, beta, lam_decay) settings within 0.0003 nmse -- layers 2-5 contribute nothing measurable.
