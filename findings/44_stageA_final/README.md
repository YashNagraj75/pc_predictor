# 44 — Stage A final: ePC vs BP I-JEPA predictors on the frozen V0 ViT encoder (CIFAR-10), depths 6 and 10

Correct numerics (Triton GEMMs off, GPU-vs-CPU gate before every stage). Training code 8ee37d9. One seed per ePC arm; BP seeds 0-4
at its selected lr. Test: 2000 CIFAR-10 test images x 40 mask shapes; per-image sums -> paired bootstrap over images (2000 draws).
Runs: A4000 (BP, epc_k4 d10, epc_alm_k4 d10), L40 #1 (all depth 6), L40 #2 (epc_k16 d10, epc_alm_blk4 d10, BP d10 cross-check).
Raw data: pc_predictor-runs (l40a/, l40b/) and the A4000 runs_vit/stageA/.

- Parity (pre-registered: |gap| <= 0.01 AND bootstrap CI includes 0): no arm, either depth. Best: per-block rho at d10,
  clean nmse 0.1987 vs BP 0.1897 (+0.0090, CI [+0.0088,+0.0092]): inside the 0.01 margin but reliably worse (4.7% relative).
- Depth helps ePC far more than BP: BP 0.1921 -> 0.1897 (d6 -> d10); ePC-ALM median 0.2155 -> 0.2021, per-block 0.2128 -> 0.1987.
- Multipliers matter at d10 only: at d6 the no-multiplier gain/4 arm (0.2066) beats both ALM arms.
- Context dropout and the occluding square: BP better everywhere; the gap grows with the fraction of context removed.
- Noise: ePC far better raw (-0.04..-0.18) but mostly prediction size; after per-condition calibration ePC is still better by
  0.006-0.016 in 7 of 8 arms (not median-ALM d6), above BP's 5-seed range (0.004-0.005). Candidate; ePC seed spread unknown.
- kNN on predictions: all ePC arms within 0.011 of BP (0.535-0.550 vs 0.548; ceiling 0.594).
- Stability: no run diverged. Median-ALM d10 collapsed transiently (step ~2600, loss 0.10 -> 0.45, rho to its floor, after a gain
  spike at 2200) and recovered by step 3200. Per-block rho: no 10x gain jumps, highest minimum credit. No-multiplier arms: gain
  grows to 1e9-1e10, gradient ratio 0.24-0.35, cosine < 0.9 at 48-92% of logged points. Logs have one point per 200 steps.
- 5-epoch selection under-predicts the final gap for the median rule (d6 val gap 0.003 -> test 0.023).
- BP is identical on the L40 and the A4000 to 4 decimals at all three lrs: mixing GPUs across arms did not affect results.
