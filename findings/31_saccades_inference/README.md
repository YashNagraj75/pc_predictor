# 31 — MPC-style saccades for the PC predictors, inference only (2026-10-27)

Host job 0a364854 (commit 70c853b), `Q1-PhaseA/diag_saccade.py`, GPU fix on, 63/63 gates passed. 2000 test images,
3 seeds, one corruption draw. PC predictors get glimpses; frozen BP references get one augmented whole view only.
Table: `q1_saccade_results.csv` (test nmse, mean of 3 seeds, grid4 geometry, full pool; seed sd <= 0.05).

- Clean: saccades beat the single clean whole-image crop (G d10 0.255 -> 0.214 at K=32) but lose to the same number of
  plain whole views (192 views: 0.199 clean, 0.193 augmented). Random glimpses beat the planner.
- The planner's error signal does nothing: epi == shuffled errors == bottom-up + IOR tour (within 0.002 everywhere).
- The only planner gain is its FIRST glimpse at the bright centre: under occlusion K=1 1.12-1.22 vs random 1.48-1.68,
  matching the occlusion oracle. Later glimpses tour the edges (into the occluder) and erase it (K=32 1.37-1.61).
- Parafoveal-only pools do best on clean (G d10 0.208): small foveal crops dilute the mean.
- PC with any multi-view input beats single-view BP, but that mixes input count with the learning rule.
