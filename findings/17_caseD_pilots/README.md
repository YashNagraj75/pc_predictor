# 17 — Case D (sparse side heads, K = 1..5): every pilot pair fails the gate, both arms

Job 4faac68e (2026-10-07), PILOT_ONLY, seed 0, 2000 steps, T = 16, eta_h = 0.0625, alpha = 1.
Heads explain the normalised copy sqrt(n) f_l/||f_l||; head strength calibrated so the head
gradient on the encoder is 0.25x the task gradient at init (measured 0.23-0.26 at init).
Spec: wiki 'Method - Sparse side-head arm plan'. Review outcomes: caseD_review_outcomes.csv.

Outcome (step 1999, trainer kNN; no-head seed-0 references at step 2000: PC-ALM 0.3475, BP 0.3775):

| K | PC-D kNN | BP-D kNN | PC-D ||f_5|| | BP-D ||f_5|| | PC-D supports | BP-D supports |
|---|---|---|---|---|---|---|
| 1 | 0.3005 | 0.3430 | 28.6 | 13.2 | 3 | 105 |
| 2 | 0.3025 | 0.3370 | 27.0 | 14.1 | 1 | 3 |
| 3 | 0.2770 | 0.3510 | 29.5 | 14.5 | 1 | 8 |
| 4 | 0.2830 | 0.3355 | 27.8 | 14.7 | 1 | 7 |
| 5 | 0.2795 | 0.3435 | 27.4 | 14.6 | 1 | 9 |

(||f_5|| starts at 2.36; layer-5 angular spread starts at 0.038 and ends ~1e-4 PC, ~1e-3 BP.)

Reading: normalising the head input closed the shrink route but left a constant-offset
route. Both arms inflate the head layers and push them toward one input-independent
direction, which one fixed sparse code explains; the head gradient on the encoder then
falls to ~0 (agree_pilot_K1.json). BP keeps the embedding informative (spread 0.02-0.23,
kNN flat at its init value) but stops learning; PC also loses embedding information
(kNN 0.343 -> 0.28-0.30, embedding norm 1.5 -> 11-16). Per D5 nothing re-tuned, no full runs.
