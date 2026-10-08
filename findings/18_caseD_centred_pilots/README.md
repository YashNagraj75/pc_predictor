# 18 — Case D revision 2 (centred heads): every pilot fails, both arms, by a rank-1 route

Job a62a3a24 (2026-10-08), PILOT_ONLY, seed 0, 2000 steps. Heads explain
sqrt(n)(f_l - mu_l)/||f_l - mu_l||, mu_l an EMA (0.99) of the head-layer mean.
Everything else as folder 17. Revision-1 pilot dirs archived on the host in
Q1-PhaseA/runs_archive/caseD_rev1_normalised.

Step 1999 (start: kNN 0.343, ||f_5|| 2.36, centred participation ratio 7.0; no-head
seed-0 references at step 2000: PC-ALM 0.3475, BP 0.3775):

| K | PC-D kNN | BP-D kNN | PC-D ||f_5|| | BP-D ||f_5|| | PC-D PR | BP-D PR | PC-D clipped steps |
|---|---|---|---|---|---|---|---|
| 1 | 0.2910 | 0.3425 | 42.0 | 23.2 | 1.01 | 1.05 | 88% |
| 2 | 0.2455 | 0.3450 | 38.3 | 26.8 | 1.01 | 1.03 | 81% |
| 3 | 0.2510 | 0.3460 | 40.1 | 29.8 | 1.01 | 1.03 | 83% |
| 4 | 0.2480 | 0.3275 | 37.7 | 28.9 | 1.01 | 1.03 | 81% |
| 5 | 0.2410 | 0.3255 | 37.9 | 28.8 | 1.01 | 1.03 | 83% |

Centring closed the constant-offset route; both arms then put all the variation of the
head layers on ONE direction (PR -> 1.0 within 250 steps), so +-v explains every input.
BP clipping 0%. PC also clips on >80% of steps and loses embedding information
(kNN to 0.24-0.29). The 120-step CPU smoke had BP healthy (PR 5.5): it did not
predict the real schedule. Per D5 nothing re-tuned, no full runs.
