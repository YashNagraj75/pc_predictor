# 16 — Sparse PC-ALM arms (cases B and C): pilots fail the pre-registered gate

Job 629aa534 (2026-10-07), after three code reviews and two aborted submissions
(8cd0a70c: gates not machine-independent; b3c717c6: NaN from an all-black view).
CPU gates all pass; selection gives beta = 0.2365, gamma = 0.0820 (active fraction
0.20 at init). 2000-step pilot, seed 0, one PC/BP pair per case.

Rule D5 (pre-registered): a pair whose deployed active fraction leaves [0.1, 0.3]
is skipped and reported, not re-tuned. Both pairs failed on the PC arm:

| arm | deployed active | distinct supports | embedding std | embedding norm | kNN start -> end |
|---|---|---|---|---|---|
| PC-B  | 0.008 (1 of 128 units) | 1   | 0.0    | 0.0   | 0.207 -> 0.1525 |
| C1-B  | 0.039                  | 115 | 0.0058 | 0.096 | 0.207 -> 0.3165 |
| PC-C  | 0.707                  | 2   | 0.0019 | 1.18  | 0.3175 -> 0.248 |
| C2    | 0.104                  | 914 | 0.0068 | 0.755 | 0.3185 -> 0.381 |

PC-B's deployed embedding is exactly zero for every input (its kNN value is a
tie-breaking artefact below the 0.171 majority rate). PC-C is near-collapsed.
The BP controls learn. n = 1 seed, 2000 of 17200 steps; no full runs were made.

Files: q1_sparse_pilots.png, collapse_check.json (1000 test images),
gate_B.json / gate_C.json, sparse_selected.json, runs/*/log.json + config.json,
sparse_review_outcomes.csv (the 20 de-duplicated review findings).
