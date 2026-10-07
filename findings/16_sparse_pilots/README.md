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

## Collapse diagnostic (job 74b3d810, 300 steps, eval every 10, same init/keys as the pilots)

`diag_collapse.json`, `q1_sparse_collapse_diag.png`. Seed 0, n = 1.

- Input variation entering layer s (1 - median cosine to the mean of z_{s-1}), start 0.0376 in all arms:
  PC-B 0.0030, C1-B 0.0074, PC-C 0.0020, C2 0.0161 at step 299. PC erases it 13-19x; BP 2.3-5x.
- PC-B embedding norm, relaxed vs deployed: 0.37 vs 0.26 at step 0, 0.23 vs 0.0028 at step 299.
  The deployed network dies by step ~100 while the relaxed state PC trains on stays alive.
- Gap at s, ||z_s^relax - z_s^deploy|| / ||z_s^deploy||: PC-B 0.18 -> 0.07, PC-C 0.066 -> 0.053 (shrinks).
- Relaxation movement of layers below s (mean): PC-B 0.021 -> 0.007, PC-C 0.006 -> 0.002.
Reading: the shared initial condition (thin input variation at s) is not the differentiator. The
collapse is driven below s, where PC's relaxation hardly moves (weak task credit through the
multiplier-free link at s) while the sparse-layer energy acts at full strength. In case B the
relaxed-vs-deployed mismatch then hides the dead deployed network from training.
