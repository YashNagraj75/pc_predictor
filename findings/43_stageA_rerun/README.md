# 43 — Stage A rerun under correct numerics (Triton GEMMs off; findings/41): A0 and BP depth 10

Commit 8ee37d9. V0 retrained (gate identical to findings/38). Smoke ok for all five arms. GPU-vs-CPU gate passed.

## A0 (pre-set rule: cheapest (T, power its) with cos >= ref - 0.005 AND weakest-leaf cos >= ref - 0.01 for every
## ePC arm, ref = T 20 / 12 its) -> T 20, 12 its (14.4 s/step; BP 0.66 s/step)
| arm | T20/g12 cos, min leaf | T10/g4 cos, min leaf | s/step T20g12 / T10g4 |
|---|---|---|---|
| ePC-ALM median/4 | 0.9979, 0.985 | 0.9943, 0.973 | 14.4 / 7.0 |
| ePC median/4 (no mult.) | 0.891, 0.250 | 0.908, 0.363 | 14.3 / 7.0 |
| ePC median/16 | 0.719, 0.051 | 0.743, 0.221 | 14.4 / 7.0 |
T10/g4 missed the rule only on the ALM arm's weakest leaf (0.0112 below the reference, limit 0.01).
With correct gradients ePC-ALM at the planned median rule is at 0.998 credit on a 2-epoch BP predictor.

## BP depth 10, 25 epochs (test nmse; kNN higher is better)
| lr | clean | drop 25% | drop 50% | square cal | noise | noise cal | kNN |
|---|---|---|---|---|---|---|---|
| 3e-4 | 0.1997 | 0.2354 | 0.2973 | 0.4779 | 1.0555 | 0.8363 | 0.546 |
| 1e-3 | 0.1929 | 0.2266 | 0.2850 | 0.4750 | 1.0829 | 0.8373 | 0.544 |
| 3e-3 (selected, EDGE) | 0.1897 | 0.2222 | 0.2782 | 0.4741 | 1.1251 | 0.8416 | 0.548 |
These match the findings/40 BP runs to <= 0.001 nmse: the intermittent fault evidently did not hit the old BP
training processes. Whether it hit the old ePC runs cannot be determined, so those stay invalid.
