# 36 — Learning-rate-matched backprop vs residual feedforward PC-ALM (d10)

Host job 360339a5 (commit 5d2ef81), 2026-11-02. Diagnostic, user-approved; the frozen BPf references are unchanged.
BP got PC-ALM's selection: lr {1e-3, 3e-3, 1e-2} on the Galaxy 90/10 holdout, seed 0. Held-out nmse 0.350 / 0.314 /
0.289 -> lr 1e-2 (grid edge, as PC-ALM's was: 0.296 at 1e-2). Then 3 seeds on Galaxy10, MNIST, CIFAR-10, single-view tests.
PC-ALM = rho 16, T 400, lr 1e-2 (findings/29, 33, 34). diff = PC - BP; CIFAR PC per-seed values were not kept locally, so
its range is mean PC minus each BP seed.

| dataset | clean PC / BP | noise PC / BP | occlusion PC / BP |
|---|---|---|---|
| Galaxy10 | 0.2996 / **0.2929** (+0.007 every seed) | 1.411 / 1.426 (mixed, BP seed sd 0.020) | 1.892 / 1.832 (mixed, BP sd 0.050) |
| MNIST | 0.1911 / **0.1828** (+0.007..+0.009) | 0.3063 / **0.2985** (+0.006..+0.009) | 1.1225 / 1.1256 (-0.003) |
| CIFAR-10 | 0.2385 / **0.2304** (+0.008..+0.009) | 0.3763 / **0.3697** (+0.006..+0.007) | 0.9697 / 0.9714 (tie) |

- The feedforward PC win of findings/29/33/34 (0.04-0.07) was the learning rate: at the same lr BP is better on clean
  and noise by 0.006-0.009 on every dataset. Per the pre-registered classes this is "no detectable effect" (< 0.01), i.e.
  PC-ALM reaches near-parity; by the seed-spread parity rule BP wins narrowly (diffs >> seed sd ~0.001).
- Occlusion: tie (MNIST, CIFAR) or mixed (Galaxy).
- Both arms selected the top of the lr grid; a wider grid could move both.
