# 34 — CIFAR-10 transfer, and the three-dataset summary (Galaxy10, MNIST, CIFAR-10)

Host job a639250a (commit 3a61799), 2026-10-31/11-01. 39/39 gates; 12/12 runs rc 0; eval rc 0; 3 seeds per model.
Frozen CIFAR-10 encoder: kNN 0.385 vs 0.2745 untrained (class-error cut 15%); raw pixels 0.292. Training sets 15,675
images on every dataset; Galaxy10 settings everywhere. BP inference settled (4x steps: |change| <= 0.010).
Verdicts are paired by seed with the pre-registered classes: all 3 seed differences beyond 0.01 = PC or BP win,
beyond 0.03 = reportable, all within 0.01 = tie. Tables: q1_three_dataset_summary.csv, q1_cifar_mnist_details.csv.

| predictor | condition | Galaxy 192 views PC/BP | MNIST 192 PC/BP | CIFAR-10 192 PC/BP |
|---|---|---|---|---|
| generative amortised PC-ALM d10 | clean | 0.193 / 0.169 BP | 0.118 / 0.094 BP | 0.141 / 0.102 BP |
| | noise | 0.615 / 0.618 tie | 0.157 / 0.137 BP | 0.202 / 0.170 BP |
| | occlusion | 1.458 / 1.623 **PC** | 1.096 / 1.114 **PC** | 0.896 / 0.946 **PC** |
| residual ff PC-ALM d10 (lr 1e-2 vs BP 1e-3) | clean | 0.227 / 0.264 **PC** | 0.125 / 0.167 **PC** | 0.151 / 0.209 **PC** |
| | noise | 0.529 / 0.577 **PC** | 0.167 / 0.214 **PC** | 0.215 / 0.272 **PC** |
| | occlusion | 1.298 / 1.352 PC | 1.077 / 1.050 BP | 0.825 / 0.821 tie |

Consistent across all three datasets:
- the generative PC predictor beats frozen BP under occlusion with many views (CIFAR-10 by 0.048-0.052 on every seed);
  it loses on clean and noisy views everywhere;
- the residual feedforward PC predictor beats BP on clean and noisy views everywhere (0.04-0.07, reportable; lr caveat);
  its occlusion result does not transfer (BP on MNIST, tie on CIFAR-10).
- kNN class accuracy of the predictions is the same for PC and BP within ~0.02 in every condition.
