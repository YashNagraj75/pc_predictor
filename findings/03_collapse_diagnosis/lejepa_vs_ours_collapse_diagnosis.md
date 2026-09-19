# Why our backprop arm collapses when LeJEPA's ResNets don't

## Short answer

Two separate things, and only the second one is a "bug":

1. **Total collapse is an exact fixed point of SIGReg.** Once the embedding is
   at zero, the SIGReg gradient is *identically* zero, so no learning rate can
   pull it back out. This is why the 300x LR sweep found no escape.
2. **Our invariance term is 128x over-weighted relative to LeJEPA's**, because
   we reduce it with `.sum(-1).mean()` where the reference uses `.mean()`. That
   is what drives the embedding into the fixed point in the first place.

LeJEPA never hits either problem because SIGReg in the reference implementation
acts on the output of a **BatchNorm projector**, not on the raw encoder output.
BatchNorm divides by the batch standard deviation, so the collapsed state is
unreachable by construction.

## Evidence for (1)

Numerically evaluating our own `sigreg.py` Epps-Pulley statistic on synthetic
projections, at N=256, t_max=5.0, n_points=17, scale_by_n=True:

| embedding state                | SIGReg statistic | max abs gradient |
|--------------------------------|------------------|------------------|
| exactly collapsed to 0         | **104.6837**     | **0.0**          |
| collapsed to a constant c=0.7  | 209.3127         | 1.098, identical for every sample |
| std = 0.0026 (what we observed)| 104.6809         | 3.22e-2          |
| true N(0,1) (the target)       | 1.0528           | 6.96e-2          |

**104.6837 is exactly the value every backprop run froze at**, at every learning
rate from 1e-4 to 3e-2. The runs were not "slowly converging" -- they were
sitting in an absorbing state.

Two structural facts fall out of the table:

- At exact collapse the gradient is exactly zero. Absorbing state.
- At collapse to a *non-zero* constant the gradient is non-zero but **identical
  for every sample**, so all points translate together and the spread never
  grows. The whole collapsed set is invariant under SIGReg gradient flow.

Near collapse the statistic behaves as `104.6837 - C*sigma^2` with C = 409.87,
so the restoring force on the embedding scale vanishes quadratically as
sigma -> 0. The basin is not just a fixed point; it has a wide flat approach.

## Evidence for (2)

Reference implementation (`MINIMAL.md` line 172):

    inv_loss = (proj.mean(0) - proj).square().mean()

Ours (`sigreg.py`, `prediction_loss`):

    return ((mu[None] - Zv) ** 2).sum(-1).mean()

The reference averages over views, batch **and the feature dimension**; we sum
over the feature dimension. With `embed_dim = 128` our invariance term is
exactly **128x larger** than the reference's for the same embeddings, while
SIGReg is on the same scale in both.

Both codebases then combine them identically
(`lejepa_loss = sigreg * lambda + inv * (1 - lambda)`), so at our `lambd = 0.05`
the effective balance is off by two orders of magnitude. **To reproduce
LeJEPA's lambda = 0.05 balance under our sum-reduction you would need
lambda = 0.871.**

An invariance term that dominant has one global optimum -- map every view of
every image to the same point -- and SIGReg, weighted 128x too weakly, cannot
hold the embedding away from it long enough to matter.

## Recipe comparison

Reference values from `MINIMAL.md` / `launch_inet10.py` in the LeJEPA repo;
ours from `Q1-PhaseA/config.py` and `encoder.py`.

| | LeJEPA reference | Ours (Q1-PhaseA) |
|---|---|---|
| invariance reduction | `.square().mean()` | `.sum(-1).mean()` -- **128x larger** |
| what SIGReg sees | projector output: `MLP(512, [2048, 2048, proj_dim], norm_layer=BatchNorm1d)` | **raw encoder output, no projector, no normalisation** |
| proj_dim into SIGReg | 16 (minimal) / 128 (default) | 128 (= embed_dim) |
| backbone | ViT-S/8; Galaxy10 table also ResNet18/34, convnextv2_nano, levit_128 | fully-connected muPC residual MLP, width 128, depth 8, no bias, no norm |
| input | 128-224 px RGB images, convolutional | 32x32 RGB **flattened to a 3072-vector** |
| LR schedule | `LinearLR` warmup (1 epoch, start_factor 0.01) then `CosineAnnealingLR` to eta_min=1e-3 | constant `param_lr`, **no warmup, no decay** |
| optimiser | AdamW, weight_decay 5e-2 (backbone) | `optax.adamw` at default weight decay |
| lambda | 0.02 (minimal) / 0.05 (Alg. 1) | 0.05 -- but see the 128x above |
| views | 4-8, heavy multi-crop + colour jitter + blur + solarise | 4, crop 0.4-1.0 + rotation + hflip + brightness/contrast 0.2 |
| budget | 400-800 epochs | 2000 steps |
| quadrature | `[0, t_max]` doubled by symmetry | `[-t_max, t_max]`, 17 points |
| precision | bf16 | fp32 |
| online probe | yes, `LayerNorm(512) + Linear(512,10)` trained jointly | offline eval only |

## Can we pull their Galaxy10 weights?

**No -- they were never published, and they would not transfer anyway.**

- The GitHub repo has **no releases** and the README links no checkpoints.
- On the HuggingFace Hub, the only model under the authors' organisation is
  `galilai-group/LeVJEPA-VideoMix-Large`, a video model from a different paper.
  Every other "lejepa" model on the Hub is a third-party reimplementation
  (`OK-AI/lejepa-vits16-pretrain-in1k`, `caiovicentino1/lejepa-v1-tinyimagenet`,
  and similar), none of them Galaxy10.
- The Galaxy10 models on the Hub (`matthieulel/*-finetuned-galaxy10-decals`) are
  unrelated supervised fine-tunes, not LeJEPA.

Even if they existed, the dimensions do not match on any axis that matters: they
are **convolutional** ResNet18/34 on multi-hundred-pixel 2D images; ours is a
**fully-connected** residual MLP of width 128 on a flattened 3072-vector. There
is no weight tensor in common.

## What the paper actually reports for Galaxy10

In-domain pretraining, 11,008 training samples, 400 epochs, linear probe:
ResNet18 75.32, ResNet34 77.29, convnextv2_nano 76.05. Note these are
*from-scratch in-domain* numbers -- the point of that table is that LeJEPA works
on a small domain-shifted dataset, not that these are strong absolute scores.

## Consequence for the PC-vs-BP claim

This matters for the result, not just for tidiness. Right now the comparison is
**confounded**: the backprop arm did not merely learn worse representations, it
fell into a degenerate fixed point that SIGReg is structurally unable to escape,
while the PC arm -- running the *same* loss -- did not. Reporting "PC gives
richer latents" off this grid would be unearned, because the BP arm never had a
working objective balance.

The honest reading is the reverse and is still interesting: **PC's relaxation
avoided an attractor that plain gradient descent fell into.** But that is a
claim about optimisation dynamics, not about representation richness, and it
needs a non-collapsing BP baseline to be worth anything.

## Suggested fixes, in order of expected effect

1. Change `prediction_loss` to `.mean()` over the feature dimension (or set
   `lambd = 0.871`). One-line change; restores the reference loss balance.
2. Add a normalising projector between the encoder and SIGReg -- even a single
   `LayerNorm` or a BatchNorm-style batch-std division makes the collapsed state
   unreachable rather than merely unfavourable.
3. Add the warmup + cosine schedule.
4. Re-run the 2x2 grid. If the PC advantage survives a non-collapsing BP arm,
   the result is real.

Items 1 and 2 are the ones that decide whether the BP arm collapses at all.
