# PC-trained JEPA: design review and staged plan

Status: design note, written before any experiment. Nothing here is a result.

---

## 1. Verdict in one paragraph

The core comparison — **PC-trained vs BP-trained encoder/predictor, identical
architecture, identical objective, measure the latents** — is the right experiment
and it is feasible at small scale. Three parts of the proposal as stated need
repair before it can run: (a) SIGReg alone is not a training objective;
(b) µPC's "ResNets" are fully connected, so "PC ResNet on Galaxy10" is an
unattempted extension, not a reproduction; (c) the ViT arm is blocked by a
documented conflict between activity normalisation and PC inference convergence.
The MPC saccade encoder should be dropped from this project. Most importantly,
the *interesting* regime is deliberately **incomplete inference** (small T) —
at convergence PC provably approaches BP, so a well-converged PC arm is
predicted to produce the same latents as BP.

---

## 2. What the source papers actually support

### µPC (verified from the paper text)

- Architectures: **fully connected residual PCNs**. Not convolutional.
  Quote: "We trained fully connected residual PCNs on standard image
  classification tasks (MNIST, Fashion-MNIST and CIFAR10)."
- Depth: up to **128 layers**, stable across activation functions.
- Results: MNIST ~98% in 5 epochs; Fashion-MNIST ~89% in <15 epochs;
  CIFAR-10 "far from SOTA because of the fully connected (as opposed to
  convolutional) architectures used."
- Tuning: weight + activity learning rates only. No momentum, weight decay,
  or nudging.
- Inference budget: T = number of hidden layers. One-step inference degrades
  performance both asymptotically and with depth.
- Conv and transformer architectures are listed explicitly as **future work**.

### The activity-normalisation blocker (verified)

µPC §3.2: residual connections and activity normalisation "both of which remain
key components of the modern transformer block", but "any kind of normalisation
of the activities seems at odds with convergence of the inference dynamics to a
solution." Without normalisation, vanilla ResNet activations explode with depth.

LayerNorm is in every transformer block. This is a structural conflict, not an
engineering detail.

### Does a PC transformer exist?

- The PCN tutorial/survey notes that prior work has trained transformers and
  VAEs with PC, flagging that attention's softmax sums over all nodes in a layer
  and so falls outside the standard layer-wise PC formulation.
- "On the Infinite Width and Depth Limits of Predictive Coding Networks"
  (arXiv 2602.07697, same lab as µPC) trained 10-layer residual CNNs on
  ImageNet-1k and 12-block / 8-head nanoGPT-style transformers on character-level
  Tiny Shakespeare with PC, using "heuristic extensions for different
  architectures". Code lives in the jpc repo under `experiments/limits_paper`.
- The µPC author's PhD thesis (arXiv 2510.23323, Oct 2025) states "it remains
  unknown whether standard transformers, shallow or deep, can be trained at all
  with PC" — partially superseded by the above, but no PC-trained *ViT on images*
  recipe was found.

**Conclusion:** PC transformers exist as demonstrations at small scale with
per-architecture hacks. There is no drop-in ViT recipe.

### A framing warning from the same thesis

"PC cannot at present provide any practical benefits over BP." Do not let the
project's claim drift back toward "PC is better." The surviving question is the
one already settled on: *are PC latents different, and different in a way that
matters downstream?*

---

## 3. Three problems with the proposal as stated

### 3.1 SIGReg is not a training objective

SIGReg has no information-preservation term. An encoder that maps every input to
an independent draw from N(0, I) minimises it perfectly and carries zero
information about the input. It only works paired with a term that forces
different views of the same input together:

    L = (1-lambda) * L_pred  +  lambda * SIGReg

"Encoders PC-trained first using SIGReg" therefore needs the prediction term as
well. That is coherent — it is LeJEPA-style pretraining — but state it that way.

### 3.2 "Both predictors PC-trained" changes two things at once

LeJEPA has **no predictor**: its "prediction loss" is pure invariance, an L2
distance between view embeddings with no network mapping one to the other and no
spatial conditioning. Real I-JEPA has a predictor that consumes context
embeddings plus target position.

If both encoder and predictor switch to PC simultaneously, the experiment cannot
attribute any observed difference to either. Use a factorial design (§5).

### 3.3 Galaxy10 with fully connected residual nets will be weak

CIFAR-10 was already "far from SOTA" with FC architectures. Galaxy morphology is
fine-grained and spatially structured. Either accept that absolute accuracy is
not the point (the *contrast* between arms is), or take on the conv extension —
which is itself the µPC future-work item.

---

## 4. Implementation considerations

Ordered by how likely each is to silently ruin a run.

### 4.1 Freeze the projection directions during the inference loop  [CRITICAL]

SIGReg resamples its M directions every weight step (seeded by `global_step`).
PC runs T inference steps *between* weight updates. If directions are resampled
inside the inference loop, the energy F is non-stationary and the relaxation
chases a moving target — it will not converge, and every downstream diagnostic
becomes meaningless.

Correct order:

    for each weight update:
        sample A  (M unit directions)          <- once
        for t in 1..T:  relax activities under F with A fixed
        compute weight gradients from relaxed activities
        optimiser step

This still gives fresh directions per weight step, so LeJEPA's SGD-resampling
argument is preserved.

### 4.2 A batch-coupled loss breaks per-sample independence of inference

PC energy with a general output loss:

    F = sum_l || z_l - f_l(z_{l-1}) ||^2  +  L(z_out)
    dF/dz_out = eps_out + dL/dz_out

Mechanically fine with a batch-coupled L. But SIGReg's gradient at sample n
depends on **all other samples in the batch**, and L_pred couples all views of
one sample. So:

- PC inference is normally embarrassingly parallel over the batch. Here it is not.
- All B x V activity sets must be held in memory simultaneously for the entire
  relaxation. The inference loop cannot be microbatched.
- Activity memory ~ B * V * sum_l dim(z_l), and PC holds *every* layer's
  activities, unlike a BP forward pass which can free them.

This is the binding constraint on a single consumer GPU. Mitigation: SIGReg only
needs the K-dimensional embedding layer, so the cost is the intermediate feature
maps, not the statistic. Measure before committing to an image size.

### 4.3 Batch size is squeezed from both ends

SIGReg's empirical characteristic function needs enough samples to be a
reasonable estimate; PC activity memory scales linearly in B. Small B gives a
noisy ECF; large B may not fit. Measure the ECF noise floor against B on
synthetic data first — cheap, no training required.

### 4.4 T is the independent variable, not a nuisance parameter  [KEY INSIGHT]

At inference equilibrium PC approaches BP (µPC Theorem 1; the limits paper finds
PC converges to BP for wider-than-deep models given enough inference steps). So:

> **A well-converged PC arm is predicted to produce the same latents as BP.**

Any difference in representation must therefore live in the *non-equilibrium*
regime: small T, deeper networks, incomplete relaxation. So do not treat T as
"set it high enough and forget it". Sweep it, and report latent metrics as a
function of T. The prediction is a monotone collapse toward the BP latent as T
grows — and if that is what happens, the negative result is itself clean and
worth reporting.

### 4.5 The eval-time latent is the feedforward pass  [corrected]

An earlier draft of this note split the latent into "relaxed z_L*" and
"feedforward f_theta(x)" and claimed they diverge at small T. That is wrong at
**evaluation** time. With no target attached the energy is

    F = sum_l || z_l - f_l(z_{l-1}) ||^2   >= 0

which equals exactly 0 at the feedforward configuration -- the unique global
minimum for a DAG. Same fact as the error-wave result from this project's
graph-PC work: errors originate at the output, so with no output error there is
nothing to propagate, and relaxation from the forward initialisation does not
move.

**Consequence:** the eval latent is unambiguous and identical in procedure across
arms -- run the forward pass. No dual extraction, no extra eval compute for the
PC arm, no ambiguity in figure labels. The entire PC-vs-BP difference lives in
the learned weights.

The relaxed/feedforward distinction is real only *during training*, where the
loss term is attached and pulls activities away from the forward pass.

### 4.5b Probe every layer, not just the embedding  [KEY]

In BP a hidden activation is h_l = f(h_{l-1}) -- purely bottom-up, untouched by
the objective. In PC, z_l* is the equilibrium of an energy that *includes the
output loss*, so during training every intermediate activity is pulled toward
consistency with the objective from above as well as below. That is a structural
difference in what an intermediate representation *is*.

Direction matters: the **last** layer is where the two arms are most alike (loss
attached in both cases); the **middle** layers are where they differ most.
Probing only the final embedding samples the point of maximum similarity -- the
worst place to look for the effect. Run the full metric suite layer-by-layer.

### 4.5c Fully connected encoders and an invariance objective  [POWER RISK]

Not the generic "conv is better on images" point. The prediction term asks two
augmented views of one image to land at the same embedding. A conv net gets much
of that for free -- weight sharing is a built-in invariance prior. An FC net has
none: two crops are unrelated input vectors with no shared structure, and the
invariance must be learned from data, which is the hardest thing for FC nets on
images. So the architecture least able to represent invariance is being paired
with an objective made entirely of invariance.

The cost is **dynamic range, not bias**. Both arms suffer equally so the contrast
stays valid, but if both encoders produce weak latents every probe compresses
into a narrow band and a real PC-vs-BP difference can sit below noise -- an
underpowered experiment that cannot distinguish a null result from a null effect.

Two mitigations:

- **Tame the augmentations.** Small-jitter crops plus rotation/reflection only
  (which are the physically motivated ones for galaxies anyway). Invariance to a
  mild transform is learnable by an FC net; aggressive multi-crop probably is not.
- **Frozen shared front end.** A fixed untrained feature extractor (random conv
  filters, or patch PCA/whitening), byte-identical and frozen in both arms, with
  the FC residual PCN on top. Not a confound, since it is identical across arms.
  It *is* a scope limitation to disclose: the learning rule is never tested on the
  spatial part of the problem. MPC does effectively this with its hand-designed
  front end. Also shrinks the input the PCN sees (see 4.5d).

### 4.5d Input dimension is a PC memory problem, not just a parameter problem

FC first layer on 64x64x3 = 12,288 inputs: ~4x CIFAR-10's 3,072 and ~16x MNIST's
784. At width 512 that is 6.3M parameters in layer one alone. µPC never faced
this input dimension. Combined with 4.2 (PC holds an activity vector for every
layer x sample x view through the whole relaxation), this argues for 32-64 px, or
the frozen front end above.

### 4.5e Why staying fully connected is the conservative choice

µPC's guarantees -- learning-rate transfer, the Hessian conditioning analysis,
the reparameterisation exponents -- are derived for fully connected residual
blocks. Convolutional architectures are off the manifold where that theory holds,
which is why the paper lists them as future work. FC is defensible, not a
compromise. The price is the scope of the final claim: "PC and BP latents differ
in fully connected residual encoders on downsampled galaxy images" does not
generalise to conv or ViT, and a reviewer will say so.

### 4.6 Matched budgets, declared explicitly

- **Data**: match samples-seen (this is the data-efficiency question). Report
  wall-clock separately; do not present it as free.
- **Tuning**: PC has extra knobs (activity lr, T, ODE solver). Give both arms an
  identical search budget, tuned on validation, and report the budget. An
  under-tuned BP baseline invalidates the whole comparison.
- Earlier measurement from this project's own graph-PC work: PC at T=200 matched
  BP's loss at ~24x wall-clock. Expect that order of overhead.

### 4.7 Galaxy10 specifics

- Two versions exist (SDSS ~69x69, DECals 256x256, ~10 classes, tens of
  thousands of images). Verify counts and resolution against the astroNN docs
  before sizing anything.
- Downsample (64-96 px) for PC memory. Full 256x256 is not realistic here.
- **Scientific rationale worth stating:** galaxies have no canonical orientation,
  so rotation/reflection invariance is physically correct — unlike digits, where
  rotation changes the label. An augmentation-based invariance objective is
  better justified on this dataset than on most.
- **Counter-caution:** morphology is fine-grained. Aggressive local crops can
  remove the very structure (bar, spiral arms) that defines the class. Choose
  augmentations that preserve morphology, and check by eye.
- Class imbalance is a known issue — use balanced probe metrics.

### 4.8 Tooling

**JPC** (`github.com/thebuckleylab/jpc`) — JAX/Equinox, functional, core under
1000 lines, ships µPC, and uses ODE solvers (not just Euler) for the inference
dynamics, which can cut the number of steps needed. `jpc.make_pc_step()` is the
one-call training step. The limits-paper experiments (CNNs, transformers) are the
closest existing code to this project.

Alternative: PCX (also JAX, object-oriented).

The graph-PC implementation already built in this project is conceptually the
same construction, so JPC's internals should read easily.

---

## 5. Staged plan

Each stage is a decision point: if the effect is absent, stop rather than
escalate.

### Stage 0 — tooling check (days)
Reproduce a µPC fully-connected residual PCN on MNIST via JPC. Confirms the
install, the T convention, and lr transfer across depth. No new science.

### Stage 1 — the minimal honest experiment
Identical FC residual encoder in both arms. Objective = L_pred + lambda*SIGReg.
Dataset: Galaxy10, downsampled. Arms: **PC vs BP**, matched samples-seen and
matched tuning budget. Sweep T.

Measure:
1. Linear probe accuracy (abstraction)
2. KNN probe accuracy (local structure)
3. Retro-fit decoder MSE/SSIM (retention) — decoder trained on frozen latents,
   no gradient to the encoder
4. Covariance eigenspectrum: effective rank, sum of 1/lambda_k at matched trace
5. Shadow-projection statistics: M fresh random directions, 1-D statistic per
   direction, report the distribution (mean and max)
6. Data-efficiency curve: probe accuracy vs samples seen

Items 4-6 are task-free and cheap; they are the ones that answer "richer
latents" without depending on a probe choice.

**Pre-register which metric decides the question before running.** With six
metrics and two arms, post-hoc selection will find a difference whether or not
one exists.

### Stage 2 — introduce the predictor
Now a real JEPA: encoder + predictor + masked-target prediction. Factorial:

|  | BP predictor | PC predictor |
|---|---|---|
| **BP encoder** | baseline | isolates PC-at-predictor |
| **PC encoder** | isolates PC-at-encoder | both |

This is the design that can attribute an effect. It is also the original
research question.

### Stage 3 — convolutional layers (optional)
The µPC future-work extension. A contribution in its own right; also where
Galaxy10 accuracy becomes respectable. Only worth it if Stage 1-2 show something.

### Stage 4 — ViT (defer)
Blocked on the LayerNorm/inference-convergence conflict. Requires either
normalising the forward pass only, or replacing LayerNorm. Attempt only if
earlier stages justify it.

### Stage 5 — MPC saccade encoder (drop from this project)
Three hand-designed, non-gradient-trained components (foveated front end,
epistemic saccade planner, path-integration aggregator) with no BP counterpart.
Swapping it in changes the architecture, the input sampling, and the aggregation
at the same time as the learning rule. It is a separate paper.

---

## 6. Risks

| Risk | Why | Mitigation |
|---|---|---|
| PC latents identical to BP at convergence | Theorem 1 / limits paper | Make T the independent variable; the non-equilibrium regime is the hypothesis |
| Under-tuned BP baseline | PC has more knobs and gets more attention | Equal search budget, reported |
| Memory ceiling from batch-coupled inference | PC holds all layers x all samples | Measure early; downsample images; SIGReg on embedding only |
| Metric shopping across six probes | Small effect, many tests | Pre-register the decision metric |
| Absolute accuracy too low to be credible | FC nets on fine-grained morphology | Frame as a contrast study, or do Stage 3 |
| Wall-clock (~24x from this project's own measurement) | PC inference is sequential | Budget for it; use JPC's ODE solvers; keep depth modest |

---

## 7. Immediate next actions

1. Install JPC; run Stage 0 on MNIST.
2. Measure the ECF noise floor vs batch size on synthetic Gaussian data — no
   training needed, and it sets the minimum viable B.
3. Profile PC activity memory for the intended encoder at 64 px and 96 px to fix
   the image size.
4. Write the Stage-1 evaluation harness (six metrics above) *before* training
   anything, and fix the decision metric in writing.
