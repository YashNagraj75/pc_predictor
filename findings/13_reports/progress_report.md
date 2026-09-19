# Predictive Coding for a JEPA-Style World-Model Predictor — Progress Report

*Compiled 2026-09-11. Covers everything from the initial proposal read-through to the
T-sweep result that currently anchors the project's central finding.*

---

## 1. Where this started, and where the question actually is now

The project began from a proposal to train the **predictor inside an I-JEPA-style
joint-embedding world model with predictive coding (PC) instead of backpropagation
(BP)**, on the premise that PC's energy-based inference might give a "better gradient"
than BP.

That premise does not hold. Predictive coding is a biologically-motivated inference
scheme, but the settled theoretical result (Millidge, Tschantz & Buckley, 2020,
*"Predictive Coding Approximates Backprop along Arbitrary Computation Graphs"*) is that
PC's weight gradient **converges to the backprop gradient**, not that it improves on it.
This was verified directly rather than taken on faith (§2).

The question that survived this correction, and the one every subsequent experiment
has been aimed at, is:

> **Does training with PC's iterative energy-relaxation produce latent representations
> that differ from backprop's — and if so, are those latents *richer*, and better suited
> to the kind of patch-prediction task I-JEPA asks of its predictor?**

This reframing is the single most consequential decision in the project: it moved the
work from "does PC train faster/better" (a closed question) to "does PC's learning
*rule* leave a different signature on the representation" (an open one), and every
phase below is structured around answering it as directly and cheaply as possible
before committing to the full predictor architecture.

---

## 2. Literature foundations

Four papers were read end-to-end and extracted to text on disk so their claims could be
quoted exactly rather than from memory:

| Paper | Role in the project |
|---|---|
| Original proposal ("Towards Hierarchical Predictive Coding Predictors for Joint Embedding World Models") | Starting point; motivated the "better gradient" premise later retired |
| Millidge, Tschantz & Buckley (2020), arXiv 2006.04182 | Establishes PC ≈ BP on arbitrary computation graphs; basis for the graph-PC implementation exercise (§3) |
| **µPC** (mu-PC) — depth-µP-style reparameterisation for PCNs | Scaling predictive coding to deep residual networks; read to decide whether a PC-ResNet encoder is even trainable at the depth this project needs |
| **MPC** (Metarepresentational Predictive Coding) | Compares directly against I-JEPA on four datasets; read to check whether it already answers this project's question (it doesn't — see below) |
| LeJEPA (SIGReg) | Supplies the anti-collapse regulariser and reference training recipe adopted for the encoder study (§5) |

**Key findings from µPC**, load-bearing for everything downstream:
- Its experiments are **all fully-connected** — "ResNet" in µPC means a fully-connected
  residual network, not a convolutional one; convolutions and transformers are named
  explicitly as future work, not something the method has been shown to handle.
- It identifies two scaling pathologies of standard PC networks and fixes them with a
  depth-µP-style reparameterisation, restoring learning-rate transfer across depth and
  width.
- It proves a **backprop-equivalence theorem** for a specific regime, but that regime is
  brittle and does not explain why the method actually works in practice — the paper's
  own emphasis is on the ill-conditioning analysis, not the equivalence proof.
- Critically for the architecture plan: PC's inference dynamics are reported to be **at
  odds with activity normalisation** (LayerNorm-style), which is a required component of
  every transformer block. This blocks a straightforward PC-ViT and is documented as a
  structural conflict, not an engineering gap, in the design review (§4).

**Key findings from MPC**, and a correction worth recording because of how it was
caught: MPC was initially misidentified in conversation as µPC before its actual content
(saccadic foveal/parafoveal sampling, cross-stream lateral prediction, N-WTA sparsity)
was re-checked against the extracted text and the mistake retracted. Once correctly
read:
- MPC **does** compare head-to-head against I-JEPA, on four datasets (kNN probe,
  10 seeds): MNIST 98.10 vs 90.82, K-MNIST 93.10 vs 80.82, ETH-80 84.02 vs 69.09, NORB
  88.78 vs 74.68 — a wide, consistent margin in MPC's favour.
- That comparison cannot be used as evidence for this project's question, because it
  changes four things at once — input sampling, architecture, sparsity mechanism, and
  learning rule — in a single comparison. MPC's own internal ablation supports this:
  swapping in classical generative PC under the *same* saccade scheme (GPC-fov) already
  recovers most of MPC's advantage over I-JEPA, leaving only a small residual for the
  meta-representational machinery itself. The win is attributable mostly to the input
  sampling scheme, not to PC vs. BP.
- MPC's own I-JEPA baseline is also suspect: at 90.82% kNN on MNIST it plausibly sits
  *below* raw-pixel kNN, because the paper states I-JEPA was "adapted" to match a
  3-layer PC circuit's parameter count and training process — no ViT, no multi-block
  masking, no EMA target encoder. This was flagged as a needed sanity check, not asserted
  as fact.

**Conclusion drawn from the literature phase:** no existing paper answers the "richer
latents" question under controlled conditions. It has to be measured directly, with
architecture, data, optimiser, and objective held fixed across a PC arm and a BP arm.
This is what the rest of the project builds.

---

## 3. Computation-graph PC, implemented by hand

Before running anything at scale, the mechanics of PC on **arbitrary computation
graphs** — including residual/skip connections — were implemented and verified in
`graph_pc_walkthrough.ipynb`, a from-scratch JAX exercise notebook (not a solved
notebook — the load-bearing derivations were left as exercises and filled in through
guided debugging).

What this established, with exact numbers reproduced from the notebook's own runs:

- Three graph topologies were built: a plain `chain`, a residual net with the skip
  folded into one node (`resid_folded`), and one with the skip as an explicit graph edge
  (`resid_explicit`).
- **PC's weight gradient matches backprop's** on all three topologies (cosine similarity
  1.000000), confirming the Millidge et al. result empirically rather than by citation —
  e.g. one branch gradient came out `-11.370010` under PC against `-11.370533` under
  backprop.
- A subtlety that matters for every downstream design decision: this equivalence
  depends on a `fixed_prediction` assumption at each node. Turning it off changes the
  same gradient to `-3.413946` — i.e. the assumption is load-bearing, not decorative.
- **How many inference steps PC needs to converge is graph-structure-dependent**: the
  number of relaxation steps needed equals the graph's longest path to the output for an
  explicitly-represented residual connection (`resid_explicit` front: `[3,2,2,1,1,0]`,
  exactly matching shortest path to output), but doubles for a folded skip
  (`resid_folded`) relative to a plain chain of the same depth — a factor of 2, not the
  naively expected `L+1`.
- Trained end-to-end on a toy problem, PC at T=200 relaxation steps reaches the same
  loss as backprop (`0.284659` vs `0.285055`) at **24× the wall-clock cost** — the
  price of iterative inference made concrete on numbers we produced ourselves, not
  just asserted from the literature.

An interactive HTML visualisation (`exercise6_visualization.html`) was built on top of
these exact computed numbers to make the error-wave propagation through each topology
inspectable step-by-step.

---

## 4. Experiment design: a two-phase splice, not a modification of either paper

`pc_jepa_design_review.md` records the decision that shaped every experiment since. Its
verdict: the core comparison — *PC-trained vs. BP-trained encoder/predictor, identical
architecture, identical objective, measure the latents* — is feasible at small scale,
but three parts of the original proposal needed repair first:

- SIGReg alone is not a training objective (it needs an invariance/prediction term to
  pair with, discussed in §5); using it bare degenerates to a trivial fixed point.
- µPC's "ResNets" are fully connected, so "PC ResNet on Galaxy10" is an **unattempted
  extension**, not a reproduction of anything already published.
- A PC-ViT is blocked by the normalisation conflict noted in §2, so it was dropped as
  an arm rather than force-fit.
- The MPC saccade encoder was dropped from the comparison entirely, for the reasons in
  §2.

The resulting design is a **two-phase splice**:

- **Phase A** borrows LeJEPA's objective (one encoder, multiple augmented views,
  invariance loss + SIGReg anti-collapse term, no predictor) and runs it **twice, as
  fully independent runs** — one trained by PC, one by BP — producing two checkpoints
  that never coexist in one model. (A single model with one PC branch and one BP branch
  was explicitly rejected as incoherent; siamese branches are only comparable when
  weight-shared or EMA-coupled, which would defeat the point.)
- **Phase B** puts an I-JEPA-style **position-conditioned** predictor back on top of the
  frozen, cached Phase-A latents, and trains *that* twice (PC and BP), giving a 2×2 whose
  diagonal is full-PC vs. full-BP and whose off-diagonals attribute effects to the
  encoder vs. the predictor module specifically.
- Phase B's predictor **must** be position-conditioned (context latent + an encoding of
  which region to predict, with the target drawn from a region the context never saw).
  This matters because LeJEPA's own ablation shows that a predictor trained without
  positional asymmetry does **not** improve representation quality (e.g. ResNet-50:
  83.93% without a predictor vs. 83.57% with one), and SIGReg already handles collapse
  independently — so a plain latent→latent predictor would produce a null result for
  reasons that have nothing to do with predictive coding. With position conditioning,
  the predictor becomes a **measuring instrument**: a downstream task whose difficulty
  is sensitive to latent quality, run identically on top of both encoders.
- **Phase A alone is a go/no-go gate** for the whole plan: it was explicitly approved as
  the first thing to run because if PC and BP encoders turn out indistinguishable at
  every layer and every inference budget, Phase B's predictor experiment has nothing
  left to find.

This is also where the number of PC relaxation steps, **T**, was fixed as the
experiment's central independent variable, on the reasoning that PC at large T
provably approaches BP (§2/§3), while small T is where PC is genuinely a different
algorithm — so a flat PC-vs-BP difference *across* T would itself be a clean, reportable
negative result.

![Design diagram for the PC-vs-BP encoder study]({{artifact:f1c21f11-1f86-4d53-8bf4-a62d2f869325}})

---

## 5. Building and debugging Q1-Phase A

The Phase-A codebase (`pc_predictor/Q1-PhaseA/`: `config.py`, `data.py`, `sigreg.py`,
`metrics.py`, `encoder.py`, `train.py`) was built with the encoder forward pass and
energy delegated to the `jpc` library (JAX predictive-coding package,
`thebuckleylab/jpc`) rather than hand-rolled, so both arms share a bit-identical
forward map. Notable engineering work, kept brief here since it is infrastructure
rather than a finding:

- **Environment**: `jpc` is not on PyPI and pulls a JAX version range that would
  downgrade the CUDA-enabled install; the working recipe installs JAX/Equinox/Optax
  first, then `jpc` with `--no-deps`, pinned by hand. A 4-line compatibility shim for
  `jaxlib.xla_extension` (renamed upstream) had to be hand-written into both the local
  and the GPU-host virtual environments, and is not part of any wheel — it must be
  re-created in any new environment.
- **Architecture**: both arms are the *same* `(model, skip)` Equinox pytree, the same
  `loss_from_views`, the same optimiser (`optax.adamw` behind gradient clipping); the
  only line that differs between arms is which gradient function (`bp_grads` vs.
  `pc_grads`) produced the update. This was a deliberate structural choice, not
  discipline the implementer has to remember, specifically to keep the comparison
  clean.
- **Data**: Galaxy10 DECaLS (de-duplicated), downsampled to 69×69 and cached as an
  883 MB `.npz` (cold build 88 s, warm reload 5 s) to fit the sandbox's ~6 GB free RAM.
- **PC relaxation**: implemented via `jax.lax.scan` rather than an unrolled Python loop,
  because compile time is nearly flat in T (0.9 s at T=16 vs. 1.5 s at T=64) while
  run time scales linearly — important since T was going to be swept over more than an
  order of magnitude.
- **A performance red herring, caught and fixed**: an early smoke test measured PC at
  ~540× the wall-clock cost of BP per step; the actual cause was that `pc_relax`/
  `pc_grads` weren't JIT-compiled while the BP path was, making every PC call re-trace
  in Python. After correcting this, the ratio dropped to a genuine and stable ~187× at
  T=16 (later measured at 8 min per 2000-step run even at T=256 — cheap, not a budget
  constraint on this scale).

---

## 6. The collapse confound, diagnosed and fixed

The first full 2×2 grid (BP/PC × SIGReg on/off) produced a result that looked like a
finding but wasn't: **backprop's embedding collapsed to effective rank ≈2 regardless of
SIGReg, while PC held effective rank ≈21–23 in both conditions**, with PC's downstream
kNN probe (~0.28–0.30) beating backprop's (~0.13, chance level).

This did not survive scrutiny. Root-causing it showed:

- `sigreg.py`'s invariance term used a sum-reduction over the embedding dimension where
  the LeJEPA reference uses a mean, making the invariance term **128× too large**
  relative to the anti-collapse term at the project's embedding width — the primary
  driver of backprop's collapse.
- Near total collapse, the SIGReg statistic's gradient with respect to embedding scale
  vanishes quadratically, so **no learning rate could escape it** once collapsed — every
  backprop run in a learning-rate sweep from 1e-4 to 3e-2 froze at the exact same
  statistic value.
- PC's apparent "rank" advantage in this broken grid was a **different phenomenon
  wearing the same clothes**: PC simply hadn't managed to drive the mis-scaled loss down
  as effectively as backprop had (final feedforward loss 0.0362 for PC vs. 0.0001 for
  BP), so under-optimisation was being misread as anti-collapse resistance.

Three fixes were applied and the grid re-run (`_v2fix`, 2000 steps): correct the
invariance-loss reduction to a mean, insert a parameterless normalising projector
(BatchNorm, no learnable parameters) between the encoder and the SIGReg term so the
collapsed state becomes structurally unreachable rather than merely disfavoured, and add
a warmup + cosine learning-rate schedule matching the LeJEPA reference recipe.

**This reversed the ordering.** With backprop no longer collapsing:

| Run (2000 steps) | kNN probe | Effective rank |
|---|---|---|
| Backprop + SIGReg | 0.3605 | 21.05 |
| PC (T=16) + SIGReg | 0.3380 | 23.14 |
| Backprop, invariance only | ~0.13 (collapsed, rank ≈1) | — |
| PC, invariance only | ~0.13 (collapsed, rank ≈1) | — |

Backprop now wins the downstream probe; PC's rank advantage does not translate into
probe accuracy. Both invariance-only controls still collapse without SIGReg, confirming
the regulariser is doing necessary work in both arms, not compensating for one.

![Fixed-grid comparison after the three corrections]({{artifact:2f46225b-b1a5-4619-91b0-6c88c0f409f5}})

---

## 7. Scaling the training budget: the gap widens, not closes

The fixed grid was re-run at the full budget used in the LeJEPA reference recipe on a
comparably-sized dataset (400 epochs = 17,200 steps, tag `_e400`):

| Arm | kNN probe | Effective rank | Intrinsic dim. (TwoNN) | Dims to 95% variance |
|---|---|---|---|---|
| Random init (floor) | 0.343 | — | — | — |
| Backprop + SIGReg | **0.3910** | 18.33 | 9.22 | 20 |
| PC (T=16) + SIGReg | 0.3305 | **20.81** | **22.71** | **50** |

At this budget, backprop clearly clears the random-init floor (+0.048) and saturates
around epoch 80–150; **PC never exceeds its own random-init probe accuracy at any point
across the full 400-epoch run.** The gap between the two arms *widened* with more
training, from 0.023 at 2000 steps to 0.061 at 17,200 — the opposite of what
under-optimisation would predict.

This is the sharpest statement of the project's central empirical tension so far:
**"richer" and "better" come apart cleanly, and the separation grows with budget.**
PC's latents are measurably higher-dimensional by every representational-richness
metric used (effective rank, intrinsic dimension, dimensions needed for 95% of
variance), but that extra dimensionality is not class-discriminative — it does not help
a downstream k-NN classifier. Effective rank, in this setting, is not a proxy for usable
structure.

![400-epoch comparison: PC's latents are higher-dimensional and lower-performing at once]({{artifact:a9f3e780-09a1-4aa7-afb2-0e8dc03a9bb6}})

A cross-arm CKA analysis, run alongside this, localises where the two arms actually
diverge: hidden-layer representations inside the eight residual blocks stay highly
similar between BP and PC throughout training (CKA ≈0.92–0.94 at every block), while
the embedding head itself diverges sharply (CKA ≈0.22 with SIGReg on). The BP/PC
difference is concentrated at the readout, not distributed through the trunk.

---

## 8. The T sweep: ruling out "PC just needs more inference"

The natural next hypothesis was that PC is simply **under-optimised at T=16** — that
more relaxation steps would let it catch up. A sweep over T ∈ {16, 32, 64}, at the full
400-epoch budget with everything else held fixed, tested this directly and **refuted it**:

| T | Loss at feedforward embeddings | Loss at relaxed activities | Energy F | kNN probe | Effective rank | Dims to 95% |
|---|---|---|---|---|---|---|
| 16 | 0.6942 | 0.5744 | 0.0018 | 0.3305 | 20.81 | 50 |
| 32 | 0.7469 | 0.2544 | 0.0073 | 0.2970 | 7.33 | 29 |
| **64** | **0.9109** | **0.0371** | 0.0075 | **0.2655** | **1.71** | **2** |
| Backprop (reference) | 0.3999 | — | — | 0.3910 | 18.33 | 20 |

The pattern is monotone and unambiguous across the whole swept range: **more
relaxation steps solve the objective PC is trained on** (loss at the relaxed activities
falls 0.5744 → 0.0371, a 15× improvement) **while making everything the encoder is
actually deployed with strictly worse** — the feedforward loss rises, the downstream
probe degrades below even the T=16 result, and effective rank collapses from 20.81 to
1.71.

Checking the evaluation code confirmed the mechanism: the downstream probe is computed
from the **feedforward** pass for both arms — which is also how a joint-embedding
encoder would actually be deployed in practice. The reading this supports:

> **Activity relaxation and the feedforward weights are competing routes to satisfying
> the training objective.** More relaxation steps let the activities themselves absorb
> the objective's pressure, shrinking the residual signal that would otherwise drive a
> weight update — so the weights are never pressured into producing a good standalone
> feedforward map, and the embedding collapses because the anti-collapse regulariser's
> pressure gets absorbed by the activities too.

This is a **decoupling** between PC's inference target and its deployed representation,
not insufficient inference — and it explicitly supersedes the under-optimisation
hypothesis raised in §7. Corroborating evidence: PC's own energy term F was already
essentially converged at T=16 (0.0018) and does not meaningfully improve by T=64
(0.0075); if inference had genuinely been starved, F should have been falling sharply
with T, not sitting flat.

![T sweep at the full training budget: relaxation and the feedforward map move in opposite directions]({{artifact:4deccc48-b2ce-4667-a5ec-6375baea552c}})

---

## 9. Where this leaves the original research question

Put together, Phases A's results across three training budgets and a T sweep give a
fairly complete answer to the reframed question from §1, at least for the encoder-only
setting studied so far:

- **PC does produce different latents from backprop**, and the difference grows with
  training budget rather than shrinking — this part of the original intuition survives.
- **"Different" and "richer" are not "better."** By every spectral/dimensionality
  metric tried, PC's latents at T=16 use more of the representational space than
  backprop's. But that extra structure does not help a linear/kNN downstream probe, and
  pushing PC toward better-converged inference (higher T) makes this worse, not better,
  because it comes at the direct expense of the feedforward map that the representation
  actually needs to be good.
- **The result the original proposal hoped for — PC as a source of better gradients or
  better latents "for free" — does not hold in this setting.** The honest, reportable
  finding is more specific and arguably more interesting: the two learning rules
  diverge sharply at the readout/embedding layer while staying nearly identical through
  the residual trunk (§7's CKA result), and the divergence is driven by a genuine
  optimisation-dynamics effect (the relaxation/feedforward decoupling of §8), not a
  representational-capacity advantage.
- This does **not**, by itself, rule out predictive coding for the *predictor* module
  specifically (the project's original target) — a predictor's output is consumed as a
  prediction against a target, not deployed as a standalone embedding the way this
  encoder is, so the decoupling mechanism found here may or may not transfer. That is
  exactly what Phase B (§4) was designed to isolate, and it has not yet been run.

---

## 10. Open threads and recommended next steps

In the order they were last discussed and roughly cheapest-first:

1. **Downward T sweep (T = 4, 8)** — same cost as the sweep just run, and it closes the
   sweep into a defensible statement either way: either there is an interior optimum in
   T below 16, or the useful amount of encoder-side relaxation is effectively zero.
2. **Glimpse/saccade frontend as a validation control** — replacing the input sampling
   with MPC-style foveal/parafoveal glimpses on the *same* PC-ResNet encoder, both arms,
   changing only how the input is sampled. This was agreed as viable but is understood
   to validate the pipeline rather than rescue PC's encoder performance, since the
   decoupling found in §8 is a property of the learning rule, not the input
   distribution.
3. **Move to the predictor-only (Phase B) configuration** — this is the arm the
   encoder-level finding does *not* argue against, and it is the part of the original
   proposal this whole encoder study was designed to de-risk before committing to it.
   Position enters the predictor's computation graph as a second parent of the first
   hidden layer with its own weight matrix (reusing the multi-parent merge already built
   in the graph-PC exercises, §3), which keeps content and position weights separately
   inspectable and ablatable.
4. **Raw-pixel/random-init floor verification for MPC's I-JEPA baseline** — a cheap,
   still-outstanding check of whether MPC's reported I-JEPA number on MNIST is
   genuinely below the do-nothing floor, which would further weaken MPC's comparison as
   independent evidence either way.
5. A same-arm seed control (BP-seed-1 vs. BP-seed-2 CKA) to establish the noise floor
   that every BP-vs-PC CKA number in §7 should be read against — flagged as needed but
   not yet run.

---

## 11. Artifact index

**Literature and design**
- [pc_jepa_design_review.md]({{artifact:6d7e0c8e-ab28-485c-8cd6-16532a320bb4}}) — the design verdict and staged plan (§4)
- [pc_jepa_design_diagram.png]({{artifact:f1c21f11-1f86-4d53-8bf4-a62d2f869325}}) — design diagram
- [mpc_table1_vs_ijepa.csv]({{artifact:0ab7d637-c5e2-4625-a80b-a65858ea5722}}) — MPC vs. I-JEPA comparison table, transcribed from the paper (§2)

**Computation-graph PC exercises (§3)**
- [graph_pc_walkthrough.ipynb]({{artifact:d3b05180-65c9-4ba2-a80a-7ee59e436f00}}) — the canonical walkthrough notebook
- [graph_pc_solutions.py]({{artifact:352d0ae7-874c-47f7-8df8-e0d1320f5068}}) — end-to-end reference solutions
- [graph_pc_results.png]({{artifact:fbbbef6a-08b2-46c1-87dd-a0763eec1f0b}}) — equivalence/training results figure
- [exercise6_visualization.html]({{artifact:207316aa-1e23-48d5-906b-429cfa17099b}}) — interactive error-wave visualisation

**Q1-Phase A: collapse diagnosis and fix (§6)**
- [lejepa_vs_ours_collapse_diagnosis.md]({{artifact:1bf92f28-57b2-4ed4-b09d-dc239b093a12}})
- [sigreg_collapse_trap.png]({{artifact:50b1766e-8755-4bef-86e3-b52f21418e98}})
- [q1_fixed_grid_report.md]({{artifact:aba1fa55-5e67-4a48-b383-b729ea3b80cf}}), [q1_fixed_grid_results.csv]({{artifact:418655dd-7bd3-4996-aa4a-ed9305719c3f}}), [q1_fixed_grid.png]({{artifact:2f46225b-b1a5-4619-91b0-6c88c0f409f5}})

**Q1-Phase A: full-budget grid (§7)**
- [q1_e400_report.md]({{artifact:fe2f72b8-8dd1-4422-be55-7cec6acefac4}}), [q1_e400_results.csv]({{artifact:bc2319b6-e6ff-4f47-8351-cd8b6e3da112}}), [q1_e400_grid.png]({{artifact:a9f3e780-09a1-4aa7-afb2-0e8dc03a9bb6}})

**Q1-Phase A: T sweep (§8)**
- [q1_tsweep_report.md]({{artifact:066b9fcb-f145-4046-bb3f-df4447076baf}}), [q1_tsweep_results.csv]({{artifact:932d1ebc-a7de-4c34-8cc9-2f7ddca156b8}}), [q1_tsweep.png]({{artifact:4deccc48-b2ce-4667-a5ec-6375baea552c}})

**Codebase**
- `Q1-PhaseA/{config,data,sigreg,metrics,encoder,train}.py` — on the granted host path `/home/yash/pc_predictor`, not duplicated here as artifacts (source of truth is the repo)
