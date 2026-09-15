# Q1 / Phase A — does PC give richer latents than BP, under an identical objective?

## The question, stated so it can fail

> Train the **same encoder** on the **same objective** with the **same data budget**,
> changing **only how the weight gradient is obtained** — backprop vs predictive
> coding — and ask whether the PC-trained representation is measurably richer.

"Richer" is not a mood. It is the metric list in §4. If every metric comes out
inside run-to-run noise, the answer is **no** and that is a publishable answer.

**What this phase is NOT.** No predictor, no patch prediction, no world model.
That is Phase B, and it is *conditional* on this phase — LeJEPA's own ablation
shows adding a predictor is neutral-to-harmful for representation quality
(ResNet-50 / ImageNet-100: 83.93% without a predictor vs 83.57% with; ViT-tiny:
71.79% vs 67.74%). So the predictor cannot be the thing that makes PC look
good, and testing latents directly is the more direct experiment.

## 1. Design — the one-variable rule

|                | BP arm | PC arm |
|----------------|--------|--------|
| encoder        | identical | identical |
| objective      | identical (`sigreg.lejepa_loss`) | identical |
| optimiser, lr, clipping, batch order, seed, data budget | identical | identical |
| **gradient source** | `jax.grad` | relaxation + local error terms |

Anything else that differs is a confound. If you need to tune the PC arm's
`activity_lr` or `T`, tune it and **report the sweep**, don't quietly pick a
winner — the µPC parameterisation exists precisely so learning rates transfer,
so a required per-arm retune is itself a finding.

## 2. Why SIGReg is in here at all

Three separate jobs. Only the first is "match LeJEPA":

1. **Anti-collapse.** The view-prediction term alone has a trivial optimum:
   map everything to one point. Something has to forbid that. SIGReg forbids it
   by pushing the embedding's marginal toward `N(0, I)`.
2. **Removing the per-arm tuning confound.** The alternatives to SIGReg
   (stop-grad, EMA teacher, temperature/centering schedules) all carry
   hyperparameters that would need per-arm tuning — reintroducing exactly the
   confound §1 exists to kill. SIGReg is hyperparameter-light by comparison.
3. **Cross-arm comparability.** Metrics that depend on the scale of the
   representation are only comparable across arms if both arms' embeddings sit
   at the same scale. SIGReg pins that scale.

### The trap this creates, and the fix

SIGReg regularises the **embedding's** spectrum directly. So measuring the
embedding's eigenspectrum measures *the regulariser*, not the representation —
both arms will look flat and you will have learned nothing.

**Read spectral metrics at the HIDDEN layers.** SIGReg does not constrain them,
and they are exactly where PC and BP should diverge, since PC's relaxation
assigns activities to every hidden node rather than propagating a signal
through them. `encoder.forward` therefore returns `(z, hs)` and `hs` is
mandatory, not decoration.

Correction to a blanket claim made earlier in this project: **only
spectral/second-moment metrics are neutralised at the embedding.**
Assignment-based metrics (kNN, clustering agreement) keep full discriminative
power there, because SIGReg constrains the marginal, not which point goes where.
So kNN at the embedding is a valid headline number.

### Measured property of the statistic (from `test_helpers.py`)

At `N=1024`, `K=16`, 128 directions, paper defaults:

| alternative | statistic | vs H0 band (1.17 ± 0.20, 3sd = 1.76) |
|---|---|---|
| collapsed (0.1×) | 402 | ~230× the band edge |
| scaled 3× | 576 | ~330× |
| bimodal | 2.26 | 1.3× |
| uniform, unit variance | 1.85 | 1.05× |
| Laplace, unit variance | 1.67 | *inside* the band |

**Read this correctly.** SIGReg is a powerful detector of second-moment misfit
(collapse, scale drift) and a weak detector of shape misfit at unit variance.
That is fine — job (1) above is a second-moment job. But do not describe the
embedding as "Gaussian" on the strength of a low SIGReg value; it is
"unit-covariance and not obviously collapsed".

## 3. The one genuinely new problem

Discriminative PC clamps the output to a per-sample target. **SIGReg has no
per-sample target** — it is a statistic over the batch. So the output node
cannot be clamped, and the loss must enter through the output node's activity
gradient:

```
dF/dz_out + dL/dz_out  =  (z_out - f_out(z_L)) + dL/dz_out
```

Two treatments, both to be built and compared (`cfg.pc.output_drive`):

- **`coupled`** — recompute `dL/dz_out` at every relaxation step. Exact, but
  row *n* of the gradient depends on every other row, so the energy is *not* a
  sum of per-sample energies and the batch relaxes jointly.
  `test_helpers.py::t_batch_coupling` verifies the coupling is real: perturbing
  row 0 moves row 5's gradient by 4.8e-03.
- **`frozen`** — compute `g = dL/dz_out` once from the feedforward embeddings,
  hold it fixed for all *T* steps. Factorises over the batch.

**Identity that makes `frozen` cheap:** holding `g` fixed is *exactly*
equivalent to clamping the output to the per-sample pseudo-target
`y_n := z_out_ff[n] - g[n]`, since for `L~ = ||z_out - y_n||²/2` we get
`dL~/dz_out = z_out - y_n`, which equals `g[n]` at `z_out = z_out_ff`. So
`frozen` is ordinary discriminative PC against a pseudo-target and can reuse
jpc's existing API unchanged. `coupled` cannot, and needs a hand-written
relaxation.

This is the direct analogue of the `fixed_prediction` assumption, applied to
the *loss* instead of to the *predictions*. Earlier in this project we verified
that `fixed_prediction` is load-bearing for PC tracking backprop: with it, PC
branch gradient `-11.370010` vs backprop `-11.370533`; without it, `-3.413946`.
Whether the assumption survives a **batch statistic** driving the output is
unknown — nobody has reported it. **Measure it**: cosine between the PC weight
gradient and the true BP gradient, as a function of *T*.

The difference between `coupled` and `frozen` is itself a result.

## 4. What gets measured

At **every layer**, both arms, on a deterministic centre view (no augmentation):

| metric | function | valid where | reads as |
|---|---|---|---|
| kNN accuracy | `metrics.knn_accuracy` | all layers incl. embedding | linear-probe-free task signal |
| eigenspectrum / effective rank / log-cond | `metrics.spectrum_report` | **hidden only** | dimensional richness |
| intrinsic dimension (TwoNN) | `metrics.twonn_dimension` | all layers | manifold dimension |
| collapse diagnostics | `metrics.collapse_report` | all layers | did an arm degenerate |
| layer-wise CKA between arms | `metrics.linear_cka` | all layers | *where* the arms differ |
| projected two-sample (KS over random directions) | `metrics.projected_two_sample` | embedding | are the two embeddings the same distribution |

`train.evaluate` and `train.compare_arms` are **complete** — they run this whole
pass. Metric validity notes are in each function's docstring; trust those over
this table if they ever disagree.

Metrics deliberately **excluded**: decoder reconstruction quality. An invariance
objective *deliberately discards* pixel information, so reconstruction measures
information retained, not representation quality. If you report it, report it as
"information retained" and never as "better".

## 5. Files

| file | status | what it is |
|---|---|---|
| `config.py` | **complete** | frozen dataclasses + `q1_grid()` run generator |
| `sigreg.py` | **complete, tested** | Epps–Pulley quadrature, SIGReg, view-prediction term, full LeJEPA loss |
| `metrics.py` | **complete, tested** | the §4 metric suite, self-contained |
| `data.py` | **complete, tested** | Galaxy10/MNIST loading, multi-view augmentation, deterministic eval view |
| `test_helpers.py` | **complete, 14/14 pass** | pins down the above |
| `encoder.py` | **skeleton — ours to write** | µPC-parameterised FC residual net |
| `train.py` | **skeleton — ours to write** | BP arm, PC arm, shared loop (`evaluate`/`compare_arms` already done) |

## 6. Running it

```bash
cd /home/yash/pc_predictor/Q1-PhaseA
/home/yash/pc_predictor/.venv/bin/python3 -W ignore test_helpers.py
```

Use the **absolute** interpreter path. Two sandbox gotchas, both already
handled but worth knowing:

- `PYTHONSAFEPATH=1` is set in this environment, which strips the script's own
  directory from `sys.path`. `test_helpers.py` and `train.py` re-insert it at
  the top. Any new entry-point script needs the same three lines.
- A relative path like `../.venv/bin/python3` triggers a `sys.prefix` mismatch
  warning. Harmless, but the absolute path avoids it.
- The CUDA plugin traceback on every run is the GPU being unreachable from the
  sandbox. JAX falls back to CPU. Ignore it.

## 7. Open items — resolve before trusting any depth result

1. **RESOLVED — the `ScaledLinear` "scaling gap" is not a gap.** µPC's Table 1
   gives `b_ℓ = 1`: µPC initialises weights from a *standard* Gaussian and puts
   **all** the scaling in per-layer premultipliers, applied at every forward
   evaluation, not folded into the weight. Table 1 in full:

   | | `a_1` (input) | `a_ℓ` (hidden) | `a_L` (output) | `b_ℓ` (init var) |
   |---|---|---|---|---|
   | PC   | 1 | 1 | 1 | `N_{ℓ-1}^{-1}` |
   | µPC  | `N_0^{-1/2}` | `(N_{ℓ-1} L)^{-1/2}` | `N_{L-1}^{-1}` | 1 |

   So the standard-normal draw is correct and the misplaced comment is the only
   error. jpc applies the premultipliers inside the energy
   (`_get_param_scalings` → `scalings[l] * vmap(model[l])(...)`), which is
   load-bearing: folding `a_ℓ` into the init gives an identical forward pass at
   `t=0` but scales the gradient on `W` by `a_ℓ` instead of `a_ℓ²`, destroying
   the learning-rate transfer that is the point of the method.
   **Gate:** the depth factor is conditional —
   `al = 1/sqrt(N) if no_skips else 1/sqrt(N*L)`. Omit `skip_model=` and you
   silently get the non-residual scaling and lose depth transfer.
2. **RESOLVED — `FCResnet` and `make_mlp(param_type="mupc")` agree.** Verified
   at `D=784, N=128, depth=30`: jpc gives `a_1=0.03571429`,
   `a_ℓ=0.01613743`, `a_L=0.0078125`, exactly Table 1, and `len(model)=30`, so
   jpc's `L` *is* the `depth` argument — `FCResnet`'s `1/sqrt(width*depth)`
   matches `1/sqrt(N*L)` at **ratio 1.000000**. Hidden weight std = 1.0013
   (`b=1` confirmed). Structure also matches: `make_mlp` builds each layer as
   `Sequential([Lambda(act_fn), Linear])` — activation *before* the linear map,
   identity on layer 0 — and the energy adds the skip **unscaled**
   (`err = z_ℓ - a_ℓ·W_ℓφ(z_{ℓ-1}) - I·z_{ℓ-1}`), which is what
   `ResNetBlock.__call__` does. `make_skip_model(depth)` places identities at
   `l ∈ [1, depth-2]` (28 of 30 non-None at depth 30), matching the paper's
   `τ_ℓ = 1` for `ℓ = 2..H`.
   **Decision: use `make_mlp` + `make_skip_model`; do not use `FCResnet` here.**
   jpc's µPC branch reads the width off `model[0][1].weight.shape[0]`, i.e. it
   assumes the `Sequential` nesting — and `FCResnet` already applies its scaling
   in `__call__`, so routing it through jpc with `param_type="mupc"` would apply
   the scaling **twice**.
3. **Appendix B.6 discrepancy** in the LeJEPA paper: Eq (5)'s all-pairs form and
   Eq (7)'s distance-from-centre form are not equal; they differ by exactly the
   within-view variance (`t_b6_discrepancy` verifies: 13.3492 vs 9.4851, gap
   3.8640). `sigreg.py` implements Eq (7). Also, the paper's rendering truncates
   the statistic's final scaling factor — implemented as the `scale_by_n`
   option; **check the official codebase before quoting an absolute SIGReg
   value.**
4. **Galaxy10 is not downloaded.** zenodo.org returns 504 through the sandbox
   proxy (gateway timeout at ~30 s, not a size limit); `data/galaxy10/Galaxy10.h5`
   is a 92-byte error page — delete it before retrying. MNIST is unpacked at
   `data/MNIST/raw` and `data.load_dataset` handles both, so bring the pipeline
   up on MNIST and swap datasets once Galaxy10 lands. `h5py 3.16.0` **is**
   installed (see below), so only the download is missing.
5. **NEW — µPC's output scaling puts the embedding at the wrong scale for
   SIGReg.** Measured per-layer RMS of `init_activities_with_ffwd` at
   `depth=30, N=128, embed=64` on Gaussian input:

   ```
   hidden (29 layers): 0.9696 0.9822 0.9833 ... 1.2017 1.2136 1.2230
   embedding (last):   0.0685
   ```

   The hidden stack is flat in depth — Desideratum 1 holds empirically at
   depth 30, which is worth citing. But `a_L = 1/N` (not `1/√N`) leaves the
   embedding ~17× smaller in RMS than the hidden activations, and SIGReg's
   target is `N(0, I)`, i.e. RMS 1. So at initialisation the regulariser sees
   an embedding whose scale is wrong by more than an order of magnitude, and
   its first job will be to inflate it — a large, uninformative transient that
   contaminates the early training comparison and, worse, differs between the
   PC and BP arms because it interacts with the relaxation.
   µPC's `a_L = 1/N` is designed for a **readout to logits**, not for an
   embedding a distributional regulariser will constrain. Three options, and
   this is a real decision to make before the first run:
   (a) keep `a_L = 1/N` and accept/measure the transient;
   (b) use `gamma` — jpc's `_get_param_scalings` already supports
   `scalings[-1] /= gamma`, so `gamma = 1/√N` restores an O(1) embedding
   through the supported API, at the cost of departing from Table 1 at the
   output layer only. **Measured** (D=784, N=128, embed=64, 256 Gaussian
   inputs), embedding RMS by choice of gamma:

   | jpc depth | last hidden | `gamma=None` | `gamma=1/N` | `gamma=1/√N` |
   |---|---|---|---|---|
   | 10 | 1.2071 | 0.0733 | 9.3761 | **0.8287** |
   | 30 | 1.2599 | 0.0725 | 9.2848 | **0.8207** |
   | 60 | 1.2621 | 0.0763 | 9.7680 | **0.8634** |

   `gamma = 1/√N` gives `a_L = 1/√N = 0.088388` (the µP/ntp output scaling)
   and lands the embedding at RMS ≈ 0.82–0.86, stable across depth — so the
   fix does not reintroduce a depth dependence. `gamma = 1/N` overshoots to
   ≈ 9.3 and is **wrong**; an earlier version of this README said `1/N`.
   **Gotcha:** `gamma` is accepted by `init_activities_with_ffwd`,
   `update_pc_activities`, `update_pc_params` and `pc_energy_fn`, but **not**
   by `make_pc_step` — so the training loop must use the lower-level
   activity/param update calls, as `mu_pc.ipynb`'s `train` already does;
   (c) standardise the embedding before SIGReg — but this is a normalisation
   on the output activity and so is at odds with inference convergence
   (§3.2 of µPC makes exactly this objection to activity normalisation).
   Whichever you pick, **apply it identically to both arms** — it is a
   property of the shared forward map, not of the learning rule.

## 9. Note on the venv

`uv` is not on `PATH` in this sandbox and the project venv shipped without
`pip`, so package installs need a one-time bootstrap:

```bash
./.venv/bin/python3 -m ensurepip --upgrade
./.venv/bin/python3 -m pip install <package>
```

That is already done — `pip 26.2.1` and `h5py 3.16.0` are installed against
CPython 3.14.7.

## 8. Cost note

The µPC smoke test — 30-layer residual PCN, width 128, T=30, batch 64 — ran
**100 iterations in 6.7 s on CPU** and reached 85.97% MNIST test accuracy after
~6,400 images. So the *T* sweep and the depth sweep are minutes of compute, not
hours. Sweep them properly; there is no excuse to guess.
