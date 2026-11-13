# Q1-PhaseA: code map

All experiment code lives here; each script's results are filed under `../findings/<NN>_<experiment>/`.
Job scripts (`job_*.sh`) are what the GPU host runs; `test_*.py` are the gate tests run before each job.
`golden_ref/` is a frozen copy of the original Phase-A modules used by `test_sparse.py` as a regression reference.
The original Phase-A design notes are kept in `../findings/13_reports/README.md`.

## ViT I-JEPA predictor (stage A)

| file | what it does |
|---|---|
| `a0_vit.py` | A0 script: measures credit (cosine to BP) and step time of ePC arms at varying T/power-iteration settings on a depth-10 predictor. |
| `diag_spc.py` | Inference-only diagnostic of the standard-PC baseline (spc_u1): per-node error propagation, credit vs BP, float32 vs float64, and step-size rule comparisons. |
| `gpu_gate.py` | GPU-vs-CPU gradient gate at real training shapes and trained weights (findings/41: Triton GEMMs gave wrong batched gradients). run <root> <tag> <outdir> [depth]: save... |
| `job_latent_vit.sh` | Latent analysis (plan_latent_analysis.md, rev 2): GPU-vs-CPU gate at both depths, known-answer tests, then latent_vit.py on every finished stage-A 25-epoch run under... |
| `job_stageA.sh` | Stage A (plan_stageA_scoped.md v3). bash job_stageA.sh <project_root> <job_dir> <smoke/a0/depth/noise> [depth] Resumable: re-launch the same command after a deadline... |
| `job_vit.sh` | ViT I-JEPA (plan_vit_ijepa.md). bash job_vit.sh <project_root> <job_dir> <gates/smoke/v0> gates: test_vit.py (V0 + transformer ePC, incl. real-batch memory); smoke:... |
| `latent_vit.py` | Inference-only latent analysis of the stage-A ViT predictors: attention/context-mixing stats, probes, target-basis PCA, interventions, and CKA between runs. |
| `test_blockrho.py` | Regression + measurement script for the per-block rho precision scheme: gates against a prior commit, measures block gain spread, and decides whether the per-block... |
| `test_latent_vit.py` | Known-answer tests for latent_vit.py's CKA and residualisation helpers. |
| `test_vit.py` | Gate suite for vit_ijepa.py and vit_epc.py: mask sampling, attention masking, patchify round-trip, padding/EMA invariance, init scale, a training smoke test, and the... |
| `train_pred_vit.py` | Trains a stage-A I-JEPA ViT predictor on frozen V0 encoders via BP or ePC arms, evaluating under clean/corrupted conditions. |
| `vit_epc.py` | ePC and state-PC (sPC) relaxation/gradient rules for the I-JEPA transformer predictor, with optional ALM multipliers and chunked batch gradients. |
| `vit_ijepa.py` | I-JEPA (Assran et al. 2023) model for CIFAR-10: ViT encoder/predictor classes, I-JEPA multi-block masking, EMA target update, training loop and evaluation gate... |

## Predictors on frozen latents (residual / generative / saccades)

| file | what it does |
|---|---|
| `choose_rho.py` | Pre-registered (gpc_alm_prereg.json, rev 2) choice of the high penalty and the phase-2 budget, per depth, from diag_c2.json. m(T2) = median over scale in {1, 2}, k in... |
| `diag_c2.py` | C2 pre-launch diagnostic (review findings F1/N2/N1/D1), never feeds training. Production pipeline: s0 = 0, alm_infer T=400 on k seen views, phase 2 over all views for... |
| `diag_c3.py` | Inference-only follow-up to C2: belief-inference vs decoder-quality separation, and depth-10 multiplier-loop stability sweep. |
| `diag_c4.py` | Inference-only diagnostic (never trains) testing two fixes from Qi et al. 2025 -- spiking precision and a forward weight-update rule -- on trained generative decoders. |
| `diag_e1_credit.py` | Inference-only check of whether PC-ALM's weight gradient converges to BP's on trained E1 checkpoints as inference budget/penalty grow. |
| `diag_epc.py` | Inference-only diagnostic comparing error-based PC (ePC) against standard state PC (sPC) and BP on trained feedforward and generative checkpoints, writing... |
| `diag_saccade.py` | MPC-style saccade/glimpse diagnostic for the PC predictors: crop-table rendering, bottom-up/top-down fixation planning, glimpse pooling, and scoring against... |
| `diag_saccade2.py` | Saccade-policy diagnostic (grid4 front end): kNN class probes, precision-weighted latent pooling (leave-one-latent/-glimpse-out), the 'ig' planner, and... |
| `diag_scale.py` | Diagnostic sweeping crop area to measure how glimpse scale degrades a frozen backbone's latents: OOD (Mahalanobis) distance, kNN class accuracy, and trained... |
| `ds_check_latents.py` | Prep-time check (review N6/P7): the stored test latents must be reproduced by re-encoding the test views exactly as the robustness tests will, and the latents must... |
| `ds_gate.py` | Backbone gate for a new dataset: the frozen encoder must carry class information well above its own initialisation (kNN k=20 cosine on deterministic centre views).... |
| `gpc_sweep.py` | Step-size / precision selection sweep for the generative PC predictor arms; --collect writes the selected settings. |
| `job_alm_pred.sh` | Job script running the PC-ALM feedforward predictor pipeline on frozen latents: gates, alpha x T selection via held-out nmse, training the selected config at 3 seeds,... |
| `job_bp_lr.sh` | Shell job that selects a learning rate for a BP feedforward residual-d10 predictor on a Galaxy holdout, then trains/tests 3 seeds at that lr across three datasets. |
| `job_c2_amend_c05.sh` | C2 amendment (user-approved 2026-10-17, post hoc): depth-10 PC-ALM rho=4 at c=0.5, seeds 0,1,2. The pre-registered c=1 arm oscillated at depth 10 (seed 0 test nmse... |
| `job_datasets.sh` | The two best PC predictors and their frozen BP references on another dataset (plan_datasets_precision.md, rev 3). bash job_datasets.sh <project_root> <job_dir>... |
| `job_e123.sh` | E1 residual feedforward PC-ALM (d3, d10) + E2/E3 generative belief/credit separation (d3, d10). Spec: wiki 'Method - Deep feedforward PC-ALM and belief-credit... |
| `job_epc.sh` | ePC inference-only check (diag_epc.py): gates (test_epc.py) then the diagnostic, Galaxy10 checkpoints. bash job_epc.sh <project_root> <job_dir> |
| `job_ffres_alm_fix.sh` | Job script rerunning the ffres PC-ALM predictor (depth 10) with the GPU numerics fix: gates, audit, hyperparameter selection, 3-seed training, and testing. |
| `job_ffres_bp_fix.sh` | Retrain the frozen residual feedforward BP reference at depth 10 with the GPU numerics fix ONLY (findings/27). Same as E1: standard parameterisation, lr 1e-3 (no... |
| `job_gpc.sh` | Generative PC predictor, stage 1 (seed 0), spec rev 2. bash job_gpc.sh <project_root> <job_dir> (resumable) |
| `job_gpc_alm.sh` | C2 (rev 2, after the 3-reviewer pass): PC-ALM generative predictor at depths 3 and 10. Spec: wiki 'Method - PC-ALM generative predictor (C2)'; prereg... |
| `job_latent.sh` | Latent analysis (Phase C, inference only) on Galaxy10, MNIST and CIFAR-10. bash job_latent.sh <project_root> <job_dir> [smoke] Per dataset: gates (test_latent.py,... |
| `job_pred.sh` | Phase P on the GPU host: frozen-latent predictors. bash job_pred.sh <project_root> <job_dir> Backbone = runs/bp_sigreg_etafix_s0 (canonical BP, host data split).... |
| `job_qi.sh` | Qi et al. 2025 fixes trained into the generative PC predictor (seed 0 screen), depths 3 and 10. Spec: wiki 'Method - Qi fixes training screen'; prereg qi_prereg.json.... |
| `job_saccade.sh` | Saccade S2 (inference only): gates -> 200-image smoke -> full run (3 seeds, K <= 32, whole views <= 192). PC predictors get glimpses; frozen BP references get one... |
| `job_saccade2.sh` | Saccade follow-up (inference only): old gates -> new gates -> 200-image smoke -> full run (3 seeds, K <= 32, views <= 192). bash job_saccade2.sh <project_root>... |
| `job_sep_seeds.sh` | Seeds 0-2 for the three amortisation arms of E2/E3 (rule sep), depths 3 and 10, with the GPU numerics fix. Settings frozen from pred_runs/sep_sel/selected.json... |
| `latent_analysis.py` | Phase-C latent-analysis sweep probing why generative/residual PC predictors degrade differently than their BP counterparts under occlusion/noise, across pooling... |
| `pred_bpset.py` | Trains a permutation-invariant set predictor (pooling encoder/decoder MLP pair) mapping a context subset of glimpse embeddings to target-transform predictions. |
| `pred_data.py` | Extracts per-layer latents (trained backbone + random-init baseline) and tau features for train/test splits, saved to an npz file for training downstream set predictors. |
| `pred_eval.py` | Evaluation utilities for latent-prediction tests: case generation, nmse/cosine/retrieval metrics, pixel-view construction/encoding/corruption, and reproduction checks. |
| `pred_ff.py` | Feedforward predictor baselines (BP, plain-PC, PC-ALM trained MLPs) mapping a seen view's latent to a target view's latent; includes the pre-registered target-layer... |
| `pred_ffres.py` | Residual feedforward predictor (E1): muPC-scaled MLP with skip connections, trained by BP, PC, or PC-ALM, with its own inference, training loop, and evaluation. |
| `pred_gpc.py` | Generative-latent-network (GLN) predictor: hierarchical predictive-coding model over per-view latents, trained/evaluated under plain PC, PC-ALM, BP auto-decoding, and... |
| `pred_sdpc.py` | Sparse deep predictive-coding (SDPC) dictionary predictor: non-negative sparse-coded layers trained by alternating proximal-gradient inference and dictionary updates. |
| `run_gpc_tests.py` | Trains and evaluates one generative-PC (gpc) predictor arm: selects step scale/precision, trains with divergence-ladder retries, then runs the full eval suite (pairs,... |
| `run_pred_ff.sh` | Phase P track 1 grid: {bp, pc} x {bp, rand} backbones x seeds {0,1,2}, layer emb (pre-registered), T=20, eta_h=0.05 (pre-registered 2026-10-12 from the real-batch... |
| `run_pred_ff_adaptive.sh` | Corrected PC arm (declared after the first grid, 2026-10-12): curvature-tracking activity step eta_h = 0.9 / (1 + 2 max_l //W_l//^2), recomputed every step. BP arm... |
| `run_pred_final.sh` | Phase P final runs (detached). 1. select SDPC hyperparameters from the validation sweep (pre-registered rule: lowest val k=1 nmse); 2. save the feedforward PC models... |
| `run_pred_tests.py` | Trains or loads a latent predictor (sdpc/bpset/ff/ffres/copy) and evaluates it on clean, k-seen, T_inf-sweep, and corruption-robustness cases, writing results to JSON. |
| `sdpc_sweep.py` | Grid sweep (lambda, beta, decay) for the sparse-dictionary PC predictor on the frozen BP latents; appends to pred_runs/sdpc_sweep.json. |
| `test_datasets.py` | Gate suite for the mnist/cifar10 dataset presets and loaders: shape/dtype/balance/determinism checks, corruption (occlusion/noise) checks, and Galaxy10... |
| `test_epc.py` | Gates for diag_epc.py's episodic-PC (ePC) port: known-answer checks against sPC/BP on random weights and trained checkpoints, for both the feedforward and generative... |
| `test_ffres.py` | Gates for pred_ffres.py: checks the feedforward/PC/ALM residual network's forward pass, energy, gradients, and stability against float64 and BP references. |
| `test_latent.py` | Gate suite for latent_analysis.py: reproduces stored evaluations, verifies swap-model identities, and runs known-answer tests for each statistic. |
| `test_pred_gpc.py` | Gates for pred_gpc.py: PC/BP/ALM inference stability and equivalence, locality of weight updates, batch invariance, predictive training, and divergence detection. |
| `test_pred_sdpc.py` | Unit tests for pred_sdpc: inference monotonicity/KKT checks, dictionary-gradient agreement, batch independence, and predict()'s causal behaviour. |
| `test_qi.py` | Gates for the Qi et al. options (spike, wrule) added to pred_gpc.py, checking the default path still reproduces the pre-change code. |
| `test_saccade.py` | Gate suite for diag_saccade.py: renderer/encoder/crop-table parity, planner, surprise, policies, and occlusion checks against reference computations. |
| `test_saccade2.py` | Gates for diag_saccade2.py (revision 2): reproduces findings/31 pooling/surprise/leave-one-out formulas, checks precision pools and ig weights, and compares saccade... |
| `test_sep.py` | Gates for the 'sep' rule in pred_gpc.py (belief stage, then credit assignment at fixed belief s). |

## Phase-A encoder (PC vs BP LeJEPA encoder)

| file | what it does |
|---|---|
| `_golden_pcalm.py` | Captures golden-reference snapshots of pcalm's PC-ALM relax/grad outputs, used by test_sparse.py to pin behaviour across edits. |
| `audit_jit.py` | Audit script comparing compiled (jit) vs. eager training gradients at trained weights, across every training path in the project, to catch GPU/XLA miscompilation. |
| `config.py` | Dataclass configs for the Phase-A encoder runs (data, encoder, SIGReg, PC-ALM inference, sparse prior, optimiser) and presets. |
| `data.py` | Dataset loaders (Galaxy10, MNIST, CIFAR-10) plus augmentation (random affine crop/flip/jitter views) and a shuffled batch iterator. |
| `det_check.py` | Review D5: the stage-1 seed-0 controls (commit 7468692) are reused; check the current code reproduces their first logged training row (same init, same batch stream)... |
| `diag_agree_heads.py` | Diagnostic comparing PC vs BP gradients (cosine, magnitude ratio) on the encoder, side-dictionary heads, and generative heads of sparse-case-D runs. |
| `diag_collapse.py` | Trains sparse PC/BP configs over a sweep of cases/arms and logs per-step health diagnostics to track representation collapse. |
| `diag_dose.py` | Measures the head-loss vs. task-loss gradient ratio ('dose') for PC and BP arms at given sparse-head depths, and can rescale PC head weights to hit a target ratio. |
| `diag_heads.py` | Diagnostic probes for sparse-coding 'heads' (case D): embedding/layer/code linear and knn probes under clean/noise/occlusion corruption, written to CSV. |
| `diag_inference.py` | Diagnostic for PC relaxation hyperparameters: checks the per-sample activity step eta_h is batch-invariant, then sweeps eta_h for relaxation stability/descent and... |
| `diag_learning.py` | CLI that measures per-layer weight displacement and probe trajectory for trained runs, flagging frozen layers. |
| `diag_sparse.py` | Diagnostics for the sparse-coding predictor arm: robustness under corruption, inference curve/amortisation gap, LASSO solver validity, PC-vs-BP gradient agreement,... |
| `encoder.py` | mu-PC-parameterised fully-connected residual encoder: builds the model/skip pair, runs the zero-energy feedforward pass, and computes the PC energy for it. |
| `gate_pilot.py` | Pairwise gate between a PC-ALM run and its matched BP run: checks divergence, clipping rate, activity-growth/active-fraction bounds, and (sparse arms) case-D/E... |
| `job_heads.sh` | Pipeline script for the Case-D sparse-side-heads study (K=1..5): preflight, gates, head selection, per-K pilot/gate, full training runs, screens, floor, probes,... |
| `job_sparse.sh` | SLURM-style driver script for the sparse PC-ALM arms experiment: gates -> selection -> pilots -> full runs -> floors -> probes -> diagnostics -> collect. |
| `metrics.py` | Representation-quality metrics (linear CKA, spectral diagnostics, kNN accuracy, intrinsic dimension, KS two-sample tests, collapse report) shared by the Q1 probes. |
| `pcalm.py` | PC-ALM (predictive-coding augmented Lagrangian method): primal-dual activity relaxation and the resulting encoder weight gradient, matching jpc.pc_energy_fn... |
| `probes.py` | Evaluation suite for a frozen embedding: scene/augmentation-factor probes (ridge, MLP, classification), a pixel decoder with an SSIM floor, path straightness, and... |
| `run_arms.py` | CLI driver that parses arm/seed/sparse-case settings into RunCfg objects and trains (or initializes-only) each planned Q1-PhaseA run, refusing to overwrite existing... |
| `run_probes.py` | CLI that loads saved runs and runs the full probe suite (classification/regression/decode/kNN) on their embeddings. |
| `select_heads.py` | Calibrates per-head sparsity (gamma), head weights, and (case E) the reconstruction weight beta_r for nested sparse-coding head sets, writing the selection to json. |
| `select_sparse.py` | Selects the sparsity thresholds beta (case B) and gamma (case C) that hit a target active fraction on sampled training centre views, writing the result to json. |
| `sigreg.py` | SIGReg (sliced Epps-Pulley normality test) and the LeJEPA invariance loss used as the Phase-A embedding regulariser. |
| `sparse.py` | Sparse-coding augmentations (cases B/C/D/E) for the PC encoder: dictionary/code init, ISTA/FISTA solvers, the deployed forward pass, PC-relaxation hooks, and... |
| `test_helpers.py` | Known-answer tests for the SIGReg, spectral-metrics, and data-batching helper functions. |
| `test_pcalm.py` | Regression tests for pcalm.py: energy equivalence to jpc at zero duals, dual growth with alpha, alpha=0 matching plain PC's weight gradient, and batch-size invariance... |
| `test_sparse.py` | Gate suite for the sparse-layer arms (sparse.py/pcalm.py): case-none golden-bit match, then cases B/C/D/E math, batch invariance, and BP controls. |
| `train.py` | Training loop for the Q1-PhaseA encoder: BP and PC-ALM arms share the objective/optimiser and differ only in the weight-gradient mechanism; also holds the shared... |
