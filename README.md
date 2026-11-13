# pc_predictor

Predictive coding (PC) as the learning rule for the predictor of a joint-embedding (I-JEPA-style) world model:
PC vs backpropagation on frozen-latent residual predictors, generative PC predictors, saccadic inference, and
transformer predictors on a frozen I-JEPA ViT encoder (error-based PC with gain-matched precision).

## Layout

| path | contents |
|---|---|
| `Q1-PhaseA/` | all experiment code: models, PC/ePC inference, training, diagnostics, gate tests, host job scripts (see its README) |
| `findings/` | one directory per experiment (`NN_name/`): README with the result, tables, figures and raw outputs |
| `tools/` | host orchestration: environment setup, repo sync, run sequences, backups and their verification |
| `notebooks/` | the graph-PC walkthrough and the muPC notebook (learning material), with reference solutions |
| `PredictiveCodingBackprop/` | submodule: Millidge et al., *Predictive Coding Approximates Backprop along Arbitrary Computation Graphs* |

## Setup

Paths are never hard-coded; scripts read them from the environment. Copy `.env.example` to `.env`, edit it, then

```bash
set -a; . ./.env; set +a
bash tools/provision.sh        # or tools/rebuild.sh for the pinned versions
```

| variable | used by |
|---|---|
| `PCP_ROOT` | repository checkout on the machine that runs jobs (all `tools/` scripts) |
| `PCP_BACKUP_DIR` | `tools/fetch_merge.sh` (clone of the backup results repository) |
| `PCP_VERIFY_DIR` | `tools/verify_merge.sh` (scratch clone used for verification) |
| `UV` | `tools/provision.sh`, `tools/rebuild.sh` (uv executable, default `uv`) |

GPU jobs must set `XLA_FLAGS=--xla_gpu_enable_triton_gemm=false` and `NVIDIA_TF32_OVERRIDE=0` (findings/41: Triton GEMMs gave
wrong batched gradients); the job scripts do this and run `Q1-PhaseA/gpu_gate.py` before training.

## tools/

| file | what it does |
|---|---|
| `backup_push.sh` | Runs on a rented GPU next to its stage-A job. Every 15 min: commit run RESULTS (reports, per-image eval, final weights, logs; no training checkpoints) to... |
| `fetch_merge.sh` | Runs on the A4000 host (always on). Every 15 min: fetch main of the public backup repo and copy every FINISHED run (report.json or diverged.json) from each... |
| `l40_stage.sh` | Rented-GPU stage-A job. On a fresh instance: clone the code at 8ee37d9, install the A4000's exact package versions, fetch CIFAR-10 and the stage-A inputs from branch... |
| `provision.sh` | Creates the project .venv with uv (Python 3.11, JAX CUDA 12) under $PCP_ROOT and verifies the GPU. |
| `rebuild.sh` | Rebuilds the project .venv from scratch under $PCP_ROOT with pinned JAX/equinox/optax/jpc versions. |
| `run_latent_then_spc.sh` | One A4000 job (user 2026-11-12; rev 2 after 3 reviewers): (1) latent analysis of every finished stage-A run (its result does NOT gate training), (2) standard-PC... |
| `run_stageA_all.sh` | One host job for ALL remaining stage-A work, in order: depth 10, depth 6, BP noise floor (STAGES overrides, e.g. STAGES="gpugate:10 depth:6" on a second GPU). Each... |
| `sync_host.sh` | Bring the host checkout to GitHub main without losing host-only content. bash sync_host.sh <root> <jobdir> <expected_sha> |
| `test_backup.sh` | Local end-to-end test of backup_push.sh + fetch_merge.sh against a bare repo, under the host wrapper's bash -e. |
| `verify_backup.sh` | Runs on a rented GPU next to backup_push.sh. Every 15 min it checks GitHub itself (its own HTTPS fetch, not the push loop's copy): every pushable file must be on main... |
| `verify_merge.sh` | Runs on the A4000 next to fetch_merge.sh. Every 30 min: its own fresh fetch of the backup repo's main, then for every FINISHED rented-GPU run checks that the A4000... |
