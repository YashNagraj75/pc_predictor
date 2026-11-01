#!/usr/bin/env bash
# Latent analysis (Phase C, inference only) on Galaxy10, MNIST and CIFAR-10.
#   bash job_latent.sh <project_root> <job_dir> [smoke]
# Per dataset: gates (test_latent.py, reproduce the stored evaluation) -> latent_analysis.py.  Stops at the first failure.
set -uo pipefail
P="$1"; JOBDIR=$(readlink -f "$2"); SMOKE="${3:-}"
cd "$P/Q1-PhaseA" || exit 2
Q=$(pwd); PY="$(cd ../.venv/bin && pwd)/python"
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false NVIDIA_TF32_OVERRIDE=0 XLA_FLAGS=--xla_gpu_autotune_level=0
ST=$Q/latent_status.txt; note() { echo "$(date +%m-%d_%H:%M) [latent${SMOKE:+/smoke}] $*" | tee -a "$ST"; }
note "start at $(git rev-parse --short HEAD)"
mkdir -p "$JOBDIR/latent"
run_ds() {   # name dir backbone gate_seed ref_json seeds
  local ds=$1 dir=$2 bb=$3 gs=$4 ref=$5; shift 5
  cd "$dir" || return 2
  $PY -W ignore $Q/test_latent.py "$bb" "$gs" "$ref" > "$JOBDIR/latent/gates_$ds.log" 2>&1; local rc=$?
  note "$ds gates rc=$rc $(grep -c PASS "$JOBDIR/latent/gates_$ds.log") pass $(grep -c FAIL "$JOBDIR/latent/gates_$ds.log") fail"
  [ $rc -eq 0 ] || return 4
  if [ -n "$SMOKE" ]; then EA="--n 200 --kmax 6 --n-infl 100 --seeds $gs"; else EA="--seeds $*"; fi
  $PY -W ignore $Q/latent_analysis.py --backbone "$bb" --out "$JOBDIR/latent/latent_$ds.json" $EA > "$JOBDIR/latent/run_$ds.log" 2>&1; rc=$?
  note "$ds analysis rc=$rc"; return $rc
}
G32=$Q/../findings/32_saccades_precision_followup/diag_saccade2.json
run_ds galaxy10 "$Q" runs/bp_sigreg_etafix_s0 1 "$G32" 0 1 2 || exit $?
run_ds mnist "$Q/ds/mnist" runs/bp_sigreg_enc_s0 0 pred_runs/eval_views.json 0 1 2 || exit $?
run_ds cifar10 "$Q/ds/cifar10" runs/bp_sigreg_enc_s0 0 pred_runs/eval_views.json 0 1 2 || exit $?
note "DONE"
