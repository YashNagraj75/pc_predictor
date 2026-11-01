#!/usr/bin/env bash
# Latent analysis (Phase C, inference only) on Galaxy10, MNIST and CIFAR-10.
#   bash job_latent.sh <project_root> <job_dir> [smoke]
# Per dataset: gates (test_latent.py, every analysed seed) -> latent_analysis.py.  smoke = two tiers: 200 images at K <= 6,
# then 64 images at K = 48 with one seed (exercises the expensive paths; prints timings).  Stops at the first failure.
set -uo pipefail
P="$1"; JOBDIR=$(readlink -f "$2"); SMOKE="${3:-}"
cd "$P/Q1-PhaseA" || exit 2
Q=$(pwd); PY="$(cd ../.venv/bin && pwd)/python"
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false NVIDIA_TF32_OVERRIDE=0 XLA_FLAGS=--xla_gpu_autotune_level=0
ST=$Q/latent_status.txt; note() { echo "$(date +%m-%d_%H:%M) [latent${SMOKE:+/smoke}] $*" | tee -a "$ST"; }
note "start at $(git rev-parse --short HEAD)"
O="$JOBDIR/latent${SMOKE:+_smoke}"; mkdir -p "$O"
run_ds() {   # name dir backbone seeds(csv) ref_json
  local ds=$1 dir=$2 bb=$3 seeds=$4 ref=$5; local sp="${seeds//,/ }"
  cd "$dir" || return 2
  if [ -f "$O/latent_$ds.json" ] && grep -q '"complete": true' "$O/latent_$ds.json"; then note "$ds already complete"; return 0; fi
  $PY -W ignore $Q/test_latent.py "$bb" "$seeds" "$ref" > "$O/gates_$ds.log" 2>&1; local rc=$?
  note "$ds gates rc=$rc $(grep -c '\[PASS\]' "$O/gates_$ds.log") pass $(grep -c '\[FAIL\]' "$O/gates_$ds.log") fail"
  [ $rc -eq 0 ] || return 4
  if [ -n "$SMOKE" ]; then
    local s1="${sp%% *}"
    timeout 3h $PY -W ignore $Q/latent_analysis.py --backbone "$bb" --out "$O/latent_${ds}_t1.json" --n 200 --kmax 6 --n-infl 100 --seeds $s1 > "$O/run_${ds}_t1.log" 2>&1; rc=$?
    note "$ds smoke tier 1 rc=$rc"; [ $rc -eq 0 ] || return $rc
    timeout 3h $PY -W ignore $Q/latent_analysis.py --backbone "$bb" --out "$O/latent_${ds}_t2.json" --n 64 --kmax 48 --n-infl 32 --seeds $s1 > "$O/run_${ds}_t2.log" 2>&1; rc=$?
    note "$ds smoke tier 2 rc=$rc $(grep ' done ' "$O/run_${ds}_t2.log" | tr '\n' ';' | cut -c1-400)"; return $rc
  fi
  rm -f "$O/latent_$ds.json" "$O/latent_${ds}_pred.npz"
  timeout 30h $PY -W ignore $Q/latent_analysis.py --backbone "$bb" --out "$O/latent_$ds.json" --seeds $sp > "$O/run_$ds.log" 2>&1; rc=$?
  note "$ds analysis rc=$rc"; return $rc
}
G32=$Q/../findings/32_saccades_precision_followup/diag_saccade2.json
run_ds galaxy10 "$Q" runs/bp_sigreg_etafix_s0 1,2 "$G32" || exit $?
run_ds mnist "$Q/ds/mnist" runs/bp_sigreg_enc_s0 0,1,2 pred_runs/eval_views.json || exit $?
run_ds cifar10 "$Q/ds/cifar10" runs/bp_sigreg_enc_s0 0,1,2 pred_runs/eval_views.json || exit $?
note "DONE"
