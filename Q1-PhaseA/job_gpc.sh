#!/usr/bin/env bash
# Generative PC predictor on the GPU host (spec: wiki 'Method - Generative PC predictor plan').
#   bash job_gpc.sh <project_root> <job_dir>        resumable; reuses pred_latents.npz from job_pred.sh
set -uo pipefail
P="$1"; JOBDIR="$2"; cd "$P/Q1-PhaseA" || exit 2
PY=../.venv/bin/python; export JAX_PLATFORMS=${JAX_PLATFORMS:-cuda}
ST=gpc_status.txt; note() { echo "$(date +%H:%M) $*" | tee -a $ST; }
collect() { tar czf "$JOBDIR/q1_gpc.tar.gz" $ST gpc_gates.log pred_runs/gpc_sweep.json pred_runs/gpc_selected.json pred_runs/tests/gpc_*.json pred_runs/test_gpc_*.log 2>/dev/null; }
trap collect EXIT
[ -f pred_latents.npz ] || { note "pred_latents.npz missing (run job_pred.sh first)"; exit 3; }
python3 -c "import numpy as np; d=np.load('pred_latents.npz'); assert str(d['backbone_run'])=='runs/bp_sigreg_etafix_s0', d['backbone_run']" || { note "latents not from bp_sigreg_etafix_s0"; exit 3; }
note "start backend $($PY -c 'import jax;print(jax.default_backend())' 2>/dev/null | tail -1)"
JAX_PLATFORMS=cpu $PY -W ignore test_pred_gpc.py > gpc_gates.log 2>&1 && JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 $PY -W ignore test_pred_gpc.py >> gpc_gates.log 2>&1 \
  || { note "GATES FAILED"; tail -30 gpc_gates.log; exit 4; }
note "gates: $(grep -c PASS gpc_gates.log) PASS"
mkdir -p pred_runs/tests
$PY -W ignore gpc_sweep.py > pred_runs/gpc_sweep.log 2>&1 || { note "sweep failed"; tail -20 pred_runs/gpc_sweep.log; exit 5; }
note "selected: $(tr -d '\n ' < pred_runs/gpc_selected.json)"
run() { out=$1; shift; [ -f pred_runs/tests/$out.json ] && return
  $PY -W ignore run_pred_tests.py gpc "$@" > pred_runs/test_$out.log 2>&1 && note "done $out" || note "FAILED $out"; }
for s in 0 1 2; do for st in zero amort; do for d in 3 10; do for r in pc bp; do
  run gpc_${r}_d${d}_${st}_bp_s$s --rule $r --depth $d --start $st --backbone bp --seed $s
done; done; done; done
for d in 3 10; do for r in pc bp; do run gpc_${r}_d${d}_zero_rand_s0 --rule $r --depth $d --start zero --backbone rand --seed 0; done; done
note "ALL DONE"
