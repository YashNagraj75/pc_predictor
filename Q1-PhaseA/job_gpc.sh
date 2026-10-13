#!/usr/bin/env bash
# Generative PC predictor, stage 1 (seed 0), spec rev 2.  bash job_gpc.sh <project_root> <job_dir>   (resumable)
set -uo pipefail
P="$1"; JOBDIR="$2"; cd "$P/Q1-PhaseA" || exit 2
PY=../.venv/bin/python; export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.3
PAR=${PAR:-3}; ST=gpc_status.txt; note() { echo "$(date +%m-%d_%H:%M) $*" | tee -a $ST; }
collect() { tar czf "$JOBDIR/q1_gpc.tar.gz" $ST gpc_gates*.log gpc_prereg.json pred_runs/gpc_sweep pred_runs/gpc_selected.json \
            pred_runs/tests/gpc_*.json pred_runs/tests_smoke pred_runs/logs_gpc 2>/dev/null; }
trap collect EXIT
[ -f pred_latents.npz ] || { note "pred_latents.npz missing"; exit 3; }
python3 -c "import numpy as np; d=np.load('pred_latents.npz'); assert str(d['backbone_run'])=='runs/bp_sigreg_etafix_s0'" || { note "wrong backbone"; exit 3; }
BK=$($PY -c 'import jax;print(jax.default_backend())' 2>/dev/null | tail -1); note "backend $BK"; [ "$BK" = gpu ] || { note "no GPU backend"; exit 3; }
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader | tee -a $ST
# gates: CPU float32, GPU float32 (TF32 off via highest precision, review N1), CPU float64
JAX_PLATFORMS=cpu $PY -W ignore test_pred_gpc.py > gpc_gates_cpu.log 2>&1 && $PY -W ignore test_pred_gpc.py > gpc_gates_gpu.log 2>&1 \
  && JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 $PY -W ignore test_pred_gpc.py > gpc_gates_f64.log 2>&1 \
  || { note "GATES FAILED"; grep -h FAIL gpc_gates_*.log | head; exit 4; }
note "gates pass: $(cat gpc_gates_*.log | grep -c PASS)"
mkdir -p pred_runs/logs_gpc pred_runs/tests
# pre-flight: every code path once, 30 steps, throwaway selection
[ -f pred_runs/tests_smoke/.ok ] || {
  echo '{}' | python3 -c "import json;print(json.dumps({f'gpc_{r}_d{d}_{s}':{'c':1.0} for r in ('pc','bp') for d in (3,10) for s in ('zero','amort')}))" > pred_runs/smoke_sel.json
  for args in "--rule pc --depth 10 --start amort" "--rule bp --depth 3 --start zero --hid-prec 1"; do
    $PY -W ignore run_gpc_tests.py $args --steps 30 --gsel pred_runs/smoke_sel.json --outdir pred_runs/tests_smoke > pred_runs/logs_gpc/smoke.log 2>&1 \
      || { note "PREFLIGHT FAILED ($args)"; tail -25 pred_runs/logs_gpc/smoke.log; exit 5; }
  done; touch pred_runs/tests_smoke/.ok; note "preflight ok"; }
# step-scale selection, PAR arms at a time
ARMS="gpc_pc_d3_zero gpc_bp_d3_zero gpc_pc_d10_zero gpc_bp_d10_zero gpc_pc_d3_amort gpc_bp_d3_amort gpc_pc_d10_amort gpc_bp_d10_amort"
echo $ARMS | tr ' ' '\n' | xargs -P $PAR -I{} sh -c "$PY -W ignore gpc_sweep.py --arm {} > pred_runs/logs_gpc/sweep_{}.log 2>&1 || echo 'sweep {} FAILED' >> $ST"
$PY -W ignore gpc_sweep.py --collect > pred_runs/logs_gpc/sweep_collect.log 2>&1; note "selected: $(python3 -c "import json;print({k:(v['c'],v.get('edge')) for k,v in json.load(open('pred_runs/gpc_selected.json')).items()})")"
# seed 0: 8 main arms, width control (d3 at width 256), precision sensitivity (d10 PC, hidden precision 9), random-backbone controls
{ for st in zero amort; do for d in 3 10; do for r in pc bp; do echo "--rule $r --depth $d --start $st --backbone bp"; done; done; done
  echo "--rule pc --depth 3 --start zero --backbone bp --width 256"; echo "--rule bp --depth 3 --start zero --backbone bp --width 256"
  echo "--rule pc --depth 10 --start zero --backbone bp --hid-prec 9"
  for d in 3 10; do for r in pc bp; do echo "--rule $r --depth $d --start zero --backbone rand"; done; done; } > gpc_stage1_arms.txt
run_one() { a="$1"; tag=$(echo "$a" | tr -d '-' | tr ' ' '_')
  $PY -W ignore run_gpc_tests.py $a --seed 0 > "pred_runs/logs_gpc/s0_$tag.log" 2>&1; rc=$?
  echo "$(date +%m-%d_%H:%M) rc=$rc $a" >> gpc_status.txt; }
export -f run_one; export PY
cat gpc_stage1_arms.txt | xargs -P $PAR -I{} bash -c 'run_one "{}"'
note "STAGE 1 DONE: $(ls pred_runs/tests/gpc_*_s0.json 2>/dev/null | wc -l) results, $(ls pred_runs/tests/gpc_*.diverged.json 2>/dev/null | wc -l) diverged"
