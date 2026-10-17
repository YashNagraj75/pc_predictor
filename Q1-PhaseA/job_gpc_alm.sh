#!/usr/bin/env bash
# C2: PC-ALM generative predictor at depths 3 and 10, plus the 3-seed controls it is compared with.
# Spec: wiki 'Method - PC-ALM generative predictor (C2)'; prereg gpc_alm_prereg.json.  Resumable.
#   bash job_gpc_alm.sh <project_root> <job_dir>
set -uo pipefail
P="$1"; JOBDIR=$(realpath "$2"); cd "$P/Q1-PhaseA" || exit 2
PY=../.venv/bin/python; export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.3
PAR=${PAR:-3}; ST=gpc_alm_status.txt; note() { echo "$(date +%m-%d_%H:%M) $*" | tee -a $ST; }
SEL=pred_runs/gpc_alm_selected.json
collect() { tar czf "$JOBDIR/q1_gpc_alm.tar.gz" $ST gpc_alm_gates*.log gpc_alm_prereg.json pred_runs/gpc_alm_sweep $SEL \
            pred_runs/tests/gpc_alm_*.json pred_runs/tests/gpc_pc_d*_zero_bp_s[012]*.json pred_runs/tests/gpc_bp_d*_zero_bp_s[012]*.json \
            pred_runs/gpc_selected.json gpc_alm_arms.txt pred_runs/c2_s0_rerun \
            pred_runs/logs_gpc_alm 2>/dev/null || true; }
trap collect EXIT
[ -f pred_latents.npz ] || { note "pred_latents.npz missing"; exit 3; }
$PY -c "import numpy as np; d=np.load('pred_latents.npz'); assert str(d['backbone_run'])=='runs/bp_sigreg_etafix_s0'" || { note "wrong backbone"; exit 3; }
[ -f pred_runs/gpc_selected.json ] || { note "stage-1 selection missing (controls need it)"; exit 3; }
BK=$($PY -c 'import jax;print(jax.default_backend())' 2>/dev/null | tail -1); note "start at $(git rev-parse --short HEAD) backend $BK"
[ "$BK" = gpu ] || { note "no GPU backend"; exit 3; }
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader | tee -a $ST || true
JAX_PLATFORMS=cpu $PY -W ignore test_pred_gpc.py > gpc_alm_gates_cpu.log 2>&1 && $PY -W ignore test_pred_gpc.py > gpc_alm_gates_gpu.log 2>&1 \
  && JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 $PY -W ignore test_pred_gpc.py > gpc_alm_gates_f64.log 2>&1 \
  || { note "GATES FAILED"; grep -h FAIL gpc_alm_gates_*.log | head; exit 4; }
note "gates pass: $(cat gpc_alm_gates_*.log | grep -c PASS)"
mkdir -p pred_runs/logs_gpc_alm pred_runs/tests
# 1. selection: rho x c on a 90/10 train split, 800 of 4000 scheduled steps, seed 0, per depth
for d in 3 10; do echo gpc_alm_d${d}_zero; done | xargs -P 2 -I{} sh -c \
  "$PY -W ignore gpc_sweep.py --arm {} --dir pred_runs/gpc_alm_sweep > pred_runs/logs_gpc_alm/sweep_{}.log 2>&1 || echo 'sweep {} FAILED' >> $ST"
$PY -W ignore gpc_sweep.py --collect --dir pred_runs/gpc_alm_sweep --out $SEL > pred_runs/logs_gpc_alm/collect.log 2>&1 \
  || { note "SELECTION FAILED"; exit 6; }
note "selected: $(python3 -c "import json;print({k:(v.get('rho'),v['c'],v.get('edge_c'),v.get('edge_rho')) for k,v in json.load(open('$SEL')).items()})")"
# 2. full runs: selected ALM x 3 seeds per depth; rho = 1 ALM (seed 0) if not selected; BP and plain PC seeds 1, 2
{ for s in 0 1 2; do for d in 3 10; do echo "--rule alm --depth $d --start zero --backbone bp --seed $s --gsel $SEL"; done; done
  for d in 3 10; do
    python3 -c "import json,sys;sys.exit(0 if json.load(open('$SEL'))['gpc_alm_d${d}_zero']['rho']!=1.0 else 1)" \
      && echo "--rule alm --depth $d --start zero --backbone bp --seed 0 --rho 1 --gsel $SEL"; done
  for s in 1 2; do for d in 3 10; do for r in bp pc; do echo "--rule $r --depth $d --start zero --backbone bp --seed $s"; done; done; done
} > gpc_alm_arms.txt
note "arms: $(wc -l < gpc_alm_arms.txt)"
run_one() { a="$1"; tag=$(echo "$a" | sed 's/--gsel [^ ]*//' | tr -d '-' | tr ' ' '_')
  $PY -W ignore run_gpc_tests.py $a > "pred_runs/logs_gpc_alm/$tag.log" 2>&1; rc=$?
  echo "$(date +%m-%d_%H:%M) rc=$rc $a" >> gpc_alm_status.txt; }
export -f run_one; export PY
cat gpc_alm_arms.txt | xargs -P $PAR -I{} bash -c 'run_one "{}"'
note "ALL DONE: $(ls pred_runs/tests/gpc_alm_*_s?.json 2>/dev/null | wc -l) alm results, $(ls pred_runs/tests/gpc_*.diverged.json 2>/dev/null | wc -l) diverged files"
