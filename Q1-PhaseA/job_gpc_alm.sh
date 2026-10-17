#!/usr/bin/env bash
# C2 (rev 2, after the 3-reviewer pass): PC-ALM generative predictor at depths 3 and 10.
# Spec: wiki 'Method - PC-ALM generative predictor (C2)'; prereg gpc_alm_prereg.json.
#   gates -> credit/stability diagnostic -> rho_hi and T2 by the pre-registered rule -> c sweep per rho
#   -> determinism check of the stage-1 controls -> 26 runs (3 seeds each arm; controls seeds 1, 2).
#   bash job_gpc_alm.sh <project_root> <job_dir>        (resumable: finished results are skipped)
set -uo pipefail
P="$1"; JOBDIR=$(realpath "$2"); cd "$P/Q1-PhaseA" || exit 2
PY=../.venv/bin/python; export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.3
PAR=${PAR:-3}; ST=gpc_alm_status.txt; note() { echo "$(date +%m-%d_%H:%M) $*" | tee -a $ST; }
SEL=pred_runs/gpc_alm_selected.json; CH=pred_runs/c2_choice.json; DG=pred_runs/c2_diag
collect() { tar czf "$JOBDIR/q1_gpc_alm.tar.gz" $ST gpc_alm_gates*.log gpc_alm_prereg.json gpc_alm_arms.txt $DG $CH $SEL \
            pred_runs/gpc_alm_sweep pred_runs/c2_det_check.json pred_runs/gpc_selected.json \
            pred_runs/tests/gpc_alm_*.json pred_runs/tests/gpc_pc_d*_zero_bp_s[12]*.json pred_runs/tests/gpc_bp_d*_zero_bp_s[12]*.json \
            pred_runs/logs_gpc_alm 2>/dev/null || true; }
trap collect EXIT
[ -f pred_latents.npz ] || { note "pred_latents.npz missing"; exit 3; }
$PY -c "import numpy as np; d=np.load('pred_latents.npz'); assert str(d['backbone_run'])=='runs/bp_sigreg_etafix_s0'" 2>/dev/null || { note "wrong backbone"; exit 3; }
[ -f pred_runs/gpc_selected.json ] || { note "stage-1 selection missing (controls need it)"; exit 3; }
BK=$($PY -c 'import jax;print(jax.default_backend())' 2>/dev/null | tail -1); note "start at $(git rev-parse --short HEAD) backend $BK"
[ "$BK" = gpu ] || { note "no GPU backend"; exit 3; }
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader | tee -a $ST || true
# stale outputs from smoke tests / earlier attempts are moved aside once, never reused
if [ ! -f .c2_started ]; then
  STAMP=$(date +%Y%m%d_%H%M); mkdir -p pred_runs/_c2_prev_$STAMP
  for f in pred_runs/gpc_alm_sweep pred_runs/gpc_alm_selected.json pred_runs/smoke_alm pred_runs/diag_c2_run pred_runs/tests/gpc_alm_*; do
    [ -e "$f" ] && mv "$f" pred_runs/_c2_prev_$STAMP/; done
  touch .c2_started
fi
mkdir -p pred_runs/logs_gpc_alm pred_runs/tests $DG
# 0. gates: CPU f32, GPU f32, CPU f64
if [ ! -f $DG/.gates_ok ]; then
  JAX_PLATFORMS=cpu $PY -W ignore test_pred_gpc.py > gpc_alm_gates_cpu.log 2>&1 && $PY -W ignore test_pred_gpc.py > gpc_alm_gates_gpu.log 2>&1 \
    && JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 $PY -W ignore test_pred_gpc.py > gpc_alm_gates_f64.log 2>&1 \
    || { note "GATES FAILED"; grep -h FAIL gpc_alm_gates_*.log | head; exit 4; }
  touch $DG/.gates_ok; note "gates pass: $(cat gpc_alm_gates_*.log | grep -c PASS)"
fi
# 1. credit / stability diagnostic at the training configuration, then the pre-registered rho_hi / T2 rule
[ -f $DG/diag_c2.json ] && [ -f $DG/.done ] || { $PY -W ignore diag_c2.py $DG/diag_c2.json > $DG/diag.log 2>&1 && touch $DG/.done || { note "DIAG FAILED"; tail -5 $DG/diag.log; exit 5; }; }
$PY choose_rho.py $DG/diag_c2.json $CH > $DG/choose.log 2>&1 || { note "CHOICE FAILED"; cat $DG/choose.log; exit 5; }
note "choice: $(cat $DG/choose.log)"
get() { $PY -c "import json;print(json.load(open('$CH'))['$1']['$2'])"; }
# 2. c sweep per rho (rho = 1 and rho_hi), at the chosen T2, 800 of 4000 steps, 90/10 split, seed 0
for d in 3 10; do
  RH=$(get $d rho_hi); T2=$(get $d T2)
  $PY -W ignore gpc_sweep.py --arm gpc_alm_d${d}_zero --rhos 1.0,$RH --T2 $T2 --dir pred_runs/gpc_alm_sweep > pred_runs/logs_gpc_alm/sweep_d$d.log 2>&1 &
done; wait
$PY -W ignore gpc_sweep.py --collect --dir pred_runs/gpc_alm_sweep --out $SEL > pred_runs/logs_gpc_alm/collect.log 2>&1 \
  || { note "SELECTION FAILED"; tail -5 pred_runs/logs_gpc_alm/collect.log; exit 6; }
note "selected c by rho: $($PY -c "import json;print({k:v['best_c_by_rho'] for k,v in json.load(open('$SEL')).items()})")"
# 3. determinism check of the reused stage-1 seed-0 controls (review D5)
[ -f pred_runs/c2_det_check.json ] || $PY -W ignore det_check.py > pred_runs/logs_gpc_alm/det_check.log 2>&1 || note "det_check exited nonzero"
note "det check: $($PY -c "import json;print({k:v['match'] for k,v in json.load(open('pred_runs/c2_det_check.json')).items()})" 2>/dev/null)"
# 4. full runs; seed 0 first so the decisive comparisons land early
{ for s in 0 1 2; do for d in 3 10; do
    RH=$(get $d rho_hi); T2=$(get $d T2); CK=$([ $s = 0 ] && echo --ckpt || true)
    echo "--rule alm --depth $d --start zero --backbone bp --seed $s --rho $RH --alpha 1 --T2 $T2 --gsel $SEL $CK"
    echo "--rule alm --depth $d --start zero --backbone bp --seed $s --rho 1.0 --alpha 1 --T2 $T2 --gsel $SEL $CK"
    echo "--rule alm --depth $d --start zero --backbone bp --seed $s --rho $RH --alpha 0 --T2 $T2 --gsel $SEL"
    [ $s = 0 ] || { echo "--rule bp --depth $d --start zero --backbone bp --seed $s"; echo "--rule pc --depth $d --start zero --backbone bp --seed $s"; }
  done; done; } > gpc_alm_arms.txt
note "arms: $(wc -l < gpc_alm_arms.txt)"
run_one() { a="$1"; tag=$(echo "$a" | sed 's/--gsel [^ ]*//' | tr -d '-' | tr ' ' '_')
  $PY -W ignore run_gpc_tests.py $a > "pred_runs/logs_gpc_alm/$tag.log" 2>&1; rc=$?
  echo "$(date +%m-%d_%H:%M) rc=$rc $a" >> gpc_alm_status.txt; }
export -f run_one; export PY
( while sleep 1200; do collect; done ) & CPID=$!
cat gpc_alm_arms.txt | xargs -P $PAR -I{} bash -c 'run_one "{}"'
kill $CPID 2>/dev/null || true
NF=$(grep -c "rc=[^0]" gpc_alm_status.txt || true)
note "ALL DONE: $(ls pred_runs/tests/gpc_alm_*_s?.json 2>/dev/null | wc -l) alm results; nonzero-rc lines in status: $NF"
