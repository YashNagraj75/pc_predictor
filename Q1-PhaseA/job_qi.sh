#!/usr/bin/env bash
# Qi et al. 2025 fixes trained into the generative PC predictor (seed 0 screen), depths 3 and 10.
# Spec: wiki 'Method - Qi fixes training screen'; prereg qi_prereg.json.  Resumable (finished arms are skipped).
#   bash job_qi.sh <project_root> <job_dir>
set -uo pipefail
P="$1"; JOBDIR="$2"; cd "$P/Q1-PhaseA" || exit 2
PY=../.venv/bin/python; export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.3
PAR=${PAR:-3}; ST=qi_status.txt; note() { echo "$(date +%m-%d_%H:%M) $*" | tee -a $ST; }
collect() { tar czf "$JOBDIR/q1_qi.tar.gz" $ST qi_gates_*.log qi_prereg.json qi_arms.txt pred_runs/logs_qi \
            pred_runs/tests/gpc_pc_d*_zero*_spk*.json pred_runs/tests/gpc_pc_d*_zero*_fu*.json \
            pred_runs/tests/gpc_pc_d*_zero_c01*.json 2>/dev/null || true; }
trap collect EXIT
mkdir -p pred_runs/logs_qi
note "start at $(git rev-parse --short HEAD)"
JAX_PLATFORMS=cpu $PY -W ignore test_qi.py > qi_gates_cpu.log 2>&1 && $PY -W ignore test_qi.py > qi_gates_gpu.log 2>&1 \
  || { note "GATES FAILED"; grep -h FAIL qi_gates_*.log | head; exit 4; }
note "gates pass: $(cat qi_gates_*.log | grep -c PASS)"
{ for d in 10 3; do
    echo "--rule pc --depth $d --start zero --backbone bp --seed 0 --ckpt --spike lit1"
    echo "--rule pc --depth $d --start zero --backbone bp --seed 0 --ckpt --spike stable --c 0.1 --c-spike 2.0"
    echo "--rule pc --depth $d --start zero --backbone bp --seed 0 --ckpt --c 0.1 --tag _c01"
    echo "--rule pc --depth $d --start zero --backbone bp --seed 0 --ckpt --wrule qi"
    echo "--rule pc --depth $d --start zero --backbone bp --seed 0 --ckpt --wrule fixed"
    echo "--rule pc --depth $d --start zero --backbone bp --seed 0 --ckpt --spike stable --c 0.1 --c-spike 2.0 --wrule fixed"
  done; } > qi_arms.txt
note "arms: $(wc -l < qi_arms.txt)"
run_one() { a="$1"; tag=$(echo "$a" | tr -d '-' | tr ' ' '_' | cut -c1-120)
  $PY -W ignore run_gpc_tests.py $a > "pred_runs/logs_qi/$tag.log" 2>&1; rc=$?
  echo "$(date +%m-%d_%H:%M) rc=$rc $a" >> qi_status.txt; }
export -f run_one; export PY
( while sleep 1200; do collect; done ) & CPID=$!
cat qi_arms.txt | xargs -P $PAR -I{} bash -c 'run_one "{}"'
kill $CPID 2>/dev/null || true
note "ALL DONE: $(grep -c 'rc=0' $ST) ok, $(grep -c 'rc=3' $ST) diverged, $(grep -cE 'rc=[^03]' $ST) other"
