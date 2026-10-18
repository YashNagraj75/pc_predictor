#!/usr/bin/env bash
# Qi et al. 2025 fixes trained into the generative PC predictor (seed 0 screen), depths 3 and 10.
# Spec: wiki 'Method - Qi fixes training screen'; prereg qi_prereg.json.  Resumable (finished arms are skipped).
#   bash job_qi.sh <project_root> <job_dir>
set -uo pipefail
P="$1"; JOBDIR="$2"; cd "$P/Q1-PhaseA" || exit 2
PY=../.venv/bin/python; export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.3
PAR=${PAR:-3}; ST=qi_status.txt; OUT=pred_runs/tests_qi; note() { echo "$(date +%m-%d_%H:%M) $*" | tee -a $ST; }
collect() { tar czf "$JOBDIR/q1_qi.tar.gz" $ST qi_gates_*.log qi_prereg.json qi_arms.txt pred_runs/logs_qi \
            $OUT/*.json 2>/dev/null || true; }
trap collect EXIT
mkdir -p pred_runs/logs_qi
note "start at $(git rev-parse --short HEAD)"
free -g | head -2 >> $ST
for d in 3 10; do   # one depth per process: the CPU gate run hit an LLVM out-of-memory with both in one process
  JAX_PLATFORMS=cpu $PY -W ignore test_qi.py $d > qi_gates_cpu_d$d.log 2>&1 && $PY -W ignore test_qi.py $d > qi_gates_gpu_d$d.log 2>&1 \
    || { note "GATES FAILED d$d"; grep -h -E "FAIL|Cannot allocate|Error" qi_gates_*_d$d.log | head; exit 4; }
done
note "gates pass: $(cat qi_gates_*.log | grep -c PASS)"
# smoke of the runner itself, one combined arm, 8 steps (review N3f)
$PY -W ignore run_gpc_tests.py --rule pc --depth 3 --start zero --backbone bp --seed 0 --steps 8 --spike stable --c 0.1 --c-spike 2.0 \
    --wrule fixed --T 3 --T2 3 --tag _smoke --outdir pred_runs/tests_qi_smoke > pred_runs/logs_qi/smoke.log 2>&1 \
  || { note "RUNNER SMOKE FAILED"; tail -5 pred_runs/logs_qi/smoke.log; exit 5; }
note "runner smoke ok"
# arms (review fixes): same-job plain-PC rerun as the primary comparator (D6); c01_fufixed control (D1); Qi-regime arm
# T = T2 = H+1 with stable spike + fixed forward update (D4).  Results go to a per-launch outdir (N4).
{ for d in 10 3; do
    H=$((d - 1)); Q=$((H + 1)); B="--rule pc --depth $d --start zero --backbone bp --seed 0 --ckpt --outdir $OUT"
    echo "$B"
    echo "$B --spike lit1"
    echo "$B --spike stable --c 0.1 --c-spike 2.0"
    echo "$B --c 0.1 --tag _c01"
    echo "$B --wrule qi"
    echo "$B --wrule fixed"
    echo "$B --c 0.1 --wrule fixed --tag _c01"
    echo "$B --spike stable --c 0.1 --c-spike 2.0 --wrule fixed"
    echo "$B --spike stable --c 0.1 --c-spike 2.0 --wrule fixed --T $Q --T2 $Q --tag _qireg"
  done; } > qi_arms.txt
note "arms: $(wc -l < qi_arms.txt)"
run_one() { a="$1"; tag=$(echo "$a" | sed 's/--outdir [^ ]*//' | tr -d '-' | tr ' /' '__' | cut -c1-120)
  $PY -W ignore run_gpc_tests.py $a >> "pred_runs/logs_qi/$tag.log" 2>&1; rc=$?
  echo "$(date +%m-%d_%H:%M) rc=$rc $a" >> qi_status.txt; }
export -f run_one; export PY OUT
( while sleep 1200; do collect; done ) & CPID=$!
cat qi_arms.txt | xargs -P $PAR -I{} bash -c 'run_one "{}"'
kill $CPID 2>/dev/null || true
note "ALL DONE: $(sed -n "/start at $(git rev-parse --short HEAD)/,\$p" $ST | grep -c 'rc=0') ok, $(sed -n "/start at $(git rev-parse --short HEAD)/,\$p" $ST | grep -c 'rc=3') diverged, $(sed -n "/start at $(git rev-parse --short HEAD)/,\$p" $ST | grep -cE 'rc=[^03]') other (this launch)"
