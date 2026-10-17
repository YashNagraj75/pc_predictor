#!/usr/bin/env bash
# C2 amendment (user-approved 2026-10-17, post hoc): depth-10 PC-ALM rho=4 at c=0.5, seeds 0,1,2.
# The pre-registered c=1 arm oscillated at depth 10 (seed 0 test nmse 0.987). Everything else is as in job_gpc_alm.sh.
# Waits for the main C2 job to finish so both never share the GPU; runs from the host checkout (a8402d3) unchanged.
#   bash job_c2_amend_c05.sh <project_root> <job_dir>
set -uo pipefail
P="$1"; JOBDIR=$(realpath "$2"); cd "$P/Q1-PhaseA" || exit 2
PY=../.venv/bin/python; export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.3
ST=gpc_alm_amend_status.txt; note() { echo "$(date +%m-%d_%H:%M) $*" | tee -a $ST; }
OUT=pred_runs/tests_amend_c05; SEL=pred_runs/gpc_alm_selected_amend_c05.json; LOGD=pred_runs/logs_gpc_alm_amend
collect() { tar czf "$JOBDIR/q1_gpc_alm_amend.tar.gz" $ST $SEL $OUT $LOGD 2>/dev/null || true; }
trap collect EXIT
note "waiting for the main C2 job (ALL DONE in gpc_alm_status.txt)"
until grep -q "ALL DONE" gpc_alm_status.txt 2>/dev/null; do sleep 300; done
while pgrep -f 'pytho[n].*run_gpc_tests[.]py' > /dev/null; do sleep 60; done
note "main job done; start at $(git rev-parse --short HEAD)"
$PY -c "
import json; s = json.load(open('pred_runs/gpc_alm_selected.json'))
s['gpc_alm_d10_zero']['best_c_by_rho']['4.0'] = 0.5
s['gpc_alm_d10_zero']['amendment'] = 'c for rho=4 set to 0.5 post hoc (2026-10-17): c=1 oscillated at depth 10'
json.dump(s, open('$SEL', 'w'), indent=1)" || { note "selection copy failed"; exit 3; }
mkdir -p $OUT $LOGD
run_one() { s="$1"; CK=$([ "$s" = 0 ] && echo --ckpt || true)
  $PY -W ignore run_gpc_tests.py --rule alm --depth 10 --start zero --backbone bp --seed $s --rho 4.0 --alpha 1 --T2 100 \
      --gsel pred_runs/gpc_alm_selected_amend_c05.json --outdir pred_runs/tests_amend_c05 $CK > pred_runs/logs_gpc_alm_amend/s$s.log 2>&1
  echo "$(date +%m-%d_%H:%M) rc=$? amend d10 rho4 c0.5 s$s" >> gpc_alm_amend_status.txt; }
export -f run_one; export PY
( while sleep 1200; do collect; done ) & CPID=$!
printf "0\n1\n2\n" | xargs -P 3 -I{} bash -c 'run_one {}'
kill $CPID 2>/dev/null || true
note "AMEND DONE: $(ls $OUT/*.json 2>/dev/null | wc -l) results"
