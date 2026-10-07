#!/usr/bin/env bash
# Sparse PC-ALM arms (cases B and C): gates -> selection -> pilots+gate -> full runs
# -> untrained floors -> probes -> inference analysis -> learning diagnostic -> collect.
# Spec: vault PCWM/Methods/'Method - Sparse PC-ALM arms spec'.
#
#   bash job_sparse.sh <project_root> <job_dir>
#
# Run from anywhere: everything happens in <project_root>/Q1-PhaseA and the
# results tarball is written to <job_dir> (captured BEFORE any cd -- $OLDPWD is
# unset in this host's job shell).
set -uo pipefail
P="$1"; JOBDIR="$2"
cd "$P/Q1-PhaseA"
PY=../.venv/bin/python
export JAX_PLATFORMS=cuda
CLEAN='cuda_|plugin|xla_bridge|versions_helpers|cuStreamGetGreenCtx|decoder step|^\s*File |^\s*~|^\s*\^'
STEPS=${STEPS:-17200}; PILOT=${PILOT:-2000}; SEEDS=${SEEDS:-"0 1 2"}
COMMON="--T 16 --eta-h 0.0625 --alpha 1 --eval-every 2000"
REUSED="runs/pc_sigreg_T16_coupled_alm_s0 runs/pc_sigreg_T16_coupled_alm_s1 runs/pc_sigreg_T16_coupled_alm_s2 runs/bp_sigreg_etafix_s0 runs/bp_sigreg_etafix_s1 runs/bp_sigreg_etafix_s2"
STATUS=sparse_status.txt; : > $STATUS
note() { echo "$*" | tee -a $STATUS; }

collect() {
  tar czf "$JOBDIR/q1_sparse.tar.gz" sparse_status.txt sparse_selected.json \
      gates_sparse.log select_sparse.log train_sparse*.log gate_*.json \
      probes_sparse.log probes_sparse_summary.csv diag_sparse.csv diag_sparse.log \
      diag_learning_sparse.csv diag_learning_sparse.log \
      runs/*/DIVERGED \
      runs/*spB*/config.json runs/*spB*/log.json runs/*spB*/probes.json \
      runs/*spC*/config.json runs/*spC*/log.json runs/*spC*/probes.json \
      runs/*randB*/config.json runs/*randB*/probes.json runs/*randC*/config.json runs/*randC*/probes.json \
      2>/dev/null
  ls -lh "$JOBDIR/q1_sparse.tar.gz"
}
trap collect EXIT          # results come back even if a later stage dies

echo "##### 0. backend"
$PY -c "import jax;print('backend', jax.default_backend(), jax.devices())" 2>&1 | tail -1

echo "##### 1. gates"
{ $PY -W ignore test_pcalm.py && $PY -W ignore test_sparse.py; } > gates_sparse.log 2>&1 \
  || { grep -E "PASS|FAIL" gates_sparse.log | grep FAIL; note "GATES FAILED"; exit 3; }
grep -cE "\[PASS\]" gates_sparse.log | xargs -I{} note "gates: {} PASS, 0 FAIL"

echo "##### 2. selection (rule D5, training images only)"
$PY -W ignore select_sparse.py --out sparse_selected.json > select_sparse.log 2>&1 \
  || { tail -5 select_sparse.log; note "SELECTION FAILED"; exit 4; }
BETA=$($PY -c "import json;print(json.load(open('sparse_selected.json'))['beta'])")
GAMMA=$($PY -c "import json;print(json.load(open('sparse_selected.json'))['gamma'])")
note "selected beta=$BETA gamma=$GAMMA"

echo "##### 3. pilots + gate (BOTH arms of each pair; a failing pair is skipped, never re-tuned)"
CASES=""
for C in B C; do
  if [ $C = B ]; then HP="--beta $BETA"; else HP="--gamma $GAMMA"; fi
  $PY -W ignore run_arms.py --steps $PILOT --arms pc bp --seeds 0 --T 16 --eta-h 0.0625 --alpha 1 \
      --eval-every 250 --sparse-case $C $HP --tag sp${C}pilot > train_sparse_pilot_$C.log 2>&1
  if $PY gate_pilot.py runs/pc_sigreg_T16_coupled_sp${C}pilot_s0 runs/bp_sigreg_sp${C}pilot_s0 \
        > gate_$C.json 2>/dev/null; then
    CASES="$CASES $C"; note "pilot pair $C: PASS"
  else
    note "pilot pair $C: FAIL -- case $C SKIPPED"
  fi
  $PY -c "import json;d=json.load(open('gate_$C.json'));print('  reasons:',d['reasons']);print('  pc:',{k:v for k,v in d['pc'].items() if k!='run'});print('  bp:',{k:v for k,v in d['bp'].items() if k!='run'})" | tee -a $STATUS
done
[ -n "$CASES" ] || { note "no case passed its pilot"; exit 5; }

echo "##### 4. full runs:$CASES  (PC arm + BP control, seeds $SEEDS)"
NEW=""
for C in $CASES; do
  if [ $C = B ]; then HP="--beta $BETA"; else HP="--gamma $GAMMA"; fi
  $PY -W ignore run_arms.py --steps $STEPS --arms pc bp --seeds $SEEDS $COMMON \
      --sparse-case $C $HP --tag sp$C >> train_sparse_$C.log 2>&1 || note "train $C exited nonzero"
  grep -E "done in|REFUSING" train_sparse_$C.log | tail -8
  for s in $SEEDS; do NEW="$NEW runs/pc_sigreg_T16_coupled_sp${C}_s$s runs/bp_sigreg_sp${C}_s$s"; done
done

echo "##### 5. untrained floors (same architecture, same beta/gamma)"
FLOORS=""
for C in $CASES; do
  if [ $C = B ]; then HP="--beta $BETA"; else HP="--gamma $GAMMA"; fi
  $PY -W ignore run_arms.py --init-only --arms pc --seeds $SEEDS --T 16 --eta-h 0.0625 --alpha 1 \
      --sparse-case $C $HP \
      --tag rand$C >> train_sparse_floors.log 2>&1
  for s in $SEEDS; do FLOORS="$FLOORS runs/pc_sigreg_T16_coupled_rand${C}_s$s"; done
done

echo "##### 6. probe suite"
$PY -W ignore run_probes.py $NEW $FLOORS > probes_sparse.log 2>&1 || note "run_probes exited nonzero"
cp probes_summary.csv probes_sparse_summary.csv 2>/dev/null || true
grep -viE "$CLEAN" probes_sparse.log | grep -iE "===|class probe|knn|decode  ssim" | tail -60

echo "##### 7. inference analysis (groups A-D; reused arms get group C as references)"
$PY -W ignore diag_sparse.py $NEW $FLOORS $REUSED --csv diag_sparse.csv > diag_sparse.log 2>&1 \
  || note "diag_sparse exited nonzero"
grep -viE "$CLEAN" diag_sparse.log | grep -E "===|delta_inf_linear|wrote" | tail -60

echo "##### 8. learning diagnostic"
$PY -W ignore diag_learning.py $NEW --csv diag_learning_sparse.csv > diag_learning_sparse.log 2>&1 \
  || note "diag_learning exited nonzero"
grep -viE "$CLEAN" diag_learning_sparse.log | grep -iE "===|BEST-FIRST|per-layer" | tail -30
note "DONE"
