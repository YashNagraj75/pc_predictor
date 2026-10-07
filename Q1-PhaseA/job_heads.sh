#!/usr/bin/env bash
# Case D: sparse side heads, K = 1..5 nested heads outward from layer 5, all
# multipliers kept.  Spec: vault PCWM/Methods/'Method - Sparse side-head arm plan'.
#   gates -> selection -> pilot pair per K + gate -> full runs (staged seeds)
#   -> untrained floor -> probes -> head analysis -> learning diagnostic -> collect.
#
#   bash job_heads.sh <project_root> <job_dir>
#
# Seeds (user decision 2026-10-07): K=1 and K=5 at 3 seeds, K=2,3,4 at 1 seed.
# Full runs go K=1, K=5, then K=2,3,4, so the decisive ends land first.
# A pair failing its pilot gate has its FULL runs skipped and is reported; the
# other K still run.  Nothing is re-tuned after a pilot.
set -uo pipefail
P="$1"; JOBDIR="$2"
cd "$P/Q1-PhaseA"
PY=../.venv/bin/python
export JAX_PLATFORMS=cuda
CLEAN='cuda_|plugin|xla_bridge|versions_helpers|cuStreamGetGreenCtx|decoder step|^\s*File |^\s*~|^\s*\^'
STEPS=${STEPS:-17200}; PILOT=${PILOT:-2000}
OP="--T 16 --eta-h 0.0625 --alpha 1"
REUSED="runs/pc_sigreg_T16_coupled_alm_s0 runs/pc_sigreg_T16_coupled_alm_s1 runs/pc_sigreg_T16_coupled_alm_s2 runs/bp_sigreg_etafix_s0 runs/bp_sigreg_etafix_s1 runs/bp_sigreg_etafix_s2"
STATUS=heads_status.txt; : > $STATUS
note() { echo "$*" | tee -a $STATUS; }
seeds_for() { case $1 in 1|5) echo "0 1 2";; *) echo "0";; esac; }

collect() {
  tar czf "$JOBDIR/q1_heads.tar.gz" heads_status.txt heads_selected.json \
      gates_heads.log select_heads.log train_heads*.log gate_D*.json \
      probes_heads.log probes_heads_summary.csv diag_heads.csv diag_heads.log \
      diag_learning_heads.csv diag_learning_heads.log \
      runs/*hK*/DIVERGED runs/*hK*/config.json runs/*hK*/log.json runs/*hK*/probes.json \
      runs/*randD*/config.json runs/*randD*/probes.json 2>/dev/null
  ls -lh "$JOBDIR/q1_heads.tar.gz"
}
trap collect EXIT

echo "##### 0. backend"
$PY -c "import jax;print('backend', jax.default_backend(), jax.devices())" 2>&1 | tail -1
for r in $REUSED; do [ -f "$r/params.eqx" ] || note "WARNING: reused run $r missing"; done

echo "##### 1. gates (CPU: the golden reference is regenerated and checked bit for bit)"
{ JAX_PLATFORMS=cpu $PY -W ignore test_pcalm.py && JAX_PLATFORMS=cpu $PY -W ignore test_sparse.py; } > gates_heads.log 2>&1 \
  || { grep FAIL gates_heads.log; note "GATES FAILED"; exit 3; }
grep -cE "\[PASS\]" gates_heads.log | xargs -I{} note "gates: {} PASS, 0 FAIL"

echo "##### 2. selection (training images only, at initialisation)"
$PY -W ignore select_heads.py --out heads_selected.json > select_heads.log 2>&1 \
  || { tail -5 select_heads.log; note "SELECTION FAILED"; exit 4; }
grep -E "^head" select_heads.log | tee -a $STATUS

echo "##### 3. pilot pair per K + gate"
PASSED=""
for K in 1 2 3 4 5; do
  $PY -W ignore run_arms.py --steps $PILOT --eval-every 250 --arms pc bp --seeds 0 $OP \
      --sparse-case D --heads-json heads_selected.json --K $K --tag hK${K}pilot > train_heads_pilot_K$K.log 2>&1
  if $PY gate_pilot.py runs/pc_sigreg_T16_coupled_hK${K}pilot_s0 runs/bp_sigreg_hK${K}pilot_s0 > gate_D$K.json 2>/dev/null; then
    PASSED="$PASSED $K"; note "pilot K=$K: PASS"
  else
    note "pilot K=$K: FAIL -- full runs for K=$K SKIPPED"
  fi
  $PY -c "import json;d=json.load(open('gate_D$K.json'));print('  reasons:',d['reasons'])" | tee -a $STATUS
done
[ -n "$PASSED" ] || { note "no K passed its pilot"; exit 5; }

echo "##### 4. full runs (order K=1, 5, 2, 3, 4)"
NEW=""
for K in 1 5 2 3 4; do
  case " $PASSED " in *" $K "*) ;; *) continue;; esac
  S=$(seeds_for $K)
  $PY -W ignore run_arms.py --steps $STEPS --eval-every 2000 --arms pc bp --seeds $S $OP \
      --sparse-case D --heads-json heads_selected.json --K $K --tag hK$K >> train_heads_K$K.log 2>&1 \
      || note "run_arms K=$K exited nonzero"
  for s in $S; do NEW="$NEW runs/pc_sigreg_T16_coupled_hK${K}_s$s runs/bp_sigreg_hK${K}_s$s"; done
  grep -iE "done in|DIVERG|REFUSING" train_heads_K$K.log | tail -6 | tee -a $STATUS
done

echo "##### 5. untrained floor (original encoder at init + the K=5 heads at init)"
$PY -W ignore run_arms.py --init-only --arms pc --seeds 0 1 2 $OP --sparse-case D \
    --heads-json heads_selected.json --K 5 --tag randD > train_heads_floor.log 2>&1
FLOORS="runs/pc_sigreg_T16_coupled_randD_s0 runs/pc_sigreg_T16_coupled_randD_s1 runs/pc_sigreg_T16_coupled_randD_s2"

echo "##### 6. probe suite (embedding)"
$PY -W ignore run_probes.py $NEW $FLOORS > probes_heads.log 2>&1 || note "run_probes exited nonzero"
cp probes_summary.csv probes_heads_summary.csv 2>/dev/null || true
grep -viE "$CLEAN" probes_heads.log | grep -iE "===|class probe|knn" | tail -60

echo "##### 7. head analysis (codes, robustness, spread); reused arms get init heads attached"
$PY -W ignore diag_heads.py $NEW $FLOORS $REUSED --heads-json heads_selected.json \
    --csv diag_heads.csv > diag_heads.log 2>&1 || note "diag_heads exited nonzero"
grep -viE "$CLEAN" diag_heads.log | grep -E "===|_linear|wrote" | tail -80

echo "##### 8. learning diagnostic"
$PY -W ignore diag_learning.py $NEW --csv diag_learning_heads.csv > diag_learning_heads.log 2>&1 \
  || note "diag_learning exited nonzero"
note "DONE"
