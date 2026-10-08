#!/usr/bin/env bash
# Case D: sparse side heads, K = 1..5 nested heads outward from layer 5, all
# multipliers kept.  Spec: vault PCWM/Methods/'Method - Sparse side-head arm plan'.
#   preflight -> gates -> selection -> pilot pair per K (+ agreement, + gate)
#   -> full runs per seed (K=1, 5 first; analysis of those immediately)
#   -> K=2,3,4 screens (time-guarded) -> floor -> probes -> analysis -> collect.
#
#   bash job_heads.sh <project_root> <job_dir>
#
# User decisions 2026-10-07: K=1 and K=5 at 3 seeds, K=2,3,4 at 1 seed; heads
# explain a normalised copy of the layer; BP-D uses a detached code; K is reported
# as a package contrast.  A pair failing its pilot gate has its full runs skipped
# and is reported; nothing is re-tuned after a pilot.
# Review fixes (2026-10-07): fail fast on pre-existing run dirs (N1); per-seed runs
# and a completed-run check (N5); gate crash != gate FAIL (D13); the gate is
# re-applied, report-only, to every full run (D2); probes_summary.csv is never
# reused stale and the reused arms are probed with the same protocol (N4/D11);
# weights are collected (D12); time guard before the K screens (N5).
set -uo pipefail
P="$1"; JOBDIR="$2"
cd "$P/Q1-PhaseA" || exit 2
PY=../.venv/bin/python
export JAX_PLATFORMS=cuda
CLEAN='cuda_|plugin|xla_bridge|versions_helpers|cuStreamGetGreenCtx|decoder step|^\s*File |^\s*~|^\s*\^'
TAG=${TAG:-hK}; KS=${KS:-"1 2 3 4 5"}; SELECT_ARGS=${SELECT_ARGS:-}
# case E: TAG=hE KS="1 5" SELECT_ARGS=--anchor
STEPS=${STEPS:-17200}; PILOT=${PILOT:-2000}; PILOT_ONLY=${PILOT_ONLY:-0}; REUSE_PILOTS=${REUSE_PILOTS:-0}; BUDGET=${BUDGET:-27000}   # s of training before K screens stop
OP="--T 16 --eta-h 0.0625 --alpha 1"
REUSED="runs/pc_sigreg_T16_coupled_alm_s0 runs/pc_sigreg_T16_coupled_alm_s1 runs/pc_sigreg_T16_coupled_alm_s2 runs/bp_sigreg_etafix_s0 runs/bp_sigreg_etafix_s1 runs/bp_sigreg_etafix_s2"
STATUS=heads_status.txt; : > $STATUS
note() { echo "$*" | tee -a $STATUS; }
seeds_for() { case $1 in 1|5) echo "0 1 2";; *) echo "0";; esac; }
pcdir() { echo "runs/pc_sigreg_T16_coupled_${TAG}$1_s$2"; }
bpdir() { echo "runs/bp_sigreg_${TAG}$1_s$2"; }
complete() {   # params.eqx, no DIVERGED, log reaches the last step
  [ -f "$1/params.eqx" ] && [ ! -f "$1/DIVERGED" ] && \
  $PY -c "import json,sys;l=json.load(open('$1/log.json'));sys.exit(0 if l and l[-1]['step']==$2-1 else 1)" 2>/dev/null
}

collect() {
  tar czf "$JOBDIR/q1_heads.tar.gz" heads_status.txt heads_selected.json \
      gates_heads.log select_heads.log train_heads*.log gate_D*.json gate_D*.err gate_full_*.json \
      agree_*.json agree_*.log runs/*${TAG}*pilot*/params.eqx probes_heads.log probes_heads_summary.csv \
      diag_heads*.csv diag_heads*.log diag_learning_heads.csv diag_learning_heads.log config_diff.txt \
      runs/*${TAG}*/DIVERGED runs/*${TAG}*/config.json runs/*${TAG}*/log.json runs/*${TAG}*/probes.json runs/*${TAG}*/params.eqx \
      runs/*randD*/config.json runs/*randD*/probes.json runs/*randD*/params.eqx \
      $(for r in $REUSED; do echo $r/probes.json; done) 2>/dev/null
  ls -lh "$JOBDIR/q1_heads.tar.gz"
}
trap collect EXIT

echo "##### 0. preflight"
$PY -c "import jax;print('backend', jax.default_backend(), jax.devices(), 'matmul_precision', jax.config.jax_default_matmul_precision)" 2>&1 | tail -1 | tee -a $STATUS
for r in $REUSED; do [ -f "$r/params.eqx" ] || { note "FATAL: reused run $r missing"; exit 6; }; done
# PILOT_ONLY=1: stop after stage 3.  REUSE_PILOTS=1 (the later full job): keep the
# pilot dirs and their gate files, skip selection + pilots, refuse only full-run dirs.
if [ "$REUSE_PILOTS" = 1 ]; then EXIST=$(ls -d runs/*_${TAG}[1-5]_s* runs/*randD_s* 2>/dev/null)
else EXIST=$(ls -d runs/*_${TAG}*_s* runs/*randD_s* 2>/dev/null); fi
if [ -n "$EXIST" ]; then
  note "FATAL: case-D run dirs already exist (stale smoke or an earlier attempt) -- move them aside first:"
  echo "$EXIST" | tee -a $STATUS; exit 7
fi
for r in $REUSED; do [ -f "$r/probes.json" ] && [ ! -f "$r/probes.json.pre_caseD" ] && cp "$r/probes.json" "$r/probes.json.pre_caseD"; done

echo "##### 1. gates (CPU: golden reference regenerated and checked bit for bit)"
{ JAX_PLATFORMS=cpu $PY -W ignore test_pcalm.py && JAX_PLATFORMS=cpu $PY -W ignore test_sparse.py; } > gates_heads.log 2>&1 \
  || { grep FAIL gates_heads.log; note "GATES FAILED"; exit 3; }
grep -cE "\[PASS\]" gates_heads.log | xargs -I{} note "gates: {} PASS, 0 FAIL"

if [ "$REUSE_PILOTS" = 1 ]; then
  [ -s heads_selected.json ] || { note "FATAL: REUSE_PILOTS but no heads_selected.json"; exit 4; }
  PASSED=""
  for K in $KS; do
    $PY -c "import json,sys;sys.exit(0 if json.load(open('gate_D$K.json')).get('pass') is True else 1)" 2>/dev/null \
      && PASSED="$PASSED $K"
  done
  note "reusing pilots; passed:$PASSED"
else
echo "##### 2. selection (training images only, at initialisation)"
$PY -W ignore select_heads.py $SELECT_ARGS --out heads_selected.json > select_heads.log 2>&1 \
  || { tail -5 select_heads.log; note "SELECTION FAILED"; exit 4; }
grep -E "^head" select_heads.log | tee -a $STATUS

echo "##### 3. pilot pair per K: agreement at init, train, gate, agreement at end"
PASSED=""
for K in $KS; do
  if ! $PY -W ignore run_arms.py --steps $PILOT --eval-every 250 --arms pc bp --seeds 0 $OP \
        --sparse-case D --heads-json heads_selected.json --K $K --tag ${TAG}${K}pilot > train_heads_pilot_K$K.log 2>&1; then
    note "pilot K=$K: run_arms FAILED -- K=$K SKIPPED"; tail -3 train_heads_pilot_K$K.log | tee -a $STATUS; continue
  fi
  PCP=runs/pc_sigreg_T16_coupled_${TAG}${K}pilot_s0
  $PY -W ignore diag_agree_heads.py $PCP --init --json agree_init_K$K.json > agree_init_K$K.log 2>&1 || note "agreement (init) K=$K errored"
  $PY -W ignore diag_agree_heads.py $PCP --json agree_pilot_K$K.json > agree_pilot_K$K.log 2>&1 || note "agreement (pilot end) K=$K errored"
  $PY gate_pilot.py $PCP runs/bp_sigreg_${TAG}${K}pilot_s0 > gate_D$K.json 2> gate_D$K.err; rc=$?
  if [ $rc -eq 0 ]; then PASSED="$PASSED $K"; note "pilot K=$K: PASS"
  elif [ $rc -eq 1 ]; then note "pilot K=$K: FAIL -- full runs SKIPPED"
  else note "pilot K=$K: GATE ERROR (rc=$rc, see gate_D$K.err) -- full runs SKIPPED"; fi
  $PY -c "import json;d=json.load(open('gate_D$K.json'));print('  ',{k:d.get(k) for k in ('reasons','error')})" 2>&1 | tee -a $STATUS
done
fi   # REUSE_PILOTS
if [ "$PILOT_ONLY" = 1 ]; then note "PILOT_ONLY: stopping after the pilots (passed:${PASSED:- none})"; exit 0; fi
[ -n "$PASSED" ] || { note "no K passed its pilot"; exit 5; }

NEW=""; FLOORS="runs/pc_sigreg_T16_coupled_randD_s0 runs/pc_sigreg_T16_coupled_randD_s1 runs/pc_sigreg_T16_coupled_randD_s2"
run_K() {   # $1 = K; one run_arms call per seed so a failure is isolated
  local K=$1
  for s in $(seeds_for $K); do
    $PY -W ignore run_arms.py --steps $STEPS --eval-every 2000 --arms pc bp --seeds $s $OP \
        --sparse-case D --heads-json heads_selected.json --K $K --tag ${TAG}$K >> train_heads_K$K.log 2>&1 \
        || note "run_arms K=$K seed $s exited nonzero"
    for d in $(pcdir $K $s) $(bpdir $K $s); do
      if complete $d $STEPS; then NEW="$NEW $d"; else note "INCOMPLETE: $d (excluded from analysis)"; fi
    done
    if complete $(pcdir $K $s) $STEPS && complete $(bpdir $K $s) $STEPS; then
      $PY gate_pilot.py $(pcdir $K $s) $(bpdir $K $s) > gate_full_K${K}_s$s.json 2>/dev/null; rc=$?
      note "full-run gate (report only) K=$K seed $s: $([ $rc -eq 0 ] && echo PASS || echo "FLAG rc=$rc")"
    fi
  done
}
analyse_now() {   # $1 = label; incremental head analysis on what exists so far
  $PY -W ignore diag_heads.py $NEW --heads-json heads_selected.json --csv diag_heads_$1.csv > diag_heads_$1.log 2>&1 \
    || note "diag_heads ($1) exited nonzero"
}

echo "##### 4. decisive full runs: K=1 and K=5, 3 seeds each"
for K in 1 5; do case " $PASSED " in *" $K "*) run_K $K;; *) note "K=$K not run (pilot)";; esac; done
analyse_now decisive

echo "##### 5. K=2,3,4 screens (1 seed), time-guarded"
for K in 2 3 4; do
  case " $KS " in *" $K "*) ;; *) continue;; esac
  case " $PASSED " in *" $K "*) ;; *) note "K=$K not run (pilot)"; continue;; esac
  [ $SECONDS -gt $BUDGET ] && { note "time guard: K=$K and later screens SKIPPED at ${SECONDS}s"; break; }
  run_K $K
done

echo "##### 6. untrained floor (original encoder at init + the K=5 heads at init)"
$PY -W ignore run_arms.py --init-only --arms pc --seeds 0 1 2 $OP --sparse-case D \
    --heads-json heads_selected.json --K 5 --tag randD > train_heads_floor.log 2>&1 || note "floor init-only failed"

echo "##### 7. config check: reused baselines vs the new runs"
TAG=$TAG $PY - > config_diff.txt 2>&1 <<'EOF'
import json, glob, os
TAG = os.environ['TAG']
def flat(d, p=""):
    o = {}
    for k, v in d.items():
        o.update(flat(v, f"{p}{k}.") if isinstance(v, dict) else {f"{p}{k}": v})
    return o
skip = ("sparse.", "tag", "arm", "name", "out_dir")
for ref, new in (("runs/pc_sigreg_T16_coupled_alm_s0", f"runs/pc_sigreg_T16_coupled_{TAG}1_s0"),
                 ("runs/bp_sigreg_etafix_s0", f"runs/bp_sigreg_{TAG}1_s0")):
    try:
        a, b = (flat(json.load(open(f"{x}/config.json"))) for x in (ref, new))
    except FileNotFoundError as e:
        print("missing", e); continue
    d = sorted(k for k in set(a) | set(b) if not k.startswith(skip) and a.get(k) != b.get(k))
    print(ref, "vs", new, "differ in:", {k: (a.get(k), b.get(k)) for k in d} or "nothing")
EOF
cat config_diff.txt | tee -a $STATUS

echo "##### 8. probe suite (embedding): new runs, floors AND reused arms, one protocol"
rm -f probes_summary.csv
$PY -W ignore run_probes.py $NEW $FLOORS $REUSED > probes_heads.log 2>&1 || note "run_probes exited nonzero"
if [ -f probes_summary.csv ]; then mv probes_summary.csv probes_heads_summary.csv; else note "no probes summary written"; fi
grep -viE "$CLEAN" probes_heads.log | grep -iE "===|class probe|knn" | tail -80

echo "##### 9. head analysis: all runs, floors, reused arms (+ initial heads)"
$PY -W ignore diag_heads.py $NEW $FLOORS $REUSED --heads-json heads_selected.json \
    --csv diag_heads.csv > diag_heads.log 2>&1 || note "diag_heads exited nonzero"
grep -viE "$CLEAN" diag_heads.log | grep -E "===|wrote" | tail -40

echo "##### 10. learning diagnostic"
$PY -W ignore diag_learning.py $NEW --csv diag_learning_heads.csv > diag_learning_heads.log 2>&1 \
  || note "diag_learning exited nonzero"

MISSING=0
for K in 1 5; do case " $PASSED " in *" $K "*) for s in 0 1 2; do
  complete $(pcdir $K $s) $STEPS && complete $(bpdir $K $s) $STEPS || MISSING=$((MISSING+1)); done;; esac; done
note "DONE (decisive pairs missing: $MISSING; elapsed ${SECONDS}s)"
[ $MISSING -eq 0 ] || exit 8
