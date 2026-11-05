#!/usr/bin/env bash
# Stage A (plan_stageA_scoped.md v3).  bash job_stageA.sh <project_root> <job_dir> <smoke|a0|depth|noise> [depth]
#   Resumable: re-launch the same command after a deadline exit (rc 75); finished and diverged runs are skipped.
#   depth D: per arm, 5-epoch lr selection on the 90/10 split at {3e-4,1e-3,3e-3}, then 25 epochs at the winner
#   (BP also runs 25 epochs at all three lrs: lr sensitivity).  noise: BP seeds 1-4 at BP's selected lr, both depths.
set -uo pipefail
P="$1"; JOBDIR=$(readlink -f "$2"); MODE="$3"; DEPTH="${4:-10}"; cd "$P/Q1-PhaseA" || exit 2
PY="$(cd ../.venv/bin && pwd)/python"
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false NVIDIA_TF32_OVERRIDE=0; unset XLA_FLAGS
ST=stageA_status.txt; note() { echo "$(date +%m-%d_%H:%M) [stageA/$MODE$DEPTH] $*" | tee -a $ST >&2; }   # stderr: run1 returns its dir on stdout
B=runs_vit/stageA; mkdir -p $B; FAIL=$B/.fail_$MODE$DEPTH; rm -f $FAIL
exec 9>"$B/.gpu.lock"; flock -n 9 || { note "another stage-A job holds the GPU lock"; exit 9; }
trap 'kill 0' TERM INT
JOB_MIN=${JOB_MIN:-610}; T_START=$(date +%s)
left() { echo $(( JOB_MIN - ($(date +%s) - T_START) / 60 )); }
[ -f runs_vit/v0_d6_s0/v0_report.json ] || { note "no V0 report"; exit 7; }
$PY -c "import json,sys; sys.exit(0 if json.load(open('runs_vit/v0_d6_s0/v0_report.json'))['gate']['pass'] else 1)" || { note "V0 gate not passed"; exit 7; }
note "start at $(git rev-parse --short HEAD), $(left) min budget"
TG() { $PY -c "import json; d=json.load(open('$B/a0/a0_choice.json')); print(d['T'], d['gn_iters'])"; }
run1() {  # arm depth lr epochs split seed [extra args] -> echoes dir; rc 0 done | 3 diverged | 75 deadline | other fail
  local T GI; read T GI < <(TG)
  local tag="$1_d$2_lr$3_e$4_$5_s$6"; [ "$1" = bp ] || tag="${tag}_T${T}g${GI}"
  local d="$B/$tag"; echo "$d"
  [ -f "$d/report.json" ] && return 0
  [ -f "$d/diverged.json" ] && { note "$tag previously diverged (skipped)"; return 3; }
  local m=$(left); [ $m -gt 30 ] || { note "$tag not started: job budget used"; return 75; }
  $PY -W ignore train_pred_vit.py --arm $1 --depth $2 --lr $3 --epochs $4 --split $5 --seed $6 --T $T --gn-iters $GI --deadline-min $((m - 20)) --out "$d" "${@:7}" >> "$d.log" 2>&1
  local rc=$?; note "$tag rc=$rc"
  [ $rc -eq 0 ] || [ $rc -eq 3 ] || [ $rc -eq 75 ] || touch $FAIL
  return $rc
}
best_lr() {  # arm depth -> "lr" or "incomplete"/"none"; table and EDGE flag to the status file
  local T GI; read T GI < <(TG)
  $PY - "$B" "$1" "$2" "$T" "$GI" <<'PYX' | tee -a "$ST" | tail -1
import json, math, os, sys
B, arm, d, T, GI = sys.argv[1:]; tab = {}
for lr in ("3e-4", "1e-3", "3e-3"):
    t = f"{B}/{arm}_d{d}_lr{lr}_e5_val_s0" + ("" if arm == "bp" else f"_T{T}g{GI}")
    if os.path.exists(t + "/report.json"):
        v = json.load(open(t + "/report.json"))["eval"]["clean"]["nmse"]; tab[lr] = v if math.isfinite(v) else math.inf
    elif os.path.exists(t + "/diverged.json"):
        tab[lr] = math.inf
print(f"   lr table {arm} d{d}: " + ", ".join(f"{k}: {v:.4f}" for k, v in tab.items()))
if len(tab) < 3: print("incomplete")
elif all(math.isinf(v) for v in tab.values()): print("none")
else:
    lr = min(tab, key=tab.get); print(f"   selected {lr}" + ("  EDGE" if lr in ("3e-4", "3e-3") else "")); print(lr)
PYX
}
case "$MODE" in
  smoke)   # every arm at depth 6 for 12 steps (via run1, with resume at step 6), and the depth-10 ePC memory/time case
    mkdir -p $B/a0; [ -f $B/a0/a0_choice.json ] || { echo '{"T": 20, "gn_iters": 12}' > $B/a0/a0_choice.json; touch $B/a0/.provisional; }
    for arm in bp epc_alm_k4 epc_k4 epc_k16; do
      d=$(run1 $arm 6 1e-3 2 val 90 --max-steps 6); rc=$?; [ $rc -eq 0 ] || exit 5
      rm -f "$d/report.json"; d=$(run1 $arm 6 1e-3 2 val 90 --max-steps 12); rc=$?; [ $rc -eq 0 ] || exit 5
      grep -q "RESUME at step 6" "$d.log" || { note "smoke $arm: resume did not start from step 6"; exit 6; }
      cp "$d/report.json" "$JOBDIR/smoke_$arm.json"
    done
    d=$(run1 epc_k16 10 1e-3 2 val 91 --max-steps 3); rc=$?; [ $rc -eq 0 ] || exit 5; cp "$d/report.json" "$JOBDIR/smoke_epc_k16_d10.json"
    rm -rf $B/*_s90* $B/*_s91*; [ -f $B/a0/.provisional ] && rm -f $B/a0/a0_choice.json $B/a0/.provisional; note "smoke ok" ;;
  a0)
    $PY -W ignore a0_vit.py $B/a0 > "$JOBDIR/a0.log" 2>&1; rc=$?; note "a0 rc=$rc"; cp $B/a0/a0_choice.json "$JOBDIR/" 2>/dev/null; exit $rc ;;
  depth)
    [ -f $B/a0/a0_choice.json ] || { note "run a0 first"; exit 7; }
    for arm in ${STAGEA_ARMS:-bp epc_alm_k4 epc_k4 epc_k16 epc_alm_blk4}; do   # epc_alm_blk4: per-block rho (after findings/40)
      for lr in 3e-4 1e-3 3e-3; do run1 $arm $DEPTH $lr 5 val 0 > /dev/null; [ $? -eq 75 ] && exit 75; done
      lr=$(best_lr $arm $DEPTH); note "$arm d$DEPTH selected lr: $lr"
      case "$lr" in incomplete|none) touch $FAIL; continue ;; esac
      lrs=$lr; [ $arm = bp ] && lrs="3e-4 1e-3 3e-3"
      for l in $lrs; do
        d=$(run1 $arm $DEPTH $l 25 full 0); rc=$?; [ $rc -eq 75 ] && exit 75
        [ $rc -eq 3 ] && note "FINAL DIVERGED $arm d$DEPTH lr $l: needs a decision (not re-tuned)"
        cp "$d/report.json" "$JOBDIR/report_$(basename $d).json" 2>/dev/null
      done
    done ;;
  noise)
    for D in 10 6; do lr=$(best_lr bp $D); case "$lr" in incomplete|none) touch $FAIL; continue ;; esac
      for s in 1 2 3 4; do d=$(run1 bp $D $lr 25 full $s); rc=$?; [ $rc -eq 75 ] && exit 75; cp "$d/report.json" "$JOBDIR/report_$(basename $d).json" 2>/dev/null; done; done ;;
  *) note "unknown mode $MODE"; exit 64 ;;
esac
[ ! -e $FAIL ] || { note "finished with failures"; exit 1; }
note "done"
