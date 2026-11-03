#!/usr/bin/env bash
# Stage A (plan_stageA_scoped.md v3).  bash job_stageA.sh <project_root> <job_dir> <a0|depth|noise|smoke> [depth]
#   a0: cost/credit check -> runs_vit/stageA/a0/a0_choice.json.   depth D: for every arm, 5-epoch lr selection on the
#   90/10 split at lr {3e-4,1e-3,3e-3}, then 25 epochs on the full train set at the arm's winner (finished runs skipped).
#   noise: BP seeds 1,2 at depths 10 and 6 at BP's selected lr.   smoke: a few steps of every arm at depth 6.
set -uo pipefail
P="$1"; JOBDIR=$(readlink -f "$2"); MODE="$3"; DEPTH="${4:-10}"; cd "$P/Q1-PhaseA" || exit 2
PY="$(cd ../.venv/bin && pwd)/python"
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false NVIDIA_TF32_OVERRIDE=0; unset XLA_FLAGS
ST=stageA_status.txt; note() { echo "$(date +%m-%d_%H:%M) [stageA/$MODE$DEPTH] $*" | tee -a $ST >&2; }   # stderr: run1 returns its dir on stdout
B=runs_vit/stageA; mkdir -p $B
[ -f runs_vit/v0_d6_s0/v0_report.json ] || { note "no V0 report"; exit 7; }
$PY -c "import json,sys; sys.exit(0 if json.load(open('runs_vit/v0_d6_s0/v0_report.json'))['gate']['pass'] else 1)" || { note "V0 gate not passed"; exit 7; }
note "start at $(git rev-parse --short HEAD)"
TG() { $PY -c "import json; d=json.load(open('$B/a0/a0_choice.json')); print(d['T'], d['gn_iters'])"; }
run1() {  # arm depth lr epochs split seed -> dir
  local d="$B/$1_d$2_lr$3_e$4_$5_s$6"; [ -f "$d/report.json" ] && { echo "$d"; return 0; }
  read T GI < <(TG)
  $PY -W ignore train_pred_vit.py --arm $1 --depth $2 --lr $3 --epochs $4 --split $5 --seed $6 --T $T --gn-iters $GI --out "$d" >> "$d.log" 2>&1
  local rc=$?; note "$1 d$2 lr$3 e$4 $5 s$6 rc=$rc"; echo "$d"
}
best_lr() {  # arm depth -> lr with the lowest holdout nmse (5-epoch runs)
  $PY - "$B" "$1" "$2" <<'PYX'
import json, os, sys
B, arm, d = sys.argv[1:]; best = None
for lr in ("3e-4", "1e-3", "3e-3"):
    p = f"{B}/{arm}_d{d}_lr{lr}_e5_val_s0/report.json"
    if os.path.exists(p):
        v = json.load(open(p))["test"]["clean"]["nmse"]
        if best is None or v < best[1]: best = (lr, v)
print(best[0] if best else "none")
PYX
}
case "$MODE" in
  smoke)
    for arm in bp epc_alm_k4 epc_k4 epc_k16; do
      d=$B/smoke_$arm; rm -rf $d
      $PY -W ignore train_pred_vit.py --arm $arm --depth 6 --lr 1e-3 --epochs 1 --split val --max-steps 30 --out $d > "$JOBDIR/smoke_$arm.log" 2>&1; rc=$?
      note "smoke $arm rc=$rc"; [ $rc -eq 0 ] || exit 5
    done ;;
  a0)
    $PY -W ignore a0_vit.py $B/a0 > "$JOBDIR/a0.log" 2>&1; rc=$?; note "a0 rc=$rc"; cp $B/a0/a0_choice.json "$JOBDIR/" 2>/dev/null; exit $rc ;;
  depth)
    [ -f $B/a0/a0_choice.json ] || { note "run a0 first"; exit 7; }
    for arm in bp epc_alm_k4 epc_k4 epc_k16; do
      for lr in 3e-4 1e-3 3e-3; do run1 $arm $DEPTH $lr 5 val 0 > /dev/null; done
      lr=$(best_lr $arm $DEPTH); note "$arm d$DEPTH selected lr $lr"
      [ "$lr" = none ] && continue
      d=$(run1 $arm $DEPTH $lr 25 full 0); cp "$d/report.json" "$JOBDIR/report_${arm}_d${DEPTH}.json" 2>/dev/null
    done ;;
  noise)
    for D in 10 6; do lr=$(best_lr bp $D); [ "$lr" = none ] && continue
      for s in 1 2; do d=$(run1 bp $D $lr 25 full $s); cp "$d/report.json" "$JOBDIR/report_bp_d${D}_s$s.json" 2>/dev/null; done; done ;;
  *) note "unknown mode"; exit 64 ;;
esac
