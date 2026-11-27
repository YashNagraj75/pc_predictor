#!/usr/bin/env bash
# Stage A (plan_stageA_scoped.md v3).  bash job_stageA.sh <project_root> <job_dir> <smoke|a0|depth|noise|seeds> [depth]
#   Resumable: re-launch the same command after a deadline exit (rc 75); finished and diverged runs are skipped.
#   depth D: per arm, 5-epoch lr selection on the 90/10 split at {3e-4,1e-3,3e-3}, then 25 epochs at the winner
#   (BP also runs 25 epochs at all three lrs).  noise: BP seeds 1-4.  seeds: SEEDS of SEED_ARMS at DEPTH, seed-0 lr.
set -uo pipefail
P="$1"; JOBDIR=$(readlink -f "$2"); MODE="$3"; DEPTH="${4:-$([ "${ARCH:-vit}" = resnet ] && echo 20 || echo 10)}";   # default = the deep depth of each architecture
 cd "$P/Q1-PhaseA" || exit 2
PY="$(cd ../.venv/bin && pwd)/python"
# Triton GEMMs give WRONG batched training gradients on this host (findings/41); cuBLAS with autotuning is correct and as fast
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false NVIDIA_TF32_OVERRIDE=0 XLA_FLAGS=--xla_gpu_enable_triton_gemm=false
ST=stageA_status.txt; note() { echo "$(date +%m-%d_%H:%M) [stageA/$MODE$DEPTH] $*" | tee -a $ST >&2; }   # stderr: run1 returns its dir on stdout
ARCH=${ARCH:-vit}; export ARCH; B=runs_vit/stageA; mkdir -p $B; LOCK=$B/.gpu.lock    # one GPU lock for both architectures
if [ "$ARCH" = resnet ]; then B=runs_vit/stageA_rn; mkdir -p $B/a0         # ResNet runs: own folder; same T/gn and arm decision
  for f in a0/a0_choice.json blockrho_decision.json; do [ -f $B/$f ] || cp runs_vit/stageA/$f $B/$f || exit 7; done; fi
FAIL=$B/.fail_$MODE$DEPTH; rm -f $FAIL
exec 9>"$LOCK"; flock -n 9 || { note "another stage-A job holds the GPU lock"; exit 9; }
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
  $PY -W ignore train_pred_vit.py --arm $1 --depth $2 --lr $3 --epochs $4 --split $5 --seed $6 --T $T --gn-iters $GI --deadline-min $((m - 20)) --arch $ARCH --out "$d" "${@:7}" >> "$d.log" 2>&1
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
# GPU-vs-CPU gradient gate before any training (findings/41); skipped only for the gate mode itself
gpu_gate() {
  G=$B/gpu_gate_$$; rm -rf $G
  $PY -W ignore gpu_gate.py run "$P" gpu $G $DEPTH > "$JOBDIR/gpu_gate_${MODE}${DEPTH}.log" 2>&1 || { note "gpu gate: GPU run failed"; return 1; }
  ( JAX_PLATFORMS=cpu XLA_FLAGS= $PY -W ignore gpu_gate.py run "$P" cpu $G $DEPTH ) >> "$JOBDIR/gpu_gate_${MODE}${DEPTH}.log" 2>&1 || { note "gpu gate: CPU run failed"; return 1; }
  $PY gpu_gate.py compare $G >> "$JOBDIR/gpu_gate_${MODE}${DEPTH}.log" 2>&1; rc=$?; rm -rf $G
  note "gpu gate rc=$rc"; return $rc
}
case "$MODE" in
  gpugate) gpu_gate; exit $? ;;
  depth|noise|seeds|a0|smoke_spc|smoke_rn) gpu_gate || exit 4 ;;
esac
case "$MODE" in
  smoke)   # every arm at depth 6 for 12 steps (via run1, with resume at step 6), and the depth-10 ePC memory/time case
    mkdir -p $B/a0; [ -f $B/a0/a0_choice.json ] || { echo '{"T": 20, "gn_iters": 12}' > $B/a0/a0_choice.json; touch $B/a0/.provisional; }
    for arm in bp epc_alm_k4 epc_k4 epc_k16 epc_alm_blk4; do
      d=$(run1 $arm 6 1e-3 2 val 90 --max-steps 6); rc=$?; [ $rc -eq 0 ] || exit 5
      rm -f "$d/report.json"; d=$(run1 $arm 6 1e-3 2 val 90 --max-steps 12); rc=$?; [ $rc -eq 0 ] || exit 5
      grep -q "RESUME at step 6" "$d.log" || { note "smoke $arm: resume did not start from step 6"; exit 6; }
      cp "$d/report.json" "$JOBDIR/smoke_$arm.json"
    done
    d=$(run1 epc_k16 10 1e-3 2 val 91 --max-steps 3); rc=$?; [ $rc -eq 0 ] || exit 5; cp "$d/report.json" "$JOBDIR/smoke_epc_k16_d10.json"
    rm -rf $B/*_s90* $B/*_s91*; [ -f $B/a0/.provisional ] && rm -f $B/a0/a0_choice.json $B/a0/.provisional; note "smoke ok" ;;
  smoke_rn)   # ResNet: known-answer tests, then bp / epc_alm_blk4 / spc_u1 6 + 6 steps at 12 blocks with resume, 3 steps at 20
    [ "$ARCH" = resnet ] || { note "smoke_rn needs ARCH=resnet"; exit 64; }
    $PY -W ignore test_resnet.py > "$JOBDIR/test_resnet.log" 2>&1 || { note "test_resnet FAILED"; exit 5; }
    rm -rf "$B"/*_s90* "$B"/*_s91*
    for arm in bp epc_alm_blk4 spc_u1; do
      d=$(run1 $arm 12 1e-3 2 val 90 --max-steps 6); [ $? -eq 0 ] || exit 5
      rm -f "$d/report.json"; d=$(run1 $arm 12 1e-3 2 val 90 --max-steps 12); [ $? -eq 0 ] || exit 5
      grep -q "RESUME at step 6" "$d.log" || { note "smoke_rn $arm: resume did not start from step 6"; exit 6; }
      cp "$d/report.json" "$JOBDIR/smoke_rn_${arm}_d12.json"
      d=$(run1 $arm 20 1e-3 2 val 91 --max-steps 3); [ $? -eq 0 ] || exit 5; cp "$d/report.json" "$JOBDIR/smoke_rn_${arm}_d20.json"
    done
    rm -rf "$B"/*_s90* "$B"/*_s91*; note "smoke_rn ok" ;;
  smoke_spc)  # spc_u1: 6 + 6 steps at depth 6 with a resume check, then 3 steps at depth 10 (memory, s/step)
    rm -rf "$B"/spc_u1_d6_lr1e-3_e2_val_s90_* "$B"/spc_u1_d10_lr1e-3_e2_val_s91_*     # only this smoke's own runs
    d=$(run1 spc_u1 6 1e-3 2 val 90 --max-steps 6); [ $? -eq 0 ] || exit 5
    rm -f "$d/report.json"; d=$(run1 spc_u1 6 1e-3 2 val 90 --max-steps 12); [ $? -eq 0 ] || exit 5
    grep -q "RESUME at step 6" "$d.log" || { note "smoke spc_u1: resume did not start from step 6"; exit 6; }
    cp "$d/report.json" "$JOBDIR/smoke_spc_u1_d6.json"; cp "$d.log" "$JOBDIR/smoke_spc_u1_d6.log"
    d=$(run1 spc_u1 10 1e-3 2 val 91 --max-steps 3); [ $? -eq 0 ] || exit 5
    cp "$d/report.json" "$JOBDIR/smoke_spc_u1_d10.json"; cp "$d.log" "$JOBDIR/smoke_spc_u1_d10.log"
    rm -rf "$B"/spc_u1_d6_lr1e-3_e2_val_s90_* "$B"/spc_u1_d10_lr1e-3_e2_val_s91_*; note "smoke_spc ok" ;;
  a0)
    $PY -W ignore a0_vit.py $B/a0 > "$JOBDIR/a0.log" 2>&1; rc=$?; note "a0 rc=$rc"; cp $B/a0/a0_choice.json "$JOBDIR/" 2>/dev/null; exit $rc ;;
  depth)
    [ -f $B/a0/a0_choice.json ] || { note "run a0 first"; exit 7; }
    for arm in ${STAGEA_ARMS:-bp epc_alm_k4 epc_k4 epc_k16 epc_alm_blk4}; do   # blk arm needs blockrho_decision.json (user decision 2026-11-06)
      if [[ $arm == epc_alm_blk* ]]; then
        $PY -c "import json,sys; d=json.load(open('$B/blockrho_decision.json')); sys.exit(0 if d['train'] and d['arm'] == '$arm' else 1)" 2>/dev/null || { note "epc_alm_blk4: no passing screen decision ($B/blockrho_decision.json); skipped"; touch $FAIL; continue; }
      fi
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
    for D in ${NOISE_DEPTHS:-10 6}; do lr=$(best_lr bp $D); case "$lr" in incomplete|none) touch $FAIL; continue ;; esac
      for s in 1 2 3 4; do d=$(run1 bp $D $lr 25 full $s); rc=$?; [ $rc -eq 75 ] && exit 75; cp "$d/report.json" "$JOBDIR/report_$(basename $d).json" 2>/dev/null; done; done ;;
  seeds)  # extra seeds SEEDS (default 1 2) of SEED_ARMS (default epc_alm_blk4) at DEPTH, each at its seed-0 selected lr
    read T0 GI0 < <(TG)
    for arm in ${SEED_ARMS:-epc_alm_blk4}; do lr=$(best_lr $arm $DEPTH)
      case "$lr" in 3e-4|1e-3|3e-3) ;; *) note "seeds: no valid selected lr for $arm d$DEPTH ('$lr')"; touch $FAIL; continue ;; esac
      s0="$B/${arm}_d${DEPTH}_lr${lr}_e25_full_s0"; [ "$arm" = bp ] || s0="${s0}_T${T0}g${GI0}"
      [ -f "$s0/report.json" ] || { note "seeds: seed-0 final $s0 missing, not starting"; touch $FAIL; continue; }
      for s in ${SEEDS:-1 2}; do d=$(run1 "$arm" "$DEPTH" "$lr" 25 full "$s"); rc=$?; [ $rc -eq 75 ] && exit 75
        [ $rc -eq 3 ] && note "SEED DIVERGED $arm d$DEPTH s$s: reported, not re-tuned"; cp "$d/report.json" "$JOBDIR/report_$(basename $d).json" 2>/dev/null; done
      nd=0; nv=0; nm=0; for s in ${SEEDS:-1 2}; do d="${s0%_s0*}_s$s${s0#*_s0}"
        if [ -f "$d/report.json" ]; then nd=$((nd + 1)); elif [ -f "$d/diverged.json" ]; then nv=$((nv + 1)); else nm=$((nm + 1)); fi; done
      note "seeds summary $arm d$DEPTH: finished $nd, diverged $nv, missing $nm"; [ $nm -eq 0 ] || touch $FAIL; done ;;
  *) note "unknown mode $MODE"; exit 64 ;;
esac
[ ! -e $FAIL ] || { note "finished with failures"; exit 1; }
note "done"
