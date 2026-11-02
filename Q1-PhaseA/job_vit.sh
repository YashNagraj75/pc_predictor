#!/usr/bin/env bash
# ViT I-JEPA (plan_vit_ijepa.md).  bash job_vit.sh <project_root> <job_dir> <gates|smoke|v0>
#   gates: test_vit.py (V0 + transformer ePC, incl. real-batch memory); smoke: gates, then a compressed V0 schedule
#   (3 epochs, 1 warmup: reaches peak lr) plus a resume check; v0: the full encoder run (resumable), ending in the gate.
set -uo pipefail
P="$1"; JOBDIR=$(readlink -f "$2"); MODE="$3"; cd "$P/Q1-PhaseA" || exit 2
PY="$(cd ../.venv/bin && pwd)/python"
# XLA GEMM autotuning ON for the ViT (3-4x faster; compiled == eager audits pass under it, run 4d5d69e2); TF32 stays off
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false NVIDIA_TF32_OVERRIDE=0; unset XLA_FLAGS
ST=vit_status.txt; note() { echo "$(date +%m-%d_%H:%M) [vit/$MODE] $*" | tee -a $ST; }
mkdir -p runs_vit
note "start at $(git rev-parse --short HEAD)"
case "$MODE" in
  gates|smoke)
    $PY -W ignore test_vit.py > "$JOBDIR/gates.log" 2>&1; rc=$?
    note "gates rc=$rc $(grep -cF '[PASS]' "$JOBDIR/gates.log") pass $(grep -cF '[FAIL]' "$JOBDIR/gates.log") fail"; [ $rc -eq 0 ] || exit 4
    # speed with XLA GEMM autotuning back on (TF32 stays off), with the compiled-vs-eager audits re-run: report only
    [ "$MODE" = gates ] && exit 0
    rm -rf runs_vit/smoke
    $PY -W ignore vit_ijepa.py --out runs_vit/smoke --epochs 3 --warmup 1 --max-steps 400 --smoke > "$JOBDIR/smoke.log" 2>&1; rc=$?
    note "smoke part 1 (400 of 585 steps) rc=$rc"; [ $rc -eq 0 ] || exit 5
    $PY -W ignore vit_ijepa.py --out runs_vit/smoke --epochs 3 --warmup 1 --smoke >> "$JOBDIR/smoke.log" 2>&1; rc=$?
    grep -q "RESUME at step 400" "$JOBDIR/smoke.log" || { note "resume did not start from the last checkpoint"; exit 6; }
    cp runs_vit/smoke/v0_report.json "$JOBDIR/" 2>/dev/null; note "smoke part 2 (resume to the end + gate report) rc=$rc"; exit $rc ;;
  v0)
    flock -n runs_vit/v0.lock $PY -W ignore vit_ijepa.py --out runs_vit/v0_d6_s0 >> "$JOBDIR/v0.log" 2>&1; rc=$?
    cp runs_vit/v0_d6_s0/v0_report.json runs_vit/v0_d6_s0/train_log.json "$JOBDIR/" 2>/dev/null; note "v0 rc=$rc"; exit $rc ;;
  *) note "unknown mode $MODE"; exit 64 ;;
esac
