#!/usr/bin/env bash
# ViT I-JEPA (plan_vit_ijepa.md).  bash job_vit.sh <project_root> <job_dir> <gates|smoke|v0>
#   gates: test_vit.py (V0 + transformer ePC); smoke: gates + 300 V0 steps with the full schedule (speed, finiteness,
#   gate report); v0: the full encoder run (resumable), ending in the kNN / linear-probe / collapse gate.
set -uo pipefail
P="$1"; JOBDIR=$(readlink -f "$2"); MODE="$3"; cd "$P/Q1-PhaseA" || exit 2
PY="$(cd ../.venv/bin && pwd)/python"
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false NVIDIA_TF32_OVERRIDE=0 XLA_FLAGS=--xla_gpu_autotune_level=0
ST=vit_status.txt; note() { echo "$(date +%m-%d_%H:%M) [vit/$MODE] $*" | tee -a $ST; }
note "start at $(git rev-parse --short HEAD)"
if [ "$MODE" = gates ] || [ "$MODE" = smoke ]; then
  $PY -W ignore test_vit.py > "$JOBDIR/gates.log" 2>&1; rc=$?
  note "gates rc=$rc $(grep -c PASS "$JOBDIR/gates.log") pass $(grep -c FAIL "$JOBDIR/gates.log") fail"; [ $rc -eq 0 ] || exit 4
fi
if [ "$MODE" = smoke ]; then
  rm -rf runs_vit/smoke
  $PY -W ignore vit_ijepa.py --out runs_vit/smoke --max-steps 300 > "$JOBDIR/smoke.log" 2>&1; rc=$?
  cp runs_vit/smoke/v0_report.json "$JOBDIR/" 2>/dev/null; note "smoke rc=$rc"; exit $rc
fi
if [ "$MODE" = v0 ]; then
  $PY -W ignore vit_ijepa.py --out runs_vit/v0_d6_s0 > "$JOBDIR/v0.log" 2>&1; rc=$?
  cp runs_vit/v0_d6_s0/v0_report.json runs_vit/v0_d6_s0/*.json "$JOBDIR/" 2>/dev/null; note "v0 rc=$rc"; exit $rc
fi
