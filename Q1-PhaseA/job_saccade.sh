#!/usr/bin/env bash
# Saccade S2 (inference only): gates -> 200-image smoke -> full run (3 seeds, K <= 32, whole views <= 192).
# PC predictors get glimpses; frozen BP references get one augmented whole view only (user rule 2026-10-27).
#   bash job_saccade.sh <project_root> <job_dir>
set -uo pipefail
P="$1"; JOBDIR=$(readlink -f "$2"); cd "$P/Q1-PhaseA" || exit 2
PY=../.venv/bin/python
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false NVIDIA_TF32_OVERRIDE=0 XLA_FLAGS=--xla_gpu_autotune_level=0
ST=saccade_status.txt; note() { echo "$(date +%m-%d_%H:%M) $*" | tee -a $ST; }
note "start at $(git rev-parse --short HEAD)"
$PY -W ignore test_saccade.py > "$JOBDIR/gates.log" 2>&1; rc=$?
note "gates rc=$rc $(grep -c PASS "$JOBDIR/gates.log") pass $(grep -c FAIL "$JOBDIR/gates.log") fail"
[ $rc -eq 0 ] || exit 4
$PY -W ignore diag_saccade.py --out "$JOBDIR/smoke.json" --n 200 --kmax 8 --vmax 12 --seeds 0 > "$JOBDIR/smoke.log" 2>&1; rc=$?
note "smoke rc=$rc"; [ $rc -eq 0 ] || exit 5
$PY -W ignore diag_saccade.py --out "$JOBDIR/diag_saccade.json" > "$JOBDIR/full.log" 2>&1; rc=$?
note "full rc=$rc"
exit $rc
