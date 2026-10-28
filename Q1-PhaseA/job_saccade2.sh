#!/usr/bin/env bash
# Saccade follow-up (inference only): old gates -> new gates -> 200-image smoke -> full run (3 seeds, K <= 32, views <= 192).
#   bash job_saccade2.sh <project_root> <job_dir> [gates_only]
set -uo pipefail
P="$1"; JOBDIR=$(readlink -f "$2"); ONLY="${3:-}"; cd "$P/Q1-PhaseA" || exit 2
PY=../.venv/bin/python
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false NVIDIA_TF32_OVERRIDE=0 XLA_FLAGS=--xla_gpu_autotune_level=0
ST=saccade2_status.txt; note() { echo "$(date +%m-%d_%H:%M) $*" | tee -a $ST; }
note "start at $(git rev-parse --short HEAD) ${ONLY}"
$PY -W ignore test_saccade.py > "$JOBDIR/gates_old.log" 2>&1; rc=$?; note "old gates rc=$rc"; [ $rc -eq 0 ] || exit 4
$PY -W ignore test_saccade2.py > "$JOBDIR/gates.log" 2>&1; rc=$?
note "gates rc=$rc $(grep -c PASS "$JOBDIR/gates.log") pass $(grep -c FAIL "$JOBDIR/gates.log") fail"; [ $rc -eq 0 ] || exit 4
$PY -W ignore diag_saccade2.py --out "$JOBDIR/smoke.json" --n 200 --kmax 4 --vmax 12 --seeds 0 1 > "$JOBDIR/smoke.log" 2>&1; rc=$?
note "smoke rc=$rc"; [ $rc -eq 0 ] || exit 5
[ "$ONLY" = "gates_only" ] && exit 0
$PY -W ignore diag_saccade2.py --out "$JOBDIR/diag_saccade2.json" > "$JOBDIR/full.log" 2>&1; rc=$?
note "full rc=$rc"; exit $rc
