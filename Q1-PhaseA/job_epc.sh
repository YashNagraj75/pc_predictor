#!/usr/bin/env bash
# ePC inference-only check (diag_epc.py): gates (test_epc.py) then the diagnostic, Galaxy10 checkpoints.
#   bash job_epc.sh <project_root> <job_dir>
set -uo pipefail
P="$1"; JOBDIR=$(readlink -f "$2"); cd "$P/Q1-PhaseA" || exit 2
PY="$(cd ../.venv/bin && pwd)/python"
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false NVIDIA_TF32_OVERRIDE=0 XLA_FLAGS=--xla_gpu_autotune_level=0
ST=epc_status.txt; note() { echo "$(date +%m-%d_%H:%M) [epc] $*" | tee -a $ST; }
note "start at $(git rev-parse --short HEAD)"
$PY -W ignore test_epc.py > "$JOBDIR/gates.log" 2>&1; rc=$?
note "gates rc=$rc $(grep -c PASS "$JOBDIR/gates.log") pass $(grep -c FAIL "$JOBDIR/gates.log") fail"; [ $rc -eq 0 ] || exit 4
$PY -W ignore diag_epc.py "$JOBDIR/diag_epc.json" > "$JOBDIR/diag.log" 2>&1; rc=$?
note "diag rc=$rc"; exit $rc
