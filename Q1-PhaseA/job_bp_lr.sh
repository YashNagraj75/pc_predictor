#!/usr/bin/env bash
# Diagnostic (user-approved 2026-11-02): BP feedforward residual-d10 predictor (same muPC net as
# frozen BPf). Selects lr from {1e-3,3e-3,1e-2} on the Galaxy holdout as PC-ALM did in findings/29
# (seed 0, lowest held-out nmse wins), then runs 3 seeds at that lr on Galaxy10/MNIST/CIFAR-10.
# BPf itself is untouched; outputs go to pred_runs/ffres_bplr(_sel)/tests_ffres_bplr.
#   bash job_bp_lr.sh <project_root> <job_dir>
set -uo pipefail
P="$1"; JOBDIR=$(readlink -f "$2"); cd "$P/Q1-PhaseA" || exit 2
Q=$(pwd); PY="$(cd ../.venv/bin && pwd)/python"
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.15 NVIDIA_TF32_OVERRIDE=0 XLA_FLAGS=--xla_gpu_autotune_level=0
# Logs a timestamped, [bplr]-tagged message to stdout and the status file.
ST=$Q/bplr_status.txt; note() { echo "$(date +%m-%d_%H:%M) [bplr] $*" | tee -a "$ST"; }
note "start at $(git rev-parse --short HEAD)"
mkdir -p "$JOBDIR/bplr" pred_runs/ffres_bplr_sel
# 1. selection on the Galaxy holdout (same grid, steps, seed as PC's selection)
for lr in 0.001 0.003 0.01; do
  o=pred_runs/ffres_bplr_sel/bp_lr$lr.json
  [ -f $o ] || echo "$PY -W ignore $Q/pred_ffres.py --arm bp --depth 10 --lr $lr --holdout --out $o > pred_runs/ffres_bplr_sel/bp_lr$lr.log 2>&1; echo \$? > pred_runs/ffres_bplr_sel/bp_lr$lr.rc"
done | xargs -d '\n' -P 3 -I{} bash -c '{}'
LR=$($PY - <<'PY'
import json, glob
rows = [json.load(open(f)) for f in sorted(glob.glob("pred_runs/ffres_bplr_sel/bp_lr*.json"))]
assert len(rows) == 3, len(rows)
ok = [r for r in rows if r["finite"]]
b = min(ok, key=lambda r: r["test"]["nmse"])
out = {"lr": b["cfg"]["lr"], "val_nmse": b["test"]["nmse"], "edge_lr": b["cfg"]["lr"] in (0.001, 0.01),
       "all": [{"lr": r["cfg"]["lr"], "val_nmse": r["test"]["nmse"], "finite": r["finite"]} for r in rows]}
json.dump(out, open("pred_runs/ffres_bplr_sel/selected.json", "w"), indent=1)
print(out["lr"])
PY
) || { note "selection failed"; exit 4; }
cp pred_runs/ffres_bplr_sel/selected.json "$JOBDIR/bplr/selected.json"
note "selected lr $LR: $(cat pred_runs/ffres_bplr_sel/selected.json | tr -d '\n ' | cut -c1-300)"
# 2. 3 seeds at the selected lr on every dataset, then the single-view test suite (clean, noise, occlusion)
for ds in galaxy10 mnist cifar10; do
  d=$Q; [ $ds = galaxy10 ] || d=$Q/ds/$ds
  for s in 0 1 2; do
    echo "cd $d && mkdir -p pred_runs/ffres_bplr pred_runs/tests_ffres_bplr && { [ -f pred_runs/ffres_bplr/bp_s$s.eqx ] || $PY -W ignore $Q/pred_ffres.py --arm bp --depth 10 --seed $s --lr $LR --steps 6000 --save --out pred_runs/ffres_bplr/bp_s$s.json; } && $PY -W ignore $Q/run_pred_tests.py ffres --arm bp --depth 10 --seed $s --model pred_runs/ffres_bplr/bp_s$s.eqx --outdir pred_runs/tests_ffres_bplr > pred_runs/ffres_bplr/test_s$s.log 2>&1; echo \$? > $JOBDIR/bplr/${ds}_s$s.rc"
  done
done | xargs -d '\n' -P 3 -I{} bash -c '{}'
NOK=$(grep -lx 0 "$JOBDIR"/bplr/*.rc | wc -l); note "runs ok: $NOK of 9"
for ds in galaxy10 mnist cifar10; do
  d=$Q; [ $ds = galaxy10 ] || d=$Q/ds/$ds
  mkdir -p "$JOBDIR/bplr/$ds" && cp $d/pred_runs/tests_ffres_bplr/*.json $d/pred_runs/ffres_bplr/*.json "$JOBDIR/bplr/$ds/" 2>/dev/null
done
[ "$NOK" -eq 9 ] || exit 8
note "DONE"
