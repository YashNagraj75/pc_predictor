#!/usr/bin/env bash
# Retrain the frozen residual feedforward BP reference at depth 10 with the GPU numerics fix ONLY (findings/27).
# Same as E1: standard parameterisation, lr 1e-3 (no tuning: user rule), 6000 steps, seeds 0-2. No PC changes.
# Frozen afterwards in pred_runs/frozen_bp_fix (MANIFEST with sha256); the old faulty frozen_bp/ffres_bp_d10_* is kept, not used.
#   bash job_ffres_bp_fix.sh <project_root> <job_dir>
set -uo pipefail
P="$1"; JOBDIR=$(readlink -f "$2"); cd "$P/Q1-PhaseA" || exit 2
PY=../.venv/bin/python
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
export NVIDIA_TF32_OVERRIDE=0 XLA_FLAGS=--xla_gpu_autotune_level=0
ST=bpfix_status.txt; FZ=pred_runs/frozen_bp_fix; TF=pred_runs/tests_ffres_bpfix; LG=pred_runs/logs_bpfix
note() { echo "$(date +%m-%d_%H:%M) $*" | tee -a $ST; }
collect() { tar czf "$JOBDIR/q1_bpfix.tar.gz" --exclude='*.eqx' $ST $FZ $TF $LG bpfix_audit.json bpfix_gates.log 2>/dev/null || true; }
trap collect EXIT
mkdir -p $FZ $TF $LG
note "start at $(git rev-parse --short HEAD), env TF32=$NVIDIA_TF32_OVERRIDE XLA_FLAGS=$XLA_FLAGS"
$PY -W ignore test_ffres.py 10 > bpfix_gates.log 2>&1 || { note "GATES FAILED"; grep -h FAIL bpfix_gates.log | head; exit 4; }
$PY -W ignore audit_jit.py bpfix_audit.json ffres > $LG/audit.log 2>&1 || { note "AUDIT FAILED TO RUN"; exit 4; }
python3 -c "
import json,sys; c=json.load(open('bpfix_audit.json'))['cases']; print({k:round(v.get('cos',-1),5) for k,v in c.items()})
sys.exit(1 if any('error' in v or v['cos']<0.999 for v in c.values()) else 0)" >> $ST || { note "AUDIT GATE FAILED"; exit 4; }
note "gates and audit passed"
for s in 0 1 2; do
  o=$FZ/ffres_bp_d10_s$s.json
  ( [ -f $o ] || $PY -W ignore pred_ffres.py --arm bp --depth 10 --seed $s --lr 0.001 --save --out $o > $LG/bp_d10_s$s.log 2>&1; echo "$(date +%m-%d_%H:%M) rc=$? bp_d10_s$s" >> $ST ) &
done; wait
for s in 0 1 2; do m=$FZ/ffres_bp_d10_s$s.eqx
  [ -f $m ] && $PY -W ignore run_pred_tests.py ffres --arm bp --depth 10 --seed $s --model $m --outdir $TF > $LG/test_bp_d10_s$s.log 2>&1 || note "TEST FAILED s$s"
done
python3 - <<'PYEOF'
import json, glob, hashlib
m = {}
for j in sorted(glob.glob("pred_runs/frozen_bp_fix/ffres_bp_d10_s?.json")):
    r = json.load(open(j)); e = j.replace(".json", ".eqx")
    m[j] = {"eqx_sha256": hashlib.sha256(open(e, "rb").read()).hexdigest(), "commit": r.get("commit"), "lr": r["cfg"]["lr"], "test_nmse": r["test"]["nmse"]}
json.dump(m, open("pred_runs/frozen_bp_fix/MANIFEST_ffres_bp_d10.json", "w"), indent=1); print(len(m), "frozen")
PYEOF
note "ALL DONE: $(ls $FZ/*.eqx 2>/dev/null | wc -l) trained, $(ls $TF/*.json 2>/dev/null | wc -l) tested"
exit 0
