#!/usr/bin/env bash
# Rerun of the residual feedforward PC-ALM predictor at depth 10 with the GPU numerics fix (E1 trained on wrong
# compiled gradients, findings/27).  User decision 2026-10-25: pre-registered selection + a separate rho = 16 arm.
#   A  (pre-registered): alpha {0.25, 1} x T {20, 100} x lr {1e-3, 3e-3, 1e-2}, rho 1, on the 90/10 holdout; 3 seeds of the winner
#   R16 (new arm, from the credit check): rho 16, alpha 1, T 400, lr {1e-3, 3e-3, 1e-2} on the holdout; 3 seeds of the winner
# No BP is trained (frozen-reference rule); the old residual BP reference is invalid, so no same-network BP comparison.
#   bash job_ffres_alm_fix.sh <project_root> <job_dir>
set -uo pipefail
P="$1"; JOBDIR=$(readlink -f "$2"); cd "$P/Q1-PhaseA" || exit 2
PY=../.venv/bin/python
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.12
export NVIDIA_TF32_OVERRIDE=0 XLA_FLAGS=--xla_gpu_autotune_level=0
PAR=${PAR:-2}; ST=ffresfix_status.txt; FS=pred_runs/ffres_fix_sel; FR=pred_runs/ffres_fix; TF=pred_runs/tests_ffres_fix; LG=pred_runs/logs_ffresfix
note() { echo "$(date +%m-%d_%H:%M) $*" | tee -a $ST; }
collect() { tar czf "$JOBDIR/q1_ffresfix.tar.gz" --exclude='*.eqx' $ST $FS $FR $TF $LG ffresfix_audit.json ffresfix_gates.log 2>/dev/null || true; }
trap collect EXIT
mkdir -p $FS $FR $TF $LG
note "start at $(git rev-parse --short HEAD), env TF32=$NVIDIA_TF32_OVERRIDE XLA_FLAGS=$XLA_FLAGS"
$PY -W ignore test_ffres.py 10 > ffresfix_gates.log 2>&1 || { note "GATES FAILED"; grep -h FAIL ffresfix_gates.log | head; exit 4; }
$PY -W ignore audit_jit.py ffresfix_audit.json ffres > $LG/audit.log 2>&1 || { note "AUDIT FAILED TO RUN"; tail -5 $LG/audit.log; exit 4; }
python3 -c "
import json,sys; c=json.load(open('ffresfix_audit.json'))['cases']; print({k:round(v.get('cos',-1),5) for k,v in c.items()})
sys.exit(1 if any('error' in v or v['cos']<0.999 for v in c.values()) else 0)" >> $ST || { note "AUDIT GATE FAILED"; exit 4; }
note "gates and audit passed"
ff() { echo "$PY -W ignore pred_ffres.py $1 > $LG/$(basename $2 .json).log 2>&1; rc=\$?; echo \$(date +%m-%d_%H:%M) rc=\$rc $2 >> $ST"; }
{ for lr in 0.001 0.003 0.01; do
    for T in 20 100; do for A in 0.25 1.0; do o=$FS/A_T${T}_a${A}_lr$lr.json; [ -f $o ] || ff "--arm alm --depth 10 --T $T --alpha $A --lr $lr --holdout --out $o" $o; done; done
    o=$FS/R16_lr$lr.json; [ -f $o ] || ff "--arm alm --depth 10 --T 400 --alpha 1.0 --rho 16 --lr $lr --holdout --out $o" $o
  done; } > $LG/sel.txt
xargs -d '\n' -P $PAR -I{} bash -c '{}' < $LG/sel.txt
python3 - <<'PYEOF' || { note "SELECTION FAILED"; exit 6; }
import json, glob
out = {}
for arm in ("A", "R16"):
    rows = [json.load(open(f)) for f in glob.glob(f"pred_runs/ffres_fix_sel/{arm}_*.json")]
    tab = [{"T": r["cfg"]["T"], "alpha": r["cfg"]["alpha"], "lr": r["cfg"]["lr"], "rho": r["cfg"].get("rho", 1.0), "val_nmse": r["test"]["nmse"], "finite": r["finite"]} for r in rows]
    ok = [t for t in tab if t["finite"]]
    b = min(ok, key=lambda t: t["val_nmse"])
    out[arm] = {**{k: b[k] for k in ("T", "alpha", "lr", "rho", "val_nmse")}, "edge_lr": b["lr"] in (0.001, 0.01), "all": tab}
json.dump(out, open("pred_runs/ffres_fix_sel/selected.json", "w"), indent=1)
print({k: (v["T"], v["alpha"], v["lr"], v["rho"], round(v["val_nmse"], 4)) for k, v in out.items()})
PYEOF
note "selected: $(python3 -c "import json;d=json.load(open('$FS/selected.json'));print({k:(v['T'],v['alpha'],v['lr'],v['rho'],round(v['val_nmse'],4),v['edge_lr']) for k,v in d.items()})")"
jg() { python3 -c "import json;print(json.load(open('$FS/selected.json'))$1)"; }
{ for arm in A R16; do T=$(jg "['$arm']['T']"); A=$(jg "['$arm']['alpha']"); L=$(jg "['$arm']['lr']"); RH=$(jg "['$arm']['rho']")
    for s in 0 1 2; do o=$FR/ffres_alm${arm}_d10_s$s.json; [ -f $o ] || ff "--arm alm --depth 10 --seed $s --T $T --alpha $A --lr $L --rho $RH --save --out $o" $o; done; done; } > $LG/runs.txt
xargs -d '\n' -P $PAR -I{} bash -c '{}' < $LG/runs.txt
for arm in A R16; do for s in 0 1 2; do m=$FR/ffres_alm${arm}_d10_s$s.eqx
  [ -f $m ] && $PY -W ignore run_pred_tests.py ffres --arm alm --depth 10 --seed $s --model $m --outdir $TF/$arm > $LG/test_${arm}_s$s.log 2>&1 || note "TEST FAILED $arm s$s"
done; done
note "ALL DONE: $(ls $FR/*.json 2>/dev/null | wc -l) trained, $(find $TF -name '*.json' | wc -l) tested"
exit 0
