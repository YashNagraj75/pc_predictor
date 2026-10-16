#!/usr/bin/env bash
# PC-ALM feedforward predictor on frozen latents: is credit assignment what holds PC back?
# Same network, data, optimiser, steps and tests as the existing ff BP / ff plain-PC arms (findings/20);
# the only change is the weight gradient (pred_ff.alm_grads).  Spec: wiki 'Method - PC-ALM predictor arm'.
#   1. gates (CPU)
#   2. selection: alpha in {0.1, 0.25, 1.0} x T in {20, 100}, seed 0, 6000 steps, train on 90% of the train
#      images and score on the other 10% (never the test set); lowest held-out nmse wins
#   3. selected config at seeds 0,1,2 on the full train set (--save), plus plain PC at the SAME T if T != 20
#      (the existing plain-PC arm used T = 20), so the comparison is matched in inference budget
#   4. full pred tests (k = 1 clean, robustness) for every saved model
#   bash job_alm_pred.sh <project_root> <job_dir>
set -uo pipefail
P="$1"; JOBDIR="$2"; cd "$P/Q1-PhaseA" || exit 2
PY=../.venv/bin/python; export JAX_PLATFORMS=${JAX_PLATFORMS:-cuda}
ST=alm_pred_status.txt; note() { echo "$(date +%m-%d_%H:%M) $*" | tee -a $ST; }
collect() { tar czf "$JOBDIR/q1_alm_pred.tar.gz" $ST alm_pred_gates.log pred_runs/alm_sel pred_runs/alm_selected.json \
            pred_runs/ff_alm_bp_emb_s*.json pred_runs/ff_pcT*_bp_emb_s*.json pred_runs/tests/ffalm_*.json \
            pred_runs/tests/ffpcT*.json pred_runs/test_ffalm*.log 2>/dev/null; }
trap collect EXIT
note "start at $(git rev-parse --short HEAD) backend $($PY -c 'import jax;print(jax.default_backend())' 2>/dev/null | tail -1)"
[ -f pred_latents.npz ] || { note "pred_latents.npz missing"; exit 4; }
JAX_PLATFORMS=cpu $PY -W ignore pred_ff.py gates > alm_pred_gates.log 2>&1 || { note "GATES FAILED"; tail -20 alm_pred_gates.log; exit 3; }
note "gates: $(grep -c PASS alm_pred_gates.log) PASS"

mkdir -p pred_runs/alm_sel pred_runs/tests
COMMON="--data pred_latents.npz --backbone bp --adaptive-eta --steps 6000"
for T in 20 100; do for A in 0.1 0.25 1.0; do
  f=pred_runs/alm_sel/alm_T${T}_a${A}_s0.json
  [ -f $f ] || $PY -W ignore pred_ff.py train $COMMON --arm alm --alpha $A --T $T --seed 0 --holdout --out $f > ${f%.json}.log 2>&1 \
    || note "selection run T=$T alpha=$A FAILED"
done; done
$PY - <<'PYEOF' || { note "selection failed"; exit 5; }
import json, glob, math
rows = []
for f in sorted(glob.glob("pred_runs/alm_sel/alm_T*_a*_s0.json")):
    d = json.load(open(f)); n = d["test"]["nmse"]
    if math.isfinite(n):
        rows.append({"T": d["cfg"]["T"], "alpha": d["cfg"]["alpha"], "val_nmse": n,
                     "cos_end": d["log"][-1].get("cos_pc_bp"), "ratio_end": d["log"][-1].get("ratio_pc_bp")})
assert len(rows) >= 1, "no finite selection run"
b = min(rows, key=lambda r: r["val_nmse"])
json.dump({"T": b["T"], "alpha": b["alpha"], "val_nmse": b["val_nmse"], "all": rows,
           "rule": "lowest held-out (last 10% of train images) k=1 nmse over alpha {0.1,0.25,1.0} x T {20,100}, seed 0"},
          open("pred_runs/alm_selected.json", "w"), indent=1)
print(b)
PYEOF
T=$(python3 -c "import json;print(json.load(open('pred_runs/alm_selected.json'))['T'])")
A=$(python3 -c "import json;print(json.load(open('pred_runs/alm_selected.json'))['alpha'])")
note "selected T=$T alpha=$A"

run() { out=$1; shift; [ -f pred_runs/tests/$out.json ] && return
  $PY -W ignore run_pred_tests.py "$@" > pred_runs/test_$out.log 2>&1 && note "tested $out" || note "TEST FAILED $out"; }
for s in 0 1 2; do
  f=pred_runs/ff_alm_bp_emb_s$s.json
  [ -f $f ] || $PY -W ignore pred_ff.py train $COMMON --arm alm --alpha $A --T $T --seed $s --save --out $f > ${f%.json}.log 2>&1 \
    && note "trained alm s$s" || note "TRAIN FAILED alm s$s"
  run ffalm_bp_s$s ff --backbone bp --seed $s --arm alm
  if [ "$T" != "20" ]; then
    f=pred_runs/ff_pcT${T}_bp_emb_s$s.json
    [ -f $f ] || $PY -W ignore pred_ff.py train $COMMON --arm pc --T $T --seed $s --save --out $f > ${f%.json}.log 2>&1 \
      && note "trained plain PC T=$T s$s" || note "TRAIN FAILED pcT$T s$s"
    run ffpcT${T}_bp_s$s ff --backbone bp --seed $s --arm pcT$T
  fi
done
note "ALL DONE"
