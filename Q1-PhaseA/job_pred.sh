#!/usr/bin/env bash
# Phase P on the GPU host: frozen-latent predictors.  bash job_pred.sh <project_root> <job_dir>
# Backbone = runs/bp_sigreg_etafix_s0 (canonical BP, host data split).  Order: gates -> latents ->
# target-space check (pre-registered rule) -> ff BP/PC grid -> SDPC validation sweep -> tests for
# copy, ff, BP set, SDPC (bp x3 seeds, rand x1).  Resumable: every step skips outputs that exist.
set -uo pipefail
P="$1"; JOBDIR="$2"; cd "$P/Q1-PhaseA" || exit 2
PY=../.venv/bin/python; export JAX_PLATFORMS=${JAX_PLATFORMS:-cuda}
ST=pred_status.txt; note() { echo "$(date +%H:%M) $*" | tee -a $ST; }
collect() { tar czf "$JOBDIR/q1_pred.tar.gz" $ST pred_target_check.json pred_runs/*.json pred_runs/tests/*.json pred_runs/*.log 2>/dev/null; }
trap collect EXIT
note "start backend $($PY -c 'import jax;print(jax.default_backend())' 2>/dev/null | tail -1)"
JAX_PLATFORMS=cpu $PY -W ignore pred_ff.py gates > pred_gates.log 2>&1 && JAX_PLATFORMS=cpu $PY -W ignore test_pred_sdpc.py >> pred_gates.log 2>&1 \
  || { note "GATES FAILED"; tail -20 pred_gates.log; exit 3; }
note "gates: $(grep -c PASS pred_gates.log) PASS"
[ -f pred_latents.npz ] || $PY -W ignore pred_data.py --run runs/bp_sigreg_etafix_s0 --out pred_latents.npz > pred_runs_data.log 2>&1 || { note "latents failed"; exit 4; }
mkdir -p pred_runs/tests pred_runs/sdpc
[ -f pred_target_check.json ] || $PY -W ignore pred_ff.py check --data pred_latents.npz > pred_runs/check.log 2>&1
note "target layer: $(python3 -c "import json;print(json.load(open('pred_target_check.json'))['choice'])")"
L=$(python3 -c "import json;print(json.load(open('pred_target_check.json'))['choice'])")
[ "$L" = "emb" ] || { note "target layer is $L, not emb: downstream scripts assume emb -- STOP for review"; exit 5; }
for bb in bp rand; do for s in 0 1 2; do
  f=pred_runs/ff_bp_${bb}_emb_s$s.json; [ -f $f ] || $PY -W ignore pred_ff.py train --data pred_latents.npz --backbone $bb --arm bp --seed $s --steps 6000 --save --out $f > /dev/null 2>&1
  f=pred_runs/ff_pcA_${bb}_emb_s$s.json; [ -f $f ] || $PY -W ignore pred_ff.py train --data pred_latents.npz --backbone $bb --arm pc --adaptive-eta --T 20 --seed $s --steps 6000 --save --out $f > /dev/null 2>&1
done; done
note "ff grid done: $(ls pred_runs/ff_*_emb_s*.json | wc -l) runs"
$PY -W ignore sdpc_sweep.py > pred_runs/sdpc_sweep.log 2>&1
$PY - <<'PYEOF' || { note "selection failed"; exit 6; }
import json
rows = json.load(open("pred_runs/sdpc_sweep.json")); assert len(rows) == 8, len(rows)
b = min(rows, key=lambda r: r["val_k1"]["nmse"])
json.dump({"lam": b["lam"], "beta": b["beta"], "lam_decay": b["lam_decay"], "T_inf": 50, "steps": 4000, "val_k1_nmse": b["val_k1"]["nmse"],
           "val_k1_nmse_range": [min(r["val_k1"]["nmse"] for r in rows), max(r["val_k1"]["nmse"] for r in rows)],
           "rule": "lowest val k=1 nmse over the 8-config sweep (pred_sdpc_prereg.json)"}, open("pred_runs/sdpc_selected.json", "w"), indent=1)
PYEOF
note "sdpc selected: $(tr -d '\n ' < pred_runs/sdpc_selected.json)"
run() { out=$1; shift; [ -f pred_runs/tests/$out.json ] && return
  $PY -W ignore run_pred_tests.py "$@" > pred_runs/test_$out.log 2>&1 && note "done $out" || note "FAILED $out"; }
run copy_bp_s0 copy --backbone bp; run copy_rand_s0 copy --backbone rand
for s in 0 1 2; do run ffbp_bp_s$s ff --backbone bp --seed $s --arm bp; run ffpcA_bp_s$s ff --backbone bp --seed $s --arm pcA; done
for s in 0 1 2; do run bpset_bp_s$s bpset --backbone bp --seed $s; done
for s in 0 1 2; do run sdpc_bp_s$s sdpc --backbone bp --seed $s; done
run bpset_rand_s0 bpset --backbone rand --seed 0; run sdpc_rand_s0 sdpc --backbone rand --seed 0
note "ALL DONE"
