#!/usr/bin/env bash
# Phase P final runs (detached).  1. select SDPC hyperparameters from the validation sweep (pre-registered rule:
# lowest val k=1 nmse); 2. save the feedforward PC models (same seeds -> same runs as ff_pcA_*.json);
# 3. every method through run_pred_tests.py.  Progress: pred_runs/final_status.txt
cd "$(dirname "$0")"; export JAX_PLATFORMS=cpu PY=../.venv/bin/python; ST=pred_runs/final_status.txt; : > $ST
note() { echo "$(date +%H:%M) $*" | tee -a $ST; }
while pgrep -f "sdpc_sweep.py" > /dev/null; do sleep 30; done
$PY - <<'PYEOF' || { note "selection failed"; exit 2; }
import json
rows = json.load(open("pred_runs/sdpc_sweep.json"))
assert len(rows) == 8, len(rows)
b = min(rows, key=lambda r: r["val_k1"]["nmse"])
sel = {"lam": b["lam"], "beta": b["beta"], "lam_decay": b["lam_decay"], "T_inf": 50, "steps": 4000,
       "val_k1_nmse": b["val_k1"]["nmse"], "rule": "lowest val k=1 nmse over the 8-config sweep (pred_sdpc_prereg.json)"}
json.dump(sel, open("pred_runs/sdpc_selected.json", "w"), indent=1); print("selected", sel)
PYEOF
note "selected: $(cat pred_runs/sdpc_selected.json | tr -d '\n ')"
for s in 0 1 2; do
  $PY -W ignore pred_ff.py train --data pred_latents.npz --backbone bp --layer emb --arm pc --adaptive-eta --seed $s \
      --T 20 --steps 6000 --save --out pred_runs/ff_pcA_bp_emb_s$s.json 2>&1 | grep TEST | sed "s/^/pcA s$s /" >> $ST
done
run() { $PY -W ignore run_pred_tests.py "$@" > pred_runs/tests_$(echo "$@" | tr ' -' '__').log 2>&1 && note "done $*" || note "FAILED $*"; }
run copy --backbone rand
for s in 0 1 2; do run ff --backbone bp --seed $s --arm bp; run ff --backbone bp --seed $s --arm pcA; done
for s in 0 1 2; do run bpset --backbone bp --seed $s; done
for s in 0 1 2; do run sdpc --backbone bp --seed $s; done
run bpset --backbone rand --seed 0; run sdpc --backbone rand --seed 0
note "ALL DONE"
