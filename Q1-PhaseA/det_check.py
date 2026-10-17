import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pred_ff as PF, pred_gpc as G

# Review D5: the stage-1 seed-0 controls (commit 7468692) are reused; check the current code reproduces their first
# logged training row (same init, same batch stream) before pooling them with new seeds 1, 2.
ztr, ttr, *_ = PF.load("pred_latents.npz", "bp", "emb")
sel = json.load(open("pred_runs/gpc_selected.json"))
res = {}
for rule in ("bp", "pc"):
    for d in (3, 10):
        key = f"gpc_{rule}_d{d}_zero"
        old = json.load(open(f"pred_runs/tests/{key}_bp_s0.json"))
        o = old["train_log"][0]
        cfg = G.GCfg(rule=rule, depth=d, start="zero", c=old["c_used"], steps=4000, stop_at=1)
        _, _, log, _ = G.train(cfg, ztr, ttr, verbose=False)
        n = log[0]
        rel = {k: abs(n[k] - o[k]) / (abs(o[k]) + 1e-12) for k in ("E_init", "E", "pred_err", "gnorm")}
        res[key] = {"rel_diff": rel, "match": max(rel.values()) < 1e-3}
        print(key, json.dumps(res[key]), flush=True)
json.dump(res, open("pred_runs/c2_det_check.json", "w"), indent=1)
