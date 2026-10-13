"""Per-arm step-scale selection for the generative PC predictor (spec: 'Method - Generative PC predictor plan').
c in {0.5,1,2,4,8}; 1500 steps on the first 90% of train images, seed 0; lowest val k=1 nmse; diverged excluded.
    python gpc_sweep.py [--steps 1500] [--arms gpc_pc_d3_zero ...]
"""
import argparse, json, os, sys, time; sys.path.insert(0, ".")
import numpy as np, pred_ff as PF, pred_eval as PE, pred_gpc as G
ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=1500); ap.add_argument("--arms", nargs="*")
ap.add_argument("--cs", nargs="*", type=float, default=[0.5, 1, 2, 4, 8]); ap.add_argument("--out", default="pred_runs/gpc_sweep.json")
a = ap.parse_args()
ztr, ttr, *_ = PF.load("pred_latents.npz", "bp", "emb")
nv = int(0.9 * len(ztr)); zv, tv = ztr[nv:], ttr[nv:]
seen, tgt = PE.k_cases(len(zv), 1)
rows = json.load(open(a.out)) if os.path.exists(a.out) else []
arms = [(r, d, st) for d in (3, 10) for r in ("pc", "bp") for st in ("zero", "amort")]
for r, d, st in arms:
    name = f"gpc_{r}_d{d}_{st}"
    if a.arms and name not in a.arms: continue
    for c in a.cs:
        if any(x["arm"] == name and x["c"] == c for x in rows): continue
        cfg = G.GCfg(rule=r, depth=d, start=st, c=c, steps=a.steps)
        t0 = time.time(); p, am, log = G.train(cfg, ztr[:nv], ttr[:nv], verbose=False)
        div = any(x.get("diverged") for x in log) or not np.isfinite(log[-1]["E"])
        val = None if div else PE.run_cases(G.make_predict(p, am, cfg), zv, tv, seen, tgt)
        rows.append({"arm": name, "c": c, "diverged": bool(div), "val_k1": val, "last": log[-1], "s": time.time() - t0})
        print(json.dumps({"arm": name, "c": c, "div": div, "val_nmse": None if val is None else round(val["nmse"], 4),
                          "monotone": log[-1].get("monotone"), "s": round(time.time() - t0)}), flush=True)
        json.dump(rows, open(a.out, "w"), indent=1)
sel = {}
for r, d, st in arms:
    name = f"gpc_{r}_d{d}_{st}"; ok = [x for x in rows if x["arm"] == name and not x["diverged"]]
    if ok:
        b = min(ok, key=lambda x: x["val_k1"]["nmse"]); sel[name] = {"c": b["c"], "val_k1_nmse": b["val_k1"]["nmse"]}
json.dump(sel, open(a.out.replace("sweep", "selected"), "w"), indent=1); print("selected", json.dumps(sel))
