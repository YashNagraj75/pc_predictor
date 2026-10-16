import argparse, glob, json, os, sys, time; sys.path.insert(0, ".")
import numpy as np, pred_ff as PF, pred_eval as PE, pred_gpc as G
ap = argparse.ArgumentParser(); ap.add_argument("--arm"); ap.add_argument("--stop", type=int, default=800)
ap.add_argument("--collect", action="store_true"); ap.add_argument("--dir", default="pred_runs/gpc_sweep")
a = ap.parse_args(); os.makedirs(a.dir, exist_ok=True)
if a.collect:
    sel = {}
    for f in sorted(glob.glob(f"{a.dir}/*.json")):
        rows = json.load(open(f)); arm = rows[0]["arm"]
        ok = [r for r in rows if not r["diverged"] and not r["exploded"]]
        if not ok:
            sel[arm] = {"c": None, "note": "every c diverged"}; continue
        b = min(ok, key=lambda r: r["val_k1_nmse"])
        sel[arm] = {"c": b["c"], "val_k1_nmse": b["val_k1_nmse"], "edge": b["c"] in (min(G.C_GRID), max(G.C_GRID)),
                    "all": {r["c"]: r.get("val_k1_nmse") for r in rows}}
    G.dump_json(sel, "pred_runs/gpc_selected.json"); print(json.dumps(sel, indent=1)); sys.exit(0)
_, rule, d, start = a.arm.split("_"); d = int(d[1:])
ztr, ttr, *_ = PF.load("pred_latents.npz", "bp", "emb")
nv = int(0.9 * len(ztr)); zv, tv = ztr[nv:], ttr[nv:]
seen, tgt = PE.k_cases(len(zv), 1)
out = f"{a.dir}/{a.arm}.json"
rows = json.load(open(out)) if os.path.exists(out) else []
for c in G.C_GRID:
    if any(r["c"] == c for r in rows): continue
    cfg = G.GCfg(rule=rule, depth=d, start=start, c=c, steps=4000, stop_at=a.stop)
    t0 = time.time(); p, am, log, div = G.train(cfg, ztr[:nv], ttr[:nv], verbose=False)
    ex = G.exploded(log)
    val = None if (div or ex) else PE.run_cases(G.make_predict(p, am, cfg), zv, tv, seen, tgt)["nmse"]
    rows.append({"arm": a.arm, "c": c, "diverged": div, "exploded": ex, "val_k1_nmse": val, "last": log[-1], "s": time.time() - t0})
    print(json.dumps(G._clean({"arm": a.arm, "c": c, "div": div, "exploded": ex, "val": val, "s": round(time.time() - t0)})), flush=True)
    G.dump_json(rows, out)
