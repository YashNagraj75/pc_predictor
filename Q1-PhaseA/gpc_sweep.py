import argparse, glob, json, os, sys, time; sys.path.insert(0, ".")
import numpy as np, pred_ff as PF, pred_eval as PE, pred_gpc as G
ap = argparse.ArgumentParser(); ap.add_argument("--arm"); ap.add_argument("--stop", type=int, default=800)
ap.add_argument("--collect", action="store_true"); ap.add_argument("--dir", default="pred_runs/gpc_sweep")
ap.add_argument("--out", default="pred_runs/gpc_selected.json")
a = ap.parse_args(); os.makedirs(a.dir, exist_ok=True)
if a.collect:
    sel = {}
    for f in sorted(glob.glob(f"{a.dir}/*.json")):
        rows = json.load(open(f)); arm = rows[0]["arm"]
        grid = [(r_, c_) for r_ in G.RHO_GRID for c_ in G.C_ALM] if "rho" in rows[0] else [(None, c_) for c_ in G.C_GRID]
        have = {(r.get("rho"), r["c"]) for r in rows}
        miss = [g for g in grid if g not in have]
        if miss:
            print(f"INCOMPLETE GRID for {arm}: missing {miss}", flush=True); sys.exit(1)
        ok = [r for r in rows if not r["diverged"] and not r["exploded"] and r.get("val_k1_nmse") is not None
              and np.isfinite(r["val_k1_nmse"])]
        print(arm, "excluded:", [(r.get("rho"), r["c"]) for r in rows if r not in ok], flush=True)
        if not ok:
            print(f"NO VALID CELL for {arm}", flush=True); sys.exit(1)
        b = min(ok, key=lambda r: r["val_k1_nmse"])
        if "rho" in rows[0]:            # alm: joint (rho, c) choice; best c per rho kept for the rho = 1 ablation
            best_c = {}
            for rho in G.RHO_GRID:
                cand = [r for r in ok if r["rho"] == rho]
                if cand: best_c[str(rho)] = min(cand, key=lambda r: r["val_k1_nmse"])["c"]
            sel[arm] = {"rho": b["rho"], "c": b["c"], "val_k1_nmse": b["val_k1_nmse"],
                        "edge_c": b["c"] in (min(G.C_ALM), max(G.C_ALM)), "edge_rho": b["rho"] in (min(G.RHO_GRID), max(G.RHO_GRID)),
                        "best_c_by_rho": best_c, "all": {f"{r['rho']}_{r['c']}": r.get("val_k1_nmse") for r in rows}}
        else:
            sel[arm] = {"c": b["c"], "val_k1_nmse": b["val_k1_nmse"], "edge": b["c"] in (min(G.C_GRID), max(G.C_GRID)),
                        "all": {r["c"]: r.get("val_k1_nmse") for r in rows}}
    G.dump_json(sel, a.out); print(json.dumps(sel, indent=1)); sys.exit(0)
_, rule, d, start = a.arm.split("_"); d = int(d[1:])
import subprocess
COMMIT = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
ztr, ttr, *_ = PF.load("pred_latents.npz", "bp", "emb")
nv = int(0.9 * len(ztr)); zv, tv = ztr[nv:], ttr[nv:]
seen, tgt = PE.k_cases(len(zv), 1)
out = f"{a.dir}/{a.arm}.json"
rows = json.load(open(out)) if os.path.exists(out) else []
grid = [(rho, c) for rho in G.RHO_GRID for c in G.C_ALM] if rule == "alm" else [(None, c) for c in G.C_GRID]
for rho, c in grid:
    if any(r["c"] == c and r.get("rho") == rho and r.get("stop") == a.stop and r.get("commit") == COMMIT for r in rows): continue
    rows = [r for r in rows if not (r["c"] == c and r.get("rho") == rho)]
    extra = {"hid_prec": rho, "alpha": 1.0} if rule == "alm" else {}
    cfg = G.GCfg(rule=rule, depth=d, start=start, c=c, steps=4000, stop_at=a.stop, **extra)
    t0 = time.time(); p, am, log, div = G.train(cfg, ztr[:nv], ttr[:nv], verbose=False)
    ex = G.exploded(log)
    val = None if (div or ex) else PE.run_cases(G.make_predict(p, am, cfg), zv, tv, seen, tgt)["nmse"]
    if val is not None and not np.isfinite(val):
        val = None
    rows.append({"arm": a.arm, "c": c, "stop": a.stop, "steps": 4000, "commit": COMMIT, **({"rho": rho} if rho is not None else {}), "diverged": div, "exploded": ex, "val_k1_nmse": val, "last": log[-1], "s": time.time() - t0})
    print(json.dumps(G._clean({"arm": a.arm, "rho": rho, "c": c, "div": div, "exploded": ex, "val": val, "s": round(time.time() - t0)})), flush=True)
    G.dump_json(rows, out)
