"""Validation sweep for the SDPC predictor (bp backbone, seed 0). Selection: lowest val k=1 nmse."""
import sys, json, itertools, time; sys.path.insert(0, ".")
import numpy as np, pred_ff as PF, pred_sdpc as S, pred_eval as PE
ztr, ttr, *_ = PF.load("pred_latents.npz", "bp", "emb")
nv = int(0.9 * len(ztr)); zv, tv = ztr[nv:], ttr[nv:]
seen1, tgt1 = PE.k_cases(len(zv), 1); seen2, tgt2 = PE.k_cases(len(zv), 2)
import os
rows = json.load(open('pred_runs/sdpc_sweep.json')) if os.path.exists('pred_runs/sdpc_sweep.json') else []
limit = int(sys.argv[1]) if len(sys.argv) > 1 else 99
for lam, beta, dec in itertools.product((0.01, 0.05), (0.3, 1.0), (1.0, 0.3)):
    if any((r['lam'], r['beta'], r['lam_decay']) == (lam, beta, dec) for r in rows): continue
    if limit == 0: break
    limit -= 1
    cfg = S.SCfg(lam=lam, beta=beta, lam_decay=dec, T_inf=50, steps=1500)
    t0 = time.time(); Ds, log = S.train(cfg, ztr[:nv], ttr[:nv], verbose=False)
    pr = S.make_predict(Ds, cfg)
    r = {"lam": lam, "beta": beta, "lam_decay": dec, "val_k1": PE.run_cases(pr, zv, tv, seen1, tgt1),
         "val_k2": PE.run_cases(pr, zv, tv, seen2, tgt2), "active": log[-1]["active"], "monotone": log[-1]["monotone"], "s": time.time() - t0}
    rows.append(r); print(json.dumps(r), flush=True)
    json.dump(rows, open("pred_runs/sdpc_sweep.json", "w"), indent=1)
