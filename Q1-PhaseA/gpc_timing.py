import sys, time, json, dataclasses; sys.path.insert(0, ".")
import jax, numpy as np, pred_ff as PF, pred_gpc as G
ztr, ttr, *_ = PF.load("pred_latents.npz", "bp", "emb")
out = {}
for prec in ("default", "highest"):
    jax.config.update("jax_default_matmul_precision", None if prec == "default" else "highest")
    for d in (3, 10):
        for r in ("pc", "bp"):
            for T in (100, 400):
                cfg = G.GCfg(rule=r, depth=d, T=T, steps=60)
                t0 = time.time(); G.train(dataclasses.replace(cfg, steps=5), ztr, ttr, verbose=False); tc = time.time() - t0
                t0 = time.time(); G.train(cfg, ztr, ttr, verbose=False, log_every=10**6); dt = (time.time() - t0) / 60
                out[f"{prec} d{d} {r} T{T}"] = round(dt * 1000, 1)
                print(prec, d, r, T, f"{dt*1000:.1f} ms/step (compile+5 steps {tc:.0f}s)", flush=True)
json.dump(out, open("gpc_timing.json", "w"), indent=1)
