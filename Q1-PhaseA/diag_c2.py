import itertools, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataclasses as dc
import numpy as np, jax, jax.numpy as jnp
import pred_gpc as G, test_pred_gpc as TG

# C2 pre-launch diagnostic (review findings F1/N2/N1/D1), never feeds training.
# Production pipeline: s0 = 0, alm_infer T=400 on k seen views, phase 2 over all views for T2 steps,
# credit cosine vs the BP auto-decoder gradient at the SAME s (seen / unseen / all views),
# belief distance to a settled auto-decoder belief, residual growth from T=400 to T=2000.
OUT = sys.argv[1] if len(sys.argv) > 1 else "pred_runs/diag_c2.json"
nrm = lambda L: float(jnp.sqrt(sum(jnp.sum(x ** 2) for x in L)))
B, K = 64, 4
rng = np.random.default_rng(0)
rows = []
t0 = time.time()
for depth, scale in itertools.product((3, 10), (1.0, 2.0, 4.0)):
    base = G.GCfg(depth=depth, rule="alm", alpha=1.0, T=400)
    p = jax.tree_util.tree_map(lambda x: scale * x, G.init(jax.random.PRNGKey(depth), base))
    Z, T9, s_true = TG.planted(p, base, b=B, k=K)
    Ph = G.phi(T9); H = base.H; ii = jnp.arange(B)[:, None]
    for k in (1, 3):
        perm = np.stack([rng.permutation(K) for _ in range(B)])
        seen, uns = jnp.asarray(perm[:, :k]), jnp.asarray(perm[:, k:])
        Zs, Ps = Z[ii, seen], Ph[ii, seen]
        s0 = jnp.zeros((B, G.M))
        s_ad = G.ad_infer(p, s0, Zs, Ps, dc.replace(base, rule="bp", c=1.0, T=2000))[0]
        for rho, c in itertools.product((1.0, 4.0, 16.0, 64.0), (0.5, 1.0)):
            cfg = dc.replace(base, hid_prec=rho, c=c)
            s, xs, lam, hist, _ = G.alm_infer(p, s0, Zs, Ps, cfg)
            if not bool(jnp.all(jnp.isfinite(s))):
                rows.append({"depth": depth, "scale": scale, "k": k, "rho": rho, "c": c, "finite": False}); continue
            res400 = nrm(G.hid_residuals(p, s, xs, Ps, H)) / nrm(xs)
            s2, x2, l2, _, _ = G.alm_infer(p, s0, Zs, Ps, dc.replace(cfg, T=2000))
            fin2 = bool(jnp.all(jnp.isfinite(s2)))
            res2000 = nrm(G.hid_residuals(p, s2, x2, Ps, H)) / nrm(x2) if fin2 else float("nan")
            row = {"depth": depth, "scale": scale, "k": k, "rho": rho, "c": c, "finite": True, "finite_T2000": fin2,
                   "res_rel_T400": res400, "res_rel_T2000": res2000, "lam_rel_T400": nrm(lam) / nrm(xs),
                   "s_dist_settled_ad": float(jnp.mean(jnp.linalg.norm(s - s_ad, axis=-1) / (jnp.linalg.norm(s_ad, axis=-1) + 1e-8))),
                   "s_dist_settled_ad_T2000": float(jnp.mean(jnp.linalg.norm(s2 - s_ad, axis=-1) / (jnp.linalg.norm(s_ad, axis=-1) + 1e-8))) if fin2 else None}
            for T2 in (100, 400, 1600):
                cc = dc.replace(cfg, T2=T2)
                xs0 = [x.at[ii, seen].set(xv) for x, xv in zip(G.topdown(p, s, Ph, H)[0], xs)]
                lam0 = [jnp.zeros_like(x).at[ii, seen].set(l) for x, l in zip(xs0, lam)]
                _, xf, lf, _, _ = G.alm_infer(p, s, Z, Ph, cc, T=T2, freeze_s=True, xs0=xs0, lam0=lam0)
                for name, ix in (("seen", seen), ("unseen", uns), ("all", None)):
                    if ix is None:
                        g = G.alm_wgrads(p, s, xf, lf, Z, Ph, cc); gb = G.ad_wgrads(p, s, Z, Ph, cc)
                    else:
                        sel = lambda a: a[ii, ix]
                        g = G.alm_wgrads(p, s, [sel(x) for x in xf], [sel(l) for l in lf], sel(Z), sel(Ph), cc)
                        gb = G.ad_wgrads(p, s, sel(Z), sel(Ph), cc)
                    row[f"cos_{name}_T2_{T2}"] = TG.cos(g, gb)
                    if name == "all":
                        row[f"ratio_all_T2_{T2}"] = float(jnp.linalg.norm(TG.flat(g)) / jnp.linalg.norm(TG.flat(gb)))
            rows.append(row)
            print(json.dumps({k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in row.items()}), f"{time.time() - t0:.0f}s", flush=True)
            G.dump_json(rows, OUT)
G.dump_json(rows, OUT)
print("DONE", len(rows))
