import glob, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
X64 = "--x64" in sys.argv
import numpy as np, jax, jax.numpy as jnp, equinox as eqx
jax.config.update("jax_default_matmul_precision", "highest")
if X64: jax.config.update("jax_enable_x64", True)
import data as D, vit_ijepa as V, vit_epc as P, train_pred_vit as A
# Inference-only check of the standard-PC baseline (spc_u1) before training (3 reviewers, 2026-11-12): how far the error
# reaches down the predictor in T steps (per-node |eps|), per-leaf credit vs BP, float32 vs float64, and rho = 1 vs gain/4.
# python diag_spc.py <root> <out.json> [--x64]   (x64: same rows in float64, fewer images)
ROOT, OUT = sys.argv[1], sys.argv[2]; dt = jnp.float64 if X64 else jnp.float32
vc, enc, tenc, _ = A.load_encoders(os.path.join(ROOT, "Q1-PhaseA/runs_vit/v0_d6_s0"))
SA = os.path.join(ROOT, "Q1-PhaseA/runs_vit/stageA")
Xtr, _, _, _ = D.load_cifar10(root=os.path.join(ROOT, "data"), n_train=-1, n_test=2000, seed=0)
NB = 16 if X64 else 64
rng = np.random.default_rng(5); X = Xtr[40000:40000 + NB]
ci, cv, ti, tv = (jnp.asarray(x) for x in V.sample_masks(rng, NB, vc))
im = jnp.asarray(V.standardise(X)); ctx, Y = A.encode(enc, tenc, im, im, ci, cv, ti, vc)
cast = lambda t: jax.tree_util.tree_map(lambda x: x.astype(dt) if eqx.is_inexact_array(x) else x, t)
ctx, Y = cast(ctx), cast(Y)
leaves = lambda g: [np.asarray(x, np.float64).ravel() for x in jax.tree_util.tree_leaves(eqx.filter(g, eqx.is_inexact_array))]


def compare(g, gb):
    a, b = leaves(g), leaves(gb); fa, fb = np.concatenate(a), np.concatenate(b)
    nb = np.array([np.linalg.norm(y) for y in b]); na = np.array([np.linalg.norm(x) for x in a])
    lc = np.array([x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-300) for x, y in zip(a, b)])[nb > 1e-12]
    lr = (na / np.maximum(nb, 1e-300))[nb > 1e-12]
    return {"cos": float(fa @ fb / (np.linalg.norm(fa) * np.linalg.norm(fb))), "ratio": float(np.linalg.norm(fa) / np.linalg.norm(fb)),
            "leaf_cos_min": float(lc.min()), "leaf_cos_med": float(np.median(lc)), "leaf_ratio_min": float(lr.min()),
            "leaf_ratio_med": float(np.median(lr)), "frac_leaves_ratio_lt_1e-3": float(np.mean(lr < 1e-3)),
            "leaf_ratio_first8": [float(x) for x in lr[:8]], "leaf_ratio_last8": [float(x) for x in lr[-8:]]}


def node_profile(pred, rho, T, n=8):             # per-node |eps_j| / |eps_top| for n (image, block) pairs; node 0 = first
    prof, er = [], []
    for b in range(n):
        h, u, en, r = P.spc_one(pred, ctx[b], ci[b], cv[b], ti[b, 0], tv[b, 0], Y[b, 0], jnp.asarray(rho, dt), 0.0, T)
        nr = np.asarray(jnp.sqrt(jnp.sum(r ** 2, axis=(1, 2))), np.float64); prof.append(nr / max(nr[-1], 1e-300))
        er.append(float(en[1] / en[0]))
    p = np.median(np.stack(prof), 0)
    return {"eps_rel_by_node": [float(x) for x in p], "nodes_eps_rel_gt_1e-3": int(np.sum(p > 1e-3)), "E_end_over_E_start_med": float(np.median(er))}


CK = [("init_d10", 10, None)] + [(f"bp_d{d}", d, f"{SA}/bp_d{d}_lr3e-3_e25_full_s0/predictor.eqx") for d in (6, 10)]
res = {"x64": X64, "n_images": NB, "rows": []}
for name, depth, path in CK:
    a = A.ACfg(arm="spc_u1", depth=depth); pred = A.make_predictor(vc, a)
    if path: pred = eqx.tree_deserialise_leaves(path, pred)
    pred = cast(pred)
    gb = eqx.filter_jit(lambda q: P.grads(q, ctx, ci, cv, ti, tv, Y, "bp", chunk=16)[0])(pred)
    gm = float(np.median(np.asarray(jax.vmap(lambda a1, c1, v1, t1, w1: P.gain(pred, a1, c1, v1, t1[0], w1[0]))(ctx[:8], ci[:8], cv[:8], ti[:8], tv[:8]))))
    # accuracy of the step-size estimate vs power iterations (cost: each iteration is one Hessian-vector product)
    for rho in (1.0, max(1.0, gm / 4)):
        lam = {}
        for it in (4, 8, 20, 100):
            E = lambda h, b=0: P.local_energy(pred, ctx[b], ci[b], cv[b], ti[b, 0], tv[b, 0], h, jnp.zeros_like(h), Y[b, 0], jnp.asarray(rho, dt))
            h0 = P.feedforward_states(pred, ctx[0], ci[0], cv[0], ti[0, 0], tv[0, 0])
            lam[it] = float(P._hvp_max(E, h0, it))
        res.setdefault("lam_vs_iters", []).append({"ckpt": name, "rho": rho, **{str(k): v for k, v in lam.items()}})
        print("LAM", name, rho, {k: round(v, 3) for k, v in lam.items()}, flush=True)
    for rule, rho in (("rho1", 1.0), ("gain/4", max(1.0, gm / 4))):
        for T in ((20, 80) if X64 else (20, 80, 320)):
            t0 = time.time()
            g = eqx.filter_jit(lambda q, r: P.grads(q, ctx, ci, cv, ti, tv, Y, "spc", rho=r, alpha=0.0, T=T, chunk=16)[0])(pred, jnp.asarray(rho, dt))
            row = {"ckpt": name, "depth": depth, "n_nodes": 2 * depth, "gain_med": gm, "rule": rule, "rho": rho, "T": T, **compare(g, gb), **node_profile(pred, rho, T)}
            row["sec"] = round(time.time() - t0, 1); res["rows"].append(row)
            print("ROW", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k not in ("eps_rel_by_node", "leaf_ratio_first8", "leaf_ratio_last8")}), flush=True)
            json.dump(res, open(OUT, "w"), indent=1)
print("DONE", len(res["rows"]))
