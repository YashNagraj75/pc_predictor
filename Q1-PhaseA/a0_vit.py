import dataclasses as dc, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp, equinox as eqx
jax.config.update("jax_default_matmul_precision", "highest")
import data as D, vit_ijepa as V, vit_epc as P, train_pred_vit as A

# A0 (plan_stageA_scoped.md v3): on a depth-10 predictor trained 1 epoch by BP on the frozen V0 features, measure credit
# (cos to BP) and step time of each ePC arm at T in {10, 20} x power iterations in {4, 12}.  Pre-set rule: the cheapest
# setting whose credit is within 0.005 cos of the (T 20, 12 iterations) value for EVERY arm is used for all ePC arms.
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
# rev 2 (reviews): BP predictor at 2 epochs; credit over 3 mask draws x 64 images; time = median of 3 calls; criterion on
# overall cos (within 0.005) AND weakest-leaf cos (within 0.01) of the T 20 / 12-iteration value; epc_k16 T 100 reference.
a = A.ACfg(arm="bp", depth=10, lr=1e-3, epochs=2, split="val")
A.run(a, os.path.join(OUT, "bp_2epoch"))
vc, enc, tenc, _ = A.load_encoders()
pred = eqx.tree_deserialise_leaves(os.path.join(OUT, "bp_2epoch", "predictor.eqx"), A.make_predictor(vc, a))
Xtr, _, _, _ = D.load_cifar10(n_train=-1, n_test=2000, seed=0)
batches = []
for draw in range(3):
    rng = np.random.default_rng(100 + draw); X = Xtr[45000 + 64 * draw:45064 + 64 * draw]
    ci, cv, ti, tv = (jnp.asarray(x) for x in V.sample_masks(rng, 64, vc))
    imgs = jnp.asarray(V.standardise(X)); ctx, Y = A.encode(enc, tenc, imgs, imgs, ci, cv, ti, vc); batches.append((ctx, ci, cv, ti, tv, Y))
gs = [tuple(float(x) for x in A.gain_stats(pred, *b[:5])) for b in batches]; gmed = float(np.median([g[1] for g in gs]))
print("gain (min, med, max) per draw", gs, flush=True)
rng = np.random.default_rng(7); X = Xtr[45000:45256]
ci, cv, ti, tv = (jnp.asarray(x) for x in V.sample_masks(rng, 256, vc))
imgs = jnp.asarray(V.standardise(X)); ctxT, YT = A.encode(enc, tenc, imgs, imgs, ci, cv, ti, vc)
def timed(f, *args):
    """Times a function call, warming up once, and returns the median runtime over three trials."""
    jax.block_until_ready(f(*args)); ts = []
    for _ in range(3):
        t0 = time.time(); jax.block_until_ready(f(*args)); ts.append(time.time() - t0)
    return float(np.median(ts))
rows = []
for arm in ("epc_alm_k4", "epc_k4", "epc_k16"):
    k, alpha = A.ARMS[arm]; rho = jnp.float32(max(1.0, gmed / k))
    for T, gi in ((20, 12), (20, 4), (10, 12), (10, 4)) + (((100, 12),) if arm == "epc_k16" else ()):
        cr = [A.credit(pred, *b, rho, alpha, T, 32, gi) for b in batches]
        c1, rr, cmin = (float(np.mean([float(c[j]) for c in cr])) for j in range(3))
        dt = timed(eqx.filter_jit(lambda q, r: P.grads(q, ctxT, ci, cv, ti, tv, YT, "epc", rho=r, alpha=alpha, T=T, chunk=32, gn_iters=gi)[0]), pred, rho) if T <= 20 else None
        rows.append({"arm": arm, "T": T, "gn_iters": gi, "rho": float(rho), "cos": c1, "ratio": rr, "min_leaf_cos": cmin, "s_per_step": dt})
        print("A0", json.dumps(rows[-1]), flush=True)
tb = timed(eqx.filter_jit(lambda q: P.grads(q, ctxT, ci, cv, ti, tv, YT, "bp", chunk=32)[0]), pred)
ref = {r["arm"]: r for r in rows if r["T"] == 20 and r["gn_iters"] == 12}
cand = [(20, 12), (20, 4), (10, 12), (10, 4)]
ok = [(T, gi) for T, gi in cand if all(r["cos"] >= ref[r["arm"]]["cos"] - 0.005 and r["min_leaf_cos"] >= ref[r["arm"]]["min_leaf_cos"] - 0.01
                                       for r in rows if (r["T"], r["gn_iters"]) == (T, gi))]
cost = {c: np.mean([r["s_per_step"] for r in rows if (r["T"], r["gn_iters"]) == c]) for c in ok}
T, gi = min(cost, key=cost.get)
json.dump({"T": T, "gn_iters": gi, "gain_median": gmed, "gain_draws": gs, "bp_s_per_step": tb, "rows": rows,
           "rule": "cheapest (T, gn_iters) with cos >= ref - 0.005 and min-leaf cos >= ref - 0.01 for every ePC arm, ref = (T 20, 12 its)"},
          open(os.path.join(OUT, "a0_choice.json"), "w"), indent=1)
print("CHOICE", T, gi, "bp s/step", round(tb, 3), flush=True)
