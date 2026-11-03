import dataclasses as dc, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp, equinox as eqx
jax.config.update("jax_default_matmul_precision", "highest")
import data as D, vit_ijepa as V, vit_epc as P, train_pred_vit as A

# A0 (plan_stageA_scoped.md v3): on a depth-10 predictor trained 1 epoch by BP on the frozen V0 features, measure credit
# (cos to BP) and step time of each ePC arm at T in {10, 20} x power iterations in {4, 12}.  Pre-set rule: the cheapest
# setting whose credit is within 0.005 cos of the (T 20, 12 iterations) value for EVERY arm is used for all ePC arms.
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
a = A.ACfg(arm="bp", depth=10, lr=1e-3, epochs=1, split="val")
A.run(a, os.path.join(OUT, "bp_1epoch"))
vc, enc, tenc, _ = A.load_encoders()
pred = eqx.tree_deserialise_leaves(os.path.join(OUT, "bp_1epoch", "predictor.eqx"), A.make_predictor(vc, a))
Xtr, _, _, _ = D.load_cifar10(n_train=-1, n_test=2000, seed=0)
rng = np.random.default_rng(7); X = Xtr[45000:45256]
ci, cv, ti, tv = (jnp.asarray(x) for x in V.sample_masks(rng, 256, vc))
imgs = jnp.asarray(V.standardise(X)); ctx, Y = A.encode(enc, tenc, imgs, imgs, ci, cv, ti, vc)
gmed = float(A.gain_median(pred, ctx, ci, cv, ti, tv)); print("gain median", gmed, flush=True)
rows = []
for arm in ("epc_alm_k4", "epc_k4", "epc_k16"):
    k, alpha = A.ARMS[arm]; rho = jnp.float32(max(1.0, gmed / k))
    for T in (20, 10):
        for gi in (12, 4):
            c1, rr, cmin = A.credit(pred, ctx[:32], ci[:32], cv[:32], ti[:32], tv[:32], Y[:32], rho, alpha, T, 32, gi)
            f = eqx.filter_jit(lambda q, r: P.grads(q, ctx, ci, cv, ti, tv, Y, "epc", rho=r, alpha=alpha, T=T, chunk=32, gn_iters=gi)[0])
            jax.block_until_ready(f(pred, rho)); t0 = time.time(); jax.block_until_ready(f(pred, rho)); dt = time.time() - t0
            rows.append({"arm": arm, "T": T, "gn_iters": gi, "rho": float(rho), "cos": float(c1), "ratio": float(rr), "min_leaf_cos": float(cmin), "s_per_step": dt})
            print("A0", json.dumps(rows[-1]), flush=True)
fb = eqx.filter_jit(lambda q: P.grads(q, ctx, ci, cv, ti, tv, Y, "bp", chunk=32)[0])
jax.block_until_ready(fb(pred)); t0 = time.time(); jax.block_until_ready(fb(pred)); tb = time.time() - t0
ref = {r["arm"]: r["cos"] for r in rows if r["T"] == 20 and r["gn_iters"] == 12}
ok = [(T, gi) for T in (20, 10) for gi in (12, 4) if all(r["cos"] >= ref[r["arm"]] - 0.005 for r in rows if r["T"] == T and r["gn_iters"] == gi)]
cost = {(T, gi): np.mean([r["s_per_step"] for r in rows if r["T"] == T and r["gn_iters"] == gi]) for T, gi in ok}
T, gi = min(cost, key=cost.get)
json.dump({"T": T, "gn_iters": gi, "gain": gmed, "bp_s_per_step": tb, "rows": rows, "rule": "cheapest (T, gn_iters) with cos >= cos(T20, gn12) - 0.005 for every ePC arm"},
          open(os.path.join(OUT, "a0_choice.json"), "w"), indent=1)
print("CHOICE", T, gi, "bp s/step", round(tb, 3), flush=True)
