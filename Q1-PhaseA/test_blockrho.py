import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp, equinox as eqx
jax.config.update("jax_default_matmul_precision", "highest")
import data as D, vit_ijepa as V, vit_epc as P, train_pred_vit as A

# Gates for the per-block-rho arm (epc_alm_blk4), then the pre-set inference-only screen on the trained depth-10 ePC-ALM
# predictor of findings/40: credit of per-block rho vs the median rule.  argv: <project root> <out json>.
ROOT, OUT = sys.argv[1], sys.argv[2]; OK = [True]; res = {"gates": [], "screen": []}
def check(n, c, info=""):
    OK[0] &= bool(c); res["gates"].append({"name": n, "pass": bool(c), "info": info}); print(f"  [{'PASS' if c else 'FAIL'}] {n}  {info}", flush=True)
def flat(t): return jnp.concatenate([v.ravel() for v in jax.tree_util.tree_leaves(eqx.filter(t, eqx.is_inexact_array))])
def cos(a, b): a, b = flat(a), flat(b); return float(a @ b / (jnp.linalg.norm(a) * jnp.linalg.norm(b)))
def rel(a, b): a, b = flat(a), flat(b); return float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))
vc, enc, tenc, _ = A.load_encoders(os.path.join(ROOT, "Q1-PhaseA/runs_vit/v0_d6_s0"))
SA = os.path.join(ROOT, "Q1-PhaseA/runs_vit/stageA")
def load_pred(run):
    a = A.ACfg(arm="epc_alm_k4", depth=10)
    return eqx.tree_deserialise_leaves(os.path.join(SA, run, "predictor.eqx"), A.make_predictor(vc, a))
pred = load_pred("epc_alm_k4_d10_lr3e-3_e25_full_s0_T10g4")
Xtr, _, _, _ = D.load_cifar10(root=os.path.join(ROOT, "data"), n_train=-1, n_test=2000, seed=0)
def batch(B, seed, X0=45000):
    rng = np.random.default_rng(seed); X = Xtr[X0:X0 + B]
    ci, cv, ti, tv = (jnp.asarray(x) for x in V.sample_masks(rng, B, vc))
    im = jnp.asarray(V.standardise(X)); ctx, Y = A.encode(enc, tenc, im, im, ci, cv, ti, vc)
    return ctx, ci, cv, ti, tv, Y
b = batch(8, 3)
# 1. the returned per-block rho equals max(1, own gain / 4), with the gain recomputed independently (same iterations)
_, ci, cv, ti, tv, Y = b; ctx = b[0]
out = jax.vmap(lambda a, c1, v1, t1, w1, y1: P._relax("epc", pred, a, c1, v1, t1, w1, y1, 1.0, 1.0, 10, 1.0, 4, 4.0))(ctx, ci, cv, ti, tv, Y)
RB = out[4]
g4 = jax.vmap(lambda a, c1, v1, t1, w1: jax.vmap(lambda t2, w2: P.gain(pred, a, c1, v1, t2, w2, gn_iters=4))(t1, w1))(ctx, ci, cv, ti, tv)
check("per-block rho == max(1, own gain/4)", float(jnp.max(jnp.abs(RB - jnp.maximum(1.0, g4 / 4.0)) / RB)) < 1e-4,
      f"rho spread {float(RB.min()):.3g}-{float(RB.max()):.3g}")
# 2. rho_rel = 0 path unchanged: epc_one == epc_core[:4], and grads ignore RB when it is the scalar rho
e1 = P.epc_one(pred, ctx[0], ci[0], cv[0], ti[0, 0], tv[0, 0], Y[0, 0], 500.0, 1.0, 10, gn_iters=4)
e2 = P.epc_core(pred, ctx[0], ci[0], cv[0], ti[0, 0], tv[0, 0], Y[0, 0], 500.0, 1.0, 10, gn_iters=4)
check("rho_rel = 0: epc_one == epc_core[:4] exactly", all(bool(jnp.all(x == y)) for x, y in zip(e1[:2] + e1[3:], e2[:2] + e2[3:4])))
# 3. identical blocks -> per-block rho is uniform and equals the scalar-rho gradient at that value
rep = lambda x: jnp.broadcast_to(x[:1, :1], x.shape) if x.ndim >= 2 else x
cU, iU, vU = (jnp.broadcast_to(x[:1], x.shape) for x in (ctx, ci, cv)); tU, wU, yU = (rep(x) for x in (ti, tv, Y))
oU = jax.vmap(lambda a, c1, v1, t1, w1, y1: P._relax("epc", pred, a, c1, v1, t1, w1, y1, 1.0, 1.0, 10, 1.0, 4, 4.0))(cU, iU, vU, tU, wU, yU)
r0 = float(oU[4][0, 0])
gB = P.grads(pred, cU, iU, vU, tU, wU, yU, "epc", rho=1.0, alpha=1.0, T=10, chunk=8, gn_iters=4, rho_rel=4.0)[0]
gS = P.grads(pred, cU, iU, vU, tU, wU, yU, "epc", rho=r0, alpha=1.0, T=10, chunk=8, gn_iters=4)[0]
check("uniform blocks: per-block grads == scalar-rho grads", rel(gB, gS) < 1e-4, f"rel {rel(gB, gS):.1e}, rho {r0:.4g}")
# 4. BP limit: rho = 1000 x own gain, no multipliers
gb = P.grads(pred, *b, "bp", chunk=8)[0]
gL = P.grads(pred, *b, "epc", rho=1.0, alpha=0.0, T=200, chunk=8, gn_iters=12, rho_rel=1e-3)[0]
check("per-block rho = 1e3 x gain: weight gradient == BP (cos > 0.999)", cos(gL, gb) > 0.999, f"cos {cos(gL, gb):.5f}")
# 5. the training-time credit path runs with per-block rho
c1_, r1_, m1_, _ = A.credit(pred, *b, jnp.float32(1.0), 1.0, 10, 8, 4, 4.0)
check("per-block credit finite", bool(np.isfinite(float(c1_))), f"cos {float(c1_):.4f}")
# 6. screen (pre-set): trained d10 ePC-ALM predictor, 3 draws x 64 images, T 10 / 4 its; median rule vs per-block
for draw in range(3):
    bb = batch(64, 100 + draw, 45000 + 64 * draw)
    gs = [float(x) for x in A.gain_stats(pred, *bb[:5])]
    for mode, rho, rr in (("median", max(1.0, gs[1] / 4), 0.0), ("per-block", 1.0, 4.0)):
        cc, ra, cm, cl = A.credit(pred, *bb, jnp.float32(rho), 1.0, 10, 32, 4, rr)
        res["screen"].append({"draw": draw, "mode": mode, "gain_min_med_max": gs, "cos": float(cc), "ratio": float(ra), "min_leaf": float(cm),
                              "median_leaf": float(np.median(np.asarray(cl)))})
        print("SCREEN", json.dumps(res["screen"][-1]), flush=True)
bb = batch(256, 7)
for rr in (0.0, 4.0):
    f = eqx.filter_jit(lambda q: P.grads(q, *bb, "epc", rho=500.0, alpha=1.0, T=10, chunk=32, gn_iters=4, rho_rel=rr)[0])
    jax.block_until_ready(f(pred)); t0 = time.time(); jax.block_until_ready(f(pred)); res[f"s_per_step_rho_rel_{rr:g}"] = time.time() - t0
print("TIMING", {k: round(v, 2) for k, v in res.items() if k.startswith("s_per")})
S = {m: np.mean([r["cos"] for r in res["screen"] if r["mode"] == m]) for m in ("median", "per-block")}
res["decision"] = {"cos_mean": S, "rule": "train the arm only if per-block cos >= median cos + 0.05", "train": bool(S["per-block"] >= S["median"] + 0.05)}
res["all_gates_pass"] = OK[0]; json.dump(res, open(OUT, "w"), indent=1)
print("DECISION", res["decision"]); print("ALL PASS" if OK[0] else "SOME FAIL"); sys.exit(0 if OK[0] else 1)
