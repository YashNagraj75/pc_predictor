import importlib.util, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp, equinox as eqx
jax.config.update("jax_default_matmul_precision", "highest")
import data as D, vit_ijepa as V, vit_epc as P, train_pred_vit as A

# Per-block rho (findings/40 follow-up), rev 2 after 3 reviewers.  argv: <project root> <old worktree at 421f84f> <out json>.
# Gates, then a measurement of the 4-iteration gain estimate, then the pre-set screen that decides whether (and with which
# k) the per-block arm trains.  The decision is written to <out json> and to blockrho_decision.json next to it.
ROOT, OLD, OUT = sys.argv[1:4]; OK = [True]; res = {"gates": [], "kappa": [], "screen": []}
def check(n, c, info=""):
    OK[0] &= bool(c); res["gates"].append({"name": n, "pass": bool(c), "info": info}); print(f"  [{'PASS' if c else 'FAIL'}] {n}  {info}", flush=True)
def leaves(t): return jax.tree_util.tree_leaves(eqx.filter(t, eqx.is_inexact_array))
def flat(t): return jnp.concatenate([v.ravel() for v in leaves(t)])
def cos(a, b): a, b = flat(a), flat(b); return float(a @ b / (jnp.linalg.norm(a) * jnp.linalg.norm(b)))
def rel(a, b): a, b = flat(a), flat(b); return float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))
def leafcos(a, b):
    cs = [float(jnp.sum(x * y) / (jnp.linalg.norm(x) * jnp.linalg.norm(y) + 1e-30)) for x, y in zip(leaves(a), leaves(b)) if float(jnp.linalg.norm(y)) > 1e-12]
    return min(cs), float(np.median(cs))
vc, enc, tenc, _ = A.load_encoders(os.path.join(ROOT, "Q1-PhaseA/runs_vit/v0_d6_s0"))
SA = os.path.join(ROOT, "Q1-PhaseA/runs_vit/stageA")
def load_pred(run):
    return eqx.tree_deserialise_leaves(os.path.join(SA, run, "predictor.eqx"), A.make_predictor(vc, A.ACfg(arm="epc_alm_k4", depth=10)))
CK = {"alm_final": load_pred("epc_alm_k4_d10_lr3e-3_e25_full_s0_T10g4"), "bp_final": load_pred("bp_d10_lr3e-3_e25_full_s0")}
pred = CK["alm_final"]
Xtr, _, _, _ = D.load_cifar10(root=os.path.join(ROOT, "data"), n_train=-1, n_test=2000, seed=0)
def batch(B, seed, X0=45000):
    rng = np.random.default_rng(seed); X = Xtr[X0:X0 + B]
    ci, cv, ti, tv = (jnp.asarray(x) for x in V.sample_masks(rng, B, vc))
    im = jnp.asarray(V.standardise(X)); ctx, Y = A.encode(enc, tenc, im, im, ci, cv, ti, vc)
    return ctx, ci, cv, ti, tv, Y
def gains(p, b, it, chunk=8):                    # chunked over images (the GPU is shared with a training job)
    f = eqx.filter_jit(lambda q, *x: jax.vmap(lambda a, c1, v1, t1, w1: jax.vmap(lambda t2, w2: P.gain(q, a, c1, v1, t2, w2, gn_iters=it))(t1, w1))(*x))
    n = b[0].shape[0]; return jnp.concatenate([f(p, *(x[i:i + chunk] for x in b[:5])) for i in range(0, n, chunk)])

# G0 regression against the pre-change code (commit 421f84f): the running arms' path must be unchanged
spec = importlib.util.spec_from_file_location("vit_epc_old", os.path.join(OLD, "Q1-PhaseA/vit_epc.py")); PO = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PO); sys.path = [x for x in sys.path if not x.startswith(OLD)]
b8 = batch(8, 3)
for al in (0.0, 1.0):
    gn = P.grads(pred, *b8, "epc", rho=500.0, alpha=al, T=10, chunk=8, gn_iters=4); go = PO.grads(pred, *b8, "epc", rho=500.0, alpha=al, T=10, chunk=8, gn_iters=4)
    bit = all(bool(jnp.all(x == y)) for x, y in zip(leaves(gn[0]), leaves(go[0])))
    check(f"rho_rel 0 == commit 421f84f: weight grads and energies, alpha {al:g}", rel(gn[0], go[0]) < 1e-6 and bool(jnp.allclose(gn[2], go[2], rtol=1e-6)),
          f"rel {rel(gn[0], go[0]):.1e}, bitwise {bit}")
a1 = (pred, b8[0][0], b8[1][0], b8[2][0], b8[3][0, 0], b8[4][0, 0], b8[5][0, 0], 500.0, 1.0, 10)
en_, eo_ = P.epc_one(*a1, gn_iters=4), PO.epc_one(*a1, gn_iters=4)
check("epc_one == commit 421f84f (states, multipliers, energies, errors)", all(bool(jnp.allclose(x, y, rtol=1e-6, atol=1e-7)) for x, y in zip(en_, eo_)))
# G1 per-block rho is max(1, own 4-iteration gain / 4); report the spread
_, _, _, RB = P.grads(pred, *b8, "epc", rho=1.0, alpha=1.0, T=10, chunk=8, gn_iters=4, rho_rel=4.0, return_rho=True)
g4 = gains(pred, b8, 4)
check("per-block rho == max(1, own gain/4)", float(jnp.max(jnp.abs(RB - jnp.maximum(1.0, g4 / 4.0)) / RB)) < 1e-4, f"spread {float(RB.min()):.3g}-{float(RB.max()):.3g}")
# G2 non-uniform reference: each (image, block) alone at scalar rho = its RB, weighted by its share of valid targets
b4 = tuple(x[:4] for x in b8); _, _, _, RB4 = P.grads(pred, *b4, "epc", rho=1.0, alpha=1.0, T=10, chunk=4, gn_iters=4, rho_rel=4.0, return_rho=True)
gbat = P.grads(pred, *b4, "epc", rho=1.0, alpha=1.0, T=10, chunk=4, gn_iters=4, rho_rel=4.0)[0]
Ntot = float(jnp.sum(b4[4])); acc = None
one = eqx.filter_jit(lambda q, r, *s: P.grads(q, *s, "epc", rho=r, alpha=1.0, T=10, chunk=1, gn_iters=4)[0])
for i in range(4):
    for k in range(b4[3].shape[1]):
        sl = (b4[0][i:i + 1], b4[1][i:i + 1], b4[2][i:i + 1], b4[3][i:i + 1, k:k + 1], b4[4][i:i + 1, k:k + 1], b4[5][i:i + 1, k:k + 1])
        g = one(pred, RB4[i, k], *sl); w = float(jnp.sum(sl[4])) / Ntot
        acc = jax.tree_util.tree_map(lambda x: w * x, g) if acc is None else jax.tree_util.tree_map(lambda x, y: x + w * y, acc, g)
spread = float(RB4.max() / RB4.min())
check("per-block grads == sum of single-block grads at their own scalar rho (non-degenerate: rho spread > 3x)", rel(gbat, acc) < 1e-4 and spread > 3, f"rel {rel(gbat, acc):.1e}, spread {spread:.1f}x")
# G3 the floor, on an untrained predictor (small gain)
p0 = A.make_predictor(vc, A.ACfg(arm="epc_alm_k4", depth=10)); _, _, _, RB0 = P.grads(p0, *b8, "epc", rho=1.0, alpha=1.0, T=10, chunk=8, gn_iters=4, rho_rel=4.0, return_rho=True)
g0 = gains(p0, b8, 4)
check("untrained predictor: rho == max(1, gain/4), floor active", bool(jnp.allclose(RB0, jnp.maximum(1.0, g0 / 4.0), rtol=1e-4)) and float(jnp.mean(RB0 <= 1.0)) > 0,
      f"fraction at floor {float(jnp.mean(RB0 <= 1.0)):.2f}")
# G4 BP limit, with convergence in rho
gb8 = P.grads(pred, *b8, "bp", chunk=8)[0]; errs = []
for rr in (1e-2, 1e-3, 1e-4):
    errs.append(rel(P.grads(pred, *b8, "epc", rho=1.0, alpha=0.0, T=200, chunk=8, gn_iters=12, rho_rel=rr)[0], gb8))
check("per-block rho -> BP limit: error falls with rho, cos > 0.9999 at 1e4 x gain", (errs[1] < errs[0] / 3 or errs[1] < 1e-4) and (errs[2] < errs[1] / 3 or errs[2] < 1e-4) and 1 - errs[2] ** 2 / 2 > 0.9999,
      "rel err " + ", ".join(f"{e:.1e}" for e in errs))
# kappa: 4- and 12-iteration gain estimates vs a 100-iteration reference, both checkpoints, 64 images x 4 blocks
for name, p in CK.items():
    bb = batch(64, 11); l4, l12, l100 = (np.asarray(gains(p, bb, it)).ravel() for it in (4, 12, 100))
    for it, l in ((4, l4), (12, l12)):
        kap = l / l100; r = {"ckpt": name, "gn_iters": it, "kappa_q": np.round(np.percentile(kap, [0, 5, 50]), 3).tolist(), "frac_kappa_lt_0.4": float(np.mean(kap < 0.4))}
        for k in (1.0, 2.0, 4.0):                # contraction of the top error mode: eta (rho + lam_true) must stay < 2
            rho = np.maximum(1.0, l / k); r[f"frac_unstable_k{k:g}"] = float(np.mean(0.9 * (rho + l100) / (rho + l) >= 2.0))
        res["kappa"].append(r); print("KAPPA", json.dumps(r), flush=True)
# screen (pre-set): 2 checkpoints x 5 draws x 64 images; median and max-rule scalar rho controls vs per-block k in {1,2,4} x gn {4,12}
CFG = [("median", 0.0, 4), ("maxrule", 0.0, 4)] + [(f"blk{k:g}_g{g}", k, g) for k in (4.0, 2.0, 1.0) for g in (4, 12)]
for name, p in CK.items():
    for draw in range(5):
        bb = batch(64, 100 + draw, 45000 + 64 * draw); gb = P.grads(p, *bb, "bp", chunk=32)[0]
        g50 = np.asarray(gains(p, tuple(x[:8] for x in bb), 50)).ravel()
        for mode, rr, gi in CFG:
            rho = {"median": max(1.0, float(np.median(g50)) / 4), "maxrule": max(1.0, float(g50.max()) / 4)}.get(mode, 1.0)
            g, _, en, rb = P.grads(p, *bb, "epc", rho=rho, alpha=1.0, T=10, chunk=32, gn_iters=gi, rho_rel=rr, return_rho=True)
            ml, md = leafcos(g, gb); E2 = np.asarray(en)
            row = {"ckpt": name, "draw": draw, "mode": mode, "cos": cos(g, gb), "ratio": float(jnp.linalg.norm(flat(g)) / jnp.linalg.norm(flat(gb))),
                   "min_leaf": ml, "median_leaf": md, "frac_energy_rise": float(np.mean(E2[..., 1] > E2[..., 0])), "rho_med": float(jnp.median(rb))}
            res["screen"].append(row); print("SCREEN", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
import pandas as pd
S = pd.DataFrame(res["screen"]); M = S.groupby(["ckpt", "mode"])[["cos", "ratio", "min_leaf", "frac_energy_rise"]].mean()
res["screen_mean"] = {f"{a}|{b}": v for (a, b), v in M.round(4).to_dict("index").items()}
def passes(mode):
    ok = all(M.loc[(c, mode), "cos"] >= 0.9 and 0.8 <= M.loc[(c, mode), "ratio"] <= 1.2 and M.loc[(c, mode), "min_leaf"] > 0 and M.loc[(c, mode), "frac_energy_rise"] <= 0.01 for c in CK)
    d = S[(S.ckpt == "alm_final") & (S["mode"] == mode)].set_index("draw").cos - S[(S.ckpt == "alm_final") & (S["mode"] == "median")].set_index("draw").cos
    return ok and bool((d >= 0.05).all())
cands = [m for m in ("blk4_g4", "blk2_g4", "blk1_g4") if passes(m)]
mx_ok = passes("maxrule")
dec = {"rule": "per-block config with gn 4 passing cos >= 0.9, ratio 0.8-1.2, min leaf > 0, energy rise <= 1% on both checkpoints, and beating the median rule by >= 0.05 cos on every draw (ALM checkpoint); largest k first. If the scalar max-rule control also passes and is within 0.02 cos of the pick, a single larger rho suffices: do not train, ask the user.",
       "passing": cands, "maxrule_passes": mx_ok, "gn12_only": [m for m in ("blk4_g12", "blk2_g12", "blk1_g12") if passes(m)] if not cands else []}
pick = cands[0] if cands else None
if pick and mx_ok and abs(M.loc[("alm_final", "maxrule"), "cos"] - M.loc[("alm_final", pick), "cos"]) <= 0.02:
    dec.update(train=False, reason="max-rule scalar rho does as well", pick=pick)
else:
    dec.update(train=bool(pick) and OK[0], pick=pick, arm=("epc_alm_" + pick.split("_")[0]) if pick else None)
bb = batch(256, 7)
for rr in ((0.0,) + ((float(pick[3]),) if pick else ())):
    f = eqx.filter_jit(lambda q: P.grads(q, *bb, "epc", rho=500.0, alpha=1.0, T=10, chunk=32, gn_iters=4, rho_rel=rr)[0])
    jax.block_until_ready(f(pred)); t0 = time.time(); jax.block_until_ready(f(pred)); res[f"s_per_step_rho_rel_{rr:g}"] = time.time() - t0
res["decision"] = dec; res["all_gates_pass"] = OK[0]
json.dump(res, open(OUT, "w"), indent=1); json.dump(dec, open(os.path.join(os.path.dirname(OUT), "blockrho_decision.json"), "w"), indent=1)
print(M.round(4).to_string()); print("DECISION", json.dumps(dec)); print("ALL PASS" if OK[0] else "SOME FAIL"); sys.exit(0 if OK[0] else 1)
