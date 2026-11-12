import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp, equinox as eqx
jax.config.update("jax_default_matmul_precision", "highest")
import vit_ijepa as V, train_pred_vit as A, latent_vit as LV
# Known-answer tests for latent_vit.py rev 2 (3 reviewers); exits 1 on any failure.
OK = [True]
def check(name, ok, info=""):
    OK[0] &= bool(ok); print(f"  [{'PASS' if ok else 'FAIL'}] {name} {info}", flush=True)
rng = np.random.default_rng(0)
X = rng.standard_normal((500, 32)); Q, _ = np.linalg.qr(rng.standard_normal((32, 32)))
check("CKA(X, X) = 1, rotation/scale invariant", abs(LV.linear_cka(X, X) - 1) < 1e-9 and abs(LV.linear_cka(X, 3 * X @ Q) - 1) < 1e-9)
check("CKA of independent data is small", LV.linear_cka(X, rng.standard_normal((500, 32))) < 0.2)
# residualisation: a per-(group, block) position term identical across images is removed exactly
pos = rng.standard_normal((40, 1, 4, 3, 16)) * 10; sig = rng.standard_normal((40, 50, 4, 3, 16))
S = (pos + sig).reshape(2000, 4, 3, 16); Rr = LV.residualise(S)
check("residualise removes the position term", np.allclose(Rr.reshape(40, 50, 4, 3, 16), sig - sig.mean(1, keepdims=True)))
other = (pos + rng.standard_normal((40, 50, 4, 3, 16))).reshape(2000, 4, 3, 16)
raw = LV.linear_cka(S[:, :, 1].reshape(-1, 16), other[:, :, 1].reshape(-1, 16)); res = LV.linear_cka(Rr[:, :, 1].reshape(-1, 16), LV.residualise(other)[:, :, 1].reshape(-1, 16))
check("shared position inflates raw CKA, residualised CKA is near 0", raw > 0.8 and res < 0.1, f"raw {raw:.3f} residualised {res:.3f}")
sp = LV.spectrum(np.eye(10)); check("spectrum of identity", abs(sp["eff_rank"] - 10) < 1e-9 and abs(sp["part_ratio"] - 10) < 1e-9)
# basis: orthonormal, last column = all-ones direction, bands cover 0..dim, band errors sum to the total
Y = rng.standard_normal((3000, 24)) @ rng.standard_normal((24, 24)); Y = Y - Y.mean(1, keepdims=True)
U, e, mu, bands = LV.basis_from_moments(Y.sum(0), Y.T @ Y, len(Y), 24)
check("basis orthonormal, last = ones/sqrt(d)", np.allclose(U.T @ U, np.eye(24), atol=1e-9) and np.allclose(U[:, -1], 1 / math.sqrt(24)))
check("bands partition the components", bands[0][0] == 0 and bands[-1] == (23, 24) and all(bands[k][1] == bands[k + 1][0] for k in range(len(bands) - 1)), str(bands))
Pp = rng.standard_normal((3000, 24)); E = (Pp - Y) @ U
check("band errors sum to the total error", abs(sum((E[:, a:b] ** 2).sum() for a, b in bands) - ((Pp - Y) ** 2).sum()) < 1e-6 * ((Pp - Y) ** 2).sum())
# noise split identity
Pc, dP, Yt = (rng.standard_normal((50, 30)) for _ in range(3)); ac = 0.7
NR = np.stack([(dP ** 2).sum(1), (Pc ** 2).sum(1), (dP * Pc).sum(1), (dP * (Pc - Yt)).sum(1)], 1)
vt, cr = LV.noise_split(ac, NR); lhs = ((ac * (Pc + dP) - Yt) ** 2).sum(1) - ((ac * Pc - Yt) ** 2).sum(1)
check("noise split: var + cross = change of calibrated error", np.allclose(vt + cr, lhs))
# masks and distances
vc = V.VCfg(); g = vc.grid
ci, cv, ti, tv = V.sample_masks(np.random.default_rng(1), 8, vc)
d = LV.nearest_ctx_dist(ci, cv, ti, tv, g); ok = True
for b in range(8):
    for k in range(ti.shape[1]):
        for t in range(ti.shape[2]):
            if not tv[b, k, t]: ok &= np.isnan(d[b, k, t]); continue
            r, c = divmod(ti[b, k, t], g); ok &= abs(min(np.hypot(r - x // g, c - x % g) for x in ci[b][cv[b]]) - d[b, k, t]) < 1e-9
check("nearest-context distance == brute force", ok)
dc = LV.ctx_dist_to_targets(ci, cv, ti, tv, g); ok = True
for b in range(8):
    tt = ti[b][tv[b]]
    for n in np.where(cv[b])[0]:
        ok &= abs(min(np.hypot(ci[b, n] // g - x // g, ci[b, n] % g - x % g) for x in tt) - dc[b, n]) < 1e-9
check("context-to-target distance == brute force", ok)
cvs, nrm = LV.intervention_masks(np.random.default_rng(2), ci, cv, ti, tv, g); ok = True
for b in range(8):
    rm = {k: np.where(cv[b] & ~cvs[k][b])[0] for k in cvs}
    ok &= all(len(rm[k]) == nrm[b] for k in rm) and np.all(dc[b, rm["near"]] <= 1.5) and np.all(dc[b, rm["far"]] > 3.1) and cvs["near"][b].sum() >= 4
    ok &= not np.any(cvs["near"][b] & ~cv[b])
check("interventions: near <= 1.5, far > 3.1, equal counts, >= 4 kept", ok, f"removed {nrm.tolist()}")
# instrumented forward vs model with non-trivial weights (2-D leaves x10)
scale = lambda p: jax.tree_util.tree_map(lambda x: x * 10.0 if eqx.is_inexact_array(x) and x.ndim == 2 else x, p)
for depth in (2, 6):
    pred = A.make_predictor(vc, A.ACfg(depth=depth)); pred = eqx.tree_at(lambda q: q.blocks, pred, scale(pred.blocks))
    ctx = jnp.asarray(np.random.default_rng(2).standard_normal((8, vc.ntok, vc.dim)), jnp.float32)
    batch, jt, gn, pr = LV.make_fns(pred, g)
    Pr, pooled, att, con = batch(ctx, jnp.asarray(ci), jnp.asarray(cv), jnp.asarray(ti), jnp.asarray(tv))
    ref = pr(ctx, jnp.asarray(ci), jnp.asarray(cv), jnp.asarray(ti), jnp.asarray(tv)); m = jnp.asarray(tv)[..., None]
    rel = float(jnp.max(jnp.abs((Pr - ref) * m)) / jnp.max(jnp.abs(ref * m)))
    check(f"d{depth}: instrumented forward == model (weights x10)", rel < 1e-5, f"rel {rel:.1e}, |P| {float(jnp.max(jnp.abs(ref * m))):.3g}")
    check(f"d{depth}: shapes", pooled.shape == (8, ti.shape[1], 2 * depth + 1, 192) and att.shape == (8, ti.shape[1], depth, 3, 4) and con.shape == (8, ti.shape[1], depth))
    if depth == 2:                                # brute-force layer-0 attention statistics for image 0, block 0
        b0 = pred.blocks[0]; h = np.asarray(pred.embed_(ctx[0], jnp.asarray(ci[0]), jnp.asarray(ti[0, 0])), np.float64)
        qkv = np.asarray(b0.qkv(b0.ln1(jnp.asarray(h, jnp.float32))), np.float64); NC = ci.shape[1]; H = b0.heads; dh = 192 // H
        kval = np.r_[cv[0], tv[0, 0]]; posn = np.r_[ci[0], ti[0, 0]]; mc_ = np.zeros(H); dw_ = np.zeros(H); en_ = np.zeros(H); un_ = []
        for hh in range(H):
            num = den = 0.0
            for n in np.where(tv[0, 0])[0]:
                q = qkv[NC + n, hh * dh:(hh + 1) * dh]; ks = qkv[:, 192 + hh * dh:192 + (hh + 1) * dh]
                sc = np.where(kval, ks @ q / math.sqrt(dh), -np.inf); a = np.exp(sc - sc.max()); a /= a.sum()
                dd = np.array([np.hypot(posn[NC + n] // g - x // g, posn[NC + n] % g - x % g) for x in posn])
                mc_[hh] += a[:NC][cv[0]].sum(); num += (a[:NC] * cv[0] * dd[:NC]).sum(); den += a[:NC][cv[0]].sum()
                en_[hh] += -(a[kval] * np.log(np.maximum(a[kval], 1e-30))).sum()
                if hh == 0: un_.append(dd[:NC][cv[0]].mean())
            nv = tv[0, 0].sum(); mc_[hh] /= nv; en_[hh] /= nv; dw_[hh] = num / den
        got = np.asarray(att[0, 0, 0], np.float64)
        check("attention statistics == brute force (layer 0)", np.allclose(got[:, 0], mc_, rtol=1e-4) and np.allclose(got[:, 1], dw_, rtol=1e-4) and np.allclose(got[:, 2], np.mean(un_), rtol=1e-4)
              and np.allclose(got[:, 3], en_, rtol=1e-4), f"{got.round(4).tolist()} vs {np.c_[mc_, dw_, np.full(H, np.mean(un_)), en_].round(4).tolist()}")
        # uniform attention: zero query weights -> entropy log(valid keys), context mass = n_ctx / n_valid, distance = uniform
        W = b0.qkv.W.at[:192].set(0.0); bq = b0.qkv.b.at[:192].set(0.0)
        pu = eqx.tree_at(lambda q: (q.blocks[0].qkv.W, q.blocks[0].qkv.b), pred, (W, bq)); bu, *_ = LV.make_fns(pu, g)
        _, _, au, _ = bu(ctx, jnp.asarray(ci), jnp.asarray(cv), jnp.asarray(ti), jnp.asarray(tv)); a0 = np.asarray(au[0, 0, 0], np.float64)
        nk = cv[0].sum() + tv[0, 0].sum()
        check("uniform attention: entropy, context mass, distance", np.allclose(a0[:, 3], math.log(nk), rtol=1e-4) and np.allclose(a0[:, 0], cv[0].sum() / nk, rtol=1e-4)
              and np.allclose(a0[:, 1], a0[:, 2], rtol=1e-4), a0.round(4).tolist())
        # Jacobian top singular value vs dense SVD (image 0, block 0)
        f = lambda z: (pred(z, jnp.asarray(ci[0]), jnp.asarray(cv[0]), jnp.asarray(ti[0, 0]), jnp.asarray(tv[0, 0])) * jnp.asarray(tv[0, 0])[:, None]).ravel()
        Jm = np.asarray(jax.jacrev(f)(ctx[0]), np.float64).reshape(-1, vc.ntok, 192)[:, cv[0]].reshape(f(ctx[0]).shape[0], -1)
        s_ref = np.linalg.norm(Jm, 2); s_got = np.asarray(jt(ctx[:1], jnp.asarray(ci[:1]), jnp.asarray(cv[:1]), jnp.asarray(ti[:1, 0]), jnp.asarray(tv[:1, 0]), jax.random.split(jax.random.PRNGKey(0), 1)))[0]
        check("Jacobian top singular value (40 its) == dense SVD within 2%", abs(s_got[1] - s_ref) / s_ref < 0.02, f"{s_got[1]:.4g} vs {s_ref:.4g} (20 its {s_got[0]:.4g})")
print("ALL PASS" if OK[0] else "SOME FAIL"); sys.exit(0 if OK[0] else 1)
