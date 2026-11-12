import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp, dataclasses as dc
jax.config.update("jax_default_matmul_precision", "highest")
import vit_ijepa as V, train_pred_vit as A, latent_vit as LV
# Known-answer tests for latent_vit.py (plan_latent_analysis.md checks); exits 1 on any failure.
OK = [True]
def check(name, ok, info=""):
    OK[0] &= bool(ok); print(f"  [{'PASS' if ok else 'FAIL'}] {name} {info}", flush=True)
rng = np.random.default_rng(0)
X = rng.standard_normal((500, 32)); Q, _ = np.linalg.qr(rng.standard_normal((32, 32)))
check("CKA(X, X) = 1", abs(LV.linear_cka(X, X) - 1) < 1e-9)
check("CKA invariant to rotation and scale", abs(LV.linear_cka(X, 3.0 * X @ Q) - 1) < 1e-9)
check("CKA of independent data is small", LV.linear_cka(X, rng.standard_normal((500, 32))) < 0.2)
sp = LV.spectrum(np.eye(10)); check("spectrum of identity: eff rank 10, participation 10", abs(sp["eff_rank"] - 10) < 1e-9 and abs(sp["part_ratio"] - 10) < 1e-9)
vc = V.VCfg(); g = vc.grid
ci, cv, ti, tv = V.sample_masks(np.random.default_rng(1), 8, vc)
d = LV.nearest_ctx_dist(ci, cv, ti, tv, g); ok = True
for b in range(8):
    for k in range(ti.shape[1]):
        for t in range(ti.shape[2]):
            if not tv[b, k, t]: ok &= np.isnan(d[b, k, t]); continue
            r, c = divmod(ti[b, k, t], g); bf = min(np.hypot(r - x // g, c - x % g) for x in ci[b][cv[b]])
            ok &= abs(bf - d[b, k, t]) < 1e-9
check("nearest-context distance == brute force", ok)
for depth in (2, 6):
    a = A.ACfg(depth=depth); pred = A.make_predictor(vc, a)
    ctx = jnp.asarray(np.random.default_rng(2).standard_normal((8, vc.ntok, vc.dim)), jnp.float32)
    batch, jt, rs, gn = LV.make_fns(pred, g)
    Pr, pooled, att = batch(ctx, jnp.asarray(ci), jnp.asarray(cv), jnp.asarray(ti), jnp.asarray(tv))
    ref = A.predict(pred, ctx, jnp.asarray(ci), jnp.asarray(cv), jnp.asarray(ti), jnp.asarray(tv))
    m = jnp.asarray(tv)[..., None]
    check(f"d{depth}: instrumented forward == model", float(jnp.max(jnp.abs((Pr - ref) * m))) < 1e-5, f"{float(jnp.max(jnp.abs((Pr - ref) * m))):.1e}")
    check(f"d{depth}: shapes", pooled.shape == (8, ti.shape[1], 2 * depth + 1, 192) and att.shape == (8, ti.shape[1], depth, 3, 3), str((pooled.shape, att.shape)))
    a_ = np.asarray(att); check(f"d{depth}: attention mass on context in [0, 1], distances >= 1", np.all((a_[..., 0] >= -1e-6) & (a_[..., 0] <= 1 + 1e-6)) and np.all(a_[..., 1] >= 1 - 1e-5))
    a4 = (ctx, jnp.asarray(ci), jnp.asarray(cv), jnp.asarray(ti[:, 0]), jnp.asarray(tv[:, 0]))
    sj = np.asarray(jt(*a4)); sr = np.asarray(rs(*a4, jax.random.split(jax.random.PRNGKey(0), 8)))
    check(f"d{depth}: random-direction sensitivity <= top singular value", np.all(sr <= sj * 1.05), f"{np.median(sr):.3g} vs {np.median(sj):.3g}")
print("ALL PASS" if OK[0] else "SOME FAIL"); sys.exit(0 if OK[0] else 1)
