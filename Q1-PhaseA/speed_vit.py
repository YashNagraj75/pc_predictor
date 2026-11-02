import dataclasses as dc, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp, equinox as eqx
jax.config.update("jax_default_matmul_precision", "highest")
import vit_ijepa as V, vit_epc as P, test_vit as TV

# Speed of the predictor gradient (BP vs ePC-ALM, T 20) at the real batch under the current XLA flags, over chunk size
# and power-iteration count, plus a compiled-vs-eager audit under the same flags.  Report only.
cfg = V.VCfg(); pc = dc.replace(cfg, pdepth=6, pdim=192, pmlp=768, pheads=3, pscale=True)
enc, pred, _ = V.init(pc); X, (ci, cv, ti, tv) = TV.batch(pc, 256, seed=11)
ctx = jax.vmap(lambda im, a, b: enc(im, a, b, pc.patch))(X, ci, cv)
Y = V._ln(jax.random.normal(jax.random.PRNGKey(3), ti.shape + (pc.dim,)))
print("XLA_FLAGS", repr(os.environ.get("XLA_FLAGS", "")), flush=True)
TV.train_smoke(cfg)
with jax.disable_jit():
    ge = P.grads(pred, ctx[:4], ci[:4], cv[:4], ti[:4], tv[:4], Y[:4], "epc", rho=64.0, alpha=1.0, T=5)[0]
gj = eqx.filter_jit(lambda q: P.grads(q, ctx[:4], ci[:4], cv[:4], ti[:4], tv[:4], Y[:4], "epc", rho=64.0, alpha=1.0, T=5)[0])(pred)
TV.check("ePC compiled == eager under these flags", TV.rel(gj, ge) < 1e-4, f"rel {TV.rel(gj, ge):.1e}")
rows = []
for meth, chunk, gi in [("bp", 32, 0), ("bp", 256, 0), ("epc", 32, 12), ("epc", 128, 12), ("epc", 256, 12), ("epc", 256, 4)]:
    try:
        f = eqx.filter_jit(lambda q, r: P.grads(q, ctx, ci, cv, ti, tv, Y, meth, rho=r, alpha=1.0, T=20, want_input=True, chunk=chunk, gn_iters=max(gi, 1))[:2])
        jax.block_until_ready(f(pred, jnp.float32(64.0))); t0 = time.time()
        for _ in range(2):
            jax.block_until_ready(f(pred, jnp.float32(64.0)))
        dt = (time.time() - t0) / 2
        pk = jax.devices()[0].memory_stats().get("peak_bytes_in_use", 0) / 2 ** 30
        rows.append({"method": meth, "chunk": chunk, "gn_iters": gi, "s_per_step": round(dt, 3), "peak_gib": round(pk, 2)})
    except Exception as e:
        rows.append({"method": meth, "chunk": chunk, "gn_iters": gi, "error": str(e)[:200]})
    print("SPEED", json.dumps(rows[-1]), flush=True)
print("AUDITS PASS" if TV.OK[0] else "AUDITS FAIL")
