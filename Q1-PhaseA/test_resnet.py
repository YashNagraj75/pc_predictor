import importlib.util, os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp, equinox as eqx
jax.config.update("jax_default_matmul_precision", "highest")
import vit_ijepa as V, vit_epc as P, resnet_pred as RN, train_pred_vit as A
# Known-answer tests for the ResNet predictor and the generalised PC core (plan_resnet_predictor.md); exit 1 on any failure.
# BASE is the last commit before the core change: the transformer's PC gradients must be unchanged against it.
BASE = os.environ.get("BASE_COMMIT", "5e9ab14")
OK = [True]


def check(name, ok, info=""):
    """Record and print one test result."""
    OK[0] &= bool(ok); print(f"  [{'PASS' if ok else 'FAIL'}] {name} {info}", flush=True)


flat = lambda t: np.concatenate([np.asarray(v, np.float64).ravel() for v in jax.tree_util.tree_leaves(eqx.filter(t, eqx.is_inexact_array))])
cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
vc = V.VCfg(); rng = np.random.default_rng(0); B = 4
ci, cv, ti, tv = (jnp.asarray(x) for x in V.sample_masks(rng, B, vc))
ctx = jnp.asarray(rng.standard_normal((B, vc.ntok, vc.dim)), jnp.float32)
Y = jnp.asarray(rng.standard_normal(ti.shape + (vc.dim,)), jnp.float32)

# 1. 3x3 convolution as one GEMM == lax convolution
h = jnp.asarray(rng.standard_normal((64, 16)), jnp.float32); lin = V.Lin(jax.random.PRNGKey(1), 9 * 16, 24)
Wk = lin.W.T.reshape(3, 3, 16, 24)               # rows of the 9C columns are ordered (dy, dx, c)
ref = jax.lax.conv_general_dilated(h.reshape(1, 8, 8, 16), Wk, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC"))[0].reshape(64, 24) + lin.b
check("conv3 GEMM == lax.conv (SAME, 3x3)", rel(np.asarray(RN.conv3(lin, h, 8)), np.asarray(ref)) < 1e-5)

# 2. parameter counts match the transformer (12 vs 6 blocks, 20 vs 10 blocks)
npar = lambda m: int(sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(m, eqx.is_inexact_array))))
for dr, dv in ((12, 6), (20, 10)):
    nr = npar(A.make_predictor(vc, A.ACfg(arch="resnet", depth=dr))); nv = npar(A.make_predictor(vc, A.ACfg(depth=dv)))
    check(f"params resnet {dr} blocks vs transformer {dv} blocks within 10%", abs(nr / nv - 1) < 0.1, f"{nr:,} vs {nv:,}")

# 3. transformer PC gradients unchanged against BASE (bp, epc, epc per-block rho, spc)
src = subprocess.run(["git", "show", f"{BASE}:Q1-PhaseA/vit_epc.py"], capture_output=True, text=True, check=True).stdout
open("/tmp/vit_epc_base.py", "w").write(src); spec = importlib.util.spec_from_file_location("vit_epc_base", "/tmp/vit_epc_base.py")
PB = importlib.util.module_from_spec(spec); spec.loader.exec_module(PB)
tp = A.make_predictor(vc, A.ACfg(depth=2))
cases = {"bp": dict(), "epc": dict(rho=50.0, alpha=1.0, T=5), "epc_blk": dict(rho=1.0, alpha=1.0, T=5, rho_rel=4.0), "spc": dict(rho=1.0, alpha=0.0, T=5, hv_iters=8)}
for name, kw in cases.items():
    m = "epc" if name.startswith("epc") else name
    gn = flat(P.grads(tp, ctx, ci, cv, ti, tv, Y, m, chunk=B, **kw)[0]); go = flat(PB.grads(tp, ctx, ci, cv, ti, tv, Y, m, chunk=B, **kw)[0])
    check(f"transformer {name} gradient identical to {BASE}", np.array_equal(gn, go) or rel(gn, go) < 1e-7, f"rel {rel(gn, go):.1e}")

# 4. ResNet: padded context entries do not change the prediction or the gradients
rp = A.make_predictor(vc, A.ACfg(arch="resnet", depth=4))
ctx2 = jnp.where(cv[..., None], ctx, 100.0 * jnp.asarray(rng.standard_normal(ctx.shape), jnp.float32))
o1 = rp(ctx[0], ci[0], cv[0], ti[0, 0], tv[0, 0]); o2 = rp(ctx2[0], ci[0], cv[0], ti[0, 0], tv[0, 0])
g1 = flat(P.grads(rp, ctx, ci, cv, ti, tv, Y, "epc", rho=50.0, alpha=1.0, T=5, chunk=B)[0]); g2 = flat(P.grads(rp, ctx2, ci, cv, ti, tv, Y, "epc", rho=50.0, alpha=1.0, T=5, chunk=B)[0])
check("resnet: padding invariance (prediction, ePC gradient)", float(jnp.max(jnp.abs(o1 - o2))) == 0.0 and rel(g1, g2) < 1e-6, f"rel {rel(g1, g2):.1e}")

# 5. ResNet: batch-order invariance of BP and ePC gradients (GroupNorm is per sample)
pm = np.array([2, 0, 3, 1]); prm = lambda x: x[pm]
for m, kw in (("bp", {}), ("epc", dict(rho=50.0, alpha=1.0, T=5))):
    a1 = flat(P.grads(rp, ctx, ci, cv, ti, tv, Y, m, chunk=B, **kw)[0]); a2 = flat(P.grads(rp, *map(prm, (ctx, ci, cv, ti, tv, Y)), m, chunk=B, **kw)[0])
    check(f"resnet: batch-order invariance ({m})", rel(a2, a1) < 1e-5, f"rel {rel(a2, a1):.1e}")

# 6. ResNet: gain estimate (power iteration) == dense SVD of the output-to-node-error Jacobian, one (image, block)
kv, row, tm = P._masks(cv[0], tv[0, 0], rp)
f = lambda e: (P._head(rp, P.states_from_errors(rp, ctx[0], ci[0], cv[0], ti[0, 0], tv[0, 0], e)[-1], ti[0, 0]) * tm).ravel()
e0 = jnp.zeros((rp.n_sub, kv.shape[0], rp.pos.shape[-1])); Jm = np.asarray(jax.jacrev(f)(e0), np.float64).reshape(f(e0).shape[0], -1)
Jm = Jm[np.abs(Jm).sum(1) > 0]; lam_ref = float(np.linalg.eigvalsh(Jm @ Jm.T)[-1]); lam = float(P.gain(rp, ctx[0], ci[0], cv[0], ti[0, 0], tv[0, 0], gn_iters=50))
check("resnet: gain == dense SVD within 2%", abs(lam / lam_ref - 1) < 0.02, f"{lam:.4g} vs {lam_ref:.4g}")

# 7. ResNet: ePC weight gradient -> BP as rho >> gain (the large-precision limit), and with the training rule
gb = flat(P.grads(rp, ctx, ci, cv, ti, tv, Y, "bp", chunk=B)[0])
ge = flat(P.grads(rp, ctx, ci, cv, ti, tv, Y, "epc", rho=1e3 * max(lam, 1.0), alpha=0.0, T=50, chunk=B)[0])
check("resnet: ePC at rho = 1e3 x gain == BP (cos > 0.999)", cos(ge, gb) > 0.999, f"cos {cos(ge, gb):.5f}, ratio {np.linalg.norm(ge) / np.linalg.norm(gb):.4f}")
gk = flat(P.grads(rp, ctx, ci, cv, ti, tv, Y, "epc", rho=1.0, alpha=1.0, T=20, chunk=B, rho_rel=4.0)[0])
print(f"     report: resnet epc_alm_blk4 (T 20) cos to BP {cos(gk, gb):.4f}, ratio {np.linalg.norm(gk) / np.linalg.norm(gb):.3f}", flush=True)
gs = flat(P.grads(rp, ctx, ci, cv, ti, tv, Y, "spc", rho=1.0, alpha=0.0, T=20, chunk=B, hv_iters=8)[0])
print(f"     report: resnet spc_u1 (T 20) cos to BP {cos(gs, gb):.4f}, ratio {np.linalg.norm(gs) / np.linalg.norm(gb):.3f}", flush=True)
print("ALL PASS" if OK[0] else "SOME FAIL"); sys.exit(0 if OK[0] else 1)
