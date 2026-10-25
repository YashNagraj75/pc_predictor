import sys, os, json, dataclasses as dc
sys.path.insert(0, os.getcwd())
import numpy as np, jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp, equinox as eqx, jpc
import pred_ff as PF, pred_ffres as R, pred_gpc as G
tag = sys.argv[1]
print("TAG", tag, "device", jax.devices()[0], "jax", jax.__version__, "XLA_FLAGS", os.environ.get("XLA_FLAGS"), "TF32", os.environ.get("NVIDIA_TF32_OVERRIDE"), flush=True)
f32 = lambda t: jax.tree_util.tree_map(lambda v: v.astype(jnp.float32) if eqx.is_inexact_array(v) else v, t)
load = lambda path, like: eqx.tree_deserialise_leaves(path, f32(like))
f64 = lambda t: jax.tree_util.tree_map(lambda v: v.astype(jnp.float64) if eqx.is_inexact_array(v) else v, t)
flat = lambda t: jnp.concatenate([v.ravel().astype(jnp.float64) for v in jax.tree_util.tree_leaves(eqx.filter(t, eqx.is_inexact_array))])
def cmp(a, b):
    a, b = flat(a), flat(b)
    return {"cos": round(float(a @ b / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-300)), 6), "relmax": float(jnp.max(jnp.abs(a - b)) / (jnp.max(jnp.abs(b)) + 1e-300))}
out = {}
def case(name, fn, *args):
    ge = fn(*args); gj = eqx.filter_jit(fn)(*args); gr = fn(*[f64(a) for a in args])
    out[name] = {"jit_vs_eager": cmp(gj, ge), "eager_vs_f64": cmp(ge, gr), "jit_vs_f64": cmp(gj, gr)}
    print(name, out[name], flush=True)
ztr, ttr, *_ = PF.load("pred_latents.npz", "bp", "emb"); X, Y, *_ = PF.pairs(ztr, ttr, True)
idx = np.random.default_rng(999).integers(0, len(X), 512); x, y = jnp.asarray(X[idx]), jnp.asarray(Y[idx])
# 1. residual muPC network, trained BP and PC-ALM checkpoints, BP loss gradient
for arm, d in (("bp", 10), ("bp", 3), ("alm", 10)):
    cfg = R.RCfg(arm=arm, depth=d, seed=0)
    pth = f"pred_runs/{'frozen_bp' if arm == 'bp' else 'ffres'}/ffres_{arm}_d{d}_s0.eqx"
    p = load(pth, R.make(jax.random.PRNGKey(0), 146, 128, cfg))
    case(f"ffres_{arm}_d{d}_bpgrad", lambda p, x, y, cfg=cfg: eqx.filter_grad(R.bp_loss)(p, x, y, cfg), p, x, y)
    pi = f32(R.make(jax.random.PRNGKey(0), 146, 128, cfg))
    if arm == "bp":
        case(f"ffres_init_d{d}_bpgrad", lambda p, x, y, cfg=cfg: eqx.filter_grad(R.bp_loss)(p, x, y, cfg), pi, x, y)
cfg = R.RCfg(arm="alm", depth=10, seed=0, T=100, alpha=0.25, lr=0.01); p = load("pred_runs/ffres/ffres_alm_d10_s0.eqx", R.make(jax.random.PRNGKey(0), 146, 128, cfg))
case("ffres_alm_d10_pcalmgrad", lambda p, x, y: R.grads(p, x, y, cfg), p, x, y)
# 2. plain sp MLP as pred_ff (jpc.make_mlp sp, depth 3), BP gradient, own forward
def mlp_loss(m, x, y):
    h = x
    for l in m: h = jax.vmap(l)(h)
    return 0.5 * jnp.sum((y - h) ** 2) / x.shape[0]
m = f32(jpc.make_mlp(jax.random.PRNGKey(1), 146, 256, 3, 128, "relu", use_bias=True, param_type="sp"))
case("sp_mlp_d3_init_bpgrad", lambda m, x, y: eqx.filter_grad(mlp_loss)(m, x, y), m, x, y)
mm = f32(jpc.make_mlp(jax.random.PRNGKey(1), 146, 256, 3, 128, "relu", use_bias=True, param_type="mupc"))
case("mupc_mlp_d3_init_noscale_bpgrad", lambda m, x, y: eqx.filter_grad(mlp_loss)(m, x, y), mm, x, y)
# 3. bare matmul chain
W = [jax.random.normal(jax.random.PRNGKey(i), (256 if i else 146, 256), jnp.float32) for i in range(4)]
def chain(W, x):
    h = x
    for w in W: h = jax.nn.relu(h @ w) / 16.0
    return h
out["matmul_chain_fwd"] = {"jit_vs_eager": cmp(jax.jit(chain)(W, x), chain(W, x)), "eager_vs_f64": cmp(chain(W, x), chain(f64(W), f64(x)))}
print("matmul_chain_fwd", out["matmul_chain_fwd"], flush=True)
# 4. generative BP auto-decoder weight gradient at a frozen G1 checkpoint (d10 s1)
try:
    gc = G.GCfg(depth=10, rule="bp"); gp = load("pred_runs/tests/gpc_bp_d10_zero_bp_s1.eqx", G.init(jax.random.PRNGKey(0), gc))
    Z = jnp.asarray(ztr[:256]); Ph = G.phi(jnp.asarray(ttr[:256])); s = 0.3 * jax.random.normal(jax.random.PRNGKey(5), (256, G.M), jnp.float32)
    case("gpc_bp_d10_adwgrads", lambda gp, s, Z, Ph: G.ad_wgrads(gp, s, Z, Ph, gc), gp, s, Z, Ph)
except Exception as e:
    print("gpc case skipped:", type(e).__name__, str(e)[:200]); out["gpc_bp_d10_adwgrads"] = {"error": str(e)[:200]}
json.dump(out, open(f"jit_audit_{tag}.json", "w"), indent=1); print("DONE", tag)
