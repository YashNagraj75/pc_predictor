"""Freeze pcalm.relax / pcalm.grads outputs on a fixed input BEFORE the sparsity
edits.  test_sparse.py asserts the sparse.case == "none" path reproduces these."""
import os, sys
# GOLDEN_REF (set by test_sparse.py) points at the PRE-EDIT source; it must come
# first on the path so `import pcalm` etc. resolve to the old modules.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if os.environ.get("GOLDEN_REF"):
    sys.path.insert(0, os.environ["GOLDEN_REF"])
import dataclasses, numpy as np, jax, jax.numpy as jnp, equinox as eqx
import data as D, encoder as E, sigreg as S, pcalm as PA
from config import RunCfg, PCCfg, DataCfg, OptCfg

def setup(alpha, B=8, T=4):
    cfg = RunCfg(arm="pc", pc=PCCfg(T=T, eta_h=0.0625, alpha=alpha),
                 data=DataCfg(n_train=64, n_test=16), opt=OptCfg(batch_size=B))
    rng = np.random.default_rng(0)
    X = jnp.asarray(rng.random((B, 32, 32, 3), dtype=np.float32))
    Zin, _ = D.make_views(jax.random.PRNGKey(1), X, cfg.data)
    params = E.init(jax.random.PRNGKey(0), cfg.enc, Zin.shape[-1])
    A = S.sample_directions(jax.random.PRNGKey(2), cfg.enc.embed_dim, cfg.sig.num_slices)
    return cfg, params, Zin, A

def snapshot(alpha):
    cfg, params, Zin, A = setup(alpha)
    acts, duals, hist = PA.relax(params, Zin, A, cfg, jax.random.PRNGKey(3))
    g, aux = PA.grads(params, Zin, A, cfg, jax.random.PRNGKey(3))
    gl = jax.tree_util.tree_leaves(eqx.filter(g, eqx.is_inexact_array))
    z, hs = E.forward(params, Zin[0], cfg.enc)
    out = {"F": np.asarray(hist["F"]), "L": np.asarray(hist["L"]), "z_ff": np.asarray(z)}
    for v in range(len(acts)):
        for l, a in enumerate(acts[v]): out[f"acts_v{v}_l{l}"] = np.asarray(a)
        for l, d in enumerate(duals[v]): out[f"duals_v{v}_l{l}"] = np.asarray(d)
    for i, x in enumerate(gl): out[f"g{i}"] = np.asarray(x)
    return out

if __name__ == "__main__":
    for alpha in (0.0, 1.0):
        out = os.path.join(os.environ.get("GOLDEN_OUT", "."), f"golden_pcalm_alpha{int(alpha)}.npz")
        np.savez(out, **snapshot(alpha))
        print("wrote", out)
