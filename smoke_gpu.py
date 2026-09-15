
import sys, time
sys.path.insert(0, "/media/md_lab/hd_21/preprocessed/pc_predictor/Q1-PhaseA")
import dataclasses as dc
import jax, jax.numpy as jnp, equinox as eqx
import config as C, encoder as E, sigreg as S, train as T

print("device:", jax.devices(), "backend:", jax.default_backend())
cfg = C.RunCfg()
V, B = cfg.data.n_views, cfg.opt.batch_size
D_in = cfg.data.input_dim
print(f"V={V} B={B} input_dim={D_in} depth={cfg.enc.depth} width={cfg.enc.width} T={cfg.pc.T}")

k = jax.random.PRNGKey(0)
Zin = jax.random.normal(k, (V, B, D_in))
params = E.init(jax.random.PRNGKey(cfg.opt.seed), cfg.enc, D_in)
A = S.sample_directions(jax.random.PRNGKey(2), cfg.enc.embed_dim, cfg.sig.num_slices)

def flat(g):
    lv = jax.tree_util.tree_leaves(eqx.filter(g, eqx.is_inexact_array))
    return jnp.concatenate([x.ravel() for x in lv])

t0=time.time(); gb, auxb = T.bp_grads(params, Zin, A, cfg); jax.block_until_ready(gb)
print(f"bp_grads   OK  compile+run {time.time()-t0:6.2f}s  |g|={float(jnp.linalg.norm(flat(gb))):.6e}")
t0=time.time(); gb2,_ = T.bp_grads(params, Zin, A, cfg); jax.block_until_ready(gb2)
print(f"bp_grads   warm run        {time.time()-t0:6.3f}s")

t0=time.time(); gp, auxp = T.pc_grads(params, Zin, A, cfg, jax.random.PRNGKey(3)); jax.block_until_ready(gp)
print(f"pc_grads   OK  compile+run {time.time()-t0:6.2f}s  |g|={float(jnp.linalg.norm(flat(gp))):.6e}")
t0=time.time(); gp2,_ = T.pc_grads(params, Zin, A, cfg, jax.random.PRNGKey(3)); jax.block_until_ready(gp2)
print(f"pc_grads   warm run        {time.time()-t0:6.3f}s")

a, b = flat(gb), flat(gp)
cos = float(a@b)/(float(jnp.linalg.norm(a))*float(jnp.linalg.norm(b)))
print(f"cos(bp, pc) = {cos:.6f}")
print(f"F_final={float(auxp['F_final']):.6e}  L_final={float(auxp['L_final']):.6f}")
print("SMOKE_OK")
