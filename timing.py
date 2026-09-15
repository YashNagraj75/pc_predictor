
import sys, time, dataclasses as dc
sys.path.insert(0, "/media/md_lab/hd_21/preprocessed/pc_predictor/Q1-PhaseA")
import jax, jax.numpy as jnp, equinox as eqx
import config as C, encoder as E, sigreg as S, train as T

base = C.RunCfg()
V, B, D_in = base.data.n_views, base.opt.batch_size, base.data.input_dim
k = jax.random.PRNGKey(0)
Zin = jax.random.normal(k, (V, B, D_in))
params = E.init(jax.random.PRNGKey(base.opt.seed), base.enc, D_in)
A = S.sample_directions(jax.random.PRNGKey(2), base.enc.embed_dim, base.sig.num_slices)

def flat(g):
    lv = jax.tree_util.tree_leaves(eqx.filter(g, eqx.is_inexact_array))
    return jnp.concatenate([x.ravel() for x in lv])

gb, _ = T.bp_grads(params, Zin, A, base); jax.block_until_ready(gb)
t0=time.time()
for _ in range(5): gb2,_ = T.bp_grads(params, Zin, A, base)
jax.block_until_ready(gb2); bp_step = (time.time()-t0)/5
ref = flat(gb)
print(f"BP warm step: {bp_step*1e3:.2f} ms")
print(f"{'drive':8s} {'T':>4s} {'compile_s':>10s} {'warm_ms':>10s} {'x_BP':>8s} {'cos(bp)':>9s} {'2k_steps_h':>11s}")

for drive in ["coupled", "frozen"]:
    for Tn in [8, 16, 64, 256]:
        cfg = dc.replace(base, pc=dc.replace(base.pc, T=Tn, output_drive=drive))
        t0=time.time()
        g, aux = T.pc_grads(params, Zin, A, cfg, jax.random.PRNGKey(3)); jax.block_until_ready(g)
        comp = time.time()-t0
        t0=time.time()
        for _ in range(3): g2,_ = T.pc_grads(params, Zin, A, cfg, jax.random.PRNGKey(3))
        jax.block_until_ready(g2); step=(time.time()-t0)/3
        a=flat(g); cos=float(a@ref)/(float(jnp.linalg.norm(a))*float(jnp.linalg.norm(ref)))
        print(f"{drive:8s} {Tn:4d} {comp:10.2f} {step*1e3:10.1f} {step/bp_step:8.0f} {cos:9.6f} {step*2000/3600:11.2f}")
print("TIMING_OK")
