
import sys, time, dataclasses as dc, logging
sys.path.insert(0, "/media/md_lab/hd_21/preprocessed/pc_predictor/Q1-PhaseA")
import jax, jax.numpy as jnp, equinox as eqx
import config as C, encoder as E, sigreg as S, train as T

base = C.RunCfg()
V, B, D_in = base.data.n_views, base.opt.batch_size, base.data.input_dim
Zin = jax.random.normal(jax.random.PRNGKey(0), (V, B, D_in))
params = E.init(jax.random.PRNGKey(base.opt.seed), base.enc, D_in)
A = S.sample_directions(jax.random.PRNGKey(2), base.enc.embed_dim, base.sig.num_slices)
key = jax.random.PRNGKey(3)

print("cfg hashable:", end=" ")
try:
    hash(base); print("YES")
except Exception as e:
    print("NO ->", type(e).__name__, e)

print("\n--- per-call timing, SAME cfg object, T=16 ---")
cfg = dc.replace(base, pc=dc.replace(base.pc, T=16))
for i in range(6):
    t0=time.time(); g,_ = T.pc_grads(params, Zin, A, cfg, key); jax.block_until_ready(g)
    print(f"  call {i}: {(time.time()-t0)*1e3:8.1f} ms")

print("\n--- flat-in-T check at tiny T, same cfg reused ---")
for Tn in [1, 2, 4, 32]:
    cfg_t = dc.replace(base, pc=dc.replace(base.pc, T=Tn))
    T.pc_grads(params, Zin, A, cfg_t, key)          # warm this cfg
    t0=time.time()
    for _ in range(3): g,_ = T.pc_grads(params, Zin, A, cfg_t, key)
    jax.block_until_ready(g)
    print(f"  T={Tn:4d}: {(time.time()-t0)/3*1e3:8.1f} ms")

print("\n--- compile log: 3 identical calls (any output below = recompile) ---")
logging.basicConfig(level=logging.WARNING)
with jax.log_compiles(True):
    for i in range(3):
        g,_ = T.pc_grads(params, Zin, A, cfg, key); jax.block_until_ready(g)
        print(f"  [after call {i}]")
print("DIAG_OK")
