
import sys, time, dataclasses as dc
sys.path.insert(0, "/media/md_lab/hd_21/preprocessed/pc_predictor/Q1-PhaseA")
import jax, jax.numpy as jnp, equinox as eqx
import config as C, encoder as E, sigreg as S, train as T

base = C.RunCfg()
V, B, D_in = base.data.n_views, base.opt.batch_size, base.data.input_dim
Zin = jax.random.normal(jax.random.PRNGKey(0), (V, B, D_in))
params = E.init(jax.random.PRNGKey(base.opt.seed), base.enc, D_in)
A = S.sample_directions(jax.random.PRNGKey(2), base.enc.embed_dim, base.sig.num_slices)
key = jax.random.PRNGKey(3)

def timeit(fn, n=5):
    fn(); t0=time.time()
    for _ in range(n): out = fn()
    jax.block_until_ready(out); return (time.time()-t0)/n

tbp = timeit(lambda: T.bp_grads(params, Zin, A, base)[0])
print("BP warm step: %.2f ms" % (tbp*1e3))
print("%8s %5s %10s %10s %8s %13s" % ("drive","T","compile_s","warm_ms","x_BP","proj_2k"))
for drive in ["frozen", "coupled"]:
    for Tn in [8, 16, 64, 256]:
        cfg = dc.replace(base, pc=dc.replace(base.pc, T=Tn, output_drive=drive))
        t0=time.time(); g,_ = T.pc_grads(params, Zin, A, cfg, key); jax.block_until_ready(g)
        tc = time.time()-t0
        tw = timeit(lambda: T.pc_grads(params, Zin, A, cfg, key)[0])
        print("%8s %5d %10.1f %10.1f %8.0f %11.2fh" % (drive, Tn, tc, tw*1e3, tw/tbp, tw*2000/3600))
print("TIMING2_DONE")
