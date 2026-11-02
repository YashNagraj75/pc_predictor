import dataclasses as dc, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp, equinox as eqx
jax.config.update("jax_default_matmul_precision", "highest")
import vit_ijepa as V, vit_epc as P

# Gates for vit_ijepa.py (V0) and vit_epc.py (transformer ePC / sPC), run on the host GPU before any training.
OK = [True]


def check(name, cond, info=""):
    OK[0] &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {info}", flush=True)


def flat(t):
    return jnp.concatenate([v.ravel() for v in jax.tree_util.tree_leaves(eqx.filter(t, eqx.is_inexact_array))])


def cos(a, b):
    a, b = flat(a), flat(b); return float(a @ b / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-30))


def rel(a, b):
    a, b = flat(a) if not isinstance(a, jax.Array) else a.ravel(), flat(b) if not isinstance(b, jax.Array) else b.ravel()
    return float(jnp.linalg.norm(a - b) / (jnp.linalg.norm(b) + 1e-30))


def batch(cfg, B, seed=0):
    rng = np.random.default_rng(seed)
    X = jnp.asarray(rng.random((B, cfg.img, cfg.img, 3), dtype=np.float32))
    return X, tuple(jnp.asarray(a) for a in V.sample_masks(rng, B, cfg))


def masks(cfg):
    rng = np.random.default_rng(1); bad = []
    for it in range(200):
        cidx, cvalid, tidx, tvalid = V.sample_masks(rng, 8, cfg)
        for b in range(8):
            c = set(cidx[b][cvalid[b]].tolist())
            tb = [tidx[b, k][tvalid[b, k]] for k in range(cfg.n_targets)]
            t = set(np.concatenate(tb).tolist())
            sizes = {len(x) for x in tb}
            rect = all(len(set((x // cfg.grid).tolist())) * len(set((x % cfg.grid).tolist())) == len(x) for x in tb)
            if c & t or len(c) < cfg.min_ctx or max(c | t) >= cfg.ntok or len(sizes) != 1 or not rect or len(c) != cvalid[b].sum():
                bad.append((it, b))
    check("masks: context and targets disjoint, in range, rectangular equal-size target blocks, >= min_ctx context", not bad, f"{len(bad)} bad")


def attention(cfg):
    blk = V.Block(jax.random.PRNGKey(0), 32, 4, 64)
    x = jax.random.normal(jax.random.PRNGKey(1), (10, 32)); kv = jnp.arange(10) < 7
    y = blk.attn(x, kv)
    q, k, v = jnp.split(blk.qkv(blk.ln1(x)), 3, -1); dh = 8; outs = []
    for h in range(4):
        qh, kh, vh = q[:, h * dh:(h + 1) * dh], k[:7, h * dh:(h + 1) * dh], v[:7, h * dh:(h + 1) * dh]
        a = qh @ kh.T / math.sqrt(dh); a = jnp.exp(a - a.max(1, keepdims=True)); a = a / a.sum(1, keepdims=True)
        outs.append(a @ vh)
    ref = blk.proj(jnp.concatenate(outs, 1))
    check("attention == per-head reference with masked keys", rel(y, ref) < 1e-5, f"{rel(y, ref):.1e}")
    x2 = x.at[7:].set(100.0)
    check("padded (invalid) tokens cannot change valid outputs", float(jnp.max(jnp.abs(blk.attn(x2, kv)[:7] - y[:7]))) < 1e-5)


def patches(cfg):
    X, _ = batch(cfg, 2)
    t = V.patchify(X[0], cfg.patch)
    check("patchify / unpatchify round trip exact; token 0 is the top-left patch",
          bool(jnp.all(V.unpatchify(t, cfg.patch, cfg.img, cfg.img) == X[0])) and bool(jnp.all(t[0] == X[0, :4, :4].reshape(-1))))


def padding_and_ema(cfg):
    enc, pred, tenc = V.init(cfg); X, (ci, cv, ti, tv) = batch(cfg, 4)
    z = enc(X[0], ci[0], cv[0], cfg.patch)
    ci2 = jnp.where(cv[0], ci[0], (ci[0] + 13) % cfg.ntok)
    z2 = enc(X[0], ci2, cv[0], cfg.patch)
    check("encoder: changing padded context indices leaves valid outputs unchanged", float(jnp.max(jnp.abs((z - z2)[cv[0]]))) < 1e-5)
    p1 = pred(z, ci[0], cv[0], ti[0, 0], tv[0, 0])
    t2 = jnp.where(tv[0, 0], ti[0, 0], (ti[0, 0] + 7) % cfg.ntok)
    p2 = pred(z, ci[0], cv[0], t2, tv[0, 0])
    check("predictor: changing padded target indices leaves valid predictions unchanged", float(jnp.max(jnp.abs((p1 - p2)[tv[0, 0]]))) < 1e-5)
    enc2 = jax.tree_util.tree_map(lambda a: a + 1.0 if eqx.is_inexact_array(a) else a, enc)
    t3 = V.ema_update(tenc, enc2, jnp.float32(0.9))
    ok = all(bool(jnp.allclose(a, 0.9 * b + 0.1 * c, atol=1e-7)) for a, b, c in zip(jax.tree_util.tree_leaves(t3), jax.tree_util.tree_leaves(tenc), jax.tree_util.tree_leaves(enc2)))
    check("EMA update == m * target + (1 - m) * online, every leaf", ok)
    P_, Y_ = jax.random.normal(jax.random.PRNGKey(2), (2, 4, V.NT_MAX, 8)), jax.random.normal(jax.random.PRNGKey(3), (2, 4, V.NT_MAX, 8))
    m = np.asarray(tv[:2]); manual = 0.5 * np.mean(((np.asarray(P_) - np.asarray(Y_)) ** 2)[m])
    check("masked_mse == manual mean over valid tokens", abs(float(V.masked_mse(P_, Y_, tv[:2])) - manual) < 1e-6)


def train_smoke(cfg):
    enc, pred, tenc = V.init(cfg); X, (ci, cv, ti, tv) = batch(cfg, 32, seed=5)
    opt, _ = V.optimiser(dc.replace(cfg, warmup=0, epochs=1), 100)
    ost = opt.init(eqx.filter((enc, pred), eqx.is_inexact_array)); step = V.make_step(cfg, opt)
    l0 = None
    for i in range(40):
        (enc, pred), tenc, ost, l = step((enc, pred), tenc, ost, X, ci, cv, ti, tv, jnp.float32(0.996))
        l0 = float(l) if l0 is None else l0
    check("training step: loss falls on a fixed batch (40 steps) and stays finite", float(l) < 0.7 * l0 and np.isfinite(float(l)), f"{l0:.4f} -> {float(l):.4f}")
    gj = eqx.filter_jit(eqx.filter_grad(V.loss_fn))((enc, pred), tenc, X, ci, cv, ti, tv, cfg)
    with jax.disable_jit():
        ge = eqx.filter_grad(V.loss_fn)((enc, pred), tenc, X, ci, cv, ti, tv, cfg)
    check("compiled == eager loss gradient (GPU audit)", rel(gj, ge) < 1e-4 and cos(gj, ge) > 0.99999, f"rel {rel(gj, ge):.1e}")


def epc_gates(cfg):
    pc = dc.replace(cfg, pdepth=2, pdim=64, pmlp=128, pheads=4, pscale=True)
    enc, pred, _ = V.init(pc); X, (ci, cv, ti, tv) = batch(pc, 4, seed=7)
    ctx = jax.vmap(lambda im, a, b: enc(im, a, b, pc.patch))(X, ci, cv)
    Y = 0.5 * jax.random.normal(jax.random.PRNGKey(9), ti.shape + (pc.dim,))
    # zero errors == forward pass
    hs = P.feedforward_states(pred, ctx[0], ci[0], cv[0], ti[0, 0], tv[0, 0])
    yf = pred.head(hs[-1], ti.shape[-1]); yp = pred(ctx[0], ci[0], cv[0], ti[0, 0], tv[0, 0])
    check("ePC: zero errors reproduce the predictor's forward pass", rel(yf, yp) < 1e-6)
    # error energy == local energy at the ePC state
    hs, u, hist = P.epc_one(pred, ctx[0], ci[0], cv[0], ti[0, 0], tv[0, 0], Y[0, 0], 4.0, 0.0, 30)
    El = P.local_energy(pred, ctx[0], ci[0], cv[0], ti[0, 0], tv[0, 0], hs, u, Y[0, 0], 4.0)
    check("ePC: error energy == local (state) energy at the relaxed state", abs(float(hist[-1] - El)) / float(El) < 1e-5, f"{float(hist[-1]):.5f} vs {float(El):.5f}")
    hh = np.asarray(hist); check("ePC: energy non-increasing (alpha 0, smooth GELU/softmax network)", np.all(np.diff(hh) <= 1e-6 * np.abs(hh[:-1])), f"{hh[0]:.4f} -> {hh[-1]:.4f}")
    # same minimum as sPC (smooth energy; both from the feedforward state)
    he, ue, _ = P.epc_one(pred, ctx[0], ci[0], cv[0], ti[0, 0], tv[0, 0], Y[0, 0], 1.0, 0.0, 3000)
    hp, up, _ = P.spc_one(pred, ctx[0], ci[0], cv[0], ti[0, 0], tv[0, 0], Y[0, 0], 1.0, 0.0, 30000)
    gE = jax.grad(lambda h: P.local_energy(pred, ctx[0], ci[0], cv[0], ti[0, 0], tv[0, 0], h, jnp.zeros_like(h), Y[0, 0], 1.0))
    s0 = float(jnp.linalg.norm(gE(P.feedforward_states(pred, ctx[0], ci[0], cv[0], ti[0, 0], tv[0, 0]))))
    st = float(jnp.linalg.norm(gE(he))) / s0
    check("ePC and sPC reach the same minimum (rho 1), ePC stationary", rel(he, hp) < 1e-3 and st < 1e-3, f"state rel {rel(he, hp):.1e}, stationarity {st:.1e}")
    # large-rho limit == BP, weights and context input
    gb, xb, _ = P.grads(pred, ctx, ci, cv, ti, tv, Y, "bp", want_input=True)
    gL, xL, _ = P.grads(pred, ctx, ci, cv, ti, tv, Y, "epc", rho=1e4, alpha=0.0, T=60, want_input=True)
    check("ePC at rho 1e4: weight gradient == BP (cos > 0.999, ratio within 2%)", cos(gL, gb) > 0.999 and abs(float(jnp.linalg.norm(flat(gL)) / jnp.linalg.norm(flat(gb))) - 1) < 0.02,
          f"cos {cos(gL, gb):.5f}")
    cx = float(jnp.sum(xL * xb) / (jnp.linalg.norm(xL) * jnp.linalg.norm(xb)))
    check("ePC at rho 1e4: context-input gradient == BP's (cos > 0.999)", cx > 0.999, f"cos {cx:.5f}")
    for rho, al in ((16.0, 1.0), (64.0, 1.0), (4.0, 0.0), (1.0, 0.0)):
        g, _, _ = P.grads(pred, ctx, ci, cv, ti, tv, Y, "epc", rho=rho, alpha=al, T=20)
        print(f"     report: ePC rho {rho:g} alpha {al:g} T 20 (random weights): cos to BP {cos(g, gb):.4f}, ratio {float(jnp.linalg.norm(flat(g)) / jnp.linalg.norm(flat(gb))):.3f}")
    with jax.disable_jit():
        ge = P.grads(pred, ctx, ci, cv, ti, tv, Y, "epc", rho=64.0, alpha=1.0, T=5)[0]
    gj5 = eqx.filter_jit(lambda q: P.grads(q, ctx, ci, cv, ti, tv, Y, "epc", rho=64.0, alpha=1.0, T=5)[0])(pred)
    check("ePC gradient: compiled == eager (GPU audit, T 5)", rel(gj5, ge) < 1e-4, f"rel {rel(gj5, ge):.1e}")


def timing(cfg):
    for depth in (6, 10):
        pc = dc.replace(cfg, pdepth=depth, pdim=192, pmlp=768, pheads=3, pscale=True)
        enc, pred, _ = V.init(pc); X, (ci, cv, ti, tv) = batch(pc, 64, seed=11)
        ctx = jax.vmap(lambda im, a, b: enc(im, a, b, pc.patch))(X, ci, cv)
        Y = jax.random.normal(jax.random.PRNGKey(3), ti.shape + (pc.dim,))
        for meth in ("bp", "epc"):
            f = eqx.filter_jit(lambda q: P.grads(q, ctx, ci, cv, ti, tv, Y, meth, rho=64.0, alpha=1.0, T=20)[0])
            jax.block_until_ready(f(pred)); t0 = time.time()
            for _ in range(3):
                jax.block_until_ready(f(pred))
            print(f"     report: predictor depth {depth} width 192, batch 64 x 4 blocks, {meth}: {(time.time() - t0) / 3:.3f} s/step", flush=True)


def main():
    cfg = V.VCfg()
    masks(cfg); attention(cfg); patches(cfg); padding_and_ema(cfg); train_smoke(cfg); epc_gates(cfg); timing(cfg)
    print("ALL PASS" if OK[0] else "SOME FAIL")
    return OK[0]


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
