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


def leafwise(g, gb):                             # per-leaf cosine and norm ratio over array leaves (review EPC F2)
    out = []
    for a, b in zip(jax.tree_util.tree_leaves(eqx.filter(g, eqx.is_inexact_array)), jax.tree_util.tree_leaves(eqx.filter(gb, eqx.is_inexact_array))):
        nb = float(jnp.linalg.norm(b))
        if nb < 1e-12:
            continue
        out.append((float(jnp.sum(a * b) / (jnp.linalg.norm(a) * nb + 1e-30)), float(jnp.linalg.norm(a)) / nb))
    return min(c for c, _ in out), max(abs(r - 1) for _, r in out)


def setup(pc, B=4, seed=7, trained=0):
    enc, pred, _ = V.init(pc); X, (ci, cv, ti, tv) = batch(pc, B, seed=seed)
    ctx = jax.vmap(lambda im, a, b: enc(im, a, b, pc.patch))(X, ci, cv)
    Y = V._ln(jax.random.normal(jax.random.PRNGKey(9), ti.shape + (pc.dim,)))
    if trained:                                  # non-trivial weights: BP steps on (ctx, Y) so softmax/LN/head are active
        import optax
        opt = optax.adam(3e-3); st = opt.init(eqx.filter(pred, eqx.is_inexact_array))
        @eqx.filter_jit
        def stp(q, st):
            g = P.grads(q, ctx, ci, cv, ti, tv, Y, "bp")[0]
            u, st = opt.update(g, st, eqx.filter(q, eqx.is_inexact_array)); return eqx.apply_updates(q, u), st
        for _ in range(trained):
            pred, st = stp(pred, st)
    return pred, ctx, ci, cv, ti, tv, Y


def epc_gates(cfg):
    for tag, pc, trained in (("depth 2, init", dc.replace(cfg, pdepth=2, pdim=64, pmlp=128, pheads=4, pscale=True), 0),
                             ("depth 6, trained 150 BP steps", dc.replace(cfg, pdepth=6, pdim=64, pmlp=128, pheads=4, pscale=True), 150)):
        pred, ctx, ci, cv, ti, tv, Y = setup(pc, trained=trained)
        a1 = (pred, ctx[0], ci[0], cv[0], ti[0, 0], tv[0, 0])
        hs = P.feedforward_states(*a1)
        check(f"[{tag}] ePC: zero errors reproduce the predictor's forward pass", rel(pred.head(hs[-1], ti.shape[-1]), pred(*a1)) < 1e-5)
        for al in (0.0, 1.0):
            hs, u, h = P.epc_one(*a1, Y[0, 0], 4.0, al, 30, hist=True)
            El = P.local_energy(*a1, hs, u, Y[0, 0], 4.0); hh = np.asarray(h)
            check(f"[{tag}] ePC alpha {al:g}: error energy == local (state) energy, multipliers included", abs(float(h[-1] - El)) / float(El) < 1e-5, f"{float(h[-1]):.5f} vs {float(El):.5f}")
            if al == 0.0:
                check(f"[{tag}] ePC: energy non-increasing", np.all(np.diff(hh) <= 1e-5 * hh[0]), f"{hh[0]:.4f} -> {hh[-1]:.4f}")
        # same minimum as sPC (rho 1, no multipliers)
        he, _, _ = P.epc_one(*a1, Y[0, 0], 1.0, 0.0, 3000)
        hp, _, _ = P.spc_one(*a1, Y[0, 0], 1.0, 0.0, 30000)
        gE = jax.grad(lambda h_: P.local_energy(*a1, h_, jnp.zeros_like(h_), Y[0, 0], 1.0))
        st = float(jnp.linalg.norm(gE(he))) / float(jnp.linalg.norm(gE(P.feedforward_states(*a1))))
        check(f"[{tag}] ePC and sPC reach the same minimum (rho 1), ePC stationary", rel(he, hp) < 1e-3 and st < 1e-3, f"state rel {rel(he, hp):.1e}, stationarity {st:.1e}")
        gb, xb, _ = P.grads(pred, ctx, ci, cv, ti, tv, Y, "bp", want_input=True)
        gL, xL, _ = P.grads(pred, ctx, ci, cv, ti, tv, Y, "epc", rho=1e4, alpha=0.0, T=60, want_input=True)
        cmin, rmax = leafwise(gL, gb)
        check(f"[{tag}] ePC at rho 1e4: every weight leaf == BP (cos > 0.999, ratio within 2%)", cmin > 0.999 and rmax < 0.02, f"min cos {cmin:.5f}, max |ratio-1| {rmax:.4f}")
        cx = float(jnp.sum(xL * xb) / (jnp.linalg.norm(xL) * jnp.linalg.norm(xb))); rx = float(jnp.linalg.norm(xL) / jnp.linalg.norm(xb))
        check(f"[{tag}] ePC at rho 1e4: context-input gradient == BP's", cx > 0.999 and abs(rx - 1) < 0.02, f"cos {cx:.5f} ratio {rx:.4f}")
        # ALM gate (review EPC F1): multipliers must bring moderate-rho credit close to BP, and beat the alpha 0 control
        gA, _, _ = P.grads(pred, ctx, ci, cv, ti, tv, Y, "epc", rho=4.0, alpha=1.0, T=400)
        g0, _, _ = P.grads(pred, ctx, ci, cv, ti, tv, Y, "epc", rho=4.0, alpha=0.0, T=400)
        rA = float(jnp.linalg.norm(flat(gA)) / jnp.linalg.norm(flat(gb)))
        check(f"[{tag}] ePC-ALM rho 4, T 400: weight gradient == BP (cos > 0.99, ratio within 5%) and alpha 0 is clearly worse",
              cos(gA, gb) > 0.99 and abs(rA - 1) < 0.05 and cos(g0, gb) < cos(gA, gb) - 0.01, f"ALM cos {cos(gA, gb):.4f} ratio {rA:.3f}; alpha 0 cos {cos(g0, gb):.4f}")
        for rho, al in ((16.0, 1.0), (64.0, 1.0), (4.0, 0.0), (1.0, 0.0)):
            g, _, en = P.grads(pred, ctx, ci, cv, ti, tv, Y, "epc", rho=rho, alpha=al, T=20)
            fin = all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree_util.tree_leaves(eqx.filter(g, eqx.is_inexact_array)))
            check(f"[{tag}] ePC rho {rho:g} alpha {al:g} T 20: finite, energy lower at the end for every block", fin and bool(jnp.all(en[..., 1] <= en[..., 0] * (1 + 1e-5))),
                  f"cos to BP {cos(g, gb):.4f}, ratio {float(jnp.linalg.norm(flat(g)) / jnp.linalg.norm(flat(gb))):.3f}")
    # padding invariance of the PC gradients (review EPC F4)
    pc = dc.replace(cfg, pdepth=2, pdim=64, pmlp=128, pheads=4, pscale=True)
    pred, ctx, ci, cv, ti, tv, Y = setup(pc)
    ci2 = jnp.where(cv, ci, (ci + 5) % pc.ntok); ti2 = jnp.where(tv, ti, (ti + 3) % pc.ntok)
    ctx2 = jnp.where(cv[..., None], ctx, 100.0); Y2 = jnp.where(tv[..., None], Y, 100.0)
    for meth in ("epc", "spc"):
        g1, x1, _ = P.grads(pred, ctx, ci, cv, ti, tv, Y, meth, rho=4.0, alpha=1.0, T=20, want_input=True)
        g2, x2, _ = P.grads(pred, ctx2, ci2, cv, ti2, tv, Y2, meth, rho=4.0, alpha=1.0, T=20, want_input=True)
        pad0 = float(jnp.max(jnp.abs(jnp.where(cv[..., None], 0.0, x2))))
        check(f"{meth}: padded indices, contexts and targets cannot change the gradients; padded context rows get zero gradient",
              rel(g2, g1) < 1e-5 and float(jnp.max(jnp.abs(jnp.where(cv[..., None], x2 - x1, 0.0)))) < 1e-5 * float(jnp.max(jnp.abs(x1))) and pad0 == 0.0,
              f"rel {rel(g2, g1):.1e}")
    with jax.disable_jit():
        ge = P.grads(pred, ctx, ci, cv, ti, tv, Y, "epc", rho=64.0, alpha=1.0, T=5)[0]
    gj = eqx.filter_jit(lambda q, r, a: P.grads(q, ctx, ci, cv, ti, tv, Y, "epc", rho=r, alpha=a, T=5)[0])(pred, jnp.float32(64.0), jnp.float32(1.0))
    check("ePC gradient: compiled (traced rho, alpha) == eager (GPU audit, T 5)", rel(gj, ge) < 1e-4, f"rel {rel(gj, ge):.1e}")


def scale(cfg):                                  # real batch: memory and time per step (review N3/N4)
    cap = jax.devices()[0].memory_stats().get("bytes_limit", 0) if jax.devices()[0].platform == "gpu" else 0
    for depth in (6, 10):
        pc = dc.replace(cfg, pdepth=depth, pdim=192, pmlp=768, pheads=3, pscale=True)
        enc, pred, _ = V.init(pc); X, (ci, cv, ti, tv) = batch(pc, 256, seed=11)
        ctx = jax.vmap(lambda im, a, b: enc(im, a, b, pc.patch))(X, ci, cv)
        Y = V._ln(jax.random.normal(jax.random.PRNGKey(3), ti.shape + (pc.dim,)))
        for meth in ("bp", "epc"):
            f = eqx.filter_jit(lambda q, r, a: P.grads(q, ctx, ci, cv, ti, tv, Y, meth, rho=r, alpha=a, T=20, want_input=True)[:2])
            jax.block_until_ready(f(pred, jnp.float32(64.0), jnp.float32(1.0))); t0 = time.time()
            for _ in range(2):
                jax.block_until_ready(f(pred, jnp.float32(64.0), jnp.float32(1.0)))
            dt = (time.time() - t0) / 2
            peak = jax.devices()[0].memory_stats().get("peak_bytes_in_use", 0) if cap else 0
            print(f"     report: predictor depth {depth} width 192, batch 256 x 4 blocks, {meth}: {dt:.3f} s/step, peak {peak / 2**30:.2f} GiB", flush=True)
            if cap:
                check(f"depth {depth} {meth}: real batch fits in memory (peak < 85% of {cap / 2**30:.1f} GiB)", peak < 0.85 * cap, f"{peak / 2**30:.2f} GiB")


def main():
    cfg = V.VCfg()
    masks(cfg); attention(cfg); patches(cfg); padding_and_ema(cfg); train_smoke(cfg); epc_gates(cfg); scale(cfg)
    print("ALL PASS" if OK[0] else "SOME FAIL")
    return OK[0]


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
