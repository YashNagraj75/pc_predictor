import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax, jax.numpy as jnp, equinox as eqx
import vit_ijepa as V

# ePC (Goemaere et al., ICML 2026) and state PC (sPC) for the I-JEPA transformer predictor, with optional ALM multipliers.
# One PC node per residual sublayer: h_j = h_{j-1} + s f_j(h_{j-1}) + eps_j, j = 1..2L, h_0 = embed(ctx); the head
# reads h_{2L}.  Energy per (image, target block):
#   E = rho/2 sum_j ||(eps_j + u_j) * row||^2 + 1/2 sum_valid ||head(h_2L) - y||^2,  u_j += alpha eps_j after each step.
# Weight (and context-input) gradients are the local PC gradients at the relaxed states (pc_e.py E_local), divided by
# N = (#valid target tokens) x dim, the normalisation of the BP loss V.masked_mse.  Inference is vmapped per sample.


def _gn_max(f, z0, iters):                       # lambda_max(J^T J) of f at z0, one sample, fixed start
    _, jvp = jax.linearize(f, z0)                # primal computed once, outside the loop (review N3)
    _, vjp = jax.vjp(f, z0)
    v = jnp.broadcast_to(jnp.linspace(1.0, 2.0, z0[0].size).reshape(z0[0].shape), z0.shape)
    v = v / jnp.linalg.norm(v)
    def body(_, v):
        w = vjp(jvp(v))[0]
        return w / (jnp.linalg.norm(w) + 1e-12)
    v = jax.lax.fori_loop(0, iters, body, v)
    return jnp.linalg.norm(vjp(jvp(v))[0])


def _hvp_max(E, z0, iters):                      # lambda_max of the Hessian of scalar E at z0 (power iteration on HVPs)
    g = jax.grad(E)
    hv = lambda v: jax.jvp(g, (z0,), (v,))[1]
    v = jnp.broadcast_to(jnp.linspace(1.0, 2.0, z0[0].size).reshape(z0[0].shape), z0.shape)
    v = v / jnp.linalg.norm(v)
    v = jax.lax.fori_loop(0, iters, lambda _, v: (lambda w: w / (jnp.linalg.norm(w) + 1e-12))(hv(v)), v)
    return jnp.linalg.norm(hv(v))


def _masks(cvalid, tvalid):
    kv = jnp.concatenate([cvalid, tvalid])
    return kv, kv[:, None].astype(jnp.float32), tvalid[:, None].astype(jnp.float32)


def states_from_errors(p, ctx, cidx, cvalid, tidx, tvalid, e):
    kv, row, _ = _masks(cvalid, tvalid)
    h, hs = p.embed_(ctx, cidx, tidx), []
    for j in range(p.n_sub):
        h = h + p.scale * p.f(j, h, kv) + e[j] * row; hs.append(h)
    return jnp.stack(hs)


def feedforward_states(p, ctx, cidx, cvalid, tidx, tvalid):
    z = jnp.zeros((p.n_sub, cidx.shape[0] + tidx.shape[0], p.pos.shape[1]))
    return states_from_errors(p, ctx, cidx, cvalid, tidx, tvalid, z)


# Local PC energy at given states hs (L2, N, w) and multipliers u: the only function weights and inputs are trained from.
def local_energy(p, ctx, cidx, cvalid, tidx, tvalid, hs, u, y, rho):
    kv, row, tm = _masks(cvalid, tvalid)
    prev, E = p.embed_(ctx, cidx, tidx), 0.0
    for j in range(p.n_sub):
        pred = prev + p.scale * p.f(j, prev, kv)
        E = E + 0.5 * rho * jnp.sum(((hs[j] - pred + u[j]) * row) ** 2)
        prev = hs[j]
    return E + 0.5 * jnp.sum(tm * (p.head(hs[-1], tidx.shape[0]) - y) ** 2)


# Returns states, multipliers and (energy at start, energy at end); the energy is not evaluated inside the loop (cost).
def epc_one(p, ctx, cidx, cvalid, tidx, tvalid, y, rho, alpha, T, c=1.0, gn_iters=12, hist=False):
    kv, row, tm = _masks(cvalid, tvalid)
    nt = tidx.shape[0]
    yhat = lambda e: p.head(states_from_errors(p, ctx, cidx, cvalid, tidx, tvalid, e)[-1], nt)
    e0 = jnp.zeros((p.n_sub, kv.shape[0], p.pos.shape[1]))
    lam = _gn_max(lambda e: yhat(e) * tm, e0, gn_iters)
    eta = c * 0.9 / (rho + lam)
    E = lambda e, u: 0.5 * rho * jnp.sum(((e + u) * row) ** 2) + 0.5 * jnp.sum(tm * (yhat(e) - y) ** 2)
    gE = jax.grad(E)
    def step(cu, t):
        e, u = cu
        e = e - eta * gE(e, u)
        u = jnp.where(t < T - 1, u + alpha * e * row, u)
        return (e, u), (E(e, u) if hist else None)
    u0 = jnp.zeros_like(e0)
    (e, u), H = jax.lax.scan(step, (e0, u0), jnp.arange(T))
    out = H if hist else jnp.stack([E(e0, u0), E(e, u)])
    return states_from_errors(p, ctx, cidx, cvalid, tidx, tvalid, e), u, out


def spc_one(p, ctx, cidx, cvalid, tidx, tvalid, y, rho, alpha, T, c=1.0, hv_iters=20, hist=False):
    kv, row, tm = _masks(cvalid, tvalid)
    h0 = feedforward_states(p, ctx, cidx, cvalid, tidx, tvalid)
    u0 = jnp.zeros_like(h0)
    E = lambda h, u: local_energy(p, ctx, cidx, cvalid, tidx, tvalid, h, u, y, rho)
    lam = _hvp_max(lambda h: E(h, u0), h0, hv_iters)
    eta = c * 0.9 / lam
    gE = jax.grad(E)
    def resid(h):
        prev, r = p.embed_(ctx, cidx, tidx), []
        for j in range(p.n_sub):
            r.append((h[j] - prev - p.scale * p.f(j, prev, kv)) * row); prev = h[j]
        return jnp.stack(r)
    def step(cu, t):
        h, u = cu
        h = h - eta * gE(h, u) * row
        u = jnp.where(t < T - 1, u + alpha * resid(h), u)
        return (h, u), (E(h, u) if hist else None)
    (h, u), H = jax.lax.scan(step, (h0, u0), jnp.arange(T))
    return h, u, (H if hist else jnp.stack([E(h0, u0), E(h, u)]))


# Output gain of one (image, target block): lambda_max(J^T J) of the valid predictions with respect to all node errors at
# zero error.  ePC behaves like BP only when rho >> gain; reported by the gates and logged in training.
def gain(p, ctx, cidx, cvalid, tidx, tvalid, gn_iters=50):
    kv, row, tm = _masks(cvalid, tvalid)
    f = lambda e: p.head(states_from_errors(p, ctx, cidx, cvalid, tidx, tvalid, e)[-1], tidx.shape[0]) * tm
    return _gn_max(f, jnp.zeros((p.n_sub, kv.shape[0], p.pos.shape[-1])), gn_iters)


def _relax(method, p, ctx, cidx, cvalid, tidx, tvalid, Y, rho, alpha, T, c):
    one = epc_one if method == "epc" else spc_one
    inner = lambda ti, tv, y: one(p, ctx, cidx, cvalid, ti, tv, y, rho, alpha, T, c)
    return jax.vmap(inner)(tidx, tvalid, Y)       # over the K target blocks of one image


def _chunked(xs, chunk):                         # (B, ...) -> (B/chunk, chunk, ...)
    return jax.tree_util.tree_map(lambda a: a.reshape((a.shape[0] // chunk, chunk) + a.shape[1:]), xs)


# Weight gradient (and, if want_input, the gradient with respect to the context embeddings) for a batch, processed in
# chunks of `chunk` images with lax.map so activation memory is bounded (review N4).  ctx (B, NC, dim) context-encoder
# output; Y (B, K, NT, dim) layer-normed targets.  method: bp | epc | spc.  Returns (grad_params, grad_ctx | None,
# energy (B, K, 2) at relaxation start/end | None).
def grads(p, ctx, cidx, cvalid, tidx, tvalid, Y, method, rho=64.0, alpha=1.0, T=20, c=1.0, want_input=False, chunk=32):
    B = ctx.shape[0]; chunk = min(chunk, B)
    assert B % chunk == 0, (B, chunk)
    N = jnp.sum(tvalid) * Y.shape[-1]

    def per_chunk(xs):
        cx, ci, cv, ti, tv, y = xs
        if method == "bp":
            def L(q, a):
                P = jax.vmap(lambda a1, c1, v1, t1, w1: jax.vmap(lambda tt, ww: q(a1, c1, v1, tt, ww))(t1, w1))(a, ci, cv, ti, tv)
                return 0.5 * jnp.sum(tv[..., None] * (P - y) ** 2)
            en = jnp.zeros((cx.shape[0], ti.shape[1], 2))
        else:
            HS, U, en = jax.vmap(lambda a, c1, v1, t1, w1, y1: _relax(method, p, a, c1, v1, t1, w1, y1, rho, alpha, T, c))(cx, ci, cv, ti, tv, y)
            HS, U = jax.lax.stop_gradient(HS), jax.lax.stop_gradient(U)
            def L(q, a):
                per_img = lambda a1, c1, v1, t1, w1, h1, u1, y1: jnp.sum(jax.vmap(
                    lambda tt, ww, hh, uu, yy: local_energy(q, a1, c1, v1, tt, ww, hh, uu, yy, rho))(t1, w1, h1, u1, y1))
                return jnp.sum(jax.vmap(per_img)(a, ci, cv, ti, tv, HS, U, y))
        gp = eqx.filter_grad(L)(p, cx)
        gx = jax.grad(lambda a: L(p, a))(cx) if want_input else jnp.zeros((0,))
        return gp, gx, en

    gps, gxs, ens = jax.lax.map(per_chunk, _chunked((ctx, cidx, cvalid, tidx, tvalid, Y), chunk))
    gp = jax.tree_util.tree_map(lambda x: x.sum(0) / N, gps)
    gx = gxs.reshape((B,) + gxs.shape[2:]) / N if want_input else None
    return gp, gx, (None if method == "bp" else ens.reshape((B,) + ens.shape[2:]))
