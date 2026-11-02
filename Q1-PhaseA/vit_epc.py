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
    _, vjp = jax.vjp(f, z0)
    v = jnp.broadcast_to(jnp.linspace(1.0, 2.0, z0[0].size).reshape(z0[0].shape), z0.shape)
    v = v / jnp.linalg.norm(v)
    def body(_, v):
        w = vjp(jax.jvp(f, (z0,), (v,))[1])[0]
        return w / (jnp.linalg.norm(w) + 1e-12)
    v = jax.lax.fori_loop(0, iters, body, v)
    return jnp.linalg.norm(vjp(jax.jvp(f, (z0,), (v,))[1])[0])


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


def epc_one(p, ctx, cidx, cvalid, tidx, tvalid, y, rho, alpha, T, c=1.0, gn_iters=20):
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
        return (e, u), E(e, u)
    (e, u), hist = jax.lax.scan(step, (e0, jnp.zeros_like(e0)), jnp.arange(T))
    return states_from_errors(p, ctx, cidx, cvalid, tidx, tvalid, e), u, hist


def spc_one(p, ctx, cidx, cvalid, tidx, tvalid, y, rho, alpha, T, c=1.0, hv_iters=20):
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
        return (h, u), E(h, u)
    (h, u), hist = jax.lax.scan(step, (h0, u0), jnp.arange(T))
    return h, u, hist


def _relax(method, p, ctx, cidx, cvalid, tidx, tvalid, Y, rho, alpha, T, c):
    one = epc_one if method == "epc" else spc_one
    inner = lambda ti, tv, y: one(p, ctx, cidx, cvalid, ti, tv, y, rho, alpha, T, c)
    return jax.vmap(inner)(tidx, tvalid, Y)       # over the K target blocks of one image


# Weight gradient (and, if want_input, the gradient with respect to the context embeddings) for a batch.
# ctx (B, NC, dim) context-encoder output; Y (B, K, NT, dim) layer-normed targets.  method: bp | epc | spc.
def grads(p, ctx, cidx, cvalid, tidx, tvalid, Y, method, rho=64.0, alpha=1.0, T=20, c=1.0, want_input=False):
    N = jnp.sum(tvalid) * Y.shape[-1]
    if method == "bp":
        def L(q, cx):
            P = jax.vmap(lambda a, ci, cv, ti, tv: jax.vmap(lambda t1, t2: q(a, ci, cv, t1, t2))(ti, tv))(cx, cidx, cvalid, tidx, tvalid)
            return 0.5 * jnp.sum(tvalid[..., None] * (P - Y) ** 2) / N
        hist = None
    else:
        HS, U, hist = jax.vmap(lambda a, ci, cv, ti, tv, y: _relax(method, p, a, ci, cv, ti, tv, y, rho, alpha, T, c))(
            ctx, cidx, cvalid, tidx, tvalid, Y)
        HS, U = jax.lax.stop_gradient(HS), jax.lax.stop_gradient(U)
        def L(q, cx):
            one = lambda a, ci, cv, ti, tv, hs, u, y: local_energy(q, a, ci, cv, ti, tv, hs, u, y, rho)
            per_img = lambda a, ci, cv, ti, tv, hs, u, y: jnp.sum(jax.vmap(lambda t1, t2, h1, u1, y1: one(a, ci, cv, t1, t2, h1, u1, y1))(ti, tv, hs, u, y))
            return jnp.sum(jax.vmap(per_img)(cx, cidx, cvalid, tidx, tvalid, HS, U, Y)) / N
    if want_input:
        gp, gx = eqx.filter_grad(lambda q, cx: L(q, cx))(p, ctx), jax.grad(lambda cx: L(p, cx))(ctx)
        return gp, gx, hist
    return eqx.filter_grad(lambda q: L(q, ctx))(p), None, hist
