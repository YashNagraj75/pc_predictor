import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax, jax.numpy as jnp, equinox as eqx
import vit_ijepa as V

# ePC (Goemaere et al., ICML 2026) and state PC (sPC) for the I-JEPA transformer predictor, with optional ALM multipliers.
# One PC node per residual sublayer: h_j = h_{j-1} + s f_j(h_{j-1}) + eps_j, j=1..2L, h_0=embed(ctx); head reads h_{2L}.
# E = rho/2 sum_j ||(eps_j+u_j)*row||^2 + 1/2 sum_valid ||head(h_2L)-y||^2, u_j += alpha*eps_j after each step.
# Weight/input gradients are the local PC gradients at relaxed states, divided by N = (#valid target tokens) x dim
#   (matches V.masked_mse's normalisation). Inference is vmapped per sample.


def _gn_max(f, z0, iters):                       # lambda_max(J^T J) of f at z0, one sample, fixed start
    """Returns lambda_max(J^T J) of f at z0 via power iteration on the Jacobian."""
    _, jvp = jax.linearize(f, z0)                # primal computed once, outside the loop (review N3)
    _, vjp = jax.vjp(f, z0)
    v = jnp.broadcast_to(jnp.linspace(1.0, 2.0, z0[0].size).reshape(z0[0].shape), z0.shape)
    v = v / jnp.linalg.norm(v)
    def body(_, v):
        """Applies one power-iteration step (J^T J v, renormalised) to v."""
        w = vjp(jvp(v))[0]
        return w / (jnp.linalg.norm(w) + 1e-12)
    v = jax.lax.fori_loop(0, iters, body, v)
    return jnp.linalg.norm(vjp(jvp(v))[0])


def _hvp_max(E, z0, iters):                      # lambda_max of the Hessian of scalar E at z0 (power iteration on HVPs)
    """Returns lambda_max of the Hessian of scalar function E at z0 via power iteration on Hessian-vector products."""
    g = jax.grad(E)
    hv = lambda v: jax.jvp(g, (z0,), (v,))[1]
    v = jnp.broadcast_to(jnp.linspace(1.0, 2.0, z0[0].size).reshape(z0[0].shape), z0.shape)
    v = v / jnp.linalg.norm(v)
    v = jax.lax.fori_loop(0, iters, lambda _, v: (lambda w: w / (jnp.linalg.norm(w) + 1e-12))(hv(v)), v)
    return jnp.linalg.norm(hv(v))


def _masks(cvalid, tvalid):
    """Returns the combined context+target validity mask and its row/target broadcast forms."""
    kv = jnp.concatenate([cvalid, tvalid])
    return kv, kv[:, None].astype(jnp.float32), tvalid[:, None].astype(jnp.float32)


def states_from_errors(p, ctx, cidx, cvalid, tidx, tvalid, e):
    """Returns the stacked per-sublayer states obtained by adding node errors e to the feedforward pass."""
    kv, row, _ = _masks(cvalid, tvalid)
    h, hs = p.embed_(ctx, cidx, tidx), []
    for j in range(p.n_sub):
        h = h + p.scale * p.f(j, h, kv) + e[j] * row; hs.append(h)
    return jnp.stack(hs)


def feedforward_states(p, ctx, cidx, cvalid, tidx, tvalid):
    """Returns the stacked per-sublayer states from a pure feedforward (zero-error) pass."""
    z = jnp.zeros((p.n_sub, cidx.shape[0] + tidx.shape[0], p.pos.shape[-1]))
    return states_from_errors(p, ctx, cidx, cvalid, tidx, tvalid, z)


# Local PC energy at given states hs (L2, N, w) and multipliers u: the only function weights and inputs are trained from.
# If the node errors e are given, the residual is evaluated as e + u - (pred - stop_gradient(pred)): same value and
# gradient, without recomputing e as the float32 difference hs - pred (cancellation at large rho).
def local_energy(p, ctx, cidx, cvalid, tidx, tvalid, hs, u, y, rho, e=None):
    """Returns the local PC energy at states hs and multipliers u; the only function weights/inputs train from."""
    kv, row, tm = _masks(cvalid, tvalid)
    prev, E = p.embed_(ctx, cidx, tidx), 0.0
    for j in range(p.n_sub):
        pred = prev + p.scale * p.f(j, prev, kv)
        r = (hs[j] - pred) if e is None else (e[j] - (pred - jax.lax.stop_gradient(pred)))
        E = E + 0.5 * rho * jnp.sum(((r + u[j]) * row) ** 2)
        prev = hs[j]
    return E + 0.5 * jnp.sum(tm * (p.head(hs[-1], tidx.shape[0]) - y) ** 2)


# Returns states, multipliers, (energy at start, energy at end) and this block's rho.  rho_rel > 0 (per-block precision,
# findings/40): rho = max(1, gain / rho_rel) from this block's own gain lam, which the step size already measures.
def epc_core(p, ctx, cidx, cvalid, tidx, tvalid, y, rho, alpha, T, c=1.0, gn_iters=12, hist=False, rho_rel=0.0):
    """Runs T steps of ePC relaxation and returns (states, multipliers, energy trace, final errors, rho used)."""
    kv, row, tm = _masks(cvalid, tvalid)
    nt = tidx.shape[0]
    yhat = lambda e: p.head(states_from_errors(p, ctx, cidx, cvalid, tidx, tvalid, e)[-1], nt)
    e0 = jnp.zeros((p.n_sub, kv.shape[0], p.pos.shape[-1]))
    lam = _gn_max(lambda e: yhat(e) * tm, e0, gn_iters)
    if rho_rel > 0:
        rho = jax.lax.stop_gradient(jnp.maximum(1.0, lam / rho_rel))
    rho = jnp.asarray(rho, jnp.float32)
    eta = c * 0.9 / (rho + lam)
    E = lambda e, u: 0.5 * rho * jnp.sum(((e + u) * row) ** 2) + 0.5 * jnp.sum(tm * (yhat(e) - y) ** 2)
    gE = jax.grad(E)
    def step(cu, t):
        """Applies one ePC gradient step to node errors e and updates the ALM multipliers u."""
        e, u = cu
        e = e - eta * gE(e, u)
        u = jnp.where(t < T - 1, u + alpha * e * row, u)
        return (e, u), (E(e, u) if hist else None)
    u0 = jnp.zeros_like(e0)
    (e, u), H = jax.lax.scan(step, (e0, u0), jnp.arange(T))
    out = H if hist else jnp.stack([E(e0, u0), E(e, u)])
    return states_from_errors(p, ctx, cidx, cvalid, tidx, tvalid, e), u, out, e * row, rho


def epc_one(*a, **k):                            # the original 4-tuple interface
    """Returns epc_core's (states, multipliers, energy, errors) 4-tuple, dropping the rho."""
    return epc_core(*a, **k)[:4]


def spc_one(p, ctx, cidx, cvalid, tidx, tvalid, y, rho, alpha, T, c=1.0, hv_iters=20, hist=False):
    """Runs T steps of state-PC relaxation and returns (states, multipliers, energy trace, final residuals)."""
    kv, row, tm = _masks(cvalid, tvalid)
    h0 = feedforward_states(p, ctx, cidx, cvalid, tidx, tvalid)
    u0 = jnp.zeros_like(h0)
    E = lambda h, u: local_energy(p, ctx, cidx, cvalid, tidx, tvalid, h, u, y, rho)
    lam = _hvp_max(lambda h: E(h, u0), h0, hv_iters)
    eta = c * 0.9 / lam
    gE = jax.grad(E)
    def resid(h):
        """Returns the per-sublayer feedforward residuals of states h, masked by row validity."""
        prev, r = p.embed_(ctx, cidx, tidx), []
        for j in range(p.n_sub):
            r.append((h[j] - prev - p.scale * p.f(j, prev, kv)) * row); prev = h[j]
        return jnp.stack(r)
    def step(cu, t):
        """Applies one sPC gradient step to states h and updates the ALM multipliers u."""
        h, u = cu
        h = h - eta * gE(h, u) * row
        if not (isinstance(alpha, (int, float)) and alpha == 0):   # static alpha 0: no multipliers, skip the extra pass
            u = jnp.where(t < T - 1, u + alpha * resid(h), u)
        return (h, u), (E(h, u) if hist else None)
    (h, u), H = jax.lax.scan(step, (h0, u0), jnp.arange(T))
    return h, u, (H if hist else jnp.stack([E(h0, u0), E(h, u)])), resid(h)


# Output gain of one (image, target block): lambda_max(J^T J) of the valid predictions with respect to all node errors at
# zero error.  ePC behaves like BP only when rho >> gain; reported by the gates and logged in training.
def gain(p, ctx, cidx, cvalid, tidx, tvalid, gn_iters=50):
    """Returns the output gain lambda_max(J^T J) of valid predictions w.r.t. node errors at zero error."""
    kv, row, tm = _masks(cvalid, tvalid)
    f = lambda e: p.head(states_from_errors(p, ctx, cidx, cvalid, tidx, tvalid, e)[-1], tidx.shape[0]) * tm
    return _gn_max(f, jnp.zeros((p.n_sub, kv.shape[0], p.pos.shape[-1])), gn_iters)


def _relax(method, p, ctx, cidx, cvalid, tidx, tvalid, Y, rho, alpha, T, c, gn_iters=12, rho_rel=0.0, hv_iters=20):
    """Runs the chosen relaxation method (epc or spc) vmapped over the K target blocks of one image."""
    if method == "epc":
        one = lambda *a: epc_core(*a, gn_iters=gn_iters, rho_rel=rho_rel)
    else:
        assert rho_rel == 0.0, "per-block rho is implemented for ePC only"
        one = lambda *a: spc_one(*a, hv_iters=hv_iters) + (jnp.asarray(rho, jnp.float32),)
    inner = lambda ti, tv, y: one(p, ctx, cidx, cvalid, ti, tv, y, rho, alpha, T, c)
    return jax.vmap(inner)(tidx, tvalid, Y)       # over the K target blocks of one image; rho per block last


def _chunked(xs, chunk):                         # (B, ...) -> (B/chunk, chunk, ...)
    """Reshapes every leaf of pytree xs from a leading batch axis into (B/chunk, chunk, ...)."""
    return jax.tree_util.tree_map(lambda a: a.reshape((a.shape[0] // chunk, chunk) + a.shape[1:]), xs)


# Weight gradient (and, if want_input, the gradient with respect to the context embeddings) for a batch, processed in
# chunks of `chunk` images with lax.map so activation memory is bounded (review N4).  ctx (B, NC, dim) context-encoder
# output; Y (B, K, NT, dim) layer-normed targets.  method: bp | epc | spc.  Returns (grad_params, grad_ctx | None,
# energy (B, K, 2) at relaxation start/end | None).
def grads(p, ctx, cidx, cvalid, tidx, tvalid, Y, method, rho=64.0, alpha=1.0, T=20, c=1.0, want_input=False, chunk=32, gn_iters=12,
          rho_rel=0.0, return_rho=False, hv_iters=20):
    """Returns (weight grads, input grads or None, relaxation energies or None[, rho used]) for a batch, chunked."""
    B = ctx.shape[0]; chunk = min(chunk, B)
    assert B % chunk == 0, (B, chunk)
    N = jnp.sum(tvalid) * Y.shape[-1]

    def per_chunk(xs):
        """Computes weight (and optional input) gradients for one chunk of the batch."""
        cx, ci, cv, ti, tv, y = xs
        if method == "bp":
            def L(q, a):
                """Returns the masked BP MSE loss of model q on context a over this chunk."""
                P = jax.vmap(lambda a1, c1, v1, t1, w1: jax.vmap(lambda tt, ww: q(a1, c1, v1, tt, ww))(t1, w1))(a, ci, cv, ti, tv)
                return 0.5 * jnp.sum(tv[..., None] * (P - y) ** 2)
            en = jnp.zeros((cx.shape[0], ti.shape[1], 2)); RB = jnp.zeros((cx.shape[0], ti.shape[1]))
        else:
            HS, U, en, EE, RB = jax.vmap(lambda a, c1, v1, t1, w1, y1: _relax(method, p, a, c1, v1, t1, w1, y1, rho, alpha, T, c, gn_iters, rho_rel, hv_iters))(cx, ci, cv, ti, tv, y)
            HS, U, EE, RB = (jax.lax.stop_gradient(x) for x in (HS, U, EE, RB))
            exact = method == "epc"
            def L(q, a):                         # each block's weight gradient uses that block's rho (RB, (chunk, K))
                """Returns the total local PC energy of model q on context a at the (stop-gradient) relaxed states."""
                per_img = lambda a1, c1, v1, t1, w1, h1, u1, y1, e1, r1: jnp.sum(jax.vmap(
                    lambda tt, ww, hh, uu, yy, ee, rr: local_energy(q, a1, c1, v1, tt, ww, hh, uu, yy, rr, ee if exact else None))(t1, w1, h1, u1, y1, e1, r1))
                return jnp.sum(jax.vmap(per_img)(a, ci, cv, ti, tv, HS, U, y, EE, RB))
        gp = eqx.filter_grad(L)(p, cx)
        gx = jax.grad(lambda a: L(p, a))(cx) if want_input else jnp.zeros((0,))
        return gp, gx, en, RB

    gps, gxs, ens, rbs = jax.lax.map(per_chunk, _chunked((ctx, cidx, cvalid, tidx, tvalid, Y), chunk))
    gp = jax.tree_util.tree_map(lambda x: x.sum(0) / N, gps)
    gx = gxs.reshape((B,) + gxs.shape[2:]) / N if want_input else None
    en = None if method == "bp" else ens.reshape((B,) + ens.shape[2:])
    if return_rho:                               # the rho each (image, block) actually used, (B, K)
        return gp, gx, en, rbs.reshape((B,) + rbs.shape[2:])
    return gp, gx, en
