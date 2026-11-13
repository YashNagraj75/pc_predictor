import dataclasses as dc, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp, equinox as eqx
jax.config.update("jax_default_matmul_precision", "highest")
import pred_ff as PF, pred_ffres as R, pred_gpc as G
import diag_e1_credit as DC

# Inference-only check of ePC (Goemaere et al., ICML 2026) vs standard state PC (sPC) on trained
# depth-10 checkpoints. ePC: errors eps_l are free variables, activities s_l = f_l(s_{l-1}) + eps_l,
# inference is GD on E = rho/2 sum||eps+u||^2 + 1/2||y-yhat||^2; the weight gradient is the usual
# local PC gradient at those activities (same equilibrium as sPC). Measures cosine/norm ratio of the
# weight gradient to BP's, per layer, residual size, and settling, vs budget T and precision rho.
OUT = sys.argv[1] if len(sys.argv) > 1 else "pred_runs/diag_epc.json"
TS = (20, 100, 400, 2000)


def flat(t):
    """Flattens a pytree's inexact-array leaves into a single 1D array."""
    return jnp.concatenate([v.ravel() for v in jax.tree_util.tree_leaves(eqx.filter(t, eqx.is_inexact_array))])


def cos(u, v):
    """Returns the cosine similarity between two pytrees' flattened parameters."""
    u, v = flat(u), flat(v)
    return float(u @ v / (jnp.linalg.norm(u) * jnp.linalg.norm(v) + 1e-30))


def ratio(u, v):
    """Returns the ratio of the flattened L2 norms of u and v."""
    return float(jnp.linalg.norm(flat(u)) / (jnp.linalg.norm(flat(v)) + 1e-30))


def gn_max(f, z0, iters=30):
    """Returns per-sample largest eigenvalue of J^T J for row-wise map f, via power iteration from a fixed start.
    Assumes sample i of the output depends only on sample i of each input leaf; leaves carry the batch on axis 0."""
    def pnorm(v):
        """Returns the per-sample L2 norm of pytree v (batch on axis 0)."""
        return jnp.sqrt(sum(jnp.sum(x.reshape(x.shape[0], -1) ** 2, 1) for x in jax.tree_util.tree_leaves(v)))
    def scale(v, n):
        """Scales pytree v per-sample by dividing by n."""
        return jax.tree_util.tree_map(lambda x: x / n.reshape((-1,) + (1,) * (x.ndim - 1)), v)
    _, vjp = jax.vjp(f, z0)
    v = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(jnp.linspace(1.0, 2.0, x[0].size).reshape(x[0].shape), x.shape), z0)
    v = scale(v, pnorm(v))
    def body(_, v):
        """Runs one power-iteration step and returns the rescaled updated vector."""
        w = vjp(jax.jvp(f, (z0,), (v,))[1])[0]
        return scale(w, pnorm(w) + 1e-12)
    v = jax.lax.fori_loop(0, iters, body, v)
    return pnorm(vjp(jax.jvp(f, (z0,), (v,))[1])[0])


# ------------------------------------------------------------------ feedforward (pred_ffres network)
def ff_states(params, e, x, a):
    """Returns the per-layer states h_l = f_l(h_{l-1}) + e_l for the feedforward (pred_ffres) network."""
    prev, h = x, []
    for l in range(len(e)):
        prev = R.layer_pred(params, l, prev, a) + e[l]; h.append(prev)
    return h


def ff_epc(params, x, y, cfg, T, rho, alpha, c=1.0, return_hist=False):
    """Runs T ePC inference steps on the feedforward network's errors and returns the local-PC weight
    gradient, states, duals and errors (plus energy history if return_hist)."""
    a = R.scalings(params, x, cfg)
    h0 = R.activities(params, x, cfg)[:-1]
    e0 = [jnp.zeros_like(v) for v in h0]; u0 = [jnp.zeros_like(v) for v in h0]
    yhat = lambda e: R.layer_pred(params, len(e), ff_states(params, e, x, a)[-1], a)
    lam = gn_max(yhat, e0)                                       # per sample, (B,)
    eta = (c * 0.9 / (rho + lam))[:, None]
    def E(e, u):                                                 # per-sample sums (no 1/B): separable across samples
        """Returns the per-sample ePC energy for errors e and duals u."""
        return 0.5 * (rho * sum(jnp.sum((ei + ui) ** 2) for ei, ui in zip(e, u)) + jnp.sum((y - yhat(e)) ** 2))
    gfn = jax.grad(E)
    def step(c_, t):
        """Runs one ePC primal/dual GD step and returns the new (e, u) carry and energy."""
        e, u = c_
        e = [ei - eta * gi for ei, gi in zip(e, gfn(e, u))]
        u = [jnp.where(t < T - 1, ui + alpha * ei, ui) for ui, ei in zip(u, e)]
        return (e, u), E(e, u)
    (e, u), hist = jax.lax.scan(step, (e0, u0), jnp.arange(T))
    h = jax.lax.stop_gradient(ff_states(params, e, x, a)); u = jax.lax.stop_gradient(u)
    g = eqx.filter_grad(lambda p: R.energy(p, h, x, y, u, a, rho))(params)   # local PC weight gradient (E_local)
    out = (g, h, u, e)
    return out + (hist,) if return_hist else out


# ------------------------------------------------------------------ generative (pred_gpc network)
def gen_states(p, s, e, Ph, H):                                  # e[j] is the error of hidden level j+1 (xs[j])
    """Returns the generative hidden states via top-down prediction from belief s plus errors e."""
    xs = [None] * H
    xs[H - 1] = G.mu_top(p, s, Ph) + e[H - 1]
    for l in range(H - 1, 0, -1):
        xs[l - 1] = G.mu_mid(p["mid"][l - 1], xs[l], H) + e[l - 1]
    return xs


def gen_epc(p, Z, Ph, cfg, T, s_fixed=None, c=1.0):
    """Runs T steps of block-coordinate ePC inference (belief s, errors e) on the generative network.
    Returns the settled belief, states, and energy history; s is held fixed if s_fixed is given."""
    H, b = cfg.H, Z.shape[0]
    s0 = jnp.zeros((b, G.M)) if s_fixed is None else s_fixed
    e0 = [jnp.zeros_like(v) for v in G.topdown(p, s0, Ph, H)[0]]
    # Block steps (belief s, errors e): at large hid_prec the two blocks' curvatures differ by orders of magnitude.
    # Block-Jacobi preconditioning with 2 blocks has lambda_max(P^-1 H) <= 2, so 0.45 c / D_block is stable.
    lam_s = gn_max(lambda s_: G.mu_bot(p, gen_states(p, s_, e0, Ph, H)[0]), s0)
    lam_e = gn_max(lambda e_: G.mu_bot(p, gen_states(p, s0, e_, Ph, H)[0]), e0)
    eta_s = c * 0.45 / (cfg.prior + cfg.out_w * lam_s)                    # (b,)
    eta = c * 0.45 / (cfg.hid_prec + cfg.out_w * lam_e)
    def E(v):
        """Returns the generative PC energy for state/error pair v = (s, e)."""
        xs = gen_states(p, v[0], v[1], Ph, H)
        return G.energy(p, v[0], xs, Z, Ph, H, cfg.prior, cfg.out_w, cfg.hid_prec)
    gfn = jax.grad(E)
    def step(v, _):
        """Runs one gradient step on (s, e) (s frozen if s_fixed is set) and returns the new carry and energy."""
        gs, ge = gfn(v)
        s = v[0] if s_fixed is not None else v[0] - eta_s[:, None] * gs
        return (s, [ei - eta[:, None, None] * gi for ei, gi in zip(v[1], ge)]), E(v)
    (s, e), hist = jax.lax.scan(step, (s0, e0), jnp.arange(T))
    xs = jax.lax.stop_gradient(gen_states(p, s, e, Ph, H)); s = jax.lax.stop_gradient(s)
    return s, xs, hist


def gen_metrics(p, s, xs, Z, Ph, cfg, s_ad, g_ad):
    """Returns cosine/ratio of the PC weight gradient to the auto-decoder reference, plus belief
    distance, observation nmse, and hidden residual."""
    g = G.pc_wgrads(p, s, xs, Z, Ph, cfg)
    blocks = {"top": cos(g["top"], g_ad["top"]), "bot": cos(g["bot"], g_ad["bot"]),
              "mid_first": cos(g["mid"][0], g_ad["mid"][0]), "mid_last": cos(g["mid"][-1], g_ad["mid"][-1])}
    return {"cos": cos(g, g_ad), "ratio": ratio(g, g_ad), "block_cos": blocks,
            "s_dist_ad": float(jnp.linalg.norm(s - s_ad) / (jnp.linalg.norm(s_ad) + 1e-30)),
            "obs_nmse": float(jnp.sum((Z - G.mu_bot(p, xs[0])) ** 2) / jnp.sum(Z ** 2)),
            "hid_res": float(jnp.sqrt(sum(e for e in G.level_energies(p, s, xs, Z, Ph, cfg.H)[1:]) * 2) / jnp.linalg.norm(Z))}


# ------------------------------------------------------------------ checkpoints
def ff_ckpts():
    """Returns available feedforward checkpoints (ALM-R16 seeds and the BP baseline) as (name, cfg, params) tuples."""
    out = []
    for s in (0, 1, 2):
        f = f"pred_runs/ffres_fix/ffres_almR16_d10_s{s}"
        if os.path.exists(f + ".eqx"):
            c = json.load(open(f + ".json"))["cfg"]
            cfg = R.RCfg(arm="alm", depth=10, seed=s, T=c["T"], alpha=c["alpha"], lr=c["lr"], rho=c.get("rho", 1.0))
            out.append((f"F_almR16_s{s}", cfg, R.load_model(f + ".eqx", cfg)))
    f = "pred_runs/ffres_bplr/bp_s0"
    if os.path.exists(f + ".eqx"):
        cfg = R.RCfg(arm="bp", depth=10, seed=0, lr=0.01); out.append(("BPf_lr1e-2_s0", cfg, R.load_model(f + ".eqx", cfg)))
    return out


def gen_ckpts():
    """Returns available generative checkpoints (PC and BP rule seeds) as (name, cfg, params) tuples."""
    out = []
    for rule, seeds in (("pc", (0, 1, 2)), ("bp", (1, 2))):
        for s in seeds:
            base = f"pred_runs/tests/gpc_{rule}_d10_zero_bp_s{s}"
            if os.path.exists(base + ".eqx"):
                d = json.load(open(base + ".json"))["cfg"]
                cfg = G.GCfg(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if k in {f.name for f in dc.fields(G.GCfg)}})
                out.append((f"gen_{rule}_d10_s{s}", cfg, eqx.tree_deserialise_leaves(base + ".eqx", G.init(jax.random.PRNGKey(0), cfg))))
    return out


def gen_batch(n=256, k=3):
    """Returns n standardized bp_emb embeddings and their k-view tau features from pred_latents.npz."""
    L = np.load("pred_latents.npz")
    raw = L["bp_emb_tr"]; mu, sd = raw.reshape(-1, 128).mean(0), raw.reshape(-1, 128).std(0) + 1e-6
    idx = np.random.default_rng(999).choice(len(raw), n, replace=False)
    return jnp.asarray((raw[idx, :k] - mu) / sd), G.phi(jnp.asarray(L["tau_tr"][idx, :k].astype(np.float32)))


def dump(res):
    """Writes res to OUT via an atomic tmp-file rename."""
    tmp = OUT + ".tmp"; json.dump(res, open(tmp, "w"), indent=1); os.replace(tmp, OUT)


def main():
    """Runs the ePC-vs-sPC/BP credit-assignment sweep over feedforward and generative checkpoints and writes OUT."""
    res = {"ff": [], "gen": [], "ckpts": {}}
    ztr, ttr, *_ = PF.load("pred_latents.npz", "bp", "emb")
    X, Y, *_ = PF.pairs(ztr, ttr, True)
    pidx = np.random.default_rng(999).integers(0, len(X), 512)
    x, y = jnp.asarray(X[pidx]), jnp.asarray(Y[pidx])
    ff = ff_ckpts(); res["ckpts"]["ff"] = [n for n, *_ in ff]
    for name, cfg, params in ff:
        gb = eqx.filter_grad(R.bp_loss)(params, x, y, cfg)
        for rho in (1.0, 4.0, 16.0, 64.0, 256.0, 1024.0):
            for alpha in (0.0, 1.0):
                if rho > 64 and alpha > 0:
                    continue
                for T in TS:
                    for meth in ("ePC", "sPC"):
                        if meth == "sPC" and rho not in (1.0, 16.0):
                            continue
                        t0 = time.time()
                        if meth == "ePC":
                            g, h, u, e = ff_epc(params, x, y, cfg, T, rho, alpha)
                        else:
                            g, h, u, _, _ = DC.run(params, x, y, cfg, T, rho, alpha)
                        fin = bool(jnp.all(jnp.isfinite(flat(g))))
                        row = {"ckpt": name, "method": meth, "rho": rho, "alpha": alpha, "T": T, "finite": fin, "sec": round(time.time() - t0, 1)}
                        if fin:
                            a = R.scalings(params, x, cfg); r = R.residuals(params, h, x, a)
                            row.update({"cos": cos(g, gb), "ratio": ratio(g, gb), "layer_cos": [round(cos(p_, q_), 4) for p_, q_ in zip(g[0], gb[0])],
                                        "res_rel": float(jnp.sqrt(sum(jnp.sum(v ** 2) for v in r) / sum(jnp.sum(v ** 2) for v in h)))})
                        res["ff"].append(row)
                        print("FF", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k != "layer_cos"}), flush=True)
            dump(res)
    Z, Ph = gen_batch(); gen = gen_ckpts(); res["ckpts"]["gen"] = [n for n, *_ in gen]
    for name, cfg0, p in gen:
        # reference: BP auto-decoder inference, settled (4 x 400 steps, BP's step scale 0.25 -- stable on all weights)
        cad = dc.replace(cfg0, c=0.25, out_w=1.0, hid_prec=1.0)
        s_ad = jax.lax.stop_gradient(G.ad_infer(p, jnp.zeros((Z.shape[0], G.M)), Z, Ph, cad, 1600)[0])
        g_ad = G.ad_wgrads(p, s_ad, Z, Ph, cad)
        # the reference belief s_ad is NOT settled at depth 10 (AD loss still falling at 25,600 steps, dbg c43fdb0b): the
        # primary credit measure is therefore the fixed-s rows (PC vs AD weight gradient at the same s); free-s rows
        # report belief distance to this reference only descriptively.
        sz = jnp.zeros((Z.shape[0], G.M)); e0 = [jnp.zeros_like(v) for v in G.topdown(p, sz, Ph, cfg0.H)[0]]
        gain = E_gain = gn_max(lambda e_: G.mu_bot(p, gen_states(p, sz, e_, Ph, cfg0.H)[0]), e0)
        res["ckpts"].setdefault("gen_gain", {})[name] = {"median": float(jnp.median(gain)), "max": float(jnp.max(gain))}
        for hp in (1.0, 4.0, 16.0, 64.0, 256.0, 4096.0):
            cfg = dc.replace(cfg0, hid_prec=hp, out_w=1.0, spike="none", rule="pc")
            for T in TS:
                for meth in ("ePC", "sPC", "ePC_fixed_s", "sPC_fixed_s"):
                    if meth.startswith("sPC") and hp not in (1.0, 16.0):
                        continue
                    t0 = time.time()
                    if meth == "ePC":
                        s, xs, hist = gen_epc(p, Z, Ph, cfg, T)
                    elif meth == "ePC_fixed_s":
                        s, xs, hist = gen_epc(p, Z, Ph, cfg, T, s_fixed=s_ad)
                    else:
                        s0 = s_ad if meth == "sPC_fixed_s" else jnp.zeros((Z.shape[0], G.M))
                        s, xs, hist, _ = G.pc_infer(p, s0, Z, Ph, cfg, T, freeze_s=(meth == "sPC_fixed_s"), spike_on=False)
                    h_ = np.asarray(hist)
                    row = {"ckpt": name, "method": meth, "hid_prec": hp, "T": T, "c_spc": cfg.c, "sec": round(time.time() - t0, 1),
                           "finite": bool(np.all(np.isfinite(h_))), "monotone": float(np.mean(np.diff(h_) <= 1e-6 * np.abs(h_[:-1]))) if T > 1 else 1.0}
                    if row["finite"]:
                        row.update(gen_metrics(p, s, xs, Z, Ph, cfg, s_ad, g_ad))
                    res["gen"].append(row)
                    print("GEN", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k != "block_cos"}), flush=True)
            dump(res)
    print("DONE")


if __name__ == "__main__":
    main()
