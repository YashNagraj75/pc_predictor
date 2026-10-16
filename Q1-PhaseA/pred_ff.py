import argparse, dataclasses, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import jax, jax.numpy as jnp, equinox as eqx, optax, jpc

# Pre-registered rule for choosing the target layer (pred_ff.py check); stored verbatim in its output JSON.
TARGET_RULE = 'target-space rule (written before the check was run, 2026-10-12):\non the BP backbone, for each layer, tau_gain = (nmse_noTau - nmse_full) / nmse_noTau.\nUse "emb" if its tau_gain >= 0.10 AND full beats copy by >= 10 % relative nmse;\notherwise the layer with the largest tau_gain among those where full beats copy by\n>= 10 %; if none qualifies, STOP -- no layer of this encoder carries enough view\ninformation and the predictor experiment needs a different target (patch level, or\na non-invariant backbone).  The same layer is then used for the untrained backbone.'

WIDTH, DEPTH, ACT = 256, 3, "relu"
PAIRS = [(a, b) for a in range(4) for b in range(4) if a != b]


@dataclasses.dataclass
class Cfg:
    backbone: str = "bp"
    layer: str = "emb"
    arm: str = "bp"
    seed: int = 0
    steps: int = 6000
    batch: int = 512
    lr: float = 1e-3
    wd: float = 1e-4
    T: int = 20
    eta_h: float = 0.05
    use_tau: bool = True
    adaptive_eta: bool = False
    alpha: float = 1.0          # PC-ALM dual rate (arm 'alm')
    holdout: bool = False       # train on the first 90% of train images, evaluate on the last 10% (selection only)


# ------------------------------------------------------------------ data
def load(path, backbone, layer):
    d = np.load(path)
    ztr, zte = d[f"{backbone}_{layer}_tr"].astype(np.float32), d[f"{backbone}_{layer}_te"].astype(np.float32)
    mu, sd = ztr.reshape(-1, ztr.shape[-1]).mean(0), ztr.reshape(-1, ztr.shape[-1]).std(0) + 1e-6
    return ((ztr - mu) / sd, d["tau_tr"].astype(np.float32), (zte - mu) / sd, d["tau_te"].astype(np.float32),
            d["ytr"], d["yte"])


# Returns X (N*12, in), Y (N*12, d), and (img, a, b) index arrays.
def pairs(z, tau, use_tau=True):
    A = np.array([p[0] for p in PAIRS]); Bv = np.array([p[1] for p in PAIRS])
    za, zb = z[:, A], z[:, Bv]                                   # (N, 12, d)
    parts = [za] + ([tau[:, A], tau[:, Bv]] if use_tau else [])
    X = np.concatenate(parts, -1).reshape(-1, sum(p.shape[-1] for p in parts))
    Y = zb.reshape(-1, z.shape[-1])
    img = np.repeat(np.arange(z.shape[0]), len(PAIRS))
    return X, Y, img, np.tile(A, z.shape[0]), np.tile(Bv, z.shape[0])


# ------------------------------------------------------------------ model
def make(key, din, dout):
    return jpc.make_mlp(key, din, WIDTH, DEPTH, dout, ACT, use_bias=True, param_type="sp")


def forward(model, x):
    h = x
    for i, layer in enumerate(model):
        h = jax.vmap(layer)(h)
    return h


def bp_loss(model, x, y):
    return 0.5 * jnp.sum((y - forward(model, x)) ** 2) / x.shape[0]


def bp_grads(model, x, y, cfg):
    return eqx.filter_grad(bp_loss)(model, x, y)


def pc_grads(model, x, y, cfg, return_acts=False):
    params = (model, None)
    acts = list(jpc.init_activities_with_ffwd(model, x, param_type="sp"))
    B = x.shape[0]
    eta = cfg.eta_h
    if getattr(cfg, "adaptive_eta", False):
        # per-sample activity Hessian of hidden layer l is bounded by 1 + ||W_{l+1}||_2^2 (ReLU slope <= 1);
        # step 0.9 / max_l of that, recomputed from the CURRENT weights (no gradient)
        sq = [jnp.linalg.norm(jax.lax.stop_gradient(l.layers[1].weight if hasattr(l, "layers") else l.weight), 2) ** 2
              for l in model[1:]]
        # joint Hessian over all hidden blocks: off-diagonal coupling adds up to max||W||^2 more (Gershgorin)
        eta = 0.9 / (1.0 + 2.0 * jnp.max(jnp.stack(sq)))

    def step(acts, _):
        g = list(jpc.compute_pc_activity_grad(params, acts, y, x=x, param_type="sp")[1])   # -> (energy, grads)
        new = [a - eta * B * ga for a, ga in zip(acts[:-1], g[:-1])] + [acts[-1]]
        return new, jpc.pc_energy_fn(params, new, y, x=x, param_type="sp")

    acts, F = jax.lax.scan(step, acts, None, length=cfg.T)
    g = jpc.compute_pc_param_grads(params, acts, y, x=x, param_type="sp")[0]
    return (g, acts, F) if return_acts else g


# f_l(z_{l-1}) for every layer, same order as acts
# jpc 'sp' mode: no scalings, no skips
def _layer_preds(model, acts, x):
    return [jax.vmap(model[l])(x if l == 0 else acts[l - 1]) for l in range(len(model))]


# PC-ALM augmented Lagrangian, completing-the-square form (pcalm.al_energy, rho=1), batch mean like jpc:
# L = 1/(2B) [ sum_{l<L} ||r_l + lam_l||^2 + ||y - f_L(z_{L-1})||^2 ],  r_l = z_l - f_l(z_{l-1}).
# Output clamped to target y, carries no multiplier (supervised PC-ALM, SakanaAI Algorithm 1).
# duals = None or all zero reproduces jpc.pc_energy_fn exactly (gated).
def alm_energy(model, acts, x, y, duals):
    preds = _layer_preds(model, acts, x)
    r = [a - f for a, f in zip(acts[:-1], preds[:-1])] + [y - preds[-1]]
    if duals is not None:
        r = [ri + li for ri, li in zip(r[:-1], duals)] + [r[-1]]
    return 0.5 * sum(jnp.sum(ri ** 2) for ri in r) / x.shape[0]


# PC-ALM weight gradient: T primal steps on L (same curvature-tracking step as plain-PC arm),
# each followed by dual step lam_l += alpha * r_l, except after the last step.
# Weights taken from L at the final state.
# alpha = 0 reduces exactly to plain PC (pc_grads).
def alm_grads(model, x, y, cfg, return_acts=False):
    acts = list(jpc.init_activities_with_ffwd(model, x, param_type="sp"))
    B = x.shape[0]
    sq = [jnp.linalg.norm(jax.lax.stop_gradient(l.layers[1].weight if hasattr(l, "layers") else l.weight), 2) ** 2
          for l in model[1:]]
    eta = 0.9 / (1.0 + 2.0 * jnp.max(jnp.stack(sq))) if cfg.adaptive_eta else cfg.eta_h
    duals = [jnp.zeros_like(a) for a in acts[:-1]]
    gfn = jax.grad(lambda h, lam: alm_energy(model, h + [acts[-1]], x, y, lam))

    def step(carry, t):
        h, lam = carry
        g = gfn(h, lam)
        h = [a - eta * B * ga for a, ga in zip(h, g)]
        preds = _layer_preds(model, h + [acts[-1]], x)
        lam = jax.lax.cond(t < cfg.T - 1, lambda L: [li + cfg.alpha * (a - f) for li, a, f in zip(L, h, preds[:-1])],
                           lambda L: L, lam)
        return (h, lam), alm_energy(model, h + [acts[-1]], x, y, lam)

    (h, lam), F = jax.lax.scan(step, (acts[:-1], duals), jnp.arange(cfg.T))
    hs = [jax.lax.stop_gradient(a) for a in h]; lam = [jax.lax.stop_gradient(l) for l in lam]
    g = eqx.filter_grad(lambda m: alm_energy(m, hs + [acts[-1]], x, y, lam))(model)
    return (g, hs + [acts[-1]], F, lam) if return_acts else g


def optimiser(cfg):
    sched = optax.warmup_cosine_decay_schedule(0.0, cfg.lr, 200, cfg.steps)
    return optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=cfg.wd))


def _flat(g):
    return jnp.concatenate([l.ravel() for l in jax.tree_util.tree_leaves(eqx.filter(g, eqx.is_inexact_array))])


# ------------------------------------------------------------------ eval
def evaluate(model, z, tau, use_tau=True, copy=False, n_retr=2000):
    X, Y, img, A, Bv = pairs(z, tau, use_tau)
    if copy:
        P = X[:, :z.shape[-1]]
    else:
        P = np.concatenate([np.asarray(forward(model, jnp.asarray(X[i:i + 8192]))) for i in range(0, len(X), 8192)])
    nmse = float(np.mean(np.sum((P - Y) ** 2, -1)) / np.mean(np.sum(Y ** 2, -1)))
    cos = float(np.mean(np.sum(P * Y, -1) / (np.linalg.norm(P, axis=-1) * np.linalg.norm(Y, axis=-1) + 1e-12)))
    # retrieval@1 among the first n_retr images, candidates = their view-B latents
    keep = img < n_retr
    Pk, ik, bk = P[keep], img[keep], Bv[keep]
    hits = []
    for b in range(4):
        sel = bk == b
        C = z[:n_retr, b]                                         # (n, d) candidates
        d2 = (np.sum(Pk[sel] ** 2, -1)[:, None] - 2 * Pk[sel] @ C.T + np.sum(C ** 2, -1)[None])
        hits.append(np.argmin(d2, 1) == ik[sel])
    return {"nmse": nmse, "cos": cos, "retr1": float(np.mean(np.concatenate(hits)))}


# ------------------------------------------------------------------ train
def train(cfg, data_path, verbose=True):
    ztr, ttr, zte, tte, _, _ = load(data_path, cfg.backbone, cfg.layer)
    if cfg.holdout:                                       # selection on held-out TRAIN images, never the test set
        nv = int(0.9 * len(ztr)); zte, tte, ztr, ttr = ztr[nv:], ttr[nv:], ztr[:nv], ttr[:nv]
    X, Y, *_ = pairs(ztr, ttr, cfg.use_tau)
    key = jax.random.PRNGKey(cfg.seed)
    model = make(key, X.shape[1], Y.shape[1])
    opt = optimiser(cfg)
    st = opt.init(eqx.filter(model, eqx.is_inexact_array))
    gfn = {"bp": bp_grads, "pc": pc_grads, "alm": alm_grads}[cfg.arm]

    @eqx.filter_jit
    def step(model, st, x, y):
        g = gfn(model, x, y, cfg)
        gn = optax.global_norm(eqx.filter(g, eqx.is_inexact_array))
        upd, st = opt.update(g, st, eqx.filter(model, eqx.is_inexact_array))
        return eqx.apply_updates(model, upd), st, gn, bp_loss(model, x, y)

    rng = np.random.default_rng(cfg.seed + 1000)          # batch stream depends on seed only, not arm
    log, t0 = [], time.time()
    for s in range(cfg.steps):
        idx = rng.integers(0, len(X), cfg.batch)
        model, st, gn, l = step(model, st, jnp.asarray(X[idx]), jnp.asarray(Y[idx]))
        if (s + 1) % 1000 == 0 or s == 0:
            row = {"step": s + 1, "train_loss": float(l), "grad_norm": float(gn)}
            if cfg.arm in ("pc", "alm"):     # how far the PC gradient is from BP's on this batch, as training proceeds
                xb, yb = jnp.asarray(X[idx]), jnp.asarray(Y[idx])
                ga, gb = gfn(model, xb, yb, cfg), bp_grads(model, xb, yb, cfg)
                a, b = _flat(ga), _flat(gb)
                row["cos_pc_bp"] = float(a @ b / (jnp.linalg.norm(a) * jnp.linalg.norm(b)))
                row["ratio_pc_bp"] = float(jnp.linalg.norm(a) / jnp.linalg.norm(b))
                lw = lambda g, i: _flat(g[i])
                row["layer_cos"] = [float(lw(ga, i) @ lw(gb, i) / (jnp.linalg.norm(lw(ga, i)) * jnp.linalg.norm(lw(gb, i)) + 1e-30)) for i in range(len(model))]
                row["layer_ratio"] = [float(jnp.linalg.norm(lw(ga, i)) / (jnp.linalg.norm(lw(gb, i)) + 1e-30)) for i in range(len(model))]
            log.append(row)
            if verbose:
                print(f"[{cfg.arm} {cfg.backbone}/{cfg.layer} s{cfg.seed}] step {s + 1} loss {float(l):.4f} "
                      f"gnorm {float(gn):.3g} {row.get('cos_pc_bp', '')}  {time.time() - t0:.0f}s", flush=True)
    res = evaluate(model, zte, tte, cfg.use_tau)
    return model, log, res


# ------------------------------------------------------------------ check
def check(data_path, out="pred_target_check.json", steps=4000):
    rows = []
    for bb in ("bp", "rand"):
        for layer in ("emb", "h7", "h3"):
            ztr, ttr, zte, tte, _, _ = load(data_path, bb, layer)
            r = {"backbone": bb, "layer": layer, "copy": evaluate(None, zte, tte, copy=True)}
            for use_tau in (True, False):
                c = Cfg(backbone=bb, layer=layer, arm="bp", steps=steps, use_tau=use_tau)
                _, _, res = train(c, data_path, verbose=False)
                r["full" if use_tau else "no_tau"] = res
            r["tau_gain"] = (r["no_tau"]["nmse"] - r["full"]["nmse"]) / r["no_tau"]["nmse"]
            r["full_vs_copy"] = (r["copy"]["nmse"] - r["full"]["nmse"]) / r["copy"]["nmse"]
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if not isinstance(v, dict)})
                  + "  " + " ".join(f"{k}:nmse={r[k]['nmse']:.4f},retr={r[k]['retr1']:.3f}" for k in ("copy", "no_tau", "full")), flush=True)
            rows.append(r)
    bp = {r["layer"]: r for r in rows if r["backbone"] == "bp"}
    ok = {l: r for l, r in bp.items() if r["full_vs_copy"] >= 0.10}
    if "emb" in ok and ok["emb"]["tau_gain"] >= 0.10:
        choice = "emb"
    elif ok:
        choice = max(ok, key=lambda l: ok[l]["tau_gain"])
    else:
        choice = None
    json.dump({"rows": rows, "choice": choice, "rule": TARGET_RULE}, open(out, "w"), indent=1)
    print("TARGET LAYER:", choice)


# ------------------------------------------------------------------ gates
def gates():
    ok = True
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (64, 20)); y = jax.random.normal(jax.random.PRNGKey(1), (64, 8))
    model = make(jax.random.PRNGKey(2), 20, 8)
    gb = bp_grads(model, x, y, None)
    leaves = lambda g: jnp.concatenate([l.ravel() for l in jax.tree_util.tree_leaves(eqx.filter(g, eqx.is_inexact_array))])
    # PC == BP only in the small-output-error limit: test there (y = f(x) + 0.01 n)
    ys = forward(model, x) + 0.01 * jax.random.normal(jax.random.PRNGKey(3), (64, 8))
    gbs = bp_grads(model, x, ys, None)
    for eta in (0.02, 0.05, 0.1, 0.2):
        gp, acts, F = pc_grads(model, x, ys, Cfg(T=300, eta_h=eta), return_acts=True)
        a, b = leaves(gp), leaves(gbs)
        cos = float(a @ b / (jnp.linalg.norm(a) * jnp.linalg.norm(b)))
        mono = bool(jnp.all(jnp.diff(F) <= 1e-4 * jnp.abs(F[0]) + 1e-8))   # round-off once converged
        print(f"   eta_h={eta}: small-error cosine {cos:.5f}, |g| ratio {float(jnp.linalg.norm(a) / jnp.linalg.norm(b)):.4f}, monotone {mono}")
        if eta == 0.05:
            good = cos > 0.998 and mono; ok &= good   # |g| ratio < 1 is the known equilibrium rescaling, reported
            print(f"[{'PASS' if good else 'FAIL'}] small output error, T=300, eta_h=0.05: PC weight gradient = BP direction (cos {cos:.5f} > 0.998), energy monotone")
    # large error (random y): report only -- PC and BP legitimately differ here
    gp = pc_grads(model, x, y, Cfg(T=20, eta_h=0.05))
    a, b = leaves(gp), leaves(gb)
    print(f"   report: random targets, T=20: cosine {float(a @ b / (jnp.linalg.norm(a) * jnp.linalg.norm(b))):.4f} (PC != BP away from the small-error limit)")
    # zero output error -> PC gradient is exactly zero (fixed point = feedforward)
    yff = forward(model, x)
    g0 = leaves(pc_grads(model, x, yff, Cfg(T=10, eta_h=0.05)))
    good = float(jnp.max(jnp.abs(g0))) < 1e-6; ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] y = forward(x): PC gradient is zero ({float(jnp.max(jnp.abs(g0))):.1e})")
    # batch invariance of the per-sample relaxation
    gp1, acts1, _ = pc_grads(model, x[:8], y[:8], Cfg(T=20), return_acts=True)
    gp2, acts2, _ = pc_grads(model, x, y, Cfg(T=20), return_acts=True)
    d = max(float(jnp.max(jnp.abs(a[:8] - b))) for a, b in zip(acts2[:-1], acts1[:-1]))
    good = d < 1e-4; ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] relaxed activities of a sample do not depend on the batch ({d:.1e})")
    ok &= alm_gates(model, x, y, leaves)
    print("ALL PASS" if ok else "SOME FAIL")
    return ok


def alm_gates(model, x, y, leaves):
    ok = True
    cos = lambda a, b: float(a @ b / (jnp.linalg.norm(a) * jnp.linalg.norm(b)))
    acts = list(jpc.init_activities_with_ffwd(model, x, param_type="sp"))
    acts = [a + 0.1 * jax.random.normal(jax.random.PRNGKey(7 + i), a.shape) for i, a in enumerate(acts[:-1])] + [y]
    e1, e2 = float(alm_energy(model, acts, x, y, None)), float(jpc.pc_energy_fn((model, None), acts, y, x=x, param_type="sp"))
    good = abs(e1 - e2) < 1e-5 * abs(e2); ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] ALM energy with no multipliers = jpc PC energy ({e1:.6f} vs {e2:.6f})")
    d = float(jnp.max(jnp.abs(leaves(alm_grads(model, x, y, Cfg(T=20, alpha=0.0))) - leaves(pc_grads(model, x, y, Cfg(T=20))))))
    good = d < 1e-5; ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] alpha = 0: ALM weight gradient = plain PC ({d:.1e})")
    # large output error (random targets): plain PC's gradient departs from BP's; ALM's should approach it.
    # Measured on the training configuration (curvature-tracking step).  Pre-registered: at T = 100, alpha = 0.25,
    # cos(ALM, BP) > 0.97 and at least 0.05 above plain PC; the T = 20 training-budget values are reported.
    gb = leaves(bp_grads(model, x, y, None))
    for T in (20, 100):
        cp = cos(leaves(pc_grads(model, x, y, Cfg(T=T, adaptive_eta=True))), gb)
        ca = {al: cos(leaves(alm_grads(model, x, y, Cfg(T=T, alpha=al, adaptive_eta=True))), gb) for al in (0.1, 0.25, 0.5, 1.0)}
        print(f"   random targets, T={T}: cos to BP  plain PC {cp:.4f} | ALM " + " ".join(f"a={k}: {v:.4f}" for k, v in ca.items()))
        if T == 100:
            good = ca[0.25] > 0.97 and ca[0.25] > cp + 0.05; ok &= good
            print(f"[{'PASS' if good else 'FAIL'}] large error, T=100, alpha=0.25: ALM matches BP (cos {ca[0.25]:.4f}) where plain PC does not ({cp:.4f})")
    g1, a1, _, _ = alm_grads(model, x[:8], y[:8], Cfg(T=20), return_acts=True)
    g2, a2, _, _ = alm_grads(model, x, y, Cfg(T=20), return_acts=True)
    d = max(float(jnp.max(jnp.abs(a[:8] - b))) for a, b in zip(a2[:-1], a1[:-1]))
    good = d < 1e-4; ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] ALM relaxed activities of a sample do not depend on the batch ({d:.1e})")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["check", "train", "gates"])
    ap.add_argument("--data", default="pred_latents.npz")
    ap.add_argument("--backbone", default="bp"); ap.add_argument("--layer", default="emb")
    ap.add_argument("--arm", default="bp"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=6000); ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--eta-h", type=float, default=0.05); ap.add_argument("--out", default=None)
    ap.add_argument("--adaptive-eta", action="store_true"); ap.add_argument("--save", action="store_true")
    ap.add_argument("--alpha", type=float, default=1.0); ap.add_argument("--holdout", action="store_true")
    a = ap.parse_args()
    if a.mode == "gates":
        sys.exit(0 if gates() else 1)
    if a.mode == "check":
        check(a.data)
    else:
        c = Cfg(backbone=a.backbone, layer=a.layer, arm=a.arm, seed=a.seed, steps=a.steps, T=a.T, eta_h=a.eta_h, adaptive_eta=a.adaptive_eta,
                alpha=a.alpha, holdout=a.holdout)
        model, log, res = train(c, a.data)
        out = a.out or f"pred_runs/ff_{a.arm}_{a.backbone}_{a.layer}_s{a.seed}.json"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump({"cfg": dataclasses.asdict(c), "log": log, "test": res}, open(out, "w"), indent=1)
        if a.save:
            eqx.tree_serialise_leaves(out.replace(".json", ".eqx"), model)
        print("TEST", json.dumps(res))


def load_model(path, din=146, dout=128, seed=0):
    return eqx.tree_deserialise_leaves(path, make(jax.random.PRNGKey(seed), din, dout))


# pred_eval-compatible predict for k=1: zhat_t = g(z_A, tau_A, tau_t) for every target t.
def make_pair_predict(model):
    f = eqx.filter_jit(lambda x: forward(model, x))
    def predict(Zs, Ts, Tt):
        Zs, Ts, Tt = np.asarray(Zs), np.asarray(Ts), np.asarray(Tt)
        assert Zs.shape[1] == 1, "the feedforward predictor takes exactly one seen view"
        m = Tt.shape[1]
        X = np.concatenate([np.repeat(Zs, m, 1), np.repeat(Ts, m, 1), Tt], -1)
        return np.asarray(f(jnp.asarray(X.reshape(-1, X.shape[-1])))).reshape(Zs.shape[0], m, -1)
    return predict
