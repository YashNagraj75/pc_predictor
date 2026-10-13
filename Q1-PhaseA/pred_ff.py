"""Phase P, track 1: feedforward predictor, PC vs backprop, on FROZEN latents.

    ./.venv/bin/python pred_ff.py check  --data pred_latents.npz            # target-space check
    ./.venv/bin/python pred_ff.py train  --data pred_latents.npz --backbone bp --layer emb --arm pc --seed 0
    ./.venv/bin/python pred_ff.py gates                                    # CPU gate tests

Task (I-JEPA analogue at image level): from view A's frozen latent and BOTH views'
transforms, predict view B's latent:   z_B_hat = g(z_A, tau_A, tau_B).
All 12 ordered pairs (A != B) of the 4 recorded views of each image.  Latents are
standardised per dimension with TRAIN statistics; the loss is 1/2 ||z_B - g||^2.

Arms share architecture, init, batches, optimiser (AdamW, same lr, clip 1.0) and
schedule; they differ ONLY in how the weight gradient is obtained:
  bp : autodiff of the loss
  pc : jpc PC energy, hidden activities initialised feedforward, T steps of activity
       GD with PER-SAMPLE step eta_h (jpc's energy is a batch mean, so the applied
       step is eta_h * B), then jpc.compute_pc_param_grads.  Output clamped to z_B.
At test time BOTH arms are scored by the same feedforward pass.

Baselines: copy (z_B_hat = z_A), mean (z_B_hat = 0, nmse 1), and a no-tau MLP g(z_A)
trained by backprop.

Metrics on the 2000 test images: nmse = E||z_hat - z||^2 / E||z||^2 (standardised, so
1 = predicting the mean), cosine, and retrieval@1 = z_hat's nearest neighbour among
the 2000 test images' view-B latents is the right image.

PRE-REGISTERED target-space rule (written before the check was run, 2026-10-12):
on the BP backbone, for each layer, tau_gain = (nmse_noTau - nmse_full) / nmse_noTau.
Use "emb" if its tau_gain >= 0.10 AND full beats copy by >= 10 % relative nmse;
otherwise the layer with the largest tau_gain among those where full beats copy by
>= 10 %; if none qualifies, STOP -- no layer of this encoder carries enough view
information and the predictor experiment needs a different target (patch level, or
a non-invariant backbone).  The same layer is then used for the untrained backbone.
"""
import argparse, dataclasses, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import jax, jax.numpy as jnp, equinox as eqx, optax, jpc

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


# ------------------------------------------------------------------ data
def load(path, backbone, layer):
    d = np.load(path)
    ztr, zte = d[f"{backbone}_{layer}_tr"].astype(np.float32), d[f"{backbone}_{layer}_te"].astype(np.float32)
    mu, sd = ztr.reshape(-1, ztr.shape[-1]).mean(0), ztr.reshape(-1, ztr.shape[-1]).std(0) + 1e-6
    return ((ztr - mu) / sd, d["tau_tr"].astype(np.float32), (zte - mu) / sd, d["tau_te"].astype(np.float32),
            d["ytr"], d["yte"])


def pairs(z, tau, use_tau=True):
    """-> X (N*12, in), Y (N*12, d), (img, a, b) index arrays."""
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
    X, Y, *_ = pairs(ztr, ttr, cfg.use_tau)
    key = jax.random.PRNGKey(cfg.seed)
    model = make(key, X.shape[1], Y.shape[1])
    opt = optimiser(cfg)
    st = opt.init(eqx.filter(model, eqx.is_inexact_array))
    gfn = bp_grads if cfg.arm == "bp" else pc_grads

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
            if cfg.arm == "pc":     # how far PC's gradient is from BP's on this batch, as training proceeds
                a, b = _flat(pc_grads(model, jnp.asarray(X[idx]), jnp.asarray(Y[idx]), cfg)), _flat(bp_grads(model, jnp.asarray(X[idx]), jnp.asarray(Y[idx]), cfg))
                row["cos_pc_bp"] = float(a @ b / (jnp.linalg.norm(a) * jnp.linalg.norm(b)))
                row["ratio_pc_bp"] = float(jnp.linalg.norm(a) / jnp.linalg.norm(b))
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
    json.dump({"rows": rows, "choice": choice, "rule": __doc__.split("PRE-REGISTERED")[1].strip()}, open(out, "w"), indent=1)
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
    print("ALL PASS" if ok else "SOME FAIL")
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
    a = ap.parse_args()
    if a.mode == "gates":
        sys.exit(0 if gates() else 1)
    if a.mode == "check":
        check(a.data)
    else:
        c = Cfg(backbone=a.backbone, layer=a.layer, arm=a.arm, seed=a.seed, steps=a.steps, T=a.T, eta_h=a.eta_h, adaptive_eta=a.adaptive_eta)
        model, log, res = train(c, a.data)
        out = a.out or f"pred_runs/ff_{a.arm}_{a.backbone}_{a.layer}_s{a.seed}.json"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump({"cfg": dataclasses.asdict(c), "log": log, "test": res}, open(out, "w"), indent=1)
        if a.save:
            eqx.tree_serialise_leaves(out.replace(".json", ".eqx"), model)
        print("TEST", json.dumps(res))


def load_model(path, din=146, dout=128, seed=0):
    return eqx.tree_deserialise_leaves(path, make(jax.random.PRNGKey(seed), din, dout))


def make_pair_predict(model):
    """pred_eval-compatible predict for k = 1: zhat_t = g(z_A, tau_A, tau_t) for every target t."""
    f = eqx.filter_jit(lambda x: forward(model, x))
    def predict(Zs, Ts, Tt):
        Zs, Ts, Tt = np.asarray(Zs), np.asarray(Ts), np.asarray(Tt)
        assert Zs.shape[1] == 1, "the feedforward predictor takes exactly one seen view"
        m = Tt.shape[1]
        X = np.concatenate([np.repeat(Zs, m, 1), np.repeat(Ts, m, 1), Tt], -1)
        return np.asarray(f(jnp.asarray(X.reshape(-1, X.shape[-1])))).reshape(Zs.shape[0], m, -1)
    return predict
