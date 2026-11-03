import argparse, dataclasses as dc, json, math, os, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp, equinox as eqx, optax
jax.config.update("jax_default_matmul_precision", "highest")
import data as D, vit_ijepa as V, vit_epc as P

# Stage A (plan_stageA_scoped.md v3): train a fresh I-JEPA predictor on the FROZEN V0 encoders (context encoder for the
# context, EMA target encoder for the layer-normed targets) with BP or ePC; evaluate the fixed test suite.  Arms:
#   bp | epc_alm_k4 (ePC + multipliers, rho = gain/4) | epc_k4 / epc_k16 (no multipliers, rho = gain/4, gain/16).
# rho tracks the output gain, re-measured every GAIN_EVERY steps (median over 16 blocks of the current batch).
ARMS = {"bp": (None, 0.0), "epc_alm_k4": (4.0, 1.0), "epc_k4": (4.0, 0.0), "epc_k16": (16.0, 0.0)}
GAIN_EVERY, LOG_EVERY, EVAL_SEED = 200, 200, 12345
V0_DIR = "runs_vit/v0_d6_s0"


@dc.dataclass(frozen=True)
class ACfg:
    arm: str = "bp"
    depth: int = 10
    lr: float = 1e-3
    epochs: int = 25
    split: str = "full"          # val: train on 45k, score on the 5k holdout; full: train on 50k, test suite on 2k test
    seed: int = 0
    T: int = 20
    gn_iters: int = 12
    chunk: int = 32
    batch: int = 256
    wd: float = 0.04
    clip: float = 1.0
    warmup_epochs: int = 1
    max_steps: int = 0           # >0: stop early (smoke); schedule unchanged


def load_encoders(v0_dir=V0_DIR):
    meta = json.load(open(os.path.join(v0_dir, "v0_report.json")))
    vc = V.VCfg(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in meta["cfg"].items()})
    tmpl = V.Encoder(jax.random.PRNGKey(0), vc)
    enc = eqx.tree_deserialise_leaves(os.path.join(v0_dir, "context_encoder.eqx"), tmpl)
    tenc = eqx.tree_deserialise_leaves(os.path.join(v0_dir, "target_encoder.eqx"), tmpl)
    return vc, enc, tenc, meta


def make_predictor(vc, a):
    return V.Predictor(jax.random.PRNGKey(10_000 + a.seed), vc, depth=a.depth, width=192, heads=3, mlp=768, scaled=True)


@eqx.filter_jit
def encode(enc, tenc, imgs_ctx, imgs_tgt, cidx, cvalid, tidx, vc):
    ctx = jax.vmap(lambda im, a, b: enc(im, a, b, vc.patch))(imgs_ctx, cidx, cvalid)
    Y = V.targets_of(tenc, imgs_tgt, tidx, vc)
    return ctx, Y


@eqx.filter_jit
def predict(pred, ctx, cidx, cvalid, tidx, tvalid):
    return jax.vmap(lambda a, ci, cv, ti, tv: jax.vmap(lambda t1, t2: pred(a, ci, cv, t1, t2))(ti, tv))(ctx, cidx, cvalid, tidx, tvalid)


@eqx.filter_jit
def gain_median(pred, ctx, cidx, cvalid, tidx, tvalid):        # 4 images x 4 blocks
    g = jax.vmap(lambda a, ci, cv, ti, tv: jax.vmap(lambda t1, t2: P.gain(pred, a, ci, cv, t1, t2))(ti, tv))(
        ctx[:4], cidx[:4], cvalid[:4], tidx[:4], tvalid[:4])
    return jnp.median(g)


def make_step(a, opt):
    k, alpha = ARMS[a.arm]
    meth = "bp" if k is None else "epc"

    @eqx.filter_jit
    def step(pred, ost, ctx, cidx, cvalid, tidx, tvalid, Y, rho):
        g, _, en = P.grads(pred, ctx, cidx, cvalid, tidx, tvalid, Y, meth, rho=rho, alpha=alpha, T=a.T, chunk=a.chunk, gn_iters=a.gn_iters)
        upd, ost = opt.update(g, ost, eqx.filter(pred, eqx.is_inexact_array))
        return eqx.apply_updates(pred, upd), ost, en
    return step


@eqx.filter_jit
def credit(pred, ctx, cidx, cvalid, tidx, tvalid, Y, rho, alpha, T, chunk, gn_iters):  # cos / ratio to BP, per leaf
    gb = P.grads(pred, ctx, cidx, cvalid, tidx, tvalid, Y, "bp", chunk=chunk)[0]
    ge = P.grads(pred, ctx, cidx, cvalid, tidx, tvalid, Y, "epc", rho=rho, alpha=alpha, T=T, chunk=chunk, gn_iters=gn_iters)[0]
    lb = jax.tree_util.tree_leaves(eqx.filter(gb, eqx.is_inexact_array)); le = jax.tree_util.tree_leaves(eqx.filter(ge, eqx.is_inexact_array))
    fb, fe = jnp.concatenate([x.ravel() for x in lb]), jnp.concatenate([x.ravel() for x in le])
    cl = jnp.stack([jnp.sum(x * y) / (jnp.linalg.norm(x) * jnp.linalg.norm(y) + 1e-30) for x, y in zip(le, lb)])
    nb = jnp.stack([jnp.linalg.norm(y) for y in lb])
    return fe @ fb / (jnp.linalg.norm(fe) * jnp.linalg.norm(fb)), jnp.linalg.norm(fe) / jnp.linalg.norm(fb), jnp.min(jnp.where(nb > 1e-12, cl, 1.0))


def optimiser(a, spe):
    total = a.epochs * spe
    sch = optax.warmup_cosine_decay_schedule(0.0, a.lr, max(1, a.warmup_epochs * spe), total, a.lr * 1e-3)
    return optax.chain(optax.clip_by_global_norm(a.clip), optax.adamw(sch, weight_decay=a.wd, mask=V.wd_mask)), total


# ------------------------------------------------------------------ evaluation (fixed masks and corruptions, paired)
def nmse_parts(Pr, Y, tvalid):
    m = np.asarray(tvalid)[..., None].astype(np.float64); Pr, Y = np.asarray(Pr, np.float64), np.asarray(Y, np.float64)
    return float(np.sum(m * (Pr - Y) ** 2)), float(np.sum(m * Y ** 2)), float(np.sum(m * Pr * Y)), float(np.sum(m * Pr ** 2))


def drop_context(rng, cidx, cvalid, frac):       # remove a fraction of the valid context tokens (re-packed to the front)
    ci, cv = cidx.copy(), cvalid.copy()
    for b in range(len(ci)):
        v = ci[b][cv[b]]; keep = rng.permutation(len(v))[:max(1, int(round(len(v) * (1 - frac))))]
        v = np.sort(v[keep]); ci[b] = 0; cv[b] = False; ci[b, :len(v)] = v; cv[b, :len(v)] = True
    return ci, cv


def corrupt(rng, X, kind):
    X = X.copy()
    if kind == "square":                         # black square, 25% of the image area (16 x 16 px)
        for b in range(len(X)):
            r, c = rng.integers(0, 17, 2); X[b, r:r + 16, c:c + 16] = 0.0
    elif kind == "noise":
        X = X + 0.2 * rng.standard_normal(X.shape).astype(np.float32)
    return X


def evaluate(pred, enc, tenc, vc, Xev, yev, Xknn, yknn, bs=200):
    rng = np.random.default_rng(EVAL_SEED)
    masks = [V.sample_masks(rng, bs, vc) for _ in range(len(Xev) // bs)]
    conds = {"clean": ("none", 0.0), "ctx_drop25": ("none", 0.25), "ctx_drop50": ("none", 0.5), "square": ("square", 0.0), "noise": ("noise", 0.0)}
    out = {}
    for name, (kind, frac) in conds.items():
        r2 = np.random.default_rng(EVAL_SEED + 1); acc = np.zeros(4); feats = []
        for i, (ci, cv, ti, tv) in enumerate(masks):
            Xb = Xev[i * bs:(i + 1) * bs]
            if frac > 0:
                ci, cv = drop_context(r2, ci, cv, frac)
            Xc = corrupt(r2, Xb, kind)
            ctx, Y = encode(enc, tenc, jnp.asarray(V.standardise(Xc)), jnp.asarray(V.standardise(Xb)), jnp.asarray(ci), jnp.asarray(cv), jnp.asarray(ti), vc)
            Pr = predict(pred, ctx, jnp.asarray(ci), jnp.asarray(cv), jnp.asarray(ti), jnp.asarray(tv))
            acc += nmse_parts(Pr, Y, tv)
            if name == "clean":
                m = np.asarray(tv)[..., None]; feats.append((np.asarray(Pr) * m).sum((1, 2)) / m.sum((1, 2)))
        alpha = acc[2] / acc[3]
        out[name] = {"nmse": acc[0] / acc[1], "nmse_cal": (acc[3] * alpha ** 2 - 2 * alpha * acc[2] + acc[1]) / acc[1], "alpha_star": alpha}
        if name == "clean":
            Fev = np.concatenate(feats)
    # kNN class accuracy of mean-pooled predicted target embeddings (reference set: predictions on Xknn, clean)
    r3 = np.random.default_rng(EVAL_SEED + 2); Fk = []
    for i in range(0, len(Xknn), bs):
        ci, cv, ti, tv = V.sample_masks(r3, min(bs, len(Xknn) - i), vc); Xb = Xknn[i:i + bs]
        ctx, _ = encode(enc, tenc, jnp.asarray(V.standardise(Xb)), jnp.asarray(V.standardise(Xb)), jnp.asarray(ci), jnp.asarray(cv), jnp.asarray(ti), vc)
        Pr = np.asarray(predict(pred, ctx, jnp.asarray(ci), jnp.asarray(cv), jnp.asarray(ti), jnp.asarray(tv)))
        m = np.asarray(tv)[..., None]; Fk.append((Pr * m).sum((1, 2)) / m.sum((1, 2)))
    import metrics as MT
    out["knn_pred"] = MT.knn_accuracy(np.concatenate(Fk), yknn, Fev, yev[:len(Fev)], k=20)
    return out


# ------------------------------------------------------------------ training
def run(a, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    vc, enc, tenc, meta = load_encoders()
    Xtr, ytr, Xte, yte = D.load_cifar10(n_train=-1, n_test=2000, seed=0)
    if a.split == "val":
        Xtr, ytr, Xev, yev = Xtr[:45000], ytr[:45000], Xtr[45000:], ytr[45000:]
    else:
        Xev, yev = Xte, yte
    Xknn, yknn = Xtr[:5000], ytr[:5000]
    spe = len(Xtr) // a.batch
    opt, total = optimiser(a, spe)
    pred = make_predictor(vc, a); ost = opt.init(eqx.filter(pred, eqx.is_inexact_array))
    k, alpha = ARMS[a.arm]
    state, step0, log, rho = (pred, ost), 0, [], 1.0
    ck, logf = os.path.join(out_dir, "ckpt.eqx"), os.path.join(out_dir, "train_log.json")
    if os.path.exists(ck):
        prev = json.load(open(logf))
        same = lambda d: {x: y for x, y in d.items() if x != "max_steps"}
        assert same(prev["cfg"]) == same(dc.asdict(a)), "config changed since the checkpoint"
        state, st = eqx.tree_deserialise_leaves(ck, (state, jnp.int32(0)))
        step0, log, rho = int(st), [r for r in prev["log"] if r["step"] <= int(st)], prev.get("rho", 1.0)
        print("RESUME at step", step0, flush=True)
    stepf = make_step(a, opt)
    stop = a.max_steps if a.max_steps > 0 else total
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    rc = np.random.default_rng([a.seed, 99]); Xcr = Xev[:32]; cr = V.sample_masks(rc, 32, vc)   # fixed credit batch
    t0, epoch, perm = time.time(), -1, None
    for it in range(step0, stop):
        if it // spe != epoch:
            epoch = it // spe; perm = np.random.default_rng([a.seed, epoch]).permutation(len(Xtr))
        rng = np.random.default_rng([a.seed, 1, it])
        imgs = jnp.asarray(V.standardise(V.augment(rng, Xtr[perm[(it % spe) * a.batch:(it % spe + 1) * a.batch]])))
        cidx, cvalid, tidx, tvalid = (jnp.asarray(x) for x in V.sample_masks(rng, a.batch, vc))
        pred, ost = state
        ctx, Y = encode(enc, tenc, imgs, imgs, cidx, cvalid, tidx, vc)
        if k is not None and (it % GAIN_EVERY == 0 or it == step0):
            gmed = float(gain_median(pred, ctx, cidx, cvalid, tidx, tvalid)); rho = max(1.0, gmed / k)
        pred, ost, en = stepf(pred, ost, ctx, cidx, cvalid, tidx, tvalid, Y, jnp.float32(rho))
        state = (pred, ost)
        if (it + 1) % LOG_EVERY == 0 or it == step0 or it + 1 == stop:
            loss = float(V.masked_mse(predict(pred, ctx, cidx, cvalid, tidx, tvalid), Y, tvalid))
            row = {"step": it + 1, "loss": loss, "rho": rho, "sec": round(time.time() - t0)}
            if k is not None:
                row["gain"] = gmed
                cx, cy = encode(enc, tenc, jnp.asarray(V.standardise(Xcr)), jnp.asarray(V.standardise(Xcr)), jnp.asarray(cr[0]), jnp.asarray(cr[1]), jnp.asarray(cr[2]), vc)
                c1, rr, cmin = credit(pred, cx, jnp.asarray(cr[0]), jnp.asarray(cr[1]), jnp.asarray(cr[2]), jnp.asarray(cr[3]), cy,
                                      jnp.float32(rho), alpha, a.T, a.chunk, a.gn_iters)
                row.update({"cos_bp": float(c1), "ratio_bp": float(rr), "min_leaf_cos": float(cmin)})
            log.append(row); print(json.dumps(row), flush=True)
            if not np.isfinite(loss):
                V.save_json(logf, {"log": log, "cfg": dc.asdict(a), "commit": commit, "diverged": it + 1, "rho": rho})
                print("DIVERGED", flush=True); return None
        if (it + 1) % (5 * spe) == 0 or it + 1 == stop:
            V.save_ckpt(ck, state, it + 1); V.save_json(logf, {"log": log, "cfg": dc.asdict(a), "commit": commit, "rho": rho})
    res = evaluate(state[0], enc, tenc, vc, Xev, yev, Xknn, yknn)
    eqx.tree_serialise_leaves(os.path.join(out_dir, "predictor.eqx"), state[0])
    V.save_json(os.path.join(out_dir, "report.json"), {"cfg": dc.asdict(a), "commit": commit, "v0_commit": meta.get("commit"),
                                                       "v0_sha256": meta.get("sha256"), "steps": stop, "eval_split": a.split, "test": res, "log": log})
    print("RESULT", json.dumps(res), flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for f in dc.fields(ACfg):
        ap.add_argument("--" + f.name.replace("_", "-"), type=type(f.default), default=f.default)
    ap.add_argument("--out", required=True)
    ns = vars(ap.parse_args()); out = ns.pop("out")
    a = ACfg(**ns); assert a.arm in ARMS, a.arm
    sys.exit(0 if run(a, out) is not None else 3)
