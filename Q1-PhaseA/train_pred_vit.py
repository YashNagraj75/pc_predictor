import argparse, dataclasses as dc, json, math, os, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp, equinox as eqx, optax
jax.config.update("jax_default_matmul_precision", "highest")
import data as D, vit_ijepa as V, vit_epc as P

# Stage A (plan_stageA_scoped.md v3): train a fresh I-JEPA predictor on the FROZEN V0 encoders (context encoder for the
# context, EMA target encoder for the layer-normed targets) with BP or ePC; evaluate the fixed test suite.  Arms:
#   bp | epc_alm_k4 (ePC + multipliers, rho = gain/4) | epc_k4 / epc_k16 (no multipliers, rho = gain/4, gain/16).
# rho tracks the output gain, re-measured every GAIN_EVERY steps (median over 16 blocks of the current batch).
# rev 2 (3 reviewers): gain re-measured every 10 steps for the first 500 (it grows ~1e4x in warmup), then every 50.
ARMS = {"bp": (None, 0.0), "epc_alm_k4": (4.0, 1.0), "epc_k4": (4.0, 0.0), "epc_k16": (16.0, 0.0),
        "epc_alm_blk4": (4.0, 1.0), "epc_alm_blk2": (2.0, 1.0), "epc_alm_blk1": (1.0, 1.0)}
# epc_alm_blk<k> (user-approved follow-up to findings/40): ePC-ALM with a per-block rho = max(1, own gain / k); k is fixed
# by the pre-set screen in test_blockrho.py before any training.
BLOCK = {"epc_alm_blk4", "epc_alm_blk2", "epc_alm_blk1"}
# spc_u1 (user 2026-11-12): textbook standard PC baseline, relaxation over the activities (vit_epc.spc_one), unit precision
# (rho fixed at 1, never gain-relative), no multipliers, same T and lr selection.  Gain and credit are still logged.
ARMS["spc_u1"] = (1.0, 0.0)
SPC = {"spc_u1"}
method_of = lambda arm: "bp" if ARMS[arm][0] is None else ("spc" if arm in SPC else "epc")
LOG_EVERY, EVAL_SEED = 200, 12345
gain_due = lambda it: it % 10 == 0 if it < 500 else it % 50 == 0
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
    deadline_min: float = 0.0    # >0: checkpoint and exit 75 before this many wall minutes (host run limit)


def load_encoders(v0_dir=V0_DIR):
    import hashlib
    meta = json.load(open(os.path.join(v0_dir, "v0_report.json")))
    for n, h in meta["sha256"].items():          # the frozen encoders must be exactly the gated ones
        assert hashlib.sha256(open(os.path.join(v0_dir, n + ".eqx"), "rb").read()).hexdigest() == h, f"{n}.eqx changed since the V0 gate"
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
def gain_stats(pred, ctx, cidx, cvalid, tidx, tvalid):         # 8 images x 4 blocks -> (min, median, max)
    g = jax.vmap(lambda a, ci, cv, ti, tv: jax.vmap(lambda t1, t2: P.gain(pred, a, ci, cv, t1, t2))(ti, tv))(
        ctx[:8], cidx[:8], cvalid[:8], tidx[:8], tvalid[:8])
    return jnp.min(g), jnp.median(g), jnp.max(g)


def make_step(a, opt):
    k, alpha = ARMS[a.arm]
    meth = method_of(a.arm)
    rr = float(k) if a.arm in BLOCK else 0.0

    @eqx.filter_jit
    def step(pred, ost, ctx, cidx, cvalid, tidx, tvalid, Y, rho):
        out = P.grads(pred, ctx, cidx, cvalid, tidx, tvalid, Y, meth, rho=rho, alpha=alpha, T=a.T, chunk=a.chunk, gn_iters=a.gn_iters,
                      rho_rel=rr, return_rho=rr > 0)
        g, en = out[0], (out[2], out[3]) if rr > 0 else out[2]
        upd, ost = opt.update(g, ost, eqx.filter(pred, eqx.is_inexact_array))
        return eqx.apply_updates(pred, upd), ost, en
    return step


@eqx.filter_jit
def credit(pred, ctx, cidx, cvalid, tidx, tvalid, Y, rho, alpha, T, chunk, gn_iters, rho_rel=0.0, meth="epc"):  # cos / ratio to BP, per leaf
    gb = P.grads(pred, ctx, cidx, cvalid, tidx, tvalid, Y, "bp", chunk=chunk)[0]
    ge = P.grads(pred, ctx, cidx, cvalid, tidx, tvalid, Y, meth, rho=rho, alpha=alpha, T=T, chunk=chunk, gn_iters=gn_iters, rho_rel=rho_rel)[0]
    lb = jax.tree_util.tree_leaves(eqx.filter(gb, eqx.is_inexact_array)); le = jax.tree_util.tree_leaves(eqx.filter(ge, eqx.is_inexact_array))
    fb, fe = jnp.concatenate([x.ravel() for x in lb]), jnp.concatenate([x.ravel() for x in le])
    cl = jnp.stack([jnp.sum(x * y) / (jnp.linalg.norm(x) * jnp.linalg.norm(y) + 1e-30) for x, y in zip(le, lb)])
    nb = jnp.stack([jnp.linalg.norm(y) for y in lb])
    cl = jnp.where(nb > 1e-12, cl, 1.0)                     # per parameter leaf, in tree order (per sublayer weights)
    lr_ = jnp.where(nb > 1e-12, jnp.stack([jnp.linalg.norm(x) for x in le]) / jnp.maximum(nb, 1e-30), 1.0)   # per-leaf |g|/|g_bp|
    return fe @ fb / (jnp.linalg.norm(fe) * jnp.linalg.norm(fb)), jnp.linalg.norm(fe) / jnp.linalg.norm(fb), jnp.min(cl), cl, lr_


def optimiser(a, spe):
    total = a.epochs * spe
    warm = max(1, min(a.warmup_epochs * spe, total // 5))      # never the whole run (optax needs decay > warmup)
    sch = optax.warmup_cosine_decay_schedule(0.0, a.lr, warm, max(total, warm + 1), a.lr * 1e-3)
    return optax.chain(optax.clip_by_global_norm(a.clip), optax.adamw(sch, weight_decay=a.wd, mask=V.wd_mask)), total


# ------------------------------------------------------------------ evaluation (fixed masks and corruptions, paired)
def per_image(Pr, Y, tvalid):                    # per image: sum err^2, sum Y^2, sum P*Y, sum P^2 over valid tokens
    m = np.asarray(tvalid)[..., None].astype(np.float64); Pr, Y = np.asarray(Pr, np.float64), np.asarray(Y, np.float64)
    f = lambda z: (m * z).sum((1, 2, 3))
    return np.stack([f((Pr - Y) ** 2), f(Y ** 2), f(Pr * Y), f(Pr ** 2)], 1)


def pooled(Z, tvalid):
    m = np.asarray(tvalid)[..., None]; return (np.asarray(Z) * m).sum((1, 2)) / m.sum((1, 2))


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


def eval_masks(n, vc, bs=200, per=50):           # one mask shape per 50 images (40 shapes on 2000), model batches of 200
    rng = np.random.default_rng(EVAL_SEED); out = []
    for _ in range(n // bs):
        parts = [V.sample_masks(rng, per, vc) for _ in range(bs // per)]
        out.append(tuple(np.concatenate([p_[k] for p_ in parts]) for k in range(4)))
    return out


def evaluate(pred, enc, tenc, vc, Xev, yev, Xknn, yknn, bs=200):
    masks = eval_masks(len(Xev), vc, bs)
    conds = {"clean": ("none", 0.0), "ctx_drop25": ("none", 0.25), "ctx_drop50": ("none", 0.5), "square": ("square", 0.0), "noise": ("noise", 0.0)}
    out, per, Fp, Fy = {}, {}, [], []           # Fp / Fy: clean-condition pooled predictions and targets (kNN)
    for name, (kind, frac) in conds.items():
        r2 = np.random.default_rng(EVAL_SEED + 1); rows = []
        for i, (ci, cv, ti, tv) in enumerate(masks):
            Xb = Xev[i * bs:(i + 1) * bs]
            if frac > 0:
                ci, cv = drop_context(r2, ci, cv, frac)
            Xc = corrupt(r2, Xb, kind)              # targets always from the CLEAN image
            ctx, Y = encode(enc, tenc, jnp.asarray(V.standardise(Xc)), jnp.asarray(V.standardise(Xb)), jnp.asarray(ci), jnp.asarray(cv), jnp.asarray(ti), vc)
            Pr = predict(pred, ctx, jnp.asarray(ci), jnp.asarray(cv), jnp.asarray(ti), jnp.asarray(tv))
            rows.append(per_image(Pr, Y, tv))
            if name == "clean":
                Fp.append(pooled(Pr, tv)); Fy.append(pooled(Y, tv))
        A_ = np.concatenate(rows); per[name] = A_; S = A_.sum(0); a = S[2] / S[3]
        out[name] = {"nmse": S[0] / S[1], "nmse_cal": (S[3] * a ** 2 - 2 * a * S[2] + S[1]) / S[1], "alpha_star": a}
    ac = out["clean"]["alpha_star"]
    for name in conds:                           # calibration fixed on clean, applied unchanged under corruption
        S = per[name].sum(0); out[name]["nmse_cal_cleanalpha"] = (S[3] * ac ** 2 - 2 * ac * S[2] + S[1]) / S[1]
        out[name]["ratio_to_clean"] = out[name]["nmse"] / out["clean"]["nmse"]
    # kNN of mean-pooled predictions (raw and centred); reference set uses the SAME mask sequence; ceiling = target features
    Fk, Fky = [], []
    for i in range(0, len(Xknn), bs):
        ci, cv, ti, tv = masks[(i // bs) % len(masks)]; Xb = Xknn[i:i + bs]; nb = len(Xb)
        ci, cv, ti, tv = ci[:nb], cv[:nb], ti[:nb], tv[:nb]
        ctx, Y = encode(enc, tenc, jnp.asarray(V.standardise(Xb)), jnp.asarray(V.standardise(Xb)), jnp.asarray(ci), jnp.asarray(cv), jnp.asarray(ti), vc)
        Pr = predict(pred, ctx, jnp.asarray(ci), jnp.asarray(cv), jnp.asarray(ti), jnp.asarray(tv))
        Fk.append(pooled(Pr, tv)); Fky.append(pooled(Y, tv))
    Fev, Fevy, Fk, Fky = np.concatenate(Fp), np.concatenate(Fy), np.concatenate(Fk), np.concatenate(Fky)
    out["knn_pred"] = V.knn_both(Fk, yknn, Fev, yev[:len(Fev)])
    out["knn_target_ceiling"] = V.knn_both(Fky, yknn, Fevy, yev[:len(Fevy)])
    return out, per


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
    state, step0, log, rho, gs = (pred, ost), 0, [], 1.0, (0.0, 0.0, 0.0)
    ck, logf = os.path.join(out_dir, "ckpt.eqx"), os.path.join(out_dir, "train_log.json")
    ckp = ck.replace(".eqx", "_prev.eqx")
    src = ck if os.path.exists(ck) else (ckp if os.path.exists(ckp) else None)
    if src:                                      # rho and the last gain are restored, so a resumed run is exact
        prev = json.load(open(logf)) if os.path.exists(logf) else {"cfg": dc.asdict(a), "log": []}
        same = lambda d: {x: y for x, y in d.items() if x not in ("max_steps", "deadline_min")}
        assert same(prev["cfg"]) == same(dc.asdict(a)), "config changed since the checkpoint"
        (state, r_, g_), st = eqx.tree_deserialise_leaves(src, ((state, jnp.float32(0), jnp.zeros(3)), jnp.int32(0)))
        step0, rho, gs = int(st), float(r_), tuple(float(x) for x in g_)
        log = [r for r in prev["log"] if r["step"] <= step0]
        print("RESUME at step", step0, "from", os.path.basename(src), flush=True)
    stepf = make_step(a, opt)
    stop = a.max_steps if a.max_steps > 0 else total
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    rc = np.random.default_rng([a.seed, 99]); Xcr = Xtr[:32]; cr = V.sample_masks(rc, 32, vc)   # fixed credit batch (train images)
    t0, epoch, perm, ep_t = time.time(), -1, None, None

    def diverged(it, why):
        V.save_json(os.path.join(out_dir, "diverged.json"), {"step": it, "why": why, "cfg": dc.asdict(a), "commit": commit})
        V.save_json(logf, {"log": log, "cfg": dc.asdict(a), "commit": commit, "diverged": it})
        print("DIVERGED", it, why, flush=True)
        return None
    for it in range(step0, stop):
        if it // spe != epoch:
            epoch = it // spe; perm = np.random.default_rng([a.seed, epoch]).permutation(len(Xtr))
        rng = np.random.default_rng([a.seed, 1, it])
        imgs = jnp.asarray(V.standardise(V.augment(rng, Xtr[perm[(it % spe) * a.batch:(it % spe + 1) * a.batch]])))
        cidx, cvalid, tidx, tvalid = (jnp.asarray(x) for x in V.sample_masks(rng, a.batch, vc))
        pred, ost = state
        ctx, Y = encode(enc, tenc, imgs, imgs, cidx, cvalid, tidx, vc)
        if k is not None and gain_due(it):
            gs = tuple(float(x) for x in gain_stats(pred, ctx, cidx, cvalid, tidx, tvalid))
            if not np.all(np.isfinite(gs)) and a.arm not in SPC:   # spc never uses the gain (review ORCH F5)
                return diverged(it, "non-finite gain")
            rho = 1.0 if a.arm in SPC else max(1.0, gs[1] / k)
        pred, ost, en = stepf(pred, ost, ctx, cidx, cvalid, tidx, tvalid, Y, jnp.float32(rho))
        state = (pred, ost)
        if (it + 1) % LOG_EVERY == 0 or it == step0 or it + 1 == stop:
            loss = float(V.masked_mse(predict(pred, ctx, cidx, cvalid, tidx, tvalid), Y, tvalid))
            row = {"step": it + 1, "loss": loss, "rho": rho, "sec": round(time.time() - t0),
                   "peak_gib": round(jax.devices()[0].memory_stats().get("peak_bytes_in_use", 0) / 2 ** 30, 2)}
            if k is not None:
                row.update({"gain_min": gs[0], "gain_med": gs[1], "gain_max": gs[2]})
                if a.arm in BLOCK:                   # "rho" above is only the median-rule reference; these are the rho the step used
                    E2, RBs = (np.asarray(x) for x in en)
                    row.update({"rho_mode": "per-block", "rho_used_min": float(RBs.min()), "rho_used_med": float(np.median(RBs)),
                                "rho_used_max": float(RBs.max()), "frac_rho_floor": float(np.mean(RBs <= 1.0)),
                                "frac_energy_rise": float(np.mean(E2[..., 1] > E2[..., 0]))})
                elif a.arm in SPC:                   # relaxation diagnostics for the standard-PC baseline (review DES F5)
                    E2 = np.asarray(en)
                    row.update({"E_end_over_E_start_med": float(np.median(E2[..., 1] / np.maximum(E2[..., 0], 1e-30))),
                                "frac_energy_rise": float(np.mean(E2[..., 1] > E2[..., 0]))})
                cx, cy = encode(enc, tenc, jnp.asarray(V.standardise(Xcr)), jnp.asarray(V.standardise(Xcr)), jnp.asarray(cr[0]), jnp.asarray(cr[1]), jnp.asarray(cr[2]), vc)
                c1, rr, cmin, cl, lrat = credit(pred, cx, jnp.asarray(cr[0]), jnp.asarray(cr[1]), jnp.asarray(cr[2]), jnp.asarray(cr[3]), cy,
                                          jnp.float32(rho), alpha, a.T, a.chunk, a.gn_iters, float(k) if a.arm in BLOCK else 0.0, method_of(a.arm))
                row.update({"cos_bp": float(c1), "ratio_bp": float(rr), "min_leaf_cos": float(cmin), "leaf_cos": [round(float(x), 4) for x in cl],
                            "leaf_ratio": [float(f"{float(x):.3g}") for x in lrat]})
            log.append(row); print(json.dumps(row), flush=True)
            if not np.isfinite(loss):
                return diverged(it + 1, "non-finite loss")
        if (it + 1) % spe == 0 or it + 1 == stop:   # checkpoint every epoch: log first, then the checkpoint
            if not np.isfinite(float(V.masked_mse(predict(pred, ctx, cidx, cvalid, tidx, tvalid), Y, tvalid))):
                return diverged(it + 1, "non-finite loss at checkpoint")
            V.save_json(logf, {"log": log, "cfg": dc.asdict(a), "commit": commit})
            try:
                V.save_ckpt(ck, (state, jnp.float32(rho), jnp.asarray(gs, jnp.float32)), it + 1)
            except AssertionError as e:
                return diverged(it + 1, str(e))
            now = time.time(); ep = (now - ep_t) if ep_t else 0.0; ep_t = now
            if a.deadline_min > 0 and it + 1 < stop and (now - t0 + 1.5 * ep) / 60 > a.deadline_min:
                print("DEADLINE checkpointed at step", it + 1, flush=True); sys.exit(75)
    res, per = evaluate(state[0], enc, tenc, vc, Xev, yev, Xknn, yknn)
    floor, _ = evaluate(make_predictor(vc, a), enc, tenc, vc, Xev, yev, Xknn, yknn)   # untrained predictor, same seed
    np.savez_compressed(os.path.join(out_dir, "per_image.npz"), **per)
    eqx.tree_serialise_leaves(os.path.join(out_dir, "predictor.eqx"), state[0])
    V.save_json(os.path.join(out_dir, "report.json"), {"cfg": dc.asdict(a), "commit": commit, "v0_commit": meta.get("commit"),
                                                       "v0_sha256": meta.get("sha256"), "steps": stop, "eval_split": a.split, "eval": res, "floor_untrained": floor,
                                                       "eval_note": "eval images un-augmented; targets from clean images; per-image sums in per_image.npz (err, Y2, PY, P2) for paired bootstrap", "log": log})
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
