import argparse, dataclasses as dc, json, math, os, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp, equinox as eqx, optax
jax.config.update("jax_default_matmul_precision", "highest")
import data as D

# Small I-JEPA (Assran et al. 2023) for CIFAR-10, stage V0 of plan_vit_ijepa.md: ViT context encoder, EMA target encoder,
# narrow transformer predictor with mask tokens, multi-block masking.  Context and target index sets are padded to fixed
# sizes (NC = all tokens, NT = NT_MAX per block) with validity flags, so every step compiles once; padded tokens are
# excluded as attention keys and from the loss.


@dc.dataclass(frozen=True)              # frozen: hashable, so it can be a static jit argument
class VCfg:
    img: int = 32
    patch: int = 4
    dim: int = 192
    depth: int = 6
    heads: int = 3
    mlp: int = 768
    pdim: int = 96              # predictor width (V0); stage A uses its own predictor
    pdepth: int = 4
    pheads: int = 3
    pmlp: int = 384
    pscale: bool = False        # predictor residual branches scaled 1/sqrt(2 L) (muPC-style; stage A, all rules)
    n_targets: int = 4
    t_scale: tuple = (0.15, 0.2)
    t_aspect: tuple = (0.75, 1.5)
    c_scale: tuple = (0.85, 1.0)
    min_ctx: int = 10            # official I-JEPA default (review F9)
    epochs: int = 300
    batch: int = 256
    lr: float = 5e-4             # I-JEPA uses 1e-3 at batch 2048; 5e-4 at batch 256 with clipping (review F5/N7)
    clip: float = 1.0
    wd: float = 0.04
    warmup: int = 15            # epochs
    ema0: float = 0.996
    seed: int = 0
    n_train: int = -1           # full CIFAR-10 training set
    max_steps: int = 0          # >0: stop early (smoke), schedule unchanged

    @property
    def grid(self):
        return self.img // self.patch

    @property
    def ntok(self):
        return self.grid ** 2

    @property
    def pin(self):
        return self.patch * self.patch * 3


NT_MAX = 16                     # largest target block the sampler can produce (4 x 4; review N11)
CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465], np.float32)
CIFAR_STD = np.array([0.2470, 0.2435, 0.2616], np.float32)


# ------------------------------------------------------------------ layers
def _w(key, o, i):
    return 0.02 * jax.random.truncated_normal(key, -2.0, 2.0, (o, i))


class Lin(eqx.Module):
    W: jax.Array
    b: jax.Array

    def __init__(self, key, i, o):
        self.W, self.b = _w(key, o, i), jnp.zeros(o)

    def __call__(self, x):
        return x @ self.W.T + self.b


class LN(eqx.Module):
    g: jax.Array
    b: jax.Array

    def __init__(self, d):
        self.g, self.b = jnp.ones(d), jnp.zeros(d)

    def __call__(self, x):
        return _ln(x) * self.g + self.b


def sincos_2d(g, d):                             # fixed 2D sin-cos table (I-JEPA / MAE), (1, g*g, d)
    def one(pos, dd):
        om = 1.0 / 10000 ** (np.arange(dd // 2) / (dd / 2.0))
        o = np.outer(pos.ravel(), om)
        return np.concatenate([np.sin(o), np.cos(o)], 1)
    yy, xx = np.meshgrid(np.arange(g, dtype=np.float64), np.arange(g, dtype=np.float64), indexing="ij")
    return jnp.asarray(np.concatenate([one(yy, d // 2), one(xx, d // 2)], 1)[None], jnp.float32)


def _ln(x, eps=1e-6):
    mu = x.mean(-1, keepdims=True)
    return (x - mu) / jnp.sqrt(((x - mu) ** 2).mean(-1, keepdims=True) + eps)


# Pre-LN transformer block, split into its two residual sublayers (attention, MLP) so ePC can put a node after each.
class Block(eqx.Module):
    ln1: LN
    qkv: Lin
    proj: Lin
    ln2: LN
    fc1: Lin
    fc2: Lin
    heads: int = eqx.field(static=True)

    def __init__(self, key, d, heads, mlp):
        k = jax.random.split(key, 4)
        self.ln1, self.qkv, self.proj = LN(d), Lin(k[0], d, 3 * d), Lin(k[1], d, d)
        self.ln2, self.fc1, self.fc2 = LN(d), Lin(k[2], d, mlp), Lin(k[3], mlp, d)
        self.heads = heads

    def attn(self, x, kvalid):                   # x (N, d) one sample; kvalid (N,) which tokens may be attended to
        N, d = x.shape
        q, k, v = jnp.split(self.qkv(self.ln1(x)), 3, -1)
        sh = lambda t: t.reshape(N, self.heads, d // self.heads)
        q, k, v = sh(q), sh(k), sh(v)
        s = jnp.einsum("nhe,mhe->hnm", q, k) / math.sqrt(d // self.heads)
        s = jnp.where(kvalid[None, None, :], s, -1e30)
        return self.proj(jnp.einsum("hnm,mhe->nhe", jax.nn.softmax(s, -1), v).reshape(N, d))

    def mlp(self, x):
        return self.fc2(jax.nn.gelu(self.fc1(self.ln2(x)), approximate=False))

    def sub(self, j, x, kvalid):                 # sublayer j in {0: attention, 1: MLP}
        return self.attn(x, kvalid) if j == 0 else self.mlp(x)


def patchify(img, p):                            # (H, W, 3) -> (H/p * W/p, p*p*3), row-major token order
    H, W, C = img.shape
    return img.reshape(H // p, p, W // p, p, C).transpose(0, 2, 1, 3, 4).reshape((H // p) * (W // p), p * p * C)


def unpatchify(t, p, H, W):
    C = t.shape[-1] // (p * p)
    return t.reshape(H // p, W // p, p, p, C).transpose(0, 2, 1, 3, 4).reshape(H, W, C)


class Encoder(eqx.Module):
    embed: Lin
    pos: jax.Array
    blocks: list
    norm: LN

    def __init__(self, key, cfg):
        k = jax.random.split(key, cfg.depth + 2)
        self.embed, self.pos = Lin(k[0], cfg.pin, cfg.dim), sincos_2d(cfg.grid, cfg.dim)
        self.blocks = [Block(k[2 + i], cfg.dim, cfg.heads, cfg.mlp) for i in range(cfg.depth)]
        self.norm = LN(cfg.dim)

    def __call__(self, img, idx, valid, p):      # one image; tokens idx (N,), valid (N,) -> (N, dim)
        x = (self.embed(patchify(img, p)) + jax.lax.stop_gradient(self.pos[0]))[idx]
        for b in self.blocks:
            x = x + b.attn(x, valid)
            x = x + b.mlp(x)
        return self.norm(x)


class Predictor(eqx.Module):
    inp: Lin
    pos: jax.Array
    mask_tok: jax.Array
    blocks: list
    norm: LN
    out: Lin
    scale: float = eqx.field(static=True)

    def __init__(self, key, cfg, depth=None, width=None, heads=None, mlp=None, scaled=None):
        depth, width = depth or cfg.pdepth, width or cfg.pdim
        heads, mlp = heads or cfg.pheads, mlp or cfg.pmlp
        scaled = cfg.pscale if scaled is None else scaled
        k = jax.random.split(key, depth + 4)
        self.inp, self.out = Lin(k[0], cfg.dim, width), Lin(k[1], width, cfg.dim)
        self.pos = sincos_2d(cfg.grid, width)
        self.mask_tok = 0.02 * jax.random.truncated_normal(k[3], -2, 2, (width,))
        self.blocks = [Block(k[4 + i], width, heads, mlp) for i in range(depth)]
        self.norm = LN(width)
        self.scale = 1.0 / math.sqrt(2 * depth) if scaled else 1.0

    @property
    def n_sub(self):
        return 2 * len(self.blocks)

    # The predictor as embed -> 2L residual sublayers -> head, for ePC: h_j = h_{j-1} + scale * f_j(h_{j-1}).
    def embed_(self, ctx, cidx, tidx):
        pos = jax.lax.stop_gradient(self.pos[0])
        xc = self.inp(ctx) + pos[cidx]
        xt = self.mask_tok[None] + pos[tidx]
        return jnp.concatenate([xc, xt], 0)

    def f(self, j, h, kvalid):
        return self.blocks[j // 2].sub(j % 2, h, kvalid)

    def head(self, h, nt):
        return self.out(self.norm(h[-nt:]))

    def __call__(self, ctx, cidx, cvalid, tidx, tvalid):
        kvalid = jnp.concatenate([cvalid, tvalid])
        h = self.embed_(ctx, cidx, tidx)
        for j in range(self.n_sub):
            h = h + self.scale * self.f(j, h, kvalid)
        return self.head(h, tidx.shape[0])


# ------------------------------------------------------------------ masks (numpy, per batch)
def _block(rng, g, scale, aspect):
    s = rng.uniform(*scale); a = math.exp(rng.uniform(math.log(aspect[0]), math.log(aspect[1])))
    h = int(round(math.sqrt(s * g * g * a))); w = int(round(math.sqrt(s * g * g / a)))
    return min(max(h, 1), g), min(max(w, 1), g)


# I-JEPA multi-block masking: one target block shape and one context block shape per batch (as the official collator);
# per image, n_targets target blocks and one context block at random positions; context = context block minus all
# targets.  Returns padded index arrays and validity flags: cidx/cvalid (B, ntok), tidx/tvalid (B, K, NT_MAX).
def sample_masks(rng, B, cfg):
    g, K = cfg.grid, cfg.n_targets
    while True:
        th, tw = _block(rng, g, cfg.t_scale, cfg.t_aspect)
        ch, cw = _block(rng, g, cfg.c_scale, (1.0, 1.0))
        assert th * tw <= NT_MAX, (th, tw)
        cidx = np.zeros((B, cfg.ntok), np.int32); cvalid = np.zeros((B, cfg.ntok), bool)
        tidx = np.zeros((B, K, NT_MAX), np.int32); tvalid = np.zeros((B, K, NT_MAX), bool)
        ok = True
        for b in range(B):
            tm = np.zeros((g, g), bool)
            for k in range(K):
                r, c = rng.integers(0, g - th + 1), rng.integers(0, g - tw + 1)
                blk = (np.arange(r, r + th)[:, None] * g + np.arange(c, c + tw)[None]).ravel()
                tidx[b, k, :len(blk)] = blk; tvalid[b, k, :len(blk)] = True; tm[r:r + th, c:c + tw] = True
            r, c = rng.integers(0, g - ch + 1), rng.integers(0, g - cw + 1)
            cm = np.zeros((g, g), bool); cm[r:r + ch, c:c + cw] = True; cm &= ~tm
            ci = np.flatnonzero(cm)
            if len(ci) < cfg.min_ctx:
                ok = False; break
            cidx[b, :len(ci)] = ci; cvalid[b, :len(ci)] = True
        if ok:
            return cidx, cvalid, tidx, tvalid


# ------------------------------------------------------------------ loss, EMA, training
def targets_of(tenc, imgs, tidx, cfg):           # layer-normed target-encoder features at the target tokens (B, K, NT, dim)
    allidx = jnp.arange(cfg.ntok); allv = jnp.ones(cfg.ntok, bool)   # imgs are already standardised
    t = _ln(jax.vmap(lambda im: tenc(im, allidx, allv, cfg.patch))(imgs))
    return jax.vmap(lambda tb, ib: tb[ib])(t, tidx)


def predict_all(enc, pred, imgs, cidx, cvalid, tidx, tvalid, cfg):
    def one(im, ci, cv, ti, tv):
        z = enc(im, ci, cv, cfg.patch)
        return jax.vmap(lambda a, b: pred(z, ci, cv, a, b))(ti, tv)
    return jax.vmap(one)(imgs, cidx, cvalid, tidx, tvalid)


def masked_mse(P, Y, tvalid):                    # 0.5 x mean squared error over valid target tokens and feature dims
    m = tvalid[..., None].astype(P.dtype)
    return 0.5 * jnp.sum(m * (P - Y) ** 2) / (jnp.sum(m) * P.shape[-1])


def loss_fn(trainable, tenc, imgs, cidx, cvalid, tidx, tvalid, cfg):
    enc, pred = trainable
    Y = jax.lax.stop_gradient(targets_of(tenc, imgs, tidx, cfg))
    return masked_mse(predict_all(enc, pred, imgs, cidx, cvalid, tidx, tvalid, cfg), Y, tvalid)


def ema_update(tenc, enc, m):
    return jax.tree_util.tree_map(lambda t, c: m * t + (1.0 - m) * c, tenc, enc)


def wd_mask(tree):                               # weight decay on weight matrices only
    return jax.tree_util.tree_map(lambda x: x.ndim == 2, tree)


def optimiser(cfg, steps_per_epoch):
    total = cfg.epochs * steps_per_epoch
    sch = optax.warmup_cosine_decay_schedule(0.0, cfg.lr, max(1, cfg.warmup * steps_per_epoch), total, cfg.lr * 1e-3)
    return optax.chain(optax.clip_by_global_norm(cfg.clip), optax.adamw(sch, weight_decay=cfg.wd, mask=wd_mask)), total


@jax.jit
def _augment(key, X):                            # area crop 0.3-1 (square), flip, bilinear to 32 px (data._one_view)
    ks = jax.random.split(key, X.shape[0])
    return jax.vmap(lambda k, im: D._one_view(k, im, im.shape[0], 0.3, 1.0, 0.0, True, 0.0, 0.0)[0])(ks, X)


def augment(rng, X):
    return _augment(jax.random.PRNGKey(int(rng.integers(0, 2 ** 31 - 1))), jnp.asarray(X))


def standardise(X):
    return (X - CIFAR_MEAN) / CIFAR_STD


def init(cfg):
    k1, k2 = jax.random.split(jax.random.PRNGKey(cfg.seed))
    enc = Encoder(k1, cfg)
    return enc, Predictor(k2, cfg), enc          # context encoder, predictor, target encoder (= copy at init)


def make_step(cfg, opt):
    @eqx.filter_jit
    def step(trainable, tenc, ost, imgs, cidx, cvalid, tidx, tvalid, m):
        loss, g = eqx.filter_value_and_grad(loss_fn)(trainable, tenc, imgs, cidx, cvalid, tidx, tvalid, cfg)
        upd, ost = opt.update(g, ost, eqx.filter(trainable, eqx.is_inexact_array))
        trainable = eqx.apply_updates(trainable, upd)
        return trainable, ema_update(tenc, trainable[0], m), ost, loss
    return step


# ------------------------------------------------------------------ evaluation
@eqx.filter_jit
def _feat(e, x, ntok, patch):
    idx, v = jnp.arange(ntok), jnp.ones(ntok, bool)
    return jax.vmap(lambda im: e(im, idx, v, patch).mean(0))(x)


def features(enc, X, cfg, chunk=500):            # mean-pooled encoder features of whole (standardised) images
    return np.concatenate([np.asarray(_feat(enc, jnp.asarray(standardise(X[i:i + chunk])), cfg.ntok, cfg.patch))
                           for i in range(0, len(X), chunk)])


def eff_rank(F):
    s = np.linalg.svd(F - F.mean(0), compute_uv=False); pr = s ** 2 / np.sum(s ** 2)
    return float(np.exp(-np.sum(pr * np.log(pr + 1e-12))))


def knn_both(Ftr, ytr, Fte, yte):                # cosine kNN on raw and on train-mean-centred features (review F3)
    import metrics as MT
    mu = Ftr.mean(0)
    return {"knn": MT.knn_accuracy(Ftr, ytr, Fte, yte, k=20), "knn_centred": MT.knn_accuracy(Ftr - mu, ytr, Fte - mu, yte, k=20)}


# V0 gate, criteria fixed before the run (plan_vit_ijepa.md rev 2): on the full 10k test set, the trained target encoder
# must beat the best of 3 random inits and raw pixels by >= 0.05 on centred kNN and on a linear probe, reach an absolute
# linear-probe floor of 0.55, and show no collapse (effective rank >= 0.1 dim, min/median dimension std >= 0.05).
def gate_report(tenc, cfg, Xtr, ytr, Xte, yte):
    import probes as PR
    out = {"n_train": int(len(Xtr)), "n_test": int(len(Xte))}
    encs = [("trained", tenc)] + [(f"random_init_{r}", Encoder(jax.random.PRNGKey(cfg.seed + 1000 + r), cfg)) for r in range(3)]
    for name, e in encs:
        Ftr, Fte = features(e, Xtr, cfg), features(e, Xte, cfg)
        sd = Fte.std(0)
        out[name] = {**knn_both(Ftr, ytr, Fte, yte), "linear": PR.class_probe(Ftr, ytr, Fte, yte)["accuracy"], "eff_rank": eff_rank(Fte),
                     "min_over_median_std": float(sd.min() / np.median(sd)), "finite": bool(np.all(np.isfinite(Fte)))}
        print("GATE", name, json.dumps(out[name]), flush=True)
    Ptr, Pte = standardise(Xtr).reshape(len(Xtr), -1), standardise(Xte).reshape(len(Xte), -1)
    out["raw_pixels"] = {**knn_both(Ptr, ytr, Pte, yte), "linear": PR.class_probe(Ptr, ytr, Pte, yte)["accuracy"]}
    t = out["trained"]; ref = [out[f"random_init_{r}"] for r in range(3)] + [out["raw_pixels"]]
    out["bar_knn_centred"] = max(r["knn_centred"] for r in ref) + 0.05
    out["bar_linear"] = max(max(r["linear"] for r in ref) + 0.05, 0.55)
    out["pass"] = bool(t["finite"] and t["knn_centred"] >= out["bar_knn_centred"] and t["linear"] >= out["bar_linear"]
                       and t["eff_rank"] >= 0.1 * cfg.dim and t["min_over_median_std"] >= 0.05)
    return out


# ------------------------------------------------------------------ main
def _fsync_replace(tmp, path):
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_ckpt(path, state, step):                # step lives inside the pytree; previous checkpoint kept (review N5)
    leaves = [x for x in jax.tree_util.tree_leaves(eqx.filter(state, eqx.is_inexact_array))]
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in leaves), "refusing to checkpoint non-finite weights"
    tmp = path + ".tmp"; eqx.tree_serialise_leaves(tmp, (state, jnp.int32(step))); 
    if os.path.exists(path):
        os.replace(path, path.replace(".eqx", "_prev.eqx"))
    _fsync_replace(tmp, path)


def save_json(path, obj):
    tmp = path + ".tmp"; json.dump(obj, open(tmp, "w"), indent=1); _fsync_replace(tmp, path)


def train(cfg, out_dir, log_every=200, ckpt_every_epochs=5, knn_every_epochs=25):
    os.makedirs(out_dir, exist_ok=True)
    Xtr, ytr, Xte, yte = D.load_cifar10(n_train=cfg.n_train, n_test=-1, seed=0)
    spe = len(Xtr) // cfg.batch
    opt, total = optimiser(cfg, spe)
    enc, pred, tenc = init(cfg)
    ost = opt.init(eqx.filter((enc, pred), eqx.is_inexact_array))
    state, step0, log = (enc, pred, tenc, ost), 0, []
    ck, logf = os.path.join(out_dir, "ckpt.eqx"), os.path.join(out_dir, "train_log.json")
    if os.path.exists(ck):                       # resume: weights, optimiser and step from the same file
        prev = json.load(open(logf))
        same = lambda d: {k: v for k, v in d.items() if k != "max_steps"}
        assert same(prev["cfg"]) == same(json.loads(json.dumps(dc.asdict(cfg)))), "config changed since the checkpoint"
        state, st = eqx.tree_deserialise_leaves(ck, (state, jnp.int32(0)))
        step0, log = int(st), [r for r in prev["log"] if r["step"] <= int(st)]
        print("RESUME at step", step0, flush=True)
    stepf = make_step(cfg, opt)
    stop = cfg.max_steps if cfg.max_steps > 0 else total
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    t0, epoch, perm = time.time(), -1, None
    yk = np.random.default_rng(1).choice(len(Xtr), 5000, replace=False)
    for it in range(step0, stop):
        if it // spe != epoch:                   # per-epoch permutation from (seed, epoch): resume is exact (review N6)
            epoch = it // spe; perm = np.random.default_rng([cfg.seed, epoch]).permutation(len(Xtr))
        rng = np.random.default_rng([cfg.seed, 1, it])
        bi = perm[(it % spe) * cfg.batch:(it % spe + 1) * cfg.batch]
        imgs = standardise(augment(rng, Xtr[bi]))
        cidx, cvalid, tidx, tvalid = (jnp.asarray(a) for a in sample_masks(rng, cfg.batch, cfg))
        m = cfg.ema0 + (1.0 - cfg.ema0) * it / total
        enc, pred, tenc, ost = state
        (enc, pred), tenc, ost, loss = stepf((enc, pred), tenc, ost, imgs, cidx, cvalid, tidx, tvalid, jnp.float32(m))
        state = (enc, pred, tenc, ost)
        if (it + 1) % log_every == 0 or it + 1 == stop:
            lv = float(loss); fs = features(tenc, Xte[:512], cfg)
            row = {"step": it + 1, "loss": lv, "ema": float(m), "feat_std": float(fs.std(0).mean()), "eff_rank": eff_rank(fs), "sec": round(time.time() - t0)}
            if (it + 1) % (knn_every_epochs * spe) < log_every:
                row.update(knn_both(features(tenc, Xtr[yk], cfg), ytr[yk], features(tenc, Xte[:2000], cfg), yte[:2000]))
            log.append(row); print(json.dumps(row), flush=True)
            if not np.isfinite(lv) or row["eff_rank"] < 5:
                print("ABORT: non-finite loss or collapsed features", flush=True)
                save_json(logf, {"log": log, "cfg": dc.asdict(cfg), "commit": commit, "aborted": True}); return None
        if (it + 1) % (ckpt_every_epochs * spe) == 0 or it + 1 == stop:
            save_ckpt(ck, state, it + 1); save_json(logf, {"log": log, "cfg": dc.asdict(cfg), "commit": commit})
    rep = gate_report(state[2], cfg, Xtr, ytr, Xte, yte)
    for name, obj in (("target_encoder", state[2]), ("context_encoder", state[0])):
        eqx.tree_serialise_leaves(os.path.join(out_dir, name + ".eqx"), obj)
    import hashlib
    hashes = {n: hashlib.sha256(open(os.path.join(out_dir, n + ".eqx"), "rb").read()).hexdigest() for n in ("target_encoder", "context_encoder")}
    save_json(os.path.join(out_dir, "v0_report.json"), {"gate": rep, "log": log, "cfg": dc.asdict(cfg), "commit": commit, "sha256": hashes, "steps": stop})
    print("GATE", json.dumps({k: rep[k] for k in ("pass", "bar_knn_centred", "bar_linear")}), json.dumps(rep["trained"]), flush=True)
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=15); ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=-1); ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch", type=int, default=256); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg = VCfg(epochs=a.epochs, warmup=a.warmup, depth=a.depth, seed=a.seed, max_steps=a.max_steps, n_train=a.n_train, lr=a.lr, batch=a.batch)
    rep = train(cfg, a.out)
    if a.smoke:                                  # smoke: training finite and features not collapsed; gate printed, not required
        sys.exit(0 if rep is not None else 3)
    sys.exit(0 if (rep is not None and rep["pass"]) else 3)
