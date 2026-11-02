import argparse, dataclasses as dc, json, math, os, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp, equinox as eqx, optax
jax.config.update("jax_default_matmul_precision", "highest")
import data as D

# Small I-JEPA (Assran et al. 2023) for CIFAR-10, stage V0 of plan_vit_ijepa.md: ViT context encoder, EMA target encoder,
# narrow transformer predictor with mask tokens, multi-block masking.  Context and target index sets are padded to fixed
# sizes (NC = all tokens, NT = NT_MAX per block) with validity flags, so every step compiles once; padded tokens are
# excluded as attention keys and from the loss.


@dc.dataclass
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
    min_ctx: int = 4
    epochs: int = 300
    batch: int = 256
    lr: float = 1e-3
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


NT_MAX = 20                     # largest target block (0.2 x 64 tokens = 12.8, aspect <= 1.5 -> at most 4 x 5)


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
        self.embed, self.pos = Lin(k[0], cfg.pin, cfg.dim), 0.02 * jax.random.truncated_normal(k[1], -2, 2, (cfg.ntok, cfg.dim))
        self.blocks = [Block(k[2 + i], cfg.dim, cfg.heads, cfg.mlp) for i in range(cfg.depth)]
        self.norm = LN(cfg.dim)

    def __call__(self, img, idx, valid, p):      # one image; tokens idx (N,), valid (N,) -> (N, dim)
        x = (self.embed(patchify(img, p)) + self.pos)[idx]
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
        self.pos = 0.02 * jax.random.truncated_normal(k[2], -2, 2, (cfg.ntok, width))
        self.mask_tok = 0.02 * jax.random.truncated_normal(k[3], -2, 2, (width,))
        self.blocks = [Block(k[4 + i], width, heads, mlp) for i in range(depth)]
        self.norm = LN(width)
        self.scale = 1.0 / math.sqrt(2 * depth) if scaled else 1.0

    @property
    def n_sub(self):
        return 2 * len(self.blocks)

    # The predictor as embed -> 2L residual sublayers -> head, for ePC: h_j = h_{j-1} + scale * f_j(h_{j-1}).
    def embed_(self, ctx, cidx, tidx):
        xc = self.inp(ctx) + self.pos[cidx]
        xt = self.mask_tok[None] + self.pos[tidx]
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
    allidx = jnp.arange(cfg.ntok); allv = jnp.ones(cfg.ntok, bool)
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
    return optax.adamw(sch, weight_decay=cfg.wd, mask=wd_mask), total


def augment(rng, X):                             # random crop (pad 4, reflect) + horizontal flip, numpy
    B, H, W, C = X.shape
    Xp = np.pad(X, ((0, 0), (4, 4), (4, 4), (0, 0)), mode="reflect")
    r, c = rng.integers(0, 9, B), rng.integers(0, 9, B)
    out = np.stack([Xp[i, r[i]:r[i] + H, c[i]:c[i] + W] for i in range(B)])
    flip = rng.random(B) < 0.5
    out[flip] = out[flip, :, ::-1]
    return out


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


def features(enc, X, cfg, chunk=500):            # mean-pooled encoder features of whole images
    return np.concatenate([np.asarray(_feat(enc, jnp.asarray(X[i:i + chunk]), cfg.ntok, cfg.patch)) for i in range(0, len(X), chunk)])


def gate_report(tenc, cfg, Xtr, ytr, Xte, yte, n_probe=10000, seed=0):
    import metrics as MT, probes as PR
    rng = np.random.default_rng(seed); it = rng.choice(len(Xtr), min(n_probe, len(Xtr)), replace=False)
    rand = Encoder(jax.random.PRNGKey(cfg.seed + 1000), cfg)
    out = {}
    for name, e in (("trained", tenc), ("random_init", rand)):
        Ftr, Fte = features(e, Xtr[it], cfg), features(e, Xte, cfg)
        s = np.linalg.svd(Fte - Fte.mean(0), compute_uv=False); pr = s ** 2 / np.sum(s ** 2)
        sd = Fte.std(0)
        out[name] = {"knn": MT.knn_accuracy(Ftr, ytr[it], Fte, yte, k=20), "linear": PR.class_probe(Ftr, ytr[it], Fte, yte)["accuracy"],
                     "eff_rank": float(np.exp(-np.sum(pr * np.log(pr + 1e-12)))), "min_over_median_std": float(sd.min() / np.median(sd)),
                     "finite": bool(np.all(np.isfinite(Fte)))}
    Ptr, Pte = Xtr[it].reshape(len(it), -1), Xte.reshape(len(Xte), -1)
    out["raw_pixels"] = {"knn": MT.knn_accuracy(Ptr, ytr[it], Pte, yte, k=20), "linear": PR.class_probe(Ptr, ytr[it], Pte, yte)["accuracy"]}
    t, r = out["trained"], out["random_init"]
    out["pass"] = bool(t["finite"] and t["knn"] > max(r["knn"], out["raw_pixels"]["knn"]) + 0.05
                       and t["linear"] > max(r["linear"], out["raw_pixels"]["linear"]) + 0.05
                       and t["eff_rank"] >= 0.1 * cfg.dim and t["min_over_median_std"] >= 0.05)
    return out


# ------------------------------------------------------------------ main
def save(path, obj, meta):
    tmp = path + ".tmp"; eqx.tree_serialise_leaves(tmp, obj); os.replace(tmp, path)
    json.dump(meta, open(path.replace(".eqx", ".json") + ".tmp", "w"), indent=1); os.replace(path.replace(".eqx", ".json") + ".tmp", path.replace(".eqx", ".json"))


def train(cfg, out_dir, log_every=200, ckpt_every_epochs=10):
    os.makedirs(out_dir, exist_ok=True)
    Xtr, ytr, Xte, yte = D.load_cifar10(n_train=cfg.n_train, n_test=2000, seed=0)
    spe = len(Xtr) // cfg.batch
    opt, total = optimiser(cfg, spe)
    enc, pred, tenc = init(cfg)
    ost = opt.init(eqx.filter((enc, pred), eqx.is_inexact_array))
    step0, log = 0, []
    ck = os.path.join(out_dir, "ckpt.eqx")
    if os.path.exists(ck):                       # resume
        meta = json.load(open(ck.replace(".eqx", ".json")))
        enc, pred, tenc, ost = eqx.tree_deserialise_leaves(ck, (enc, pred, tenc, ost))
        step0, log = meta["step"], meta["log"]
        print("RESUME at step", step0, flush=True)
    stepf = make_step(cfg, opt)
    rng = np.random.default_rng(cfg.seed * 7919 + step0)
    stop = cfg.max_steps if cfg.max_steps > 0 else total
    t0 = time.time(); perm = rng.permutation(len(Xtr)); pi = 0
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    for it in range(step0, stop):
        if pi + cfg.batch > len(perm):
            perm, pi = rng.permutation(len(Xtr)), 0
        bi = perm[pi:pi + cfg.batch]; pi += cfg.batch
        imgs = jnp.asarray(augment(rng, Xtr[bi]))
        cidx, cvalid, tidx, tvalid = (jnp.asarray(a) for a in sample_masks(rng, cfg.batch, cfg))
        m = jnp.float32(cfg.ema0 + (1.0 - cfg.ema0) * it / total)      # array, not a Python float: no recompiles
        (enc, pred), tenc, ost, loss = stepf((enc, pred), tenc, ost, imgs, cidx, cvalid, tidx, tvalid, m)
        if (it + 1) % log_every == 0 or it == step0:
            lv = float(loss)
            if not np.isfinite(lv):
                print("DIVERGED at", it + 1, flush=True); return None
            fs = features(tenc, Xte[:256], cfg)
            row = {"step": it + 1, "loss": lv, "ema": m, "feat_std": float(fs.std(0).mean()), "sec": round(time.time() - t0)}
            log.append(row); print(json.dumps(row), flush=True)
        if (it + 1) % (ckpt_every_epochs * spe) == 0 or it + 1 == stop:
            save(ck, (enc, pred, tenc, ost), {"step": it + 1, "log": log, "cfg": dc.asdict(cfg), "commit": commit})
    rep = gate_report(tenc, cfg, Xtr, ytr, Xte, yte)
    save(os.path.join(out_dir, "target_encoder.eqx"), tenc, {"cfg": dc.asdict(cfg), "commit": commit, "gate": rep, "steps": stop})
    save(os.path.join(out_dir, "context_encoder.eqx"), enc, {"cfg": dc.asdict(cfg), "commit": commit, "steps": stop})
    json.dump({"gate": rep, "log": log, "cfg": dc.asdict(cfg), "commit": commit}, open(os.path.join(out_dir, "v0_report.json"), "w"), indent=1)
    print("GATE", json.dumps(rep), flush=True)
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--depth", type=int, default=6); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0); ap.add_argument("--n-train", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--batch", type=int, default=256)
    a = ap.parse_args()
    cfg = VCfg(epochs=a.epochs, depth=a.depth, seed=a.seed, max_steps=a.max_steps, n_train=a.n_train, lr=a.lr, batch=a.batch)
    rep = train(cfg, a.out)
    sys.exit(0 if (rep is not None and (rep["pass"] or cfg.max_steps > 0)) else 3)
