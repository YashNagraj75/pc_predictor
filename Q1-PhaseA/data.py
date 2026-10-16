import os, struct, gzip
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates


# ------------------------------------------------------------------ loading
# Returns (Xtr, ytr, Xte, yte); X float32 (N,size,size,3) in [0,1].
# Uses Galaxy10_DECals_NoDuplicated.h5 (DECaLS 256x256x3) to avoid train/test leakage from dupes.
# Raw data is 3.5GB uint8/14GB f32; read in chunks + bilinear downsample. size=69 matches old SDSS res.
# H5 read+resample ~60s; downsampled array cached as .npz (~3.9s reload). Delete .npz to force re-read.
def load_galaxy10(root="../data", n_train=-1, n_test=2000, seed=0,
                  size=69, chunk=512):
    cache = os.path.join(root, "galaxy10", f"g10_{size}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        X, labs = z["X"], z["labs"]
    else:
        X, labs = _read_galaxy10_h5(root, size, chunk)
        np.savez_compressed(cache, X=X, labs=labs)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    X, labs = X[perm], labs[perm]
    Xte, yte = X[:n_test], labs[:n_test]
    Xtr, ytr = X[n_test:], labs[n_test:]
    if n_train > 0:
        Xtr, ytr = Xtr[:n_train], ytr[:n_train]
    return Xtr, ytr, Xte, yte


def _read_galaxy10_h5(root, size, chunk):
    import h5py
    path = os.path.join(root, "galaxy10", "Galaxy10_DECals_NoDuplicated.h5")
    with h5py.File(path, "r") as f:
        n = f["images"].shape[0]
        X = np.empty((n, size, size, 3), np.float32)
        for i in range(0, n, chunk):
            c = np.asarray(f["images"][i:i + chunk], np.float32) / 255.0
            X[i:i + chunk] = np.asarray(jax.image.resize(
                c, (c.shape[0], size, size, 3), "linear", antialias=True))
        labs = f["ans"][:].astype(np.int32)
    return X, labs


# Returns (Xtr, ytr, Xte, yte); X float32 shape (N,28,28,1), scaled to [0,1]
def load_mnist(root="../data", n_train=-1, n_test=2000, seed=0):
    d = os.path.join(root, "MNIST", "raw")
    def rd(name):
        p = os.path.join(d, name)
        p = p if os.path.exists(p) else p + ".gz"
        op = gzip.open if p.endswith(".gz") else open
        with op(p, "rb") as f:
            buf = f.read()
        magic = int.from_bytes(buf[:4], "big")
        ndim = magic & 0xFF
        shape = [int.from_bytes(buf[4 + 4*i: 8 + 4*i], "big") for i in range(ndim)]
        return np.frombuffer(buf[4 + 4*ndim:], np.uint8).reshape(shape)
    Xtr = rd("train-images-idx3-ubyte")[..., None].astype(np.float32) / 255.0
    ytr = rd("train-labels-idx1-ubyte").astype(np.int32)
    Xte = rd("t10k-images-idx3-ubyte")[..., None].astype(np.float32) / 255.0
    yte = rd("t10k-labels-idx1-ubyte").astype(np.int32)
    if n_test > 0:
        Xte, yte = Xte[:n_test], yte[:n_test]
    if n_train > 0:
        Xtr, ytr = Xtr[:n_train], ytr[:n_train]
    return Xtr, ytr, Xte, yte


def load_dataset(cfg):
    if cfg.dataset == "galaxy10":
        return load_galaxy10(cfg.root, cfg.n_train, cfg.n_test, cfg.seed)
    if cfg.dataset == "mnist":
        return load_mnist(cfg.root, cfg.n_train, cfg.n_test, cfg.seed)
    raise ValueError(cfg.dataset)


# ------------------------------------------------------------ augmentation
# One augmented view of one image via a single affine sampling grid.
# Returns (view (out,out,C), box (4,), params (4,))
# box = (cx, cy, w, h): centre in [-1,1], w,h = crop's linear fraction of image.
@partial(jax.jit, static_argnames=("out_size", "hflip"))
def _one_view(key, img, out_size, area_lo, area_hi, max_rot,
              hflip, brightness, contrast):
    H, W, C = img.shape
    k = jax.random.split(key, 6)
    area = jax.random.uniform(k[0], (), minval=area_lo, maxval=area_hi)
    s = jnp.sqrt(area)                                   # linear crop fraction
    theta = jax.random.uniform(k[1], (), minval=-max_rot, maxval=max_rot)
    fx = jnp.where(jax.random.bernoulli(k[2]) & hflip, -1.0, 1.0)
    lim = jnp.maximum(0.0, 1.0 - s)
    cx = jax.random.uniform(k[3], (), minval=-lim, maxval=lim)
    cy = jax.random.uniform(k[4], (), minval=-lim, maxval=lim)

    g = jnp.linspace(-1.0, 1.0, out_size)
    yy, xx = jnp.meshgrid(g, g, indexing="ij")
    ct, st = jnp.cos(theta), jnp.sin(theta)
    xs = s * (ct * xx * fx - st * yy) + cx
    ys = s * (st * xx * fx + ct * yy) + cy
    px = (xs + 1.0) * 0.5 * (W - 1)
    py = (ys + 1.0) * 0.5 * (H - 1)
    out = jnp.stack([map_coordinates(img[..., c], [py, px], order=1, mode="nearest")
                     for c in range(C)], axis=-1)

    kb = jax.random.split(k[5], 2)
    b = jax.random.uniform(kb[0], (), minval=-brightness, maxval=brightness)
    c_ = 1.0 + jax.random.uniform(kb[1], (), minval=-contrast, maxval=contrast)
    out = jnp.clip((out - 0.5) * c_ + 0.5 + b, 0.0, 1.0)
    return (out, jnp.stack([cx, cy, s, s]),
            jnp.stack([theta, jnp.where(fx < 0, 1.0, 0.0), b, c_ - 1.0]))


# imgs: (B,H,W,C) float32 in [0,1].
# Zin: (V,B,input_dim) flattened augmented views for encoder
# boxes: (V,B,4) crop boxes (cx,cy,w,h) -- unused Phase A, needed Phase B pos conditioning
# params: (V,B,4) (theta, flipped, brightness, contrast_delta) -- only probes.py reads these
# make_views is the 2-value training wrapper; keep training call sites on that one
def make_views_labeled(key, imgs, cfg):
    B = imgs.shape[0]
    keys = jax.random.split(key, cfg.n_views * B).reshape(cfg.n_views, B, 2)
    fn = jax.vmap(lambda kk, im: _one_view(
        kk, im, cfg.img_size, cfg.crop_area[0], cfg.crop_area[1],
        cfg.max_rotation, cfg.hflip, cfg.brightness, cfg.contrast))
    views, boxes, params = [], [], []
    for v in range(cfg.n_views):
        out, bx, pr = fn(keys[v], imgs)
        if cfg.grayscale and out.shape[-1] == 3:
            out = out.mean(-1, keepdims=True)
        views.append(out.reshape(B, -1))
        boxes.append(bx)
        params.append(pr)
    return jnp.stack(views), jnp.stack(boxes), jnp.stack(params)


# Training-path view generation.
# Returns (Zin, boxes). See make_views_labeled.
def make_views(key, imgs, cfg):
    views, boxes, _ = make_views_labeled(key, imgs, cfg)
    return views, boxes


# Deterministic centre view for evaluation -- no augmentation.
# Avoids contaminating probe metrics with augmentation noise.
# Output shape: (B, input_dim).
def eval_batch(imgs, cfg):
    x = jax.image.resize(imgs, (imgs.shape[0], cfg.img_size, cfg.img_size,
                                imgs.shape[-1]), method="bilinear")
    if cfg.grayscale and x.shape[-1] == 3:
        x = x.mean(-1, keepdims=True)
    return x.reshape(imgs.shape[0], -1)


# Infinite shuffled batch iterator (numpy indexing, cheap).
def batches(key, X, y, batch_size, n_steps):
    n = len(X)
    for i in range(n_steps):
        key, sub = jax.random.split(key)
        idx = np.asarray(jax.random.choice(sub, n, (batch_size,), replace=False))
        yield i, X[idx], y[idx]
