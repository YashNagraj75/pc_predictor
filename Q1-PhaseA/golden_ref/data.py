# Dataset loading and multi-view generation for Phase A. Two datasets: galaxy10 (Galaxy10 SDSS/DECaLS,
# 69x69x3, rotation-invariant) and mnist (28x28x1, max_rotation=0.0 since digits have a canonical orientation).
# View generation uses one affine sampling grid per view (crop + rotation + flip in a single bilinear
# resample, vmap/jit-able). make_views also returns each view's crop box, used by Phase B's predictor.
import os, struct, gzip
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates


# ------------------------------------------------------------------ loading
def load_galaxy10(root="../data", n_train=-1, n_test=2000, seed=0,
                  size=69, chunk=512):
    """Loads and splits Galaxy10 as (Xtr, ytr, Xte, yte), X float32 (N,size,size,3) in [0,1], caching
    the downsampled de-duplicated DECaLS array as an .npz to avoid re-reading the raw h5."""
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
    """Reads and bilinearly downsamples the raw Galaxy10 DECaLS h5 file in chunks of `chunk` images."""
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


def load_mnist(root="../data", n_train=-1, n_test=2000, seed=0):
    """-> (Xtr, ytr, Xte, yte) with X float32 (N,28,28,1) in [0,1]."""
    d = os.path.join(root, "MNIST", "raw")
    def rd(name):
        """Reads one IDX-format MNIST file (gzipped or not) and returns it as a numpy array."""
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
    """Dispatches to load_galaxy10 or load_mnist based on cfg.dataset; raises ValueError otherwise."""
    if cfg.dataset == "galaxy10":
        return load_galaxy10(cfg.root, cfg.n_train, cfg.n_test, cfg.seed)
    if cfg.dataset == "mnist":
        return load_mnist(cfg.root, cfg.n_train, cfg.n_test, cfg.seed)
    raise ValueError(cfg.dataset)


# ------------------------------------------------------------ augmentation
@partial(jax.jit, static_argnames=("out_size", "hflip"))
def _one_view(key, img, out_size, area_lo, area_hi, max_rot,
              hflip, brightness, contrast):
    """Produces one augmented view of one image via a single affine sampling grid (crop, rotation, flip, jitter).
    Returns (view (out,out,C), box (cx, cy, w, h) with centre in [-1,1], params (theta, flipped, brightness, contrast))."""
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


def make_views_labeled(key, imgs, cfg):
    """Generates cfg.n_views augmented views of imgs (B,H,W,C) and returns (Zin, boxes, params) stacked over views.
    make_views is the two-value training-path wrapper around this."""
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


def make_views(key, imgs, cfg):
    """Training-path view generation.  -> (Zin, boxes).  See make_views_labeled."""
    views, boxes, _ = make_views_labeled(key, imgs, cfg)
    return views, boxes


def eval_batch(imgs, cfg):
    """Deterministic centre view for evaluation -- no augmentation, so probe
    metrics are not contaminated by augmentation noise.  (B, input_dim)."""
    x = jax.image.resize(imgs, (imgs.shape[0], cfg.img_size, cfg.img_size,
                                imgs.shape[-1]), method="bilinear")
    if cfg.grayscale and x.shape[-1] == 3:
        x = x.mean(-1, keepdims=True)
    return x.reshape(imgs.shape[0], -1)


def batches(key, X, y, batch_size, n_steps):
    """Infinite shuffled batch iterator (numpy indexing, cheap)."""
    n = len(X)
    for i in range(n_steps):
        key, sub = jax.random.split(key)
        idx = np.asarray(jax.random.choice(sub, n, (batch_size,), replace=False))
        yield i, X[idx], y[idx]
