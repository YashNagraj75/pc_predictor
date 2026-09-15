"""Verification of the completed helpers.

These are not decoration.  sigreg.py and metrics.py are handed over as
finished code, so their behaviour has to be pinned down by tests, otherwise
they are a liability rather than a help.  Run:

    ../.venv/bin/python3 test_helpers.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # sandbox sets PYTHONSAFEPATH=1, which strips the script dir from sys.path

import numpy as np
import jax, jax.numpy as jnp, optax
import sigreg as S, metrics as M, data as D
from config import DataCfg

K = jax.random.PRNGKey(0)
ok = lambda name, v: print(f"  PASS  {name:44s} {v}")


# ------------------------------------------------------------------ sigreg
def t_directions():
    A = S.sample_directions(K, 32, 64)
    n = jnp.linalg.norm(A, axis=0)
    assert A.shape == (32, 64)
    assert jnp.allclose(n, 1.0, atol=1e-5), n
    ok("directions: shape + unit-norm columns", f"|a|=1.0 for all 64")


def t_ep_ordering():
    """Power against shape alternatives, measured against the NULL BAND.

    A single Gaussian draw fluctuates, so "gaussian < alternative" is only
    meaningful relative to the spread of the statistic under H0.  This test
    estimates that spread over 20 Gaussian draws and asks which alternatives
    sit outside it.  Worth knowing: at N=1024 with the paper's defaults the
    statistic has little power against MILD shape alternatives (a uniform with
    unit variance scores *inside* the Gaussian band), and large power against
    variance/mean misfit.  Since SIGReg is the anti-collapse mechanism, what
    matters is the second kind -- but do not read it as a general shape test.
    """
    N, Kd, M = 1024, 16, 128
    A = S.sample_directions(K, Kd, M)
    null = []
    for i in range(20):
        z = jax.random.normal(jax.random.PRNGKey(1000 + i), (N, Kd))
        null.append(float(S.sigreg(z, A)))
    mu, sd = float(np.mean(null)), float(np.std(null))
    hi = mu + 3 * sd

    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(7), 4)
    g = jax.random.normal(k1, (N, Kd))
    alt = {
        "uniform (var 1)":  jax.random.uniform(k2, (N, Kd), minval=-np.sqrt(3), maxval=np.sqrt(3)),
        "bimodal":          jnp.sign(jax.random.normal(k3, (N, Kd))) + 0.2 * g,
        "laplace":          jax.random.laplace(k4, (N, Kd)) / np.sqrt(2.0),
        "collapsed (0.1x)": g * 0.1,
        "scaled 3x":        g * 3.0,
    }
    v = {n: float(S.sigreg(z, A)) for n, z in alt.items()}
    # SECOND-MOMENT misfit -- what SIGReg exists to catch -- is detected by
    # two to three orders of magnitude.
    assert v["collapsed (0.1x)"] > 50 * hi, (v, hi)
    assert v["scaled 3x"] > 50 * hi, (v, hi)
    # SHAPE misfit at unit variance is barely detected: all three land within
    # a small factor of the null band's own edge.  This is the documented
    # blind spot; asserted so a change in behaviour gets noticed.
    for n in ("uniform (var 1)", "laplace", "bimodal"):
        assert v[n] < 3 * hi, (n, v, hi)
    ok("epps-pulley: strong vs scale, weak vs shape",
       f"H0 {mu:.2f}+-{sd:.2f} (3sd={hi:.2f}) | " +
       " ".join(f"{n.split()[0]}={x:.3g}" for n, x in v.items()))


def t_ep_not_scale_invariant():
    """SIGReg tests against N(0,1) including mean and variance -- it must NOT
    be scale invariant, or it would not constrain the embedding's scale."""
    k1, k2 = jax.random.split(K)
    A = S.sample_directions(k1, 16, 128)
    z = jax.random.normal(k2, (1024, 16))
    a, b, c = float(S.sigreg(z, A)), float(S.sigreg(z * 2, A)), float(S.sigreg(z + 2, A))
    assert b > a and c > a
    ok("epps-pulley: penalises scale and shift", f"z={a:.3f} 2z={b:.3f} z+2={c:.3f}")


def t_ep_minimisation():
    """End-to-end: descending SIGReg on the embeddings themselves must turn an
    anisotropic cloud into an isotropic one (the paper's Figure 6 setup)."""
    k1, k2 = jax.random.split(K)
    N, Kd = 512, 16
    z = jax.random.normal(k1, (N, Kd)) * jnp.linspace(0.1, 4.0, Kd)
    before = M.spectrum_report(np.asarray(z))
    opt = optax.adam(3e-2); st = opt.init(z)
    g = jax.jit(jax.grad(lambda zz, AA: S.sigreg(zz, AA)))
    key = k2
    for i in range(400):
        key, sub = jax.random.split(key)
        A = S.sample_directions(sub, Kd, 64)       # resampled per step, per paper
        gr = g(z, A)
        up, st = opt.update(gr, st)
        z = optax.apply_updates(z, up)
    after = M.spectrum_report(np.asarray(z))
    assert after["effective_rank"] > before["effective_rank"]
    assert after["log_cond"] < before["log_cond"]
    ok("sigreg descent: eff.rank rises, cond falls",
       f"eff.rank {before['effective_rank']:.1f}->{after['effective_rank']:.1f} "
       f"log10cond {before['log_cond']:.2f}->{after['log_cond']:.2f} "
       f"trace {before['trace']:.1f}->{after['trace']:.1f}")


def t_batch_coupling():
    """The claim that motivates the two output_drive variants: row n of
    dSIGReg/dZ depends on the OTHER rows.  Perturb row 0, watch row 5 move."""
    k1, k2 = jax.random.split(K)
    A = S.sample_directions(k1, 8, 64)
    z = jax.random.normal(k2, (64, 8))
    g0 = S.sigreg_grad(z, A)
    z2 = z.at[0].add(3.0)
    g1 = S.sigreg_grad(z2, A)
    moved = float(jnp.abs(g1[5] - g0[5]).max())
    assert g0.shape == z.shape
    assert moved > 1e-8, moved
    ok("sigreg_grad: batch-coupled (not per-sample)",
       f"row-5 grad moved {moved:.3e} when row 0 changed")


def t_prediction_loss():
    k1, = jax.random.split(K, 1)
    Zv = jnp.tile(jax.random.normal(k1, (1, 32, 8)), (4, 1, 1))
    assert float(S.prediction_loss(Zv, 4)) < 1e-10
    Zv2 = Zv + jax.random.normal(k1, Zv.shape) * 0.5
    assert float(S.prediction_loss(Zv2, 4)) > 0
    ok("prediction_loss: 0 iff all views identical",
       f"identical={float(S.prediction_loss(Zv,4)):.2e} perturbed={float(S.prediction_loss(Zv2,4)):.3f}")


def t_b6_discrepancy():
    """Pin down the Appendix B.6 problem found earlier, so the comment in
    sigreg.py is backed by a test rather than a memory.
    Eq (5) all-pairs form vs Eq (7) distance-from-centre form."""
    z = np.asarray(jax.random.normal(K, (6, 32, 8)))
    Vg, V = 2, 6
    mu = z[:Vg].mean(0)
    eq7 = ((mu[None] - z) ** 2).sum(-1).mean()
    eq5 = np.mean([[((z[v] - z[vp]) ** 2).sum(-1) for vp in range(V)]
                   for v in range(Vg)])
    within = ((z[:Vg] - mu[None]) ** 2).sum(-1).mean()
    assert not np.isclose(eq5, eq7), (eq5, eq7)
    assert np.isclose(eq5, eq7 + within, rtol=1e-6), (eq5, eq7, within)
    ok("B.6: Eq(5) != Eq(7); gap == within-view var",
       f"eq5={eq5:.4f} eq7={eq7:.4f} within={within:.4f}")


# ----------------------------------------------------------------- metrics
def t_cka():
    x = np.asarray(jax.random.normal(K, (500, 24)))
    q, _ = np.linalg.qr(np.asarray(jax.random.normal(K, (24, 24))))
    assert abs(M.linear_cka(x, x) - 1) < 1e-9
    assert abs(M.linear_cka(x, x @ q) - 1) < 1e-9      # rotation invariant
    assert abs(M.linear_cka(x, x * 7.0) - 1) < 1e-9    # scale invariant
    y = np.asarray(jax.random.normal(jax.random.PRNGKey(9), (500, 24)))
    lo = M.linear_cka(x, y)
    assert lo < 0.2, lo
    ok("cka: 1 for self/rotation/scale, low for independent", f"independent={lo:.3f}")


def t_spectrum():
    iso = np.asarray(jax.random.normal(K, (4000, 20)))
    r = M.spectrum_report(iso)
    assert r["effective_rank"] > 19, r
    v = np.asarray(jax.random.normal(K, (4000, 1)))
    rank1 = np.tile(v, (1, 20))
    r1 = M.spectrum_report(rank1)
    assert r1["effective_rank"] < 1.05, r1
    ok("spectrum: eff.rank ~D isotropic, ~1 rank-1",
       f"iso={r['effective_rank']:.2f}/20  rank1={r1['effective_rank']:.3f}")


def t_knn():
    k1, k2 = jax.random.split(K)
    c = np.asarray(jax.random.normal(k1, (5, 16))) * 6
    y = np.repeat(np.arange(5), 200)
    Z = c[y] + np.asarray(jax.random.normal(k2, (1000, 16))) * 0.4
    a = M.knn_accuracy(Z[::2], y[::2], Z[1::2], y[1::2], k=5)
    Zr = np.asarray(jax.random.normal(k2, (1000, 16)))
    b = M.knn_accuracy(Zr[::2], y[::2], Zr[1::2], y[1::2], k=5)
    assert a > 0.95 and b < 0.4, (a, b)
    ok("knn: separable ~1.0, random ~chance", f"separable={a:.3f} random={b:.3f} (chance=0.2)")


def t_twonn():
    """Gaussian data on a d-dim linear subspace embedded in 40-d."""
    got = {}
    for d in (2, 5, 10):
        k = jax.random.PRNGKey(d)
        lat = np.asarray(jax.random.normal(k, (2000, d)))
        B = np.asarray(jax.random.normal(k, (d, 40)))
        got[d] = M.twonn_dimension(lat @ B)
    for d, v in got.items():
        assert abs(v - d) < 0.35 * d + 0.6, (d, v)
    ok("twonn: recovers subspace dimension", {d: round(v, 2) for d, v in got.items()})


def t_two_sample():
    k1, k2 = jax.random.split(K)
    A = np.asarray(S.sample_directions(k1, 12, 64))
    a = np.asarray(jax.random.normal(k1, (800, 12)))
    b = np.asarray(jax.random.normal(k2, (800, 12)))
    c = b + 1.5
    same = M.projected_two_sample(a, b, A)
    diff = M.projected_two_sample(a, c, A)
    assert same["ks_max"] < 0.15 and diff["ks_mean"] > 0.4, (same, diff)
    ok("two-sample: null small, shifted large",
       f"same ks_max={same['ks_max']:.3f}  shifted ks_mean={diff['ks_mean']:.3f}")


# -------------------------------------------------------------------- data
def t_views():
    cfg = DataCfg(img_size=32, n_views=4, dataset="galaxy10")
    imgs = jnp.abs(jax.random.normal(K, (8, 69, 69, 3))) * 0.3
    imgs = jnp.clip(imgs, 0, 1)
    Zin, boxes = D.make_views(K, imgs, cfg)
    assert Zin.shape == (4, 8, 32 * 32 * 3), Zin.shape
    assert boxes.shape == (4, 8, 4)
    assert float(Zin.min()) >= 0.0 and float(Zin.max()) <= 1.0
    s = boxes[..., 2]
    assert float(s.min()) >= np.sqrt(cfg.crop_area[0]) - 1e-5
    assert float(s.max()) <= np.sqrt(cfg.crop_area[1]) + 1e-5
    # different views of the same image must actually differ
    d = float(jnp.abs(Zin[0] - Zin[1]).mean())
    assert d > 1e-3, d
    ok("make_views: shapes, range, boxes, views differ",
       f"Zin={tuple(Zin.shape)} boxes={tuple(boxes.shape)} mean|v0-v1|={d:.4f}")


def t_eval_batch():
    cfg = DataCfg(img_size=32)
    imgs = jnp.clip(jnp.abs(jax.random.normal(K, (5, 69, 69, 3))) * .3, 0, 1)
    a = D.eval_batch(imgs, cfg); b = D.eval_batch(imgs, cfg)
    assert a.shape == (5, 32 * 32 * 3)
    assert jnp.allclose(a, b)                      # deterministic
    ok("eval_batch: deterministic, right shape", tuple(a.shape))


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items()) if k.startswith("t_")]
    fails = 0
    print(f"\nrunning {len(tests)} helper tests\n" + "-" * 78)
    for t in tests:
        try:
            t()
        except Exception:
            fails += 1
            print(f"  FAIL  {t.__name__}")
            traceback.print_exc()
    print("-" * 78)
    print(f"{len(tests)-fails}/{len(tests)} passed")
    raise SystemExit(1 if fails else 0)
