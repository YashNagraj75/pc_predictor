# Regression tests for PC-ALM (pcalm.py): energy equivalence to jpc at zero duals, dual growth with alpha,
# alpha=0 matching plain PC's weight gradient, and batch-size invariance of the per-sample relaxation.
import dataclasses
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp

import data as D
import encoder as E
import pcalm
import sigreg as S
from config import DataCfg, EncoderCfg, OptCfg, PCCfg, RunCfg

FAILED = []


def check(name, ok, detail=""):
    """Prints a PASS/FAIL line for one gate and records its name in FAILED if it failed."""
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def small_case(depth=4, width=16, T=3, alpha=0.0, use_sigreg=True, B=8, seed=0,
               projector="bn"):
    """Builds a tiny encoder and a real-data batch, small enough to run on CPU in seconds."""
    enc = EncoderCfg(depth=depth, width=width, embed_dim=width)
    cfg = RunCfg(arm="pc", use_sigreg=use_sigreg, enc=enc, projector=projector,
                 pc=PCCfg(T=T, alpha=alpha), opt=OptCfg(seed=seed))
    params = E.init(jax.random.PRNGKey(seed), cfg.enc, cfg.data.input_dim)
    Xtr = D.load_dataset(cfg.data)[0]
    kv, ka = jax.random.split(jax.random.PRNGKey(1))
    Zin, _ = D.make_views(kv, jnp.asarray(Xtr[:B]), cfg.data)
    A = S.sample_directions(ka, cfg.enc.embed_dim, cfg.sig.num_slices)
    return params, Zin, A, cfg


def test_al_energy_matches_jpc_at_zero_duals():
    """Checks that al_energy with duals=None or all-zero duals equals jpc's plain PC energy at rho=1."""
    params, Zin, A, cfg = small_case()
    z, hs = E.forward(params, Zin[0], cfg.enc)
    acts = [*hs, z]
    # perturb off the feedforward pass, where every residual is exactly 0 and
    # the comparison would be trivially 0 == 0
    key = jax.random.PRNGKey(7)
    acts = [a + 0.3 * jax.random.normal(k, a.shape)
            for a, k in zip(acts, jax.random.split(key, len(acts)))]

    ours_none = float(pcalm.al_energy(params, acts, Zin[0], None, 1.0, cfg.enc))
    ours_zero = float(pcalm.al_energy(params, acts, Zin[0],
                                      [jnp.zeros_like(a) for a in acts],
                                      1.0, cfg.enc))
    jpcs = float(E.energy(params, acts, Zin[0], cfg.enc))
    rel = abs(ours_none - jpcs) / max(abs(jpcs), 1e-12)
    check("al_energy(duals=None) == jpc energy", rel < 1e-5,
          f"ours={ours_none:.8f} jpc={jpcs:.8f} rel={rel:.2e}")
    check("al_energy(duals=0) == al_energy(duals=None)",
          abs(ours_zero - ours_none) < 1e-6 * max(abs(ours_none), 1.0))


def test_residuals_vanish_on_the_forward_pass():
    """Checks that every layer residual is ~0 at the feedforward (zero-energy) activations."""
    params, Zin, A, cfg = small_case()
    z, hs = E.forward(params, Zin[0], cfg.enc)
    r = pcalm.residuals(params, [*hs, z], Zin[0], cfg.enc)
    worst = max(float(jnp.max(jnp.abs(ri))) for ri in r)
    check("residuals vanish at the feedforward init", worst < 1e-4,
          f"max|r| = {worst:.2e}")


def test_alpha_zero_reproduces_pc():
    """Checks that pcalm.grads at alpha=0 reproduces train.pc_grads' weight gradient."""
    import train as T
    params, Zin, A, cfg = small_case(alpha=0.0)
    g_alm, _ = pcalm.grads(params, Zin, A, cfg, jax.random.PRNGKey(0))
    g_pc, _ = T.pc_grads(params, Zin, A, cfg, jax.random.PRNGKey(0))
    la = [x for x in jax.tree_util.tree_leaves(g_alm) if jnp.issubdtype(x.dtype, jnp.floating)]
    lp = [x for x in jax.tree_util.tree_leaves(g_pc) if jnp.issubdtype(x.dtype, jnp.floating)]
    num = jnp.sqrt(sum(jnp.sum((a - b) ** 2) for a, b in zip(la, lp)))
    den = jnp.sqrt(sum(jnp.sum(b ** 2) for b in lp)) + 1e-12
    dot = sum(jnp.sum(a * b) for a, b in zip(la, lp))
    cos = float(dot / (jnp.sqrt(sum(jnp.sum(a ** 2) for a in la))
                       * jnp.sqrt(sum(jnp.sum(b ** 2) for b in lp)) + 1e-12))
    check("alpha=0 matches plain PC weight gradient", float(num / den) < 1e-4,
          f"rel diff={float(num/den):.2e}  cos={cos:.8f}")


def test_duals_stay_zero_at_alpha_zero():
    """Checks that the multipliers stay exactly zero through relaxation when alpha=0."""
    params, Zin, A, cfg = small_case(alpha=0.0)
    _, duals, _ = pcalm.relax(params, Zin, A, cfg, jax.random.PRNGKey(0))
    worst = max(float(jnp.max(jnp.abs(l))) for dv in duals for l in dv)
    check("duals stay 0 at alpha=0", worst == 0.0, f"max|lam| = {worst:.2e}")


def test_duals_grow_at_positive_alpha():
    """Checks that the multipliers become nonzero through relaxation when alpha=1."""
    params, Zin, A, cfg = small_case(alpha=1.0)
    _, duals, _ = pcalm.relax(params, Zin, A, cfg, jax.random.PRNGKey(0))
    n = float(jnp.sqrt(sum(jnp.sum(l ** 2) for dv in duals for l in dv)))
    check("duals become nonzero at alpha=1", n > 0.0, f"||lam|| = {n:.6f}")


def test_dual_reaches_deeper_than_pc():
    """Reports (not asserts) how much further credit reaches at alpha=1 vs alpha=0, layer by layer."""
    params, Zin, A, cfg = small_case(depth=8, T=16, alpha=0.0)
    out = {}
    for alpha in (0.0, 1.0):
        c = dataclasses.replace(cfg, pc=dataclasses.replace(cfg.pc, alpha=alpha))
        acts, duals, _ = pcalm.relax(params, Zin, A, c, jax.random.PRNGKey(0))
        z, hs = E.forward(params, Zin[0], c.enc)
        ffl = [*hs, z]
        per = [float(jnp.linalg.norm(acts[0][i] - ffl[i])
                     / (jnp.linalg.norm(ffl[i]) + 1e-12)) for i in range(len(ffl))]
        out[alpha] = per
    print("      alpha=0  per-layer activity move:", " ".join(f"{v:.1e}" for v in out[0.0]))
    print("      alpha=1  per-layer activity move:", " ".join(f"{v:.1e}" for v in out[1.0]))
    n0 = sum(1 for v in out[0.0] if v > 1e-3)
    n1 = sum(1 for v in out[1.0] if v > 1e-3)
    print(f"      layers moving >1e-3:  alpha=0 -> {n0},  alpha=1 -> {n1}")


def test_batch_invariance_without_sigreg():
    """Checks that relaxation is batch-size invariant when SIGReg and the bn projector are both off
    (otherwise the objective couples samples and this would measure augmentation, not invariance)."""
    sizes = (2, 8, 32)
    # Views are built ONCE at the largest batch and sliced, so sample 0 is byte-identical across batches
    # (D.make_views splits its key per sample, so a separate call per B would change sample 0's crop).
    params, Zin_full, A, cfg = small_case(use_sigreg=False, alpha=1.0,
                                          B=max(sizes), projector="none")
    moved = []
    for B in sizes:
        Zin = Zin_full[:, :B]
        acts, _, _ = pcalm.relax(params, Zin, A, cfg, jax.random.PRNGKey(0))
        z, _ = E.forward(params, Zin[0], cfg.enc)
        moved.append(float(jnp.linalg.norm(acts[0][-1][0] - z[0])
                           / (jnp.linalg.norm(z[0]) + 1e-12)))
    spread = max(moved) / max(min(moved), 1e-30)
    check("relaxation is batch-size invariant (per-sample objective)", spread < 1.001,
          "moves = " + ", ".join(f"B={b}:{m:.6f}" for b, m in zip(sizes, moved))
          + f"  spread={spread:.3f}x")


if __name__ == "__main__":
    print("PC-ALM gates")
    for fn in (test_al_energy_matches_jpc_at_zero_duals,
               test_residuals_vanish_on_the_forward_pass,
               test_alpha_zero_reproduces_pc,
               test_duals_stay_zero_at_alpha_zero,
               test_duals_grow_at_positive_alpha,
               test_batch_invariance_without_sigreg,
               test_dual_reaches_deeper_than_pc):
        try:
            fn()
        except Exception as exc:
            check(fn.__name__, False, f"raised {type(exc).__name__}: {exc}")
    print("\n" + ("ALL PASS" if not FAILED else f"FAILED: {', '.join(FAILED)}"))
    sys.exit(1 if FAILED else 0)
