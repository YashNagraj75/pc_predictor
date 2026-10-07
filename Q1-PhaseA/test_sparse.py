"""Gates on the sparse arms (cases B and C).  Must pass before any GPU job.

    ./.venv/bin/python test_sparse.py

Spec: vault PCWM/Methods/'Method - Sparse PC-ALM arms spec.md'.

Order matters.  The first block pins that the edits did not change the
ORIGINAL code path at all (golden outputs frozen before the edits by
_golden_pcalm.py).  The second pins the architecture change against jpc.  The
rest pin the maths of B and C, the batch-size bookkeeping and the controls.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dataclasses
import math
import tempfile

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import data as D
import encoder as E
import pcalm
import sigreg as S
import sparse as SP
import train as TR
from config import DataCfg, EncoderCfg, OptCfg, PCCfg, RunCfg, SparseCfg

jax.config.update("jax_enable_x64", False)
FAILED = []
HERE = os.path.dirname(os.path.abspath(__file__))


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


_X = {}


def _data():
    if "X" not in _X:
        _X["X"] = D.load_dataset(DataCfg())[0][:64]
    return _X["X"]


def small(case="none", s=2, alpha=1.0, T=3, B=8, use_sigreg=True, projector="bn",
          beta=0.3, gamma=0.05, beta_g=1.0, eta=0.0625, below_hard=True, seed=0):
    """Tiny encoder (depth 4 -> L = 6 layers, width 16) on real images."""
    enc = EncoderCfg(depth=4, width=16, embed_dim=16)
    sp = SparseCfg(case=case, layer=s, beta=beta, gamma=gamma, beta_g=beta_g,
                   T_test=8, below_hard=below_hard)
    cfg = RunCfg(arm="pc", use_sigreg=use_sigreg, projector=projector, enc=enc,
                 pc=PCCfg(T=T, alpha=alpha, eta_h=eta), opt=OptCfg(seed=seed),
                 sparse=sp)
    params = SP.init_params(jax.random.PRNGKey(seed), cfg, cfg.data.input_dim)
    kv, ka = jax.random.split(jax.random.PRNGKey(1))
    Zin, _ = D.make_views(kv, jnp.asarray(_data()[:B]), cfg.data)
    A = S.sample_directions(ka, cfg.enc.embed_dim, cfg.sig.num_slices)
    return params, Zin, A, cfg


def leaves(t):
    return [np.asarray(x) for x in jax.tree_util.tree_leaves(eqx.filter(t, eqx.is_inexact_array))]


def maxrel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-30))


# ------------------------------------------------ 1. original path unchanged

def test_golden_case_none():
    """case none reproduces the pre-edit pcalm.relax / pcalm.grads exactly."""
    import _golden_pcalm as G
    for alpha in (0.0, 1.0):
        path = os.path.join(HERE, f"golden_pcalm_alpha{int(alpha)}.npz")
        ref = np.load(path)
        got = G.snapshot(alpha)
        keys = sorted(ref.files)
        bad = [k for k in keys if not np.array_equal(ref[k], got[k])]
        worst = max((maxrel(got[k], ref[k]) for k in keys), default=0.0)
        check(f"case none == golden (alpha={alpha:g}): {len(keys)} arrays bit-identical",
              not bad and set(keys) == set(got), f"mismatched={bad[:4]} worst_rel={worst:.2e}")


def test_init_none_is_E_init():
    cfg = RunCfg()
    a = SP.init_params(jax.random.PRNGKey(3), cfg, cfg.data.input_dim)
    b = E.init(jax.random.PRNGKey(3), cfg.enc, cfg.data.input_dim)
    same = (jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b)
            and all(np.array_equal(x, y) for x, y in zip(leaves(a), leaves(b))))
    check("init_params(case none) == E.init", same)


def test_old_config_json_loads_as_case_none():
    from run_probes import cfg_from_dict
    import json
    d = json.loads(RunCfg().to_json())
    d.pop("sparse")
    check("config.json without a 'sparse' block -> case none",
          cfg_from_dict(d).sparse.case == "none")


# --------------------------------------------- 2. architecture change vs jpc

def test_architecture_change():
    params, Zin, A, cfg = small(case="B")
    base = E.init(jax.random.PRNGKey(0), cfg.enc, cfg.data.input_dim)
    s, L = cfg.sparse.layer, len(params[0])
    ok_skip = params[1][s] is None and all(
        (params[1][l] is None) == (base[1][l] is None) for l in range(L) if l != s)
    w, w0 = params[0][s].layers[1].weight, base[0][s].layers[1].weight
    ok_w = bool(jnp.allclose(w, w0 * math.sqrt(L)))
    others = [l for l in range(L) if l != s]
    ok_rest = all(np.array_equal(params[0][l].layers[1].weight, base[0][l].layers[1].weight)
                  for l in others)
    check("B/C: skip[s] removed, W_s *= sqrt(L), every other layer untouched",
          ok_skip and ok_w and ok_rest, f"skip={ok_skip} Ws={ok_w} rest={ok_rest}")
    # z_s scale is now comparable to the stream instead of ~sqrt(L) smaller
    acts, ex = SP.deploy(params, Zin[0], dataclasses.replace(
        cfg, sparse=dataclasses.replace(cfg.sparse, beta=0.0)))
    r = float(jnp.linalg.norm(acts[s]) / jnp.linalg.norm(acts[s - 1]))
    check("variance match: ||z_s|| / ||z_{s-1}|| is O(1)", 0.2 < r < 5.0, f"ratio={r:.3f}")


def test_al_energy_matches_jpc_with_missing_skip():
    """Our residuals must equal jpc's energy on the modified (skip[s]=None) net."""
    params, Zin, A, cfg = small(case="B")
    x = Zin[0]
    z, hs = E.forward(params, x, cfg.enc)
    key = jax.random.PRNGKey(5)
    acts = [a + 0.1 * jax.random.normal(jax.random.fold_in(key, i), a.shape)
            for i, a in enumerate([*hs, z])]
    ours = float(pcalm.al_energy(params, acts, x, None, 1.0, cfg.enc))
    jp = float(E.energy(params, acts, x, cfg.enc))
    check("al_energy == jpc.pc_energy_fn with skip[s] = None", abs(ours - jp) <= 1e-5 * abs(jp),
          f"ours={ours:.7f} jpc={jp:.7f}")


def test_deploy_reduces_to_jpc_forward():
    for case, kw in (("B", dict(beta=0.0)), ("C", dict(beta_g=1.0))):
        params, Zin, A, cfg = small(case=case, **kw)
        z_j, hs_j = E.forward(params, Zin[0], cfg.enc)
        if case == "B":
            acts, _ = SP.deploy(params, Zin[0], cfg)
        else:
            acts, _ = SP.deploy(params, Zin[0], cfg, beta_g=0.0)   # the dial at 0
        err = max(maxrel(a, b) for a, b in zip(acts, [*hs_j, z_j]))
        label = "beta=0" if case == "B" else "beta_g=0 dial"
        check(f"{case}: deploy({label}) == jpc forward of the modified net", err < 1e-6,
              f"max rel err={err:.2e}")


# ---------------------------------------------------------- 3. case B maths

def test_B_deploy_is_threshold():
    params, Zin, A, cfg = small(case="B", beta=0.3)
    acts, ex = SP.deploy(params, Zin[0], cfg)
    s = cfg.sparse.layer
    want = SP.soft_threshold(ex["f_s"], cfg.sparse.beta / cfg.pc.rho)
    frac = float(jnp.mean(acts[s] != 0))
    check("B: deployed z_s == S_{beta/rho}(f_s)", bool(jnp.array_equal(acts[s], want)),
          f"active frac={frac:.3f}")


def test_B_prox_iteration_converges_to_closed_form():
    """Test-time fixed point with g_s = 0: iterating the PC-B update on z_s alone
    (gradient on rho/2||z - f||^2, then the prox) must reach S_{beta/rho}(f)."""
    params, Zin, A, cfg = small(case="B", beta=0.3)
    _, ex = SP.deploy(params, Zin[0], cfg)
    f = ex["f_s"]
    z, eta, rho, beta = f, 0.25, cfg.pc.rho, cfg.sparse.beta
    for _ in range(400):
        z = SP.soft_threshold(z - eta * rho * (z - f), eta * beta)
    err = maxrel(z, SP.soft_threshold(f, beta / rho))
    check("B: prox-gradient on z_s converges to S_{beta/rho}(f_s)", err < 1e-5,
          f"rel err={err:.2e}")


def test_B_beta0_equals_plain_relax():
    """beta = 0, alpha = 0: case B is exactly the original PC relaxation on the
    modified network (the prox is the identity and no multipliers exist)."""
    params, Zin, A, cfg = small(case="B", beta=0.0, alpha=0.0)
    acts_b, _, hb = pcalm.relax(params, Zin, A, cfg, jax.random.PRNGKey(0))
    cfg0 = dataclasses.replace(cfg, sparse=SparseCfg())     # same params, old path
    acts_0, _, h0 = pcalm.relax(params, Zin, A, cfg0, jax.random.PRNGKey(0))
    err = max(maxrel(a, b) for av, bv in zip(acts_b, acts_0) for a, b in zip(av, bv))
    check("B(beta=0, alpha=0) relax == original relax on the same net", err < 1e-5,
          f"max rel err={err:.2e}")


def test_layer_alphas():
    _, _, _, cfg = small(case="B", alpha=1.0)
    a = SP.layer_alphas(cfg, 6)
    _, _, _, cfg2 = small(case="B", alpha=1.0, below_hard=False)
    b = SP.layer_alphas(cfg2, 6)
    _, _, _, cfg3 = small(case="none", alpha=1.0)
    c = SP.layer_alphas(cfg3, 6)
    check("alpha_s = 0, others alpha; below_soft zeroes l < s; none unchanged",
          a == [1, 1, 0, 1, 1, 1] and b == [0, 0, 0, 1, 1, 1] and c == [1.0] * 6,
          f"hard={a} soft={b} none={c}")


def test_B_dual_at_s_stays_zero():
    params, Zin, A, cfg = small(case="B", alpha=1.0, T=5)
    _, duals, _ = pcalm.relax(params, Zin, A, cfg, jax.random.PRNGKey(0))
    s = cfg.sparse.layer
    zs = max(float(jnp.max(jnp.abs(duals[v][s]))) for v in range(len(duals)))
    other = max(float(jnp.max(jnp.abs(duals[v][s + 1]))) for v in range(len(duals)))
    check("B: lambda_s stays exactly 0 while lambda_{s+1} grows", zs == 0.0 and other > 0,
          f"|lam_s|max={zs:.1e} |lam_s+1|max={other:.2e}")


# ---------------------------------------------------------- 4. case C maths

def test_ista_vs_fista_and_kkt():
    params, Zin, A, cfg = small(case="C", gamma=0.05)
    _, ex = SP.deploy(params, Zin[0], cfg)
    f, Dm, mu, g = ex["f_s"], SP.get_D(params), SP.lasso_mu(cfg), cfg.sparse.gamma
    eta = cfg.sparse.step_frac / (mu * SP.spectral_sq(Dm))
    energies = [float(jnp.mean(SP.lasso_energy(f, Dm, SP.ista(f, Dm, g, mu, t, eta), g, mu)))
                for t in (0, 1, 2, 5, 10, 50, 200)]
    mono = all(b <= a + 1e-6 for a, b in zip(energies, energies[1:]))
    a_ref = SP.fista(f, Dm, g, mu, 3000)
    a_ist = SP.ista(f, Dm, g, mu, 5000, eta)
    gap = float(jnp.mean(SP.lasso_energy(f, Dm, a_ist, g, mu) - SP.lasso_energy(f, Dm, a_ref, g, mu)))
    kkt = float(jnp.max(SP.kkt_violation(f, Dm, a_ref, g, mu)))
    check("C: ISTA energy non-increasing in t", mono,
          " ".join(f"{e:.4f}" for e in energies))
    check("C: ISTA(5000) matches FISTA(3000); FISTA satisfies KKT",
          abs(gap) < 1e-4 and kkt < 1e-2, f"gap={gap:.2e} kkt/gamma={kkt:.2e}")


def test_ista_t0_is_one_step_from_zero():
    params, Zin, A, cfg = small(case="C")
    _, ex = SP.deploy(params, Zin[0], cfg)
    f, Dm, mu, g = ex["f_s"], SP.get_D(params), SP.lasso_mu(cfg), cfg.sparse.gamma
    eta = 0.1
    a0 = SP.ista(f, Dm, g, mu, 0, eta)
    a1 = SP.ista(f, Dm, g, mu, 1, eta, a0=jnp.zeros_like(a0))
    check("C: a^(0) == one ISTA step from a = 0", bool(jnp.allclose(a0, a1, atol=1e-7)))


def test_blend_is_the_z_minimiser():
    params, Zin, A, cfg = small(case="C")
    acts, ex = SP.deploy(params, Zin[0], cfg)
    s, Dm, rho, bg = cfg.sparse.layer, SP.get_D(params), cfg.pc.rho, cfg.sparse.beta_g
    f, a = ex["f_s"], ex["a"]
    Ez = lambda z: 0.5 * rho * jnp.sum((z - f) ** 2) + 0.5 * bg * jnp.sum((z - a @ Dm.T) ** 2)
    gz = jax.grad(Ez)(acts[s])
    check("C: deployed z_s is the stationary point of the two quadratics",
          float(jnp.max(jnp.abs(gz))) < 1e-5, f"max|grad|={float(jnp.max(jnp.abs(gz))):.2e}")
    # and substituting it back gives the LASSO with mu = rho beta_g / (rho + beta_g)
    lhs = float(Ez(acts[s]))
    rhs = float(0.5 * SP.lasso_mu(cfg) * jnp.sum((f - a @ Dm.T) ** 2))
    check("C: min_z of the quadratics == mu/2 ||f - D a||^2", abs(lhs - rhs) < 1e-4 * abs(rhs),
          f"{lhs:.6f} vs {rhs:.6f}")


def test_C_relax_descends_at_alpha0():
    """alpha = 0 removes dual ascent, so with the coupled output drive the sum
    F + L must not increase over the relaxation (small step)."""
    params, Zin, A, cfg = small(case="C", alpha=0.0, T=12, eta=0.02, use_sigreg=False,
                                projector="none")
    _, _, h = pcalm.relax(params, Zin, A, cfg, jax.random.PRNGKey(0))
    tot = np.asarray(h["F"]) + np.asarray(h["L"])
    mono = bool(np.all(np.diff(tot) <= 1e-6 * np.abs(tot[:-1]).max()))
    check("C: F + L non-increasing over relaxation at alpha = 0", mono,
          " ".join(f"{v:.5f}" for v in tot[:6]) + " ...")


def test_C_D_gradient_matches_analytic():
    params, Zin, A, cfg = small(case="C", alpha=1.0, T=3)
    g, aux = pcalm.grads(params, Zin, A, cfg, jax.random.PRNGKey(0))
    acts, _, hist = pcalm.relax(params, Zin, A, cfg, jax.random.PRNGKey(0))
    s, Dm, bg = cfg.sparse.layer, SP.get_D(params), cfg.sparse.beta_g
    want = sum(bg * ((hist["a"][v] @ Dm.T - acts[v][s]).T @ hist["a"][v]) / Zin.shape[1]
               for v in range(Zin.shape[0]))
    err = maxrel(g[2], want)
    check("C: dF/dD == beta_g (D a - z_s) a^T / B  (summed over views)", err < 1e-4,
          f"rel err={err:.2e}  active_frac={float(aux['active_frac']):.3f}")


def test_project_and_serialise():
    params, Zin, A, cfg = small(case="C")
    p2 = SP.project((params[0], params[1], params[2] * 3.0), cfg)
    norms = jnp.linalg.norm(p2[2], axis=0)
    check("C: project() restores unit-norm columns", bool(jnp.allclose(norms, 1.0, atol=1e-6)))
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "p.eqx")
        eqx.tree_serialise_leaves(f, params)
        like = SP.init_params(jax.random.PRNGKey(99), cfg, cfg.data.input_dim)
        back = eqx.tree_deserialise_leaves(f, like)
    same = all(np.array_equal(a, b) for a, b in zip(leaves(params), leaves(back)))
    check("C: params (model, skip, D) serialise round-trip", same)


# ---------------------------------------------- 5. batch-size bookkeeping

def test_batch_invariance():
    """Per-sample objective (SIGReg and the BN projector off): sample 0's settled
    z_s and code must not depend on B.  This is the test that catches an
    eta_h * B leaking into a prox threshold or a code step."""
    sizes = (2, 8, 32)
    for case in ("B", "C"):
        params, Zin_full, A, cfg = small(case=case, alpha=1.0, T=6, B=max(sizes),
                                         use_sigreg=False, projector="none")
        s, ref, worst = cfg.sparse.layer, None, 0.0
        for B in sizes:
            acts, _, h = pcalm.relax(params, Zin_full[:, :B], A, cfg, jax.random.PRNGKey(0))
            v = np.asarray(acts[0][s][0]) if case == "B" else np.asarray(h["a"][0][0])
            if ref is None:
                ref = v
            else:
                worst = max(worst, maxrel(v, ref))
        check(f"{case}: settled sparse variable of sample 0 is batch-size invariant",
              worst < 1e-4, f"max rel diff over B={sizes}: {worst:.2e}")


# ------------------------------------------------------------ 6. controls

def test_bp_controls():
    for case in ("B", "C"):
        params, Zin, A, cfg = small(case=case, alpha=1.0)
        cfg = dataclasses.replace(cfg, arm="bp")
        g, aux = TR.bp_grads(params, Zin, A, cfg)
        lv = leaves(g)
        finite = all(np.all(np.isfinite(x)) for x in lv)
        s = cfg.sparse.layer
        gWs = float(jnp.linalg.norm(g[0][s].layers[1].weight))
        extra = ""
        ok = finite and gWs > 0
        ok = ok and "head_energy" in aux
        if case == "C":
            gD = float(jnp.linalg.norm(g[2]))
            ok = ok and gD > 0
            extra = f" |dD|={gD:.2e} head={float(aux['head_energy']):.4f}"
        check(f"{'C1-B' if case == 'B' else 'C2'}: BP through the deployed net is finite and "
              f"reaches W_s", ok, f"|dW_s|={gWs:.2e}{extra}")


def test_pc_grads_finite_and_logged():
    for case in ("B", "C"):
        params, Zin, A, cfg = small(case=case, alpha=1.0, T=4)
        g, aux = TR.pc_grads(params, Zin, A, cfg, jax.random.PRNGKey(0))
        finite = all(np.all(np.isfinite(x)) for x in leaves(g))
        check(f"PC-{case}: weight gradients finite; F_first/active_frac logged",
              finite and "F_first" in aux and "active_frac" in aux and "act_growth" in aux,
              f"F0={float(aux['F_first']):.4f} F_T={float(aux['F_final']):.4f} "
              f"active={float(aux['active_frac']):.3f} act_growth={float(aux['act_growth']):.3f}")


def test_pc_agrees_with_its_bp_control_below_s():
    """At a long relaxation the PC weight gradient on the layers BELOW s must
    point the same way as backprop through the control's objective (task loss
    + deployed sparse-layer energy, summed over views).  Measured 2026-10-06:
    0.83-0.85 (B) and 0.78-0.84 (C) with the corrected objective, 0.43-0.45 (B)
    and 0.37-0.51 (C) with the first version (head energy averaged over views /
    no Huber term for C1-B) -- so this test catches that bug class.
    Layers at and ABOVE s are not asserted: PC computes their gradient at the
    relaxed z_s (which carries task credit, g_s/rho) while the control's runs at
    the deployed z_s = S(f_s), a structural difference of removing the
    multiplier at s, not a bug."""
    def cos(a, b):
        a, b = np.ravel(a), np.ravel(b)
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    for case in ("B", "C"):
        params, Zin, A, cfg = small(case=case, alpha=1.0, T=100, eta=0.0625, B=8)
        gp, _ = pcalm.grads(params, Zin, A, cfg, jax.random.PRNGKey(0))
        gb, _ = TR.bp_grads(params, Zin, A, dataclasses.replace(cfg, arm="bp"))
        s = cfg.sparse.layer
        cs = [cos(gp[0][l].layers[1].weight, gb[0][l].layers[1].weight) for l in range(s)]
        check(f"PC-{case} vs its BP control: per-layer cosine > 0.7 on layers below s (T=100)",
              min(cs) > 0.7, " ".join(f"L{l}={c:.3f}" for l, c in enumerate(cs)))


if __name__ == "__main__":
    print("Sparse-arm gates")
    for fn in (test_golden_case_none, test_init_none_is_E_init,
               test_old_config_json_loads_as_case_none,
               test_architecture_change, test_al_energy_matches_jpc_with_missing_skip,
               test_deploy_reduces_to_jpc_forward,
               test_B_deploy_is_threshold, test_B_prox_iteration_converges_to_closed_form,
               test_B_beta0_equals_plain_relax, test_layer_alphas, test_B_dual_at_s_stays_zero,
               test_ista_vs_fista_and_kkt, test_ista_t0_is_one_step_from_zero,
               test_blend_is_the_z_minimiser, test_C_relax_descends_at_alpha0,
               test_C_D_gradient_matches_analytic, test_project_and_serialise,
               test_batch_invariance, test_bp_controls, test_pc_grads_finite_and_logged,
               test_pc_agrees_with_its_bp_control_below_s):
        try:
            fn()
        except Exception as exc:
            import traceback
            traceback.print_exc(limit=3)
            check(fn.__name__, False, f"raised {type(exc).__name__}: {exc}")
    print("\n" + ("ALL PASS" if not FAILED else f"FAILED: {', '.join(FAILED)}"))
    sys.exit(1 if FAILED else 0)
