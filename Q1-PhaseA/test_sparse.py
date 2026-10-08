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
          beta=0.3, gamma=0.05, beta_g=1.0, eta=0.0625, below_hard=True, seed=0,
          normalise_fs=True, s_step=0.5):
    """Tiny encoder (depth 4 -> L = 6 layers, width 16) on real images."""
    enc = EncoderCfg(depth=4, width=16, embed_dim=16)
    sp = SparseCfg(case=case, layer=s, beta=beta, gamma=gamma, beta_g=beta_g,
                   T_test=8, below_hard=below_hard, normalise_fs=normalise_fs,
                   s_step=s_step)
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
    """case none reproduces the PRE-EDIT pcalm.relax / pcalm.grads bit for bit.

    The reference is regenerated on THIS machine by running _golden_pcalm.py
    against the pre-sparsity source in golden_ref/ (pcalm/encoder/config as of
    the commit before the sparse work).  A reference frozen on one machine
    cannot be bit-identical on another: the GPU host's CPU differs from the
    laptop's in the 7th significant digit (335.496826 vs 335.496857), which
    failed the first GPU job's gates.  Comparing two runs on the same machine
    keeps the test strict."""
    import subprocess, tempfile
    import _golden_pcalm as G
    with tempfile.TemporaryDirectory() as d:
        env = {**os.environ, "GOLDEN_REF": os.path.join(HERE, "golden_ref"),
               "GOLDEN_OUT": d, "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS", "cpu")}
        r = subprocess.run([sys.executable, "-W", "ignore", os.path.join(HERE, "_golden_pcalm.py")],
                           env=env, capture_output=True, text=True, cwd=HERE)
        if r.returncode != 0:
            check("golden reference generation (pre-edit source)", False, r.stderr[-300:])
            return
        for alpha in (0.0, 1.0):
            ref = np.load(os.path.join(d, f"golden_pcalm_alpha{int(alpha)}.npz"))
            got = G.snapshot(alpha)
            keys = sorted(ref.files)
            bad = [k for k in keys if not np.array_equal(ref[k], got[k])]
            worst = max((maxrel(got[k], ref[k]) for k in keys), default=0.0)
            check(f"case none == pre-edit code (alpha={alpha:g}): {len(keys)} arrays bit-identical",
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
        params, Zin, A, cfg = small(case=case, normalise_fs=False, **kw)   # jpc has no norm
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
    # normalisation off and the layer-s step equal to eta_h: then case B is
    # literally the old relaxation on the modified net
    params, Zin, A, cfg = small(case="B", beta=0.0, alpha=0.0, normalise_fs=False,
                                s_step=0.0625)
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


def test_control_objective_composition():
    """The BP controls differentiate task loss + the deployed sparse-layer energy
    SUMMED over views (not averaged -- the bug all three reviewers found)."""
    for case in ("B", "C"):
        params, Zin, A, cfg = small(case=case, alpha=1.0)
        cb = dataclasses.replace(cfg, arm="bp")
        tot, aux = TR.bp_objective(params, Zin, A, cb)
        task, _ = TR.loss_from_views(params, Zin, A, cb)
        head = sum(float(SP.deployed_head_energy(params, Zin[v], cb)) for v in range(Zin.shape[0]))
        ok = abs(float(tot) - float(task) - head) <= 1e-5 * max(1.0, abs(head)) and head > 0
        if case == "B":       # and the deployed energy is the Huber envelope of f~_s
            _, ex = SP.deploy(params, Zin[0], cb)
            hb = float(jnp.mean(SP.huber_energy(ex["f_s"], cfg.sparse.beta, cfg.pc.rho)))
            ok = ok and abs(hb - float(SP.deployed_head_energy(params, Zin[0], cb))) < 1e-6
        check(f"{'C1-B' if case == 'B' else 'C2'} objective == task + sum over views of the deployed energy",
              ok, f"total={float(tot):.5f} task={float(task):.5f} sum_v head={head:.5f}")


def test_pc_agrees_with_its_bp_control_below_s():
    """PC arm vs its BP control with the task drive REMOVED (identical views,
    SIGReg off): both gradients then come from the sparse layer's energy alone.

    GATED on gradient-NORM ratios, which separate the bug class from correct
    code robustly across seeds 0-4 (measured 2026-10-07): B, layers <= s,
    0.73-1.36; C, dictionary, 0.57-0.64.  A head energy averaged over views
    would give ~4x those.  Cosines are PRINTED, not gated: on this 4-layer,
    width-16 net a single-batch cosine is seed-sensitive (B 0.65-0.95; C with
    the task drive 0.40-0.87, and it also depends on which images are used --
    the laptop's cached split differs from the GPU host's).
    Not measured here: B with the task drive, where a converged PC-B trains at
    the task-informed support S(f_s + g_s/rho) and C1-B at S(f_s) -- case B's
    train/test gap, the hypothesis under test, not a bug."""
    def cos(a, b):
        a, b = np.ravel(a), np.ravel(b)
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    nrm = lambda x: float(jnp.linalg.norm(x))
    W = lambda g, l: g[0][l].layers[1].weight
    for seed in (0, 1):
        params, Zin, A, cfg = small(case="B", alpha=1.0, T=16, use_sigreg=False,
                                    projector="none", seed=seed)
        Zin = jnp.stack([Zin[0]] * Zin.shape[0])
        gp, _ = pcalm.grads(params, Zin, A, cfg, jax.random.PRNGKey(0))
        gb, _ = TR.bp_grads(params, Zin, A, dataclasses.replace(cfg, arm="bp"))
        s = cfg.sparse.layer
        cs = [cos(W(gp, l), W(gb, l)) for l in range(s + 1)]
        rs = [nrm(W(gp, l)) / (nrm(W(gb, l)) + 1e-30) for l in range(s + 1)]
        check(f"PC-B vs C1-B, zero task drive, seed {seed}: |g_pc|/|g_bp| in [0.5, 2] on layers <= s",
              all(0.5 <= r <= 2.0 for r in rs),
              "ratio " + " ".join(f"{r:.2f}" for r in rs) + "  (cos " + " ".join(f"{c:.2f}" for c in cs) + ")")
        params, Zin, A, cfg = small(case="C", alpha=1.0, T=16, use_sigreg=False,
                                    projector="none", seed=seed)
        Zin = jnp.stack([Zin[0]] * Zin.shape[0])
        gp, _ = pcalm.grads(params, Zin, A, cfg, jax.random.PRNGKey(0))
        gb, _ = TR.bp_grads(params, Zin, A, dataclasses.replace(cfg, arm="bp"))
        r = nrm(gp[2]) / nrm(gb[2])
        check(f"PC-C vs C2, zero task drive, seed {seed}: dictionary cos > 0.9, |g_pc|/|g_bp| in [0.4, 1.5]",
              cos(gp[2], gb[2]) > 0.9 and 0.4 <= r <= 1.5, f"cos={cos(gp[2], gb[2]):.3f} ratio={r:.2f}")


def test_fs_normalisation():
    """The sparse layer's prediction is per-sample normalised to fs_norm in BOTH
    the deployed network and the relaxation energy, consistently."""
    for case in ("B", "C"):
        params, Zin, A, cfg = small(case=case, beta=0.0 if case == "B" else 0.3)
        s, c = cfg.sparse.layer, cfg.sparse.fs_norm
        acts, ex = SP.deploy(params, Zin[0], cfg)
        nr = np.asarray(jnp.linalg.norm(ex["f_s"], axis=-1))
        preds = pcalm.layer_predictions(params, acts, Zin[0], cfg.enc, SP.norm_spec(cfg))
        same = maxrel(preds[s], ex["f_s"])
        check(f"{case}: ||f_s|| == fs_norm per sample; relaxation prediction == deployed f_s",
              np.allclose(nr, c, rtol=1e-5) and same < 1e-5,
              f"norms {nr.min():.4f}..{nr.max():.4f} (c={c}) rel diff={same:.1e}")
    params, Zin, A, cfg = small(case="B", beta=0.0)
    acts, _ = SP.deploy(params, Zin[0], cfg)
    F = float(pcalm.al_energy(params, acts, Zin[0], None, 1.0, cfg.enc, None, SP.norm_spec(cfg)))
    check("B(beta=0): energy with normalisation is 0 at the deployed forward pass", F < 1e-8,
          f"F={F:.2e}")
    # the shortcut is closed: scaling W_s or silencing its input cannot shrink f_s
    p2 = (eqx.tree_at(lambda m: m[cfg.sparse.layer].layers[1].weight, params[0],
                      params[0][cfg.sparse.layer].layers[1].weight * 0.1), *params[1:])
    _, e1 = SP.deploy(params, Zin[0], cfg)
    _, e2 = SP.deploy(p2, Zin[0], cfg)
    check("normalised f_s is invariant to rescaling W_s", maxrel(e2["f_s"], e1["f_s"]) < 1e-5)


def test_B_layer_s_step_converges():
    """Case B's own step at s, on the FULL-SIZE architecture (the small test net
    is not representative here: its layer-s curvature differs, and 0.5/rho
    oscillated there).  Measured 2026-10-07 at init, B=64: spec fixed-point
    residual at s 0.324 (shared eta_h) -> 0.036 (s_step=0.5) at T=16."""
    import diag_sparse as DS
    Xtr = _data()
    res = {}
    for ss in (0.0625, 0.5):
        cfg = RunCfg(arm="pc", pc=PCCfg(T=16, eta_h=0.0625, alpha=1.0),
                     sparse=SparseCfg(case="B", beta=0.235, s_step=ss))
        p = SP.init_params(jax.random.PRNGKey(0), cfg, cfg.data.input_dim)
        res[ss] = DS.relax_fixed_point_residual(p, cfg, Xtr, Ts=(16,), B=16)[0]["fp_B"]
    check("B (full size): layer-s step brings the fixed-point residual at s below 0.1 at T=16",
          res[0.5] < 0.1 and res[0.5] < res[0.0625],
          f"shared eta_h: {res[0.0625]:.3f}  s_step=0.5: {res[0.5]:.3f}")


def test_all_zero_view_gives_finite_gradients():
    """An all-black augmented view (f_s = 0 exactly) must not produce NaN.
    Regression for job b3c717c6: one such view at step 68 made every arm's
    gradient NaN through f / ||f||."""
    for case in ("B", "C"):
        for arm in ("pc", "bp"):
            params, Zin, A, cfg = small(case=case, alpha=1.0, T=4)
            Zin = Zin.at[1, 3].set(0.0)                  # one view of one sample black
            cfg = dataclasses.replace(cfg, arm=arm)
            g, aux = (TR.pc_grads(params, Zin, A, cfg, jax.random.PRNGKey(0)) if arm == "pc"
                      else TR.bp_grads(params, Zin, A, cfg))
            ok = all(np.all(np.isfinite(x)) for x in leaves(g))
            check(f"{case}/{arm}: gradients finite with an all-zero view in the batch", ok)


# ------------------------------------------------------- case D: side heads

def smallD(heads=(2,), ws=None, gammas=None, alpha=1.0, T=8, B=8, use_sigreg=True,
           projector="bn", seed=0, eta=0.0625):
    """Tiny encoder (L = 6) with case-D side heads on residual layers."""
    # defaults calibrated like production: gamma ~ 20 % active on the normalised
    # head input, w small (head energy well below the task loss), T_test == T.
    # An uncalibrated w (head energy 2.5x task) made the tiny relaxation unstable.
    K = len(heads)
    ws = tuple(ws if ws is not None else [0.005] * K)
    gammas = tuple(gammas if gammas is not None else [1.0] * K)
    enc = EncoderCfg(depth=4, width=16, embed_dim=16)
    sp = SparseCfg(case="D", heads=tuple(heads), head_ws=ws, gammas=gammas, beta_g=1.0, T_test=T)
    cfg = RunCfg(arm="pc", use_sigreg=use_sigreg, projector=projector, enc=enc,
                 pc=PCCfg(T=T, alpha=alpha, eta_h=eta), opt=OptCfg(seed=seed), sparse=sp)
    params = SP.init_params(jax.random.PRNGKey(seed), cfg, cfg.data.input_dim)
    kv, ka = jax.random.split(jax.random.PRNGKey(1))
    Zin, _ = D.make_views(kv, jnp.asarray(_data()[:B]), cfg.data)
    A = S.sample_directions(ka, cfg.enc.embed_dim, cfg.sig.num_slices)
    params = SP.init_mu(params, D.eval_batch(jnp.asarray(_data()[:64]), cfg.data), cfg)
    return params, Zin, A, cfg


def test_D_init_and_nesting():
    """Encoder identical to case none; D_l depends only on (seed, l), so nested
    head sets share dictionaries; every multiplier kept; no normalisation."""
    pD, _, _, cD = smallD(heads=(2, 3))
    pN, _, _, cN = small(case="none")
    same_enc = all(np.array_equal(a, b) for a, b in zip(leaves((pD[0], pD[1])), leaves(pN)))
    p1, _, _, _ = smallD(heads=(3,))
    shared = np.array_equal(np.asarray(pD[2][1]), np.asarray(p1[2][0]))
    unit = all(np.allclose(np.linalg.norm(np.asarray(Dl), axis=0), 1.0, atol=1e-5) for Dl in pD[2])
    alphas = SP.layer_alphas(cD, len(pD[0]))
    check("D: encoder == case none init; D_l shared across nested sets; unit columns; all alphas kept; no norm",
          same_enc and shared and unit and all(a == cD.pc.alpha for a in alphas) and SP.norm_spec(cD) is None,
          f"enc={same_enc} shared={shared} unit={unit} alphas={alphas}")
    bad = []
    for kw in (dict(heads=(0,)), dict(heads=(5,)), dict(heads=(2, 2)), dict(heads=(2, 3), ws=[0.1])):
        try:
            smallD(**kw); bad.append(kw)
        except AssertionError:
            pass
    check("D: invalid head sets are refused (input/embedding layer, duplicates, length mismatch)",
          not bad, f"accepted: {bad}")


def test_D_zero_weight_equals_pcalm():
    """w_l = 0: the heads are inert, so relax / grads equal plain PC-ALM."""
    pD, Zin, A, cD = smallD(heads=(2, 3), ws=[0.0, 0.0])
    pN, _, _, cN = small(case="none", alpha=1.0, T=8)
    aD, _, _ = pcalm.relax(pD, Zin, A, cD, jax.random.PRNGKey(0))
    aN, _, _ = pcalm.relax(pN, Zin, A, cN, jax.random.PRNGKey(0))
    gD, _ = pcalm.grads(pD, Zin, A, cD, jax.random.PRNGKey(0))
    gN, _ = pcalm.grads(pN, Zin, A, cN, jax.random.PRNGKey(0))
    da = max(maxrel(x, y) for x, y in zip(leaves(aD), leaves(aN)))
    dg = max(maxrel(x, y) for x, y in zip(leaves((gD[0], gD[1])), leaves(gN)))
    dD = max(float(jnp.max(jnp.abs(x))) for x in gD[2])
    check("D: w = 0 reproduces PC-ALM (activities, encoder grads) and gives zero dictionary grad",
          da < 1e-5 and dg < 1e-5 and dD == 0.0, f"acts {da:.2e} grads {dg:.2e} |dD| {dD:.1e}")


def test_D_codes_solve_lasso():
    """Deployed codes approach the FISTA optimum of each head's LASSO as t grows."""
    pD, Zin, A, cD = smallD(heads=(2, 3))
    x = Zin[0]
    out = []
    for t in (4, 64, 512):
        acts, ex = SP.deploy(pD, x, cD, t=t)
        gaps = []
        for f, Dl, a, g in zip(ex["f_heads"], SP.get_D(pD), ex["a"], cD.sparse.gammas):
            astar = SP.fista(f, Dl, g, cD.sparse.beta_g, 3000)
            e = float(jnp.mean(SP.lasso_energy(f, Dl, a, g, cD.sparse.beta_g)))
            es = float(jnp.mean(SP.lasso_energy(f, Dl, astar, g, cD.sparse.beta_g)))
            gaps.append((e - es) / (abs(es) + 1e-12))
        out.append(max(gaps))
    check("D: deployed codes converge to the LASSO optimum (rel. suboptimality falls, < 1e-3 at t=512)",
          out[0] >= out[1] >= out[2] and out[2] < 1e-3 and min(out) > -1e-4,
          " ".join(f"{v:.2e}" for v in out))


def test_D_relaxation_codes_track_layer():
    """During the PC relaxation the codes descend each head's energy at the
    RELAXED z_l, and with all multipliers on z_l stays near f_l."""
    pD, Zin, A, cD = smallD(heads=(2, 3), T=40)
    acts, duals, h = pcalm.relax(pD, Zin, A, cD, jax.random.PRNGKey(0))
    preds = pcalm.layer_predictions(pD, acts[0], Zin[0], cD.enc)
    r = [float(jnp.linalg.norm(acts[0][l] - preds[l]) / (jnp.linalg.norm(preds[l]) + 1e-12))
         for l in cD.sparse.heads]
    fin = all(np.all(np.isfinite(np.asarray(a))) for a in h["a"][0])
    check("D: relaxed head layers stay near their predictions (multiplier kept), codes finite",
          fin and max(r) < 0.1, "rel resid " + " ".join(f"{v:.3f}" for v in r))


def test_D_control_objective_and_agreement():
    """BP-D = task + head energy SUMMED over views; with the task drive removed
    PC-D's gradient matches BP-D's on the dictionaries and on the layers <= max head."""
    pD, Zin, A, cD = smallD(heads=(2, 3))
    cb = dataclasses.replace(cD, arm="bp")
    tot, aux = TR.bp_objective(pD, Zin, A, cb)
    task, _ = TR.loss_from_views(pD, Zin, A, cb)
    head = sum(float(SP.deployed_head_energy(pD, Zin[v], cb)) for v in range(Zin.shape[0]))
    ok = abs(float(tot) - float(task) - head) <= 1e-5 * max(1.0, abs(head))
    check("D: BP-D objective == task + head energy summed over views", ok,
          f"tot={float(tot):.6f} task={float(task):.6f} head={head:.6f}")

    def cos(a, b):
        a, b = np.ravel(a), np.ravel(b)
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    nrm = lambda x: float(jnp.linalg.norm(x))
    # Long relaxation (T = T_test = 200): measured 2026-10-07 encoder ratio 1.0-1.4,
    # cos 0.82-0.97, dictionary cos >= 0.99.  At the production T = 16 the head
    # credit reaching the lower layers is much smaller -- that is measured, not
    # gated, by diag_agree_heads.py in the job's pilot stage.
    for seed in (0, 1):
        pD, Zin, A, cD = smallD(heads=(2, 3), T=200, use_sigreg=False, projector="none", seed=seed)
        Zin = jnp.stack([Zin[0]] * Zin.shape[0])
        gp, _ = pcalm.grads(pD, Zin, A, cD, jax.random.PRNGKey(0))
        gb, _ = TR.bp_grads(pD, Zin, A, dataclasses.replace(cD, arm="bp"))
        rD = [nrm(a) / (nrm(b) + 1e-30) for a, b in zip(gp[2], gb[2])]
        cDs = [cos(a, b) for a, b in zip(gp[2], gb[2])]
        W = lambda g, l: g[0][l].layers[1].weight
        rW = [nrm(W(gp, l)) / (nrm(W(gb, l)) + 1e-30) for l in range(max(cD.sparse.heads) + 1)]
        cW = [cos(W(gp, l), W(gb, l)) for l in range(max(cD.sparse.heads) + 1)]
        check(f"D: zero task drive, T=200, seed {seed}: dictionary cos > 0.95; encoder cos > 0.75 and "
              f"ratio in [0.7, 1.6] on layers <= max head",
              min(cDs) > 0.95 and min(cW) > 0.75 and all(0.7 <= r <= 1.6 for r in rW) and all(0.7 <= r <= 1.4 for r in rD),
              "D cos " + " ".join(f"{c:.3f}" for c in cDs) + " ratio " + " ".join(f"{r:.2f}" for r in rD)
              + " | W cos " + " ".join(f"{c:.3f}" for c in cW) + " ratio " + " ".join(f"{r:.2f}" for r in rW))


def test_D_batch_invariance_and_zero_view():
    sizes = (2, 8, 32)
    pD, Zf, A, cD = smallD(heads=(2, 3), T=6, B=max(sizes), use_sigreg=False, projector="none")
    ref, worst = None, 0.0
    for B in sizes:
        _, _, h = pcalm.relax(pD, Zf[:, :B], A, cD, jax.random.PRNGKey(0))
        v = np.concatenate([np.asarray(a[0]) for a in h["a"][0]])
        ref = v if ref is None else ref
        worst = max(worst, maxrel(v, ref))
    check("D: sample 0's settled codes are batch-size invariant", worst < 1e-4, f"{worst:.2e}")
    pD, Zin, A, cD = smallD(heads=(2, 3))
    Zin = Zin.at[:, 0].set(0.0)
    fin = []
    for arm in ("pc", "bp"):
        c = dataclasses.replace(cD, arm=arm)
        g, _ = pcalm.grads(pD, Zin, A, c, jax.random.PRNGKey(0)) if arm == "pc" else TR.bp_grads(pD, Zin, A, c)
        fin.append(all(np.all(np.isfinite(x)) for x in leaves(g)))
    check("D: an all-zero view gives finite PC and BP gradients", all(fin), str(fin))


def test_D_train_project_serialise_and_gate():
    """A few training steps per arm: finite, D re-normalised, round-trips through
    load_run, and the pair gate reads the case-D health rows."""
    import gate_pilot as GP
    from run_probes import load_run
    tmp = tempfile.mkdtemp()
    dirs = {}
    for arm in ("pc", "bp"):
        _, _, _, cD = smallD(heads=(2, 3))
        cfg = dataclasses.replace(cD, arm=arm, out_dir=tmp, tag="tD",
                                  opt=dataclasses.replace(cD.opt, n_steps=4, eval_every=2))
        p, log = TR.train(cfg, verbose=False)
        dirs[arm] = os.path.join(tmp, cfg.name)
        unit = all(np.allclose(np.linalg.norm(np.asarray(Dl), axis=0), 1.0, atol=1e-5) for Dl in p[2])
        c2, p2 = load_run(dirs[arm])
        rt = all(np.array_equal(a, b) for a, b in zip(leaves(p), leaves(p2))) and c2.sparse.heads == (2, 3)
        ev = [r for r in log if isinstance(r.get("eval"), dict)]
        has = bool(ev) and "h2_supports" in ev[-1]["eval"]["sparse"] and "embed_spread" in ev[-1]["eval"]["sparse"]
        check(f"D ({arm}): trains, keeps unit-norm D, round-trips, logs head health", unit and rt and has,
              f"unit={unit} roundtrip={rt} health={has}")
    ok, reasons, info = GP.gate(dirs["pc"], dirs["bp"])
    readable = all("missing" not in r and "fewer than two" not in r and "no " not in r[:8] for r in reasons)
    check("D: pair gate evaluates case-D criteria without schema errors", readable, str(reasons)[:300])


def test_D_head_reads_normalised_copy():
    """Heads explain sqrt(n) f/||f||: deployed codes and the head energy are
    invariant to rescaling the head layer, so shrinking it cannot lower the
    energy (the route the smoke run showed).  BP-D's code is detached."""
    pD, Zin, A, cD = smallD(heads=(2,))
    x = Zin[0]
    acts, ex = SP.deploy(pD, x, cD)
    f = acts[2]
    mu = SP.get_mu(pD)[0]
    nrm = np.linalg.norm(np.asarray(ex["f_heads"][0]), axis=1)
    rows_ok = np.allclose(nrm, np.sqrt(f.shape[-1]), rtol=1e-4)
    D0, g0 = SP.get_D(pD)[0], cD.sparse.gammas[0]
    def e(v, m):
        h = SP.head_in(v, m)
        return float(jnp.mean(SP.lasso_energy(h, D0, SP.ista(h, D0, g0, 1.0, 16, 0.9 / SP.spectral_sq(D0)), g0, 1.0)))
    # scale about mu, and a shared offset absorbed by mu, leave the head energy unchanged
    c = 5.0 * jax.random.normal(jax.random.PRNGKey(3), mu.shape)
    inv = abs(e(f, mu) - e(mu + 0.5 * (f - mu), mu)) < 1e-4 * abs(e(f, mu)) \
        and abs(e(f, mu) - e(f + c, mu + c)) < 1e-4 * abs(e(f, mu))
    # detached: the BP-D head gradient wrt D equals the partial derivative at fixed a
    a = ex["a"][0]; ft = ex["f_heads"][0]
    g_part = jax.grad(lambda Dm: cD.sparse.head_ws[0] * jnp.mean(SP.lasso_energy(ft, Dm, a, g0, 1.0)))(D0)
    g_bp = jax.grad(lambda Dm: SP.deployed_head_energy((pD[0], pD[1], (Dm,), pD[3]), x, cD))(D0)
    det = maxrel(g_bp, g_part) < 1e-4
    # mu: no gradient, untouched by the optimiser, EMA-updated toward the batch mean
    gmu = jax.grad(lambda m: SP.deployed_head_energy((pD[0], pD[1], pD[2], (m,)), x, cD))(mu)
    opt = TR.optim(dataclasses.replace(cD, arm="bp"))
    fl = eqx.filter(pD, eqx.is_inexact_array)
    st = opt.init(fl)
    rnd = jax.tree_util.tree_map(lambda v: jnp.ones_like(v), fl)
    upd, _ = opt.update(rnd, st, fl)
    p2 = eqx.apply_updates(pD, upd)
    frozen = bool(jnp.all(p2[3][0] == mu)) and not bool(jnp.all(p2[0][0].layers[1].weight == pD[0][0].layers[1].weight))
    p3 = SP.update_mu(pD, Zin, cD)
    bm = jnp.mean(jnp.stack([SP._head_means(pD, Zin[v], cD)[0] for v in range(Zin.shape[0])]), 0)
    m = cD.sparse.mu_momentum
    ema = maxrel(p3[3][0], m * mu + (1 - m) * bm) < 1e-5
    ok_mu = float(jnp.max(jnp.abs(gmu))) == 0.0 and frozen and ema
    check("D: centred head input normalised per row; energy invariant to scale about mu and to a tracked offset; "
          "BP-D code detached; mu has no gradient, is frozen by the optimiser, EMA-updated",
          rows_ok and inv and det and ok_mu,
          f"rows={rows_ok} invariant={inv} detached={det} ({maxrel(g_bp, g_part):.1e}) mu_ok={ok_mu} "
          f"(|gmu|={float(jnp.max(jnp.abs(gmu))):.1e} frozen={frozen} ema={ema})")


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
               test_control_objective_composition, test_pc_agrees_with_its_bp_control_below_s, test_fs_normalisation,
               test_B_layer_s_step_converges, test_all_zero_view_gives_finite_gradients,
               test_D_init_and_nesting, test_D_zero_weight_equals_pcalm, test_D_codes_solve_lasso,
               test_D_relaxation_codes_track_layer, test_D_control_objective_and_agreement,
               test_D_batch_invariance_and_zero_view, test_D_train_project_serialise_and_gate,
               test_D_head_reads_normalised_copy):
        try:
            fn()
        except Exception as exc:
            import traceback
            traceback.print_exc(limit=3)
            check(fn.__name__, False, f"raised {type(exc).__name__}: {exc}")
    print("\n" + ("ALL PASS" if not FAILED else f"FAILED: {', '.join(FAILED)}"))
    sys.exit(1 if FAILED else 0)
