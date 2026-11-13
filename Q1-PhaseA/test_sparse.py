# Gate suite for the sparse-layer arms (sparse.py / pcalm.py): case "none" must reproduce the pre-sparsity golden
# reference bit for bit, then cases B (soft-threshold), C (LASSO dictionary), D (side heads) and E (reconstruction
# anchor) are checked against their closed-form optima, batch invariance, and their BP controls. Run as a script
# to execute every test and exit nonzero on any failure.

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
    """Prints a PASS or FAIL line for a named check, appending optional detail text.
    Records the check in FAILED when it does not pass."""
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


_X = {}


def _data():
    """Loads and caches a 64-sample slice of the dataset in a module-level dict.
    Returns the cached data on subsequent calls."""
    if "X" not in _X:
        _X["X"] = D.load_dataset(DataCfg())[0][:64]
    return _X["X"]


def small(case="none", s=2, alpha=1.0, T=3, B=8, use_sigreg=True, projector="bn",
          beta=0.3, gamma=0.05, beta_g=1.0, eta=0.0625, below_hard=True, seed=0,
          normalise_fs=True, s_step=0.5):
    """Builds a tiny 6-layer, width-16 encoder config and returns its params, input views, directions, and run config."""
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
    """Converts the inexact-array leaves of a pytree into a list of numpy arrays."""
    return [np.asarray(x) for x in jax.tree_util.tree_leaves(eqx.filter(t, eqx.is_inexact_array))]


def maxrel(a, b):
    """Computes the maximum relative error between two arrays, using b as the reference.
    Adds a small epsilon to the denominator to avoid division by zero."""
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-30))


# ------------------------------------------------ 1. original path unchanged

def test_golden_case_none():
    """Verifies case 'none' reproduces pre-edit pcalm.relax/pcalm.grads bit for bit for alpha in (0.0, 1.0).
    Regenerates the reference on this machine via _golden_pcalm.py since cross-machine refs differ in the 7th digit."""
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
    """Checks that SP.init_params with no case falls back to E.init and produces identical parameter trees."""
    cfg = RunCfg()
    a = SP.init_params(jax.random.PRNGKey(3), cfg, cfg.data.input_dim)
    b = E.init(jax.random.PRNGKey(3), cfg.enc, cfg.data.input_dim)
    same = (jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b)
            and all(np.array_equal(x, y) for x, y in zip(leaves(a), leaves(b))))
    check("init_params(case none) == E.init", same)


def test_old_config_json_loads_as_case_none():
    """Tests that a config.json missing the "sparse" block loads with sparse.case set to "none"."""
    from run_probes import cfg_from_dict
    import json
    d = json.loads(RunCfg().to_json())
    d.pop("sparse")
    check("config.json without a 'sparse' block -> case none",
          cfg_from_dict(d).sparse.case == "none")


# --------------------------------------------- 2. architecture change vs jpc

def test_architecture_change():
    """Checks that case B/C removes the skip at the sparse layer and rescales its weight by sqrt(L), leaving other layers unchanged.
    Verifies that the sparse layer's activation norm is now O(1) relative to the previous layer's, not suppressed by sqrt(L)."""
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
    """Checks that al_energy with skip=None matches jpc's energy function on the same activations.
    Asserts the two values agree within a relative tolerance of 1e-5."""
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
    """Checks that SP.deploy matches E.forward when beta=0 or the beta_g dial is set to 0.
    Verifies this holds for both case "B" and case "C" within a 1e-6 relative error tolerance."""
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
    """Tests that deployed activations at the sparse layer equal the soft threshold of f_s by beta/rho.
    Checks the resulting active fraction matches this threshold relationship."""
    params, Zin, A, cfg = small(case="B", beta=0.3)
    acts, ex = SP.deploy(params, Zin[0], cfg)
    s = cfg.sparse.layer
    want = SP.soft_threshold(ex["f_s"], cfg.sparse.beta / cfg.pc.rho)
    frac = float(jnp.mean(acts[s] != 0))
    check("B: deployed z_s == S_{beta/rho}(f_s)", bool(jnp.array_equal(acts[s], want)),
          f"active frac={frac:.3f}")


def test_B_prox_iteration_converges_to_closed_form():
    """Iterates the PC-B gradient-then-prox update on z_s alone with g_s = 0.
    Checks the fixed point converges to the closed-form soft-threshold S_{beta/rho}(f_s)."""
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
    """Checks that case B relax with beta=0, alpha=0 matches the original PC relaxation on the same network.
    Confirms the prox is identity and no multipliers arise, within a 1e-5 relative error tolerance."""
    params, Zin, A, cfg = small(case="B", beta=0.0, alpha=0.0, normalise_fs=False,
                                s_step=0.0625)
    acts_b, _, hb = pcalm.relax(params, Zin, A, cfg, jax.random.PRNGKey(0))
    cfg0 = dataclasses.replace(cfg, sparse=SparseCfg())     # same params, old path
    acts_0, _, h0 = pcalm.relax(params, Zin, A, cfg0, jax.random.PRNGKey(0))
    err = max(maxrel(a, b) for av, bv in zip(acts_b, acts_0) for a, b in zip(av, bv))
    check("B(beta=0, alpha=0) relax == original relax on the same net", err < 1e-5,
          f"max rel err={err:.2e}")


def test_layer_alphas():
    """Tests that layer_alphas zeroes layer s for "hard" sparsity and all layers below s for "soft".
    Verifies "none" sparsity mode leaves all layer alphas unchanged."""
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
    """Checks that dual variable lambda_s remains exactly zero at the sparse layer during relaxation.
    Verifies that the dual at the next layer, lambda_{s+1}, grows nonzero as expected."""
    params, Zin, A, cfg = small(case="B", alpha=1.0, T=5)
    _, duals, _ = pcalm.relax(params, Zin, A, cfg, jax.random.PRNGKey(0))
    s = cfg.sparse.layer
    zs = max(float(jnp.max(jnp.abs(duals[v][s]))) for v in range(len(duals)))
    other = max(float(jnp.max(jnp.abs(duals[v][s + 1]))) for v in range(len(duals)))
    check("B: lambda_s stays exactly 0 while lambda_{s+1} grows", zs == 0.0 and other > 0,
          f"|lam_s|max={zs:.1e} |lam_s+1|max={other:.2e}")


# ---------------------------------------------------------- 4. case C maths

def test_ista_vs_fista_and_kkt():
    """Checks that ISTA energy decreases monotonically with iteration count.
    Verifies ISTA(5000) converges to the same energy as FISTA(3000) and that FISTA satisfies the KKT condition."""
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
    """Checks that ista at step 0 matches one explicit ISTA step starting from a zero initial code.
    Asserts the two results are numerically close within tolerance."""
    params, Zin, A, cfg = small(case="C")
    _, ex = SP.deploy(params, Zin[0], cfg)
    f, Dm, mu, g = ex["f_s"], SP.get_D(params), SP.lasso_mu(cfg), cfg.sparse.gamma
    eta = 0.1
    a0 = SP.ista(f, Dm, g, mu, 0, eta)
    a1 = SP.ista(f, Dm, g, mu, 1, eta, a0=jnp.zeros_like(a0))
    check("C: a^(0) == one ISTA step from a = 0", bool(jnp.allclose(a0, a1, atol=1e-7)))


def test_blend_is_the_z_minimiser():
    """Checks that the deployed z_s is the stationary point minimizing the blended quadratic energy.
    Verifies the minimum equals the LASSO form with mu = rho*beta_g / (rho + beta_g)."""
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
    """Checks that with alpha=0 (no dual ascent), F+L is non-increasing across relaxation steps.
    Asserts monotonic descent holds under coupled output drive with a small step size."""
    params, Zin, A, cfg = small(case="C", alpha=0.0, T=12, eta=0.02, use_sigreg=False,
                                projector="none")
    _, _, h = pcalm.relax(params, Zin, A, cfg, jax.random.PRNGKey(0))
    tot = np.asarray(h["F"]) + np.asarray(h["L"])
    mono = bool(np.all(np.diff(tot) <= 1e-6 * np.abs(tot[:-1]).max()))
    check("C: F + L non-increasing over relaxation at alpha = 0", mono,
          " ".join(f"{v:.5f}" for v in tot[:6]) + " ...")


def test_C_D_gradient_matches_analytic():
    """Checks that the gradient w.r.t. D matches the analytic formula beta_g (D a - z_s) a^T / B, summed over views.
    Fails the check if the relative error exceeds 1e-4."""
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
    """Tests that project() renormalizes dictionary columns to unit norm after scaling.
    Tests that params serialize and deserialize via eqx round-trip to identical values."""
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
    """Checks that sample 0's settled sparse variable is invariant to batch size B.
    Guards against an eta_h * B leak into a prox threshold or code step."""
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
    """Tests that backprop gradients through the deployed network are finite and reach the sparse weight.
    Checks dictionary gradients and head energy for case C, matching checks C1-B and C2."""
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
    """Checks that PC weight gradients are finite for cases B and C.
    Verifies that aux dict logs F_first, active_frac, and act_growth."""
    for case in ("B", "C"):
        params, Zin, A, cfg = small(case=case, alpha=1.0, T=4)
        g, aux = TR.pc_grads(params, Zin, A, cfg, jax.random.PRNGKey(0))
        finite = all(np.all(np.isfinite(x)) for x in leaves(g))
        check(f"PC-{case}: weight gradients finite; F_first/active_frac logged",
              finite and "F_first" in aux and "active_frac" in aux and "act_growth" in aux,
              f"F0={float(aux['F_first']):.4f} F_T={float(aux['F_final']):.4f} "
              f"active={float(aux['active_frac']):.3f} act_growth={float(aux['act_growth']):.3f}")


def test_control_objective_composition():
    """Checks that the BP objective equals task loss plus the deployed sparse-layer energy summed over views.
    Verifies the sum (not average) over views, matching the fix for the bug found by all three reviewers."""
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
    """Checks PC gradients match BP-control gradients on layers at or below the sparse layer, with task drive removed.
    Gates on grad-norm ratios (B: 0.73-1.36, C dictionary: 0.57-0.64 across seeds 0-4), not on seed-sensitive cosines."""
    def cos(a, b):
        """Computes the cosine similarity between two flattened arrays.
        Adds a small epsilon to the denominator to avoid division by zero."""
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
    """Verifies that the sparse layer's output f_s is normalised per sample to fs_norm.
    Checks this holds consistently for the deployed network, the relaxation energy, and under W_s rescaling."""
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
    """Tests Case B's layer-s step on the full-size architecture, not the unrepresentative small test net.
    Checks that s_step=0.5 drives the fixed-point residual at s below 0.1 at T=16, better than shared eta_h."""
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
    """Checks that gradients stay finite when one augmented view is all-zero (f_s = 0 exactly).
    Regression test for job b3c717c6, where a zero view caused NaN gradients via f / ||f||."""
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
    """Builds a tiny (L=6) encoder config with case-D sparse-coding side heads on residual layers.
    Returns initialized params, input views, sampled directions, and the run config, using calibrated defaults."""
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
    """Verifies that D_l depends only on (seed, l) so nested head sets share dictionaries with unit-norm columns.
    Checks that the encoder matches case "none", all alpha multipliers are kept with no normalization, and invalid head sets are rejected."""
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
    """Checks that zero head weights make sparse-coding heads inert, matching plain PC-ALM relax and grads.
    Also verifies the dictionary gradient is exactly zero in this case."""
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
    """Checks that deployed codes' relative LASSO suboptimality decreases as t grows across heads.
    Asserts the gap falls below 1e-3 at t=512, matching each head's FISTA optimum."""
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
    """Tests that sparse-code relaxation keeps codes finite and layer activations close to their predictions.
    Checks relative residual between relaxed head layers and their predictions stays below 0.1 when multipliers are on."""
    pD, Zin, A, cD = smallD(heads=(2, 3), T=40)
    acts, duals, h = pcalm.relax(pD, Zin, A, cD, jax.random.PRNGKey(0))
    preds = pcalm.layer_predictions(pD, acts[0], Zin[0], cD.enc)
    r = [float(jnp.linalg.norm(acts[0][l] - preds[l]) / (jnp.linalg.norm(preds[l]) + 1e-12))
         for l in cD.sparse.heads]
    fin = all(np.all(np.isfinite(np.asarray(a))) for a in h["a"][0])
    check("D: relaxed head layers stay near their predictions (multiplier kept), codes finite",
          fin and max(r) < 0.1, "rel resid " + " ".join(f"{v:.3f}" for v in r))


def test_D_control_objective_and_agreement():
    """Checks that BP-D's objective equals task loss plus head energy summed over views.
    Checks that with task drive removed, PC-D's gradient agrees with BP-D's on dictionaries and on layers up to the max head."""
    pD, Zin, A, cD = smallD(heads=(2, 3))
    cb = dataclasses.replace(cD, arm="bp")
    tot, aux = TR.bp_objective(pD, Zin, A, cb)
    task, _ = TR.loss_from_views(pD, Zin, A, cb)
    head = sum(float(SP.deployed_head_energy(pD, Zin[v], cb)) for v in range(Zin.shape[0]))
    ok = abs(float(tot) - float(task) - head) <= 1e-5 * max(1.0, abs(head))
    check("D: BP-D objective == task + head energy summed over views", ok,
          f"tot={float(tot):.6f} task={float(task):.6f} head={head:.6f}")

    def cos(a, b):
        """Computes the cosine similarity between two arrays after flattening them.
        Adds a small epsilon to the denominator to avoid division by zero."""
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
    """Checks that sample 0's settled codes stay invariant to batch size across sizes (2, 8, 32).
    Checks that an all-zero input view still yields finite PC and BP gradients."""
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
    """Trains both PC and BP arms for a few steps with case-D sparse heads.
    Checks D stays unit-norm, round-trips through load_run, and logs head health, then verifies the pair gate reads case-D rows cleanly."""
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
    """Checks that head input is centered on mu and row-normalised to sqrt(n), making head energy invariant to scale.
    Checks the BP-D head gradient wrt D matches the code-detached partial derivative, and that mu has no gradient and is EMA-updated."""
    pD, Zin, A, cD = smallD(heads=(2,))
    x = Zin[0]
    acts, ex = SP.deploy(pD, x, cD)
    f = acts[2]
    mu = SP.get_mu(pD)[0]
    nrm = np.linalg.norm(np.asarray(ex["f_heads"][0]), axis=1)
    rows_ok = np.allclose(nrm, np.sqrt(f.shape[-1]), rtol=1e-4)
    D0, g0 = SP.get_D(pD)[0], cD.sparse.gammas[0]
    def e(v, m):
        """Computes the mean LASSO energy after running ISTA sparse coding on the head-processed input."""
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


# ------------------------------------------------- case E: reconstruction anchor

def smallE(heads=(2, 3), beta_r=0.05, **kw):
    """Builds smallD's config and params, then sets a positive beta_r to anchor the code to the pixel space.
    Returns the resulting params, input batch, activity, and config as a 6-tuple."""
    pD, Zin, A, cD = smallD(heads=heads, **kw)
    cE = dataclasses.replace(cD, sparse=dataclasses.replace(cD.sparse, beta_r=beta_r))
    pE = SP.init_params(jax.random.PRNGKey(cE.opt.seed), cE, cE.data.input_dim)
    pE = SP.init_mu(pE, D.eval_batch(jnp.asarray(_data()[:64]), cE.data), cE)
    return pE, Zin, A, cE


def test_E_init_and_reduction():
    """Checks that case E's params match case D with beta_r=0, plus unit-norm G and pixel mean when beta_r>0.
    Verifies param_labels has 6 entries and sparse codes are identical between D and E at beta_r=0."""
    pD, Zin, A, cD = smallD(heads=(2, 3))
    pE, _, _, cE = smallE(heads=(2, 3))
    x = Zin[0]
    xb = D.eval_batch(jnp.asarray(_data()[:64]), cE.data)
    same = all(np.array_equal(a, b) for a, b in zip(leaves(pD), leaves(pE[:4])))
    unitG = all(np.allclose(np.linalg.norm(np.asarray(G), axis=0), 1.0, atol=1e-5) for G in pE[4])
    xbar = np.allclose(np.asarray(pE[5]), np.asarray(jnp.mean(xb, 0)), atol=1e-6)
    lab = SP.param_labels(pE)
    labs = len(lab) == 6 and lab[4] == ("dict", "dict") and lab[5] == "frozen"
    c0 = dataclasses.replace(cE, sparse=dataclasses.replace(cE.sparse, beta_r=0.0))
    codes_eq = all(np.array_equal(a, b) for a, b in zip(SP.deploy(pD, x, cD)[1]["a"], SP.deploy(pD, x, c0)[1]["a"]))
    check("E: beta_r=0 is case D; beta_r>0 adds unit-norm G and the pixel mean, rest identical, labels complete",
          same and unitG and xbar and labs and codes_eq and len(SP.init_params(jax.random.PRNGKey(0), c0, c0.data.input_dim)) == 4,
          f"same={same} unitG={unitG} xbar={xbar} labels={labs} codes_eq={codes_eq}")


def test_E_codes_solve_anchored_lasso():
    """Checks that deployed codes converge toward the FISTA optimum of the explicit anchored lasso energy.
    Verifies the energy gap shrinks monotonically with more deployment steps, reaching under 1e-3 at t=512."""
    pE, Zin, A, cE = smallE(heads=(2, 3))
    x = Zin[0]; sp = cE.sparse
    xc = SP.anchor_target(x, pE[5])
    def energy(f, Dl, Gl, a, g):
        """Computes the total sparse-coding energy: reconstruction loss on f and xc plus an L1 sparsity penalty on a.
        Returns the scalar mean energy over the batch."""
        return float(jnp.mean(0.5 * sp.beta_g * jnp.sum((f - a @ Dl.T) ** 2, -1)
                              + 0.5 * sp.beta_r * jnp.sum((xc - a @ Gl.T) ** 2, -1) + g * jnp.sum(jnp.abs(a), -1)))
    out = []
    for t in (4, 64, 512):
        _, ex = SP.deploy(pE, x, cE, t=t)
        gaps = []
        for f, Dl, Gl, a, g in zip(ex["f_heads"], pE[2], pE[4], ex["a"], sp.gammas):
            F, Dst, m = SP.stacked(f, Dl, xc, Gl, cE)
            astar = SP.fista(F, Dst, g, m, 3000)
            gaps.append((energy(f, Dl, Gl, a, g) - energy(f, Dl, Gl, astar, g)) / abs(energy(f, Dl, Gl, astar, g)))
        out.append(max(gaps))
    check("E: deployed codes converge to the optimum of the explicit anchored energy (< 1e-3 at t=512)",
          out[0] >= out[1] >= out[2] and out[2] < 1e-3 and min(out) > -1e-4, " ".join(f"{v:.2e}" for v in out))


def test_E_relaxation_hooks_match_autodiff():
    """Checks that head_smooth matches a manual energy including the anchor term.
    Verifies a_step equals a prox step on its exact gradient with step size set by the stacked Lipschitz constant."""
    pE, Zin, A, cE = smallE(heads=(2, 3))
    x = Zin[0]; sp = cE.sparse
    acts, ex = SP.deploy(pE, x, cE, t=3)
    a = ex["a"]
    xc = SP.anchor_target(x, pE[5])
    manual = sum(w * (0.5 * sp.beta_g * jnp.sum((SP.head_in(acts[l], mu) - al @ Dl.T) ** 2)
                      + 0.5 * sp.beta_r * jnp.sum((xc - al @ Gl.T) ** 2)) / x.shape[0]
                 for l, Dl, mu, Gl, al, w in zip(sp.heads, pE[2], pE[3], pE[4], a, sp.head_ws))
    hs = SP.head_smooth(pE, acts, a, cE, x=x)
    eta = SP.code_step(pE, cE)
    gr = jax.grad(lambda aa: SP.head_smooth(pE, acts, aa, cE, x=x))(a)
    ref = tuple(SP.soft_threshold(al - (el / (w / x.shape[0])) * g_, el * gm)
                for al, g_, el, gm, w in zip(a, gr, eta, sp.gammas, sp.head_ws))
    got = SP.a_step(pE, acts, a, cE, eta, x=x)
    err = max(maxrel(u, v) for u, v in zip(got, ref))
    lip = all(abs(float(e) * float(SP.spectral_sq(SP.stacked_D(Dl, Gl, cE))) / sp.step_frac - 1.0) < 1e-5
              for e, Dl, Gl in zip(eta, pE[2], pE[4]))
    check("E: head_smooth includes the anchor; a_step = prox step on its exact gradient; step from the stacked Lipschitz",
          maxrel(hs, manual) < 1e-5 and err < 1e-4 and lip, f"energy {maxrel(hs, manual):.1e} step {err:.1e} lip={lip}")


def test_E_pc_bp_agree_and_train():
    """Verifies PC and BP-E gradients on G and D agree closely under zero task drive with no xbar gradient.
    Trains both arms briefly and checks unit-norm G, fixed xbar, round-trip reload, logged R^2, and gate readout."""
    def cos(u, v):
        """Computes the cosine similarity between two flattened vectors.
        Adds a small epsilon to the denominator to avoid division by zero."""
        u, v = np.ravel(u), np.ravel(v); return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-30))
    pE, Zin, A, cE = smallE(heads=(2, 3), T=200, use_sigreg=False, projector="none")
    Zin = jnp.stack([Zin[0]] * Zin.shape[0])
    gp, _ = pcalm.grads(pE, Zin, A, cE, jax.random.PRNGKey(0))
    gb, _ = TR.bp_grads(pE, Zin, A, dataclasses.replace(cE, arm="bp"))
    cG = [cos(a, b) for a, b in zip(gp[4], gb[4])]; cDs = [cos(a, b) for a, b in zip(gp[2], gb[2])]
    zero_x = float(jnp.max(jnp.abs(gp[5]))) == 0.0 and float(jnp.max(jnp.abs(gb[5]))) == 0.0
    check("E: zero task drive, T=200: PC vs BP-E cosine > 0.95 on G and D; no gradient on xbar",
          min(cG + cDs) > 0.95 and zero_x, f"G {[round(c, 3) for c in cG]} D {[round(c, 3) for c in cDs]} xbar0={zero_x}")
    import gate_pilot as GP
    from run_probes import load_run
    tmp = tempfile.mkdtemp(); dirs = {}
    for arm in ("pc", "bp"):
        _, _, _, cE = smallE(heads=(2, 3))
        cfg = dataclasses.replace(cE, arm=arm, out_dir=tmp, tag="tE",
                                  opt=dataclasses.replace(cE.opt, n_steps=4, eval_every=2))
        p, log = TR.train(cfg, verbose=False)
        dirs[arm] = os.path.join(tmp, cfg.name)
        unit = all(np.allclose(np.linalg.norm(np.asarray(G), axis=0), 1.0, atol=1e-5) for G in p[4])
        xb = np.allclose(np.asarray(p[5]), np.asarray(jnp.mean(SP.mu_rows(D.load_dataset(cfg.data)[0], cfg), 0)), atol=1e-5)
        c2, p2 = load_run(dirs[arm])
        rt = all(np.array_equal(a, b) for a, b in zip(leaves(p), leaves(p2))) and c2.sparse.beta_r == cfg.sparse.beta_r
        ev = [r for r in log if isinstance(r.get("eval"), dict)]
        has = bool(ev) and "h2_recon_r2" in ev[-1]["eval"]["sparse"]
        check(f"E ({arm}): trains, unit-norm G, xbar = training mean and fixed, round-trips, logs R^2",
              unit and xb and rt and has, f"unit={unit} xbar={xb} roundtrip={rt} r2={has}")
    ok, reasons, info = GP.gate(dirs["pc"], dirs["bp"])
    check("E: pair gate reads the reconstruction criteria", "head_recon_r2_first_last" in info["pc"], str(reasons)[:200])


def test_E_target_codek_and_views():
    """Tests that anchor targets remove brightness, code_k steps match deploy count, and views use own pixels."""
    pE, Zin, A, cE = smallE(heads=(2, 3), beta_r=8.0)
    x = Zin[0]
    t = SP.anchor_target(x, pE[5])
    zero_mean = float(jnp.max(jnp.abs(jnp.mean(t, -1)))) < 1e-5
    shift = 0.3 * jnp.linspace(-1, 1, x.shape[0])[:, None]
    _, e1 = SP.deploy(pE, x, cE)
    l, Dl, mu, Gl, xc2 = SP._heads(pE, cE, x + shift)[0]
    inv_t = maxrel(xc2, SP._heads(pE, cE, x)[0][4]) < 1e-5
    # (2)
    acts, ex = SP.deploy(pE, x, cE, t=2)
    eta = SP.code_step(pE, cE)
    cK = dataclasses.replace(cE, sparse=dataclasses.replace(cE.sparse, code_k=4))
    once = SP.a_step(pE, acts, ex["a"], cK, eta, x=x)
    rep = ex["a"]
    for _ in range(4):
        rep = SP.a_step(pE, acts, rep, cE, eta, x=x)
    same = max(maxrel(u, v) for u, v in zip(once, rep)) < 1e-5
    ndep = SP.t_deploy(cK) == cK.sparse.T_test * 4
    dflt = max(maxrel(u, v) for u, v in zip(SP.deploy(pE, x, cK)[1]["a"], SP.deploy(pE, x, cK, t=cK.sparse.T_test * 4)[1]["a"])) == 0.0
    # (3) distinct views: each view's head energy uses its own pixels
    acts_v = [SP.deploy(pE, Zin[v], cE)[0] for v in range(Zin.shape[0])]
    a_v = [SP.deploy(pE, Zin[v], cE)[1]["a"] for v in range(Zin.shape[0])]
    own = sum(SP.head_smooth(pE, acts_v[v], a_v[v], cE, x=Zin[v]) for v in range(Zin.shape[0]))
    swapped = sum(SP.head_smooth(pE, acts_v[v], a_v[v], cE, x=Zin[(v + 1) % Zin.shape[0]]) for v in range(Zin.shape[0]))
    distinct = float(jnp.max(jnp.abs(Zin[0] - Zin[1]))) > 0 and float(own) < float(swapped)
    check("E: anchor target removes brightness; code_k = k equals k steps, deployed T_test*k; views use own pixels",
          zero_mean and inv_t and same and ndep and dflt and distinct,
          f"zero_mean={zero_mean} shift_inv={inv_t} codek={same} t_deploy={ndep} default={dflt} own<swapped={distinct} "
          f"({float(own):.3g} vs {float(swapped):.3g})")


def test_E_G_gradient_is_local_hebbian():
    """Verifies that the PC gradient on G_l equals -sum_v w_l beta_r (xc_v - G a_v)^T a_v / B.
    Confirms this local, Hebbian update holds at production-like beta_r across distinct views."""
    pE, Zin, A, cE = smallE(heads=(2, 3), beta_r=8.0, T=4)
    sp = cE.sparse
    acts_v = [SP.deploy(pE, Zin[v], cE)[0] for v in range(Zin.shape[0])]
    a_v = [SP.deploy(pE, Zin[v], cE)[1]["a"] for v in range(Zin.shape[0])]
    def F(G):
        """Computes smooth sparse-coding energy for parameter G across all views, then sums the results."""
        p = (pE[0], pE[1], pE[2], pE[3], G, pE[5])
        return sum(SP.head_smooth(p, acts_v[v], a_v[v], cE, x=Zin[v]) for v in range(Zin.shape[0]))
    g = jax.grad(F)(pE[4])
    ok = True; worst = 0.0
    for i, (Gl, wl) in enumerate(zip(pE[4], sp.head_ws)):
        man = 0.0
        for v in range(Zin.shape[0]):
            xc = SP.anchor_target(Zin[v], pE[5]); a = a_v[v][i]
            man = man - wl * sp.beta_r * (xc - a @ Gl.T).T @ a / Zin.shape[1]
        worst = max(worst, maxrel(g[i], man))
    check("E: G gradient = local Hebbian -w beta_r (xc - G a)^T a / B, summed over distinct views", worst < 1e-4, f"{worst:.1e}")


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
               test_D_head_reads_normalised_copy, test_E_init_and_reduction,
               test_E_codes_solve_anchored_lasso, test_E_relaxation_hooks_match_autodiff,
               test_E_pc_bp_agree_and_train, test_E_target_codek_and_views,
               test_E_G_gradient_is_local_hebbian):
        try:
            fn()
        except Exception as exc:
            import traceback
            traceback.print_exc(limit=3)
            check(fn.__name__, False, f"raised {type(exc).__name__}: {exc}")
    print("\n" + ("ALL PASS" if not FAILED else f"FAILED: {', '.join(FAILED)}"))
    sys.exit(1 if FAILED else 0)
