"""Is PC inference valid, and does it do useful work at test time?

    ./.venv/bin/python diag_sparse.py runs/<run> [runs/<run> ...] --csv diag_sparse.csv

Spec: 'Method - Sparse PC-ALM arms spec', section Analysis.  Groups:

  A  inference validity   C: LASSO KKT violation and suboptimality vs FISTA(2000)
                             against test-time steps t; every PC arm: the proximal-
                             gradient fixed-point residual of the TRAINING relaxation
                             against the inference budget T.
  B  useful work          probe accuracy against t (C: ISTA steps; B: iterated prox
                             from the dense f_s, which must converge to the closed
                             form -- the built-in negative control); amortisation gap;
                             the beta_g dial (beta_g = 0 asserted equal to the plain
                             forward pass); dictionary health; rescaling cheat (log).
  C  robustness           Gaussian noise / random-square occlusion at TEST time only,
                             identical seeds for every arm; probes fit on CLEAN train
                             features; Delta_inf = probe(t=T) - probe(t=0); kappa at s.
  D  learning             per-layer cosine of the PC weight gradient against BACKPROP
                             THROUGH THE SAME DEPLOYED NETWORK; clip fraction.

Runs with sparse.case == "none" (the reused BP / PC-ALM arms) get group C at the
feedforward readout only, so they share the robustness axes as references.
Output: one long-format CSV row per (run, group, metric, condition, t).
"""
import argparse
import csv
import dataclasses
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
import numpy as np

import data as D
import metrics as M
import pcalm as PA
import sigreg as S
import sparse as SP
import train as TR
from run_probes import load_run

T_LIST = (0, 1, 2, 5, 10, 16, 25)
NOISE = (0.05, 0.1, 0.2)
OCCL = (0.10, 0.25)
BETA_G_DIAL = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 1000.0)
CORRUPT_SEED = 12345                 # shared by every arm


# ------------------------------------------------------------- readouts

def readout(params, x, cfg, t=None, beta_g=None):
    """Deployed embedding with t test-time inference steps.

    C: t ISTA steps (deploy).  B: t iterated prox-gradient steps on z_s alone,
    starting from the DENSE f_s (eta = 0.5 / rho): t = 0 is the network with
    no inference, t -> inf is the closed form S_{beta/rho}(f_s).  none: the
    feedforward pass (t ignored).
    Returns (z_out, z_s, f_s) -- z_s, f_s are None for case none.
    """
    sp = cfg.sparse
    if sp.case == "none":
        z, _ = SP.forward(params, x, cfg)
        return z, None, None
    if sp.case == "C":
        acts, ex = SP.deploy(params, x, cfg, t=t, beta_g=beta_g)
        return acts[-1], acts[sp.layer], ex["f_s"]
    # case B
    cfg0 = dataclasses.replace(cfg, sparse=dataclasses.replace(sp, beta=0.0))
    acts, ex = SP.deploy(params, x, cfg0)               # dense pass, gives f_s
    f = ex["f_s"]
    rho = cfg.pc.rho
    if t is None:
        zs = SP.soft_threshold(f, sp.beta / rho)
    else:
        zs, eta = f, 0.5 / rho
        for _ in range(t):
            zs = SP.soft_threshold(zs - eta * rho * (zs - f), eta * sp.beta)
    sc = SP._scalings(params, x, cfg)
    prev = zs
    for l in range(sp.layer + 1, len(params[0])):
        prev = SP._pred(params, sc, l, prev)
    return prev, zs, f


def batched(fn, X, bs=1000):
    outs = [fn(X[i:i + bs]) for i in range(0, len(X), bs)]
    return [None if outs[0][k] is None else np.concatenate([np.asarray(o[k]) for o in outs])
            for k in range(len(outs[0]))]


# --------------------------------------------------------------- probes

def ridge_classifier(Ztr, ytr, Zte, yte, lams=(1e-2, 1e-1, 1, 10, 100, 1e3), seed=0):
    """Closed-form ridge on one-hot labels; lambda on a 20% split of TRAIN."""
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-6
    A, B = (Ztr - mu) / sd, (Zte - mu) / sd
    K = int(max(ytr.max(), yte.max())) + 1
    Y = np.eye(K)[ytr]
    perm = np.random.default_rng(seed).permutation(len(A))
    nv = len(A) // 5
    va, tr = perm[:nv], perm[nv:]

    def fit(X, T, lam):
        X1 = np.hstack([X, np.ones((len(X), 1))])
        return np.linalg.solve(X1.T @ X1 + lam * np.eye(X1.shape[1]), X1.T @ T)

    def acc(W, X, y):
        return float(np.mean(np.argmax(np.hstack([X, np.ones((len(X), 1))]) @ W, 1) == y))

    best = max(lams, key=lambda l: acc(fit(A[tr], Y[tr], l), A[va], ytr[va]))
    return acc(fit(A, Y, best), B, yte), best


def probe_pair(Ztr, ytr, Zte, yte):
    lin, lam = ridge_classifier(Ztr, ytr, Zte, yte)
    return {"linear": lin, "knn": float(M.knn_accuracy(Ztr, ytr, Zte, yte))}


# ------------------------------------------------------------ corruption

def corrupt(xflat, cfg, kind, level, seed=CORRUPT_SEED):
    """Test-time corruption in pixel space; xflat is eval_batch output in [0,1]."""
    if kind == "clean" or level == 0:
        return xflat
    c = 1 if cfg.data.grayscale else 3
    s = cfg.data.img_size
    x = np.asarray(xflat).reshape(-1, s, s, c).copy()
    rng = np.random.default_rng(seed)
    if kind == "noise":
        x = np.clip(x + level * rng.standard_normal(x.shape).astype(np.float32), 0.0, 1.0)
    elif kind == "occlusion":
        side = max(1, int(round(np.sqrt(level) * s)))
        for i in range(len(x)):
            r0, c0 = rng.integers(0, s - side + 1, 2)
            x[i, r0:r0 + side, c0:c0 + side, :] = 0.0     # sky is dark
    return jnp.asarray(x.reshape(len(x), -1))


# ------------------------------------------------------------- group A

def lasso_curve(params, x, cfg):
    sp, mu = cfg.sparse, SP.lasso_mu(cfg)
    Dm = SP.get_D(params)
    _, ex = SP.deploy(params, x, cfg, t=0)
    f = ex["f_s"]
    eta = sp.step_frac / (mu * SP.spectral_sq(Dm))
    a_ref = SP.fista(f, Dm, sp.gamma, mu, 2000)
    E_ref = SP.lasso_energy(f, Dm, a_ref, sp.gamma, mu)
    rows = []
    for t in T_LIST + (50, 100):
        a = SP.ista(f, Dm, sp.gamma, mu, t, eta)
        E = SP.lasso_energy(f, Dm, a, sp.gamma, mu)
        rows.append({"t": t,
                     "subopt_rel": float(jnp.mean((E - E_ref) / (E_ref + 1e-12))),
                     "kkt_over_gamma": float(jnp.mean(SP.kkt_violation(f, Dm, a, sp.gamma, mu))),
                     "active_frac": float(jnp.mean(a != 0))})
    return rows


def relax_fixed_point_residual(params, cfg, Xtr, Ts=(16, 64, 256), B=256, seed=0):
    """Proximal-gradient mapping norm at the end of the TRAINING relaxation.

    G_l = (z_l - prox_l(z_l - eta * grad_l)) / eta, grad_l the per-sample gradient
    of the smooth relaxation objective (B * dF_smooth/dz_l plus the output drive
    at the top), prox_l the identity except S_{eta beta} at s for case B.
    Reported relative to the same mapping norm at the relaxation's START
    (deployed forward, zero duals), per layer, so a value -> 0 as T grows means
    the relaxation converges to its fixed point.  Sparse PC arms only.
    """
    key = jax.random.PRNGKey(seed)
    kb, kv, ka, kg = jax.random.split(key, 4)
    idx = np.random.default_rng(seed).choice(len(Xtr), B, replace=False)
    Zin, _ = D.make_views(kv, jnp.asarray(Xtr[idx]), cfg.data)
    A = S.sample_directions(ka, cfg.enc.embed_dim, cfg.sig.num_slices)
    loss_z = PA._objective(cfg, A)
    V, eta = Zin.shape[0], cfg.pc.eta_h

    def mapping(acts, duals, a):
        def Fs(acts):
            F = sum(PA.al_energy(params, acts[v], Zin[v], duals[v], cfg.pc.rho, cfg.enc,
                                 cfg.pc.out_scale) for v in range(V))
            if a is not None:
                F = F + sum(SP.head_smooth(params, acts[v], a[v], cfg) for v in range(V))
            return F
        g = jax.grad(Fs)(acts)
        gL = jax.grad(lambda ac: loss_z(jnp.stack([ac[v][-1] for v in range(V)])))(acts)
        per = []
        for l in range(len(acts[0])):
            num, den = 0.0, 0.0
            for v in range(V):
                grad = B * g[v][l] + B * gL[v][l]
                step = acts[v][l] - eta * grad
                if cfg.sparse.case == "B" and l == cfg.sparse.layer:
                    step = SP.soft_threshold(step, eta * cfg.sparse.beta)
                G = (acts[v][l] - step) / eta
                num += float(jnp.sum(G ** 2))
                den += float(jnp.sum(grad ** 2))
            per.append((num, den))
        return per

    rows = []
    for T in Ts:
        c2 = dataclasses.replace(cfg, pc=dataclasses.replace(cfg.pc, T=T))
        acts, duals, hist = PA.relax(params, Zin, A, c2, kg)
        a = hist.get("a")
        per = mapping(acts, duals, a)
        rows.append({"T": T, "per_layer_G": [float(np.sqrt(n)) for n, _ in per],
                     "per_layer_grad": [float(np.sqrt(d)) for _, d in per]})
    # normalise by the mapping norm at the START of relaxation (deployed forward,
    # zero duals), so per_layer_rel -> 0 with T means convergence to the fixed point
    init = [SP.relax_init(params, Zin[v], cfg) for v in range(V)]
    acts0 = [i[0] for i in init]
    a0 = [i[1] for i in init] if cfg.sparse.case == "C" else None
    base = mapping(acts0, PA.zero_duals(acts0), a0)
    for r in rows:
        r["per_layer_rel"] = [g / (np.sqrt(b[0]) + 1e-12) for g, b in zip(r["per_layer_G"], base)]
    return rows


# ------------------------------------------------------------- group D

def grad_cosines(params, cfg, Xtr, B=256, seed=0):
    """Per-layer cosine: PC weight gradient vs BP through the same deployed net."""
    key = jax.random.PRNGKey(seed + 1)
    kb, kv, ka, kg = jax.random.split(key, 4)
    idx = np.random.default_rng(seed).choice(len(Xtr), B, replace=False)
    Zin, _ = D.make_views(kv, jnp.asarray(Xtr[idx]), cfg.data)
    A = S.sample_directions(ka, cfg.enc.embed_dim, cfg.sig.num_slices)
    gp, _ = TR.pc_grads(params, Zin, A, dataclasses.replace(cfg, arm="pc"), kg)
    gb, _ = TR.bp_grads(params, Zin, A, dataclasses.replace(cfg, arm="bp"))
    out = []
    for l in range(len(params[0])):
        a = np.asarray(gp[0][l].layers[1].weight).ravel()
        b = np.asarray(gb[0][l].layers[1].weight).ravel()
        out.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)))
    if cfg.sparse.case == "C":
        a, b = np.asarray(gp[2]).ravel(), np.asarray(gb[2]).ravel()
        out.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)))
    return out


# ------------------------------------------------------------------ main

def analyse(run_dir, n_train, n_test, fast=False):
    cfg, params = load_run(run_dir)
    sp = cfg.sparse
    Xtr, ytr, Xte, yte = D.load_dataset(cfg.data)
    rtr = np.random.default_rng(0).choice(len(Xtr), min(n_train, len(Xtr)), replace=False)
    rte = np.random.default_rng(1).choice(len(Xte), min(n_test, len(Xte)), replace=False)
    xtr = D.eval_batch(jnp.asarray(Xtr[rtr]), cfg.data)
    xte = D.eval_batch(jnp.asarray(Xte[rte]), cfg.data)
    ytr_, yte_ = np.asarray(ytr)[rtr], np.asarray(yte)[rte]
    tag = os.path.basename(run_dir.rstrip("/"))
    rows = []

    def add(group, metric, value, condition="clean", t="", **kw):
        rows.append({"run": tag, "case": sp.case, "arm": cfg.arm, "seed": cfg.opt.seed,
                     "group": group, "metric": metric, "condition": condition, "t": t,
                     "value": value, **kw})

    T_dep = sp.T_test if sp.case == "C" else (None if sp.case == "B" else None)
    t_readouts = [("0", 0), ("T", T_dep)] if sp.case != "none" else [("ff", None)]

    # ---- C: robustness (every run) ---------------------------------------
    conds = [("clean", 0.0)] + [("noise", s) for s in NOISE] + [("occlusion", o) for o in OCCL]
    feats_tr = {lab: batched(lambda xb, t=t: readout(params, xb, cfg, t=t), xtr)
                for lab, t in t_readouts}
    clean_te = {lab: batched(lambda xb, t=t: readout(params, xb, cfg, t=t), xte)
                for lab, t in t_readouts}
    for kind, lvl in conds:
        xc = corrupt(xte, cfg, kind, lvl)
        cname = kind if kind == "clean" else f"{kind}={lvl:g}"
        res = {}
        for lab, t in t_readouts:
            z_te, zs_te, fs_te = batched(lambda xb, t=t: readout(params, xb, cfg, t=t), xc)
            pr = probe_pair(feats_tr[lab][0], ytr_, z_te, yte_)
            res[lab] = pr
            for k, v in pr.items():
                add("C", f"probe_{k}", v, cname, lab)
            if zs_te is not None and kind != "clean":
                zs0, fs0 = clean_te[lab][1], clean_te[lab][2]
                kap = float(np.mean(np.linalg.norm(zs_te - zs0, axis=1))
                            / (np.mean(np.linalg.norm(fs_te - fs0, axis=1)) + 1e-12))
                add("C", "kappa_s", kap, cname, lab)
        if "T" in res:
            for k in ("linear", "knn"):
                add("C", f"delta_inf_{k}", res["T"][k] - res["0"][k], cname, "T-0")

    if sp.case == "none":
        return rows

    # ---- B: inference curve, amortisation gap, dial, health ---------------
    for t in T_LIST:
        ztr_t = batched(lambda xb: readout(params, xb, cfg, t=t), xtr)[0]
        zte_t = batched(lambda xb: readout(params, xb, cfg, t=t), xte)[0]
        for k, v in probe_pair(ztr_t, ytr_, zte_t, yte_).items():
            add("B", f"curve_{k}", v, "clean", t)
    if sp.case == "B":
        z_closed = batched(lambda xb: readout(params, xb, cfg, t=None), xte)[0]
        z_t25 = batched(lambda xb: readout(params, xb, cfg, t=25), xte)[0]
        add("B", "iterated_prox_vs_closed_form_relerr",
            float(np.max(np.abs(z_t25 - z_closed)) / (np.max(np.abs(z_closed)) + 1e-12)))
    if sp.case == "C":
        mu = SP.lasso_mu(cfg)
        Dm = SP.get_D(params)
        for kind, lvl in [("clean", 0.0), ("noise", 0.1), ("occlusion", 0.25)]:
            xc = corrupt(xte[:1000], cfg, kind, lvl)
            _, e0 = SP.deploy(params, xc, cfg, t=0)
            _, eT = SP.deploy(params, xc, cfg)
            E0 = SP.lasso_energy(e0["f_s"], Dm, e0["a"], sp.gamma, mu)
            ET = SP.lasso_energy(eT["f_s"], Dm, eT["a"], sp.gamma, mu)
            add("B", "amortisation_gap_rel", float(jnp.mean((E0 - ET) / (ET + 1e-12))),
                kind if kind == "clean" else f"{kind}={lvl:g}")
        # the beta_g dial (test time only); beta_g = 0 must equal the plain forward
        z_plain = np.asarray(E_plain(params, xte[:500], cfg))
        for bg in BETA_G_DIAL:
            if bg == 0.0:
                z0 = batched(lambda xb: readout(params, xb, cfg, beta_g=0.0), xte[:500])[0]
                add("B", "dial_bg0_equals_plain_forward_relerr",
                    float(np.max(np.abs(z0 - z_plain)) / (np.max(np.abs(z_plain)) + 1e-12)))
            for kind, lvl in [("clean", 0.0), ("noise", 0.1)]:
                ztr_b = batched(lambda xb: readout(params, xb, cfg, beta_g=bg), xtr)[0]
                zte_b = batched(lambda xb: readout(params, xb, cfg, beta_g=bg),
                                corrupt(xte, cfg, kind, lvl))[0]
                for k, v in probe_pair(ztr_b, ytr_, zte_b, yte_).items():
                    add("B", f"dial_{k}", v, kind if kind == "clean" else f"{kind}={lvl:g}",
                        "", beta_g=bg)
        # group A for C: LASSO validity against t, clean and noisy
        for kind, lvl in [("clean", 0.0), ("noise", 0.1)]:
            for r in lasso_curve(params, corrupt(xte[:512], cfg, kind, lvl), cfg):
                for k in ("subopt_rel", "kkt_over_gamma", "active_frac"):
                    add("A", k, r[k], kind if kind == "clean" else f"{kind}={lvl:g}", r["t"])
    for k, v in SP.health(params, xte[:1024], cfg).items():
        add("B", f"health_{k}", v)

    # ---- A: training-relaxation fixed point; D: gradient agreement (PC arms)
    if cfg.arm == "pc" and not fast:
        for r in relax_fixed_point_residual(params, cfg, Xtr):
            for l, v in enumerate(r["per_layer_rel"]):
                add("A", "relax_fixed_point_rel", v, "train_batch", r["T"], layer=l)
    if not fast:
        for l, c in enumerate(grad_cosines(params, cfg, Xtr)):
            add("D", "cos_pc_vs_bp_same_net", c, "train_batch", "", layer=l)

    # ---- rescaling cheat + clip fraction, from the training log -----------
    log = json.load(open(os.path.join(run_dir, "log.json"))) if os.path.exists(
        os.path.join(run_dir, "log.json")) else []
    key = "grad_norm_enc" if sp.case == "C" else "grad_norm_preclip"
    g = [r[key] for r in log if key in r]
    if g:
        add("D", "clip_frac", sum(v > cfg.opt.grad_clip for v in g) / len(g))
    gd = [r["grad_norm_dict"] for r in log if "grad_norm_dict" in r]
    if gd:
        add("D", "clip_frac_dict", sum(v > cfg.opt.grad_clip for v in gd) / len(gd))
    for r in log:
        h = r.get("eval", {}).get("sparse")
        if h:
            for k in ("zs_rownorm", "Ws_fro", "Wnext_fro", "active_frac"):
                add("B", f"trajectory_{k}", h[k], "train_log", r["step"])
    return rows


def E_plain(params, x, cfg):
    """Plain forward pass of the MODIFIED network (no sparse step), via jpc."""
    import encoder as E
    return E.forward(params, x, cfg.enc)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--csv", default="diag_sparse.csv")
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--fast", action="store_true", help="skip groups A-relax and D")
    args = ap.parse_args()
    all_rows = []
    for rd in args.runs:
        if not os.path.exists(os.path.join(rd, "params.eqx")):
            print(f"[skip] {rd}: no params.eqx", flush=True)
            continue
        print(f"=== {rd}", flush=True)
        rows = analyse(rd, args.n_train, args.n_test, fast=args.fast)
        all_rows += rows
        for r in rows:
            if r["group"] == "C" and r["metric"] == "probe_linear":
                print(f"   C  {r['condition']:<16} t={r['t']:<3} linear={r['value']:.4f}", flush=True)
            if r["metric"].startswith("delta_inf"):
                print(f"   C  {r['condition']:<16} {r['metric']}={r['value']:+.4f}", flush=True)
    keys = sorted({k for r in all_rows for k in r},
                  key=lambda k: ["run", "case", "arm", "seed", "group", "metric", "condition",
                                 "t", "value"].index(k) if k in
                  ["run", "case", "arm", "seed", "group", "metric", "condition", "t", "value"] else 99)
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(all_rows)
    print(f"wrote {args.csv} ({len(all_rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
