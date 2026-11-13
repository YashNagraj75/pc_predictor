import glob, json, math, os, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp, equinox as eqx
jax.config.update("jax_default_matmul_precision", "highest")
import data as D, vit_ijepa as V, vit_epc as P, train_pred_vit as A
# Latent analysis of the stage-A ViT predictors (plan_latent_analysis.md; rev 2 after 3 reviewers).  Inference only, on the
# saved predictor.eqx of each 25-epoch run plus the untrained seed-0 init, on the fixed stage-A test suite (same masks,
# corruptions and rng streams as train_pred_vit.evaluate).  python latent_vit.py <root> <outdir> [depth ...]
GROUP = 50                                       # eval_masks: one mask shape per 50 consecutive images
CONDS = {"clean": ("none", 0.0), "ctx_drop25": ("none", 0.25), "ctx_drop50": ("none", 0.5), "noise": ("noise", 0.0), "square": ("square", 0.0)}
DBINS = (1.01, 1.5, 2.1, 3.1, 99.0)              # distance (patch units) of a target patch to the nearest visible context patch
CUTS = (0.5, 0.8, 0.95, 1.0)                     # PC bands: cumulative share of centred target variance
PC_ARMS = ("epc_alm_k4", "epc_alm_blk4", "epc_k4", "spc_u1")
SHA = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()


def dump(obj, path):                             # atomic json write
    """Writes obj as JSON to a temp file, then atomically replaces path with it."""
    tmp = path + ".tmp"; json.dump(obj, open(tmp, "w")); os.replace(tmp, path)


def readable(path):
    """Loads and parses a JSON file at the given path.
    Returns None if the file cannot be read or parsed."""
    try:
        return json.load(open(path))
    except Exception:
        return None


def forward_instr(p, ctx, cidx, cvalid, tidx, tvalid, g):
    """Runs the transformer over context and target tokens, tracking per-layer attention and context-mixing stats.
    Returns target predictions, pooled hidden states per layer, stacked attention metrics, and context-contribution ratios."""
    kvalid = jnp.concatenate([cvalid, tvalid]); NC, nt = cidx.shape[0], tidx.shape[0]
    pos = jnp.concatenate([cidx, tidx]); rc = jnp.stack([pos // g, pos % g], -1).astype(jnp.float32)
    dist = jnp.sqrt(jnp.sum((rc[:, None] - rc[None]) ** 2, -1))[NC:]                    # (nt, N) from target queries
    ck = jnp.concatenate([cvalid, jnp.zeros(nt, bool)]).astype(jnp.float32)
    w = tvalid.astype(jnp.float32); nw = jnp.maximum(jnp.sum(w), 1.0)
    unif = jnp.sum(dist * ck, -1) / jnp.maximum(jnp.sum(ck), 1.0)                        # (nt,)
    h = p.embed_(ctx, cidx, tidx); hs, att, con = [h], [], []
    for j in range(p.n_sub):
        b = p.blocks[j // 2]
        if j % 2 == 0:
            N, d = h.shape; H = b.heads
            q, k, v = jnp.split(b.qkv(b.ln1(h)), 3, -1)
            sh = lambda t: t.reshape(N, H, d // H)
            q, k, v = sh(q), sh(k), sh(v)
            s = jnp.einsum("nhe,mhe->hnm", q, k) / math.sqrt(d // H)
            s = jnp.where(kvalid[None, None, :], s, -1e30)
            Aw = jax.nn.softmax(s, -1)
            out = b.proj(jnp.einsum("hnm,mhe->nhe", Aw, v).reshape(N, d))
            At = Aw[:, NC:]                                                             # (H, nt, N)
            mc = jnp.sum(At * ck, -1)
            dcw = jnp.sum(w * jnp.sum(At * ck * dist[None], -1), -1) / jnp.maximum(jnp.sum(w * mc, -1), 1e-12)
            en = -jnp.sum(jnp.where(kvalid[None, None], At * jnp.log(jnp.maximum(At, 1e-30)), 0.0), -1)
            att.append(jnp.stack([jnp.sum(mc * w, -1) / nw, dcw, jnp.full((H,), jnp.sum(unif * w) / nw), jnp.sum(en * w, -1) / nw], -1))
            cc = jnp.einsum("hnm,mhe->nhe", At * ck[None, None], v).reshape(nt, d) @ b.proj.W.T * p.scale
            con.append(jnp.sum(w * jnp.linalg.norm(cc, axis=-1) / jnp.maximum(jnp.linalg.norm(h[NC:], axis=-1), 1e-12)) / nw)
        else:
            out = b.mlp(h)
        h = h + p.scale * out; hs.append(h)
    pooled = jnp.stack([jnp.sum(x[NC:] * w[:, None], 0) / nw for x in hs])
    return p.head(h, nt), pooled, jnp.stack(att), jnp.stack(con)


def make_fns(pred, g):
    """Builds and JIT-compiles batched helper functions for a predictor: forward predictions, Jacobian top singular values, gain, and single-instance prediction.
    Estimates the top singular value of d(valid predictions)/d(valid context) via power iteration at 20 and 40 steps."""
    one = lambda c, ci, cv, ti, tv: forward_instr(pred, c, ci, cv, ti, tv, g)
    batch = eqx.filter_jit(jax.vmap(lambda c, ci, cv, ti, tv: jax.vmap(lambda t1, t2: one(c, ci, cv, t1, t2))(ti, tv)))

    def jac_top(c, ci, cv, ti, tv, key):         # top singular value of d(valid predictions)/d(valid context): iters 20 and 40
        """Estimates the top singular value of d(valid predictions)/d(valid context) via power iteration.
        Returns stacked norms at 20 and 40 iterations plus the prediction and context norms."""
        f = lambda z: pred(z, ci, cv, ti, tv) * tv[:, None]
        cm = cv[:, None].astype(c.dtype)
        v = jax.random.normal(key, c.shape) * cm; v = v / jnp.linalg.norm(v)
        def it(_, v):
            """Computes one power-iteration step for estimating the top singular vector of f's Jacobian at c.
            Applies the Jacobian via jvp, its transpose via vjp masked by cm, then renormalizes the result."""
            _, Jv = jax.jvp(f, (c,), (v,)); w = jax.vjp(f, c)[1](Jv)[0] * cm
            return w / (jnp.linalg.norm(w) + 1e-30)
        v20 = jax.lax.fori_loop(0, 20, it, v); v40 = jax.lax.fori_loop(0, 20, it, v20)
        s = lambda u: jnp.linalg.norm(jax.jvp(f, (c,), (u,))[1])
        return jnp.stack([s(v20), s(v40), jnp.linalg.norm(f(c)), jnp.linalg.norm(c * cm)])
    jt = eqx.filter_jit(jax.vmap(jac_top))
    gn = eqx.filter_jit(jax.vmap(lambda c, ci, cv, ti, tv: P.gain(pred, c, ci, cv, ti, tv)))
    pr = eqx.filter_jit(lambda c, ci, cv, ti, tv: A.predict(pred, c, ci, cv, ti, tv))
    return batch, jt, gn, pr


def nearest_ctx_dist(cidx, cvalid, tidx, tvalid, g):   # (B, K, NT) distance of each target patch to the nearest valid ctx patch
    """Computes, for each target patch, the Chebyshev-free Euclidean grid distance to the nearest valid context patch.
    Returns NaN for invalid target patches and ignores invalid context patches when taking the minimum."""
    cr, cc = cidx // g, cidx % g; tr, tc = tidx // g, tidx % g
    d = np.sqrt((tr[..., None] - cr[:, None, None, :]) ** 2 + (tc[..., None] - cc[:, None, None, :]) ** 2)
    d = np.where(cvalid[:, None, None, :], d, np.inf).min(-1)
    return np.where(tvalid, d, np.nan)


def ctx_dist_to_targets(cidx, cvalid, tidx, tvalid, g):   # (B, NC) distance of each valid ctx patch to the nearest valid target
    """Computes, for each valid context patch, the Chebyshev-like grid distance to the nearest valid target patch.
    Returns NaN for invalid context patches and ignores invalid target patches when taking the minimum."""
    cr, cc = cidx // g, cidx % g; T = tidx.reshape(len(tidx), -1); Tv = tvalid.reshape(len(tidx), -1)
    d = np.sqrt((cr[:, :, None] - (T // g)[:, None]) ** 2 + (cc[:, :, None] - (T % g)[:, None]) ** 2)
    d = np.where(Tv[:, None], d, np.inf).min(-1)
    return np.where(cvalid, d, np.nan)


def intervention_masks(rng, cidx, cvalid, tidx, tvalid, g, keep_min=4):
    """Builds three context-patch masks (near-target removed, random-matched, far-removed) for an intervention ablation.
    Returns the masks dict alongside the per-batch (near, far) removal counts, keeping at least keep_min context patches."""
    dct = ctx_dist_to_targets(cidx, cvalid, tidx, tvalid, g); out = {k: cvalid.copy() for k in ("near", "far", "random")}; n_rm = []
    for b in range(len(cidx)):
        val = np.where(cvalid[b])[0]; near = val[dct[b, val] <= 1.5]; far = val[dct[b, val] > 2.1]
        n = max(min(len(near), len(val) - keep_min), 0); nf = min(n, len(far)); n_rm.append((n, nf))
        if n > 0:
            out["near"][b, rng.choice(near, n, replace=False)] = False
            out["random"][b, rng.choice(val, n, replace=False)] = False
        if nf > 0:
            out["far"][b, rng.choice(far, nf, replace=False)] = False
    return out, np.array(n_rm).reshape(-1, 2)


def residualise(S):                              # S (n_img, K, nodes, w): remove the per-(mask shape, block) mean over images
    """Removes the per-group mean over images from S, grouping images into blocks of size GROUP.
    Truncates S to a multiple of GROUP before computing the residual."""
    n = S.shape[0] // GROUP * GROUP; X = S[:n].reshape(n // GROUP, GROUP, *S.shape[1:]).astype(np.float64)
    return (X - X.mean(1, keepdims=True)).reshape(n, *S.shape[1:])


def linear_cka(X, Y):
    """Computes the linear centered kernel alignment (CKA) similarity between X and Y.
    Returns NaN if the denominator is zero."""
    X = np.asarray(X, np.float64); Y = np.asarray(Y, np.float64); X = X - X.mean(0); Y = Y - Y.mean(0)
    den = np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)
    return float(np.linalg.norm(Y.T @ X) ** 2 / den) if den > 0 else float("nan")


def spectrum(C):
    """Computes the eigenvalue spectrum of covariance matrix C, sorted in descending order.
    Returns a dict with effective rank, participation ratio, and top-10 eigenvalue fractions."""
    e = np.clip(np.linalg.eigvalsh(C)[::-1], 0, None); p = e / e.sum()
    return {"eff_rank": float(np.exp(-np.sum(p[p > 0] * np.log(p[p > 0])))), "part_ratio": float(e.sum() ** 2 / np.sum(e ** 2)),
            "top10_frac": [float(x) for x in p[:10]]}


def ridge_probe(Ftr, ytr, Fte, yte, lam=10.0):
    """Fits a ridge-regularized linear probe on standardized features to predict one-hot labels.
    Returns the classification accuracy of the probe on the test set."""
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-8; Xtr = np.c_[(Ftr - mu) / sd, np.ones(len(Ftr))]; Xte = np.c_[(Fte - mu) / sd, np.ones(len(Fte))]
    Y = np.eye(int(ytr.max()) + 1)[ytr]; W = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ Y)
    return float(np.mean((Xte @ W).argmax(1) == yte))


def node_probes(Ste, Str, yte, ytr):             # per node: kNN (raw/centred) and ridge probe on position-residualised states
    """Computes, for each node, position-residualised and normalized features, then runs kNN and ridge probes.
    Returns a list of per-node metric dicts with kNN (raw/centred) and ridge probe results."""
    Rte, Rtr = residualise(Ste), residualise(Str); out = []
    for j in range(Ste.shape[2]):
        mu, sd = Rtr[:, :, j].reshape(-1, Rtr.shape[-1]).mean(0), Rtr[:, :, j].reshape(-1, Rtr.shape[-1]).std(0) + 1e-8
        Fte = ((Rte[:, :, j] - mu) / sd).mean(1); Ftr = ((Rtr[:, :, j] - mu) / sd).mean(1)
        r = V.knn_both(Ftr, ytr[:len(Ftr)], Fte, yte[:len(Fte)]); r["ridge"] = ridge_probe(Ftr, ytr[:len(Ftr)], Fte, yte[:len(Fte)]); out.append(r)
    return out


def target_basis(enc, tenc, vc, Xev, masks, bs):
    """Computes the centred covariance of clean layer-normed targets across all masked batches.
    Returns its 191 eigenvectors sorted descending, with the all-ones "off-manifold" direction appended last."""
    S1, S2, n = np.zeros(vc.dim), np.zeros((vc.dim, vc.dim)), 0
    for i, (ci, cv, ti, tv) in enumerate(masks):
        Y = np.asarray(V.targets_of(tenc, jnp.asarray(V.standardise(Xev[i * bs:(i + 1) * bs])), jnp.asarray(ti), vc), np.float64)[tv]
        S1 += Y.sum(0); S2 += Y.T @ Y; n += len(Y)
    return basis_from_moments(S1, S2, n, vc.dim)


def basis_from_moments(S1, S2, n, dim):
    """Computes an orthonormal basis from accumulated first/second moments via PCA on the projected covariance.
    Returns the basis, eigenvalues, mean, and frequency-band index ranges derived from cumulative variance cutoffs."""
    mu = S1 / n; C = S2 / n - np.outer(mu, mu); one = np.ones(dim) / math.sqrt(dim); Pi = np.eye(dim) - np.outer(one, one)
    e, U = np.linalg.eigh(Pi @ C @ Pi); U = U[:, ::-1][:, :dim - 1]; e = e[::-1][:dim - 1]
    U = np.linalg.qr(U - np.outer(one, one @ U))[0]; U = np.c_[U, one]
    cum = np.cumsum(e) / e.sum(); ends = [min(int(np.searchsorted(cum, c - 1e-12)) + 1, dim - 1) for c in CUTS]
    bands = [(0 if k == 0 else ends[k - 1], ends[k]) for k in range(len(ends))] + [(dim - 1, dim)]
    return U, e, mu, bands


def noise_split(ac, NR):                         # NR columns ||dP||^2, ||Pc||^2, dP.Pc, dP.(Pc - Y): (var term, cross term)
    """Computes variance and cross-noise terms by scaling NR columns with ac.
    Returns a tuple (variance_term, cross_term) derived from ||dP||^2, ||Pc||^2, and dot products."""
    return ac ** 2 * NR[:, 0], 2 * ac * (ac * NR[:, 2] - (NR[:, 2] - NR[:, 3]))


def analyse(run, pred, enc, tenc, vc, Xev, Xknn, U, bands, states_path, bs=200):
    """Runs evaluation conditions, interventions, and sensitivity analyses on the latent ViT predictor.
    Returns a results dict and a per-image metrics dict, and saves clean hidden states to states_path."""
    g = vc.grid; masks = A.eval_masks(len(Xev), vc, bs); batch, jt, gn, pr = make_fns(pred, g)
    res = {"run": run, "conds": {}}; per = {}; nb_ = len(bands)
    Sclean, Pclean, ctx_clean = [], [], []
    for name, (kind, frac) in CONDS.items():
        r2 = np.random.default_rng(A.EVAL_SEED + 1)
        rows, bandE, binE, binY, binN, binE0, binY0, attS, conS = [], [], [], [], [], [], [], [], []
        comp = np.zeros((7, vc.dim)); CP = np.zeros((vc.dim, vc.dim)); sP = np.zeros(vc.dim); nn = 0
        noise_rows = []
        for i, (ci0, cv0, ti, tv) in enumerate(masks):
            Xb = Xev[i * bs:(i + 1) * bs]; ci, cv = ci0, cv0
            if frac > 0:
                ci, cv = A.drop_context(r2, ci0, cv0, frac)
            Xc = A.corrupt(r2, Xb, kind)
            J = lambda x: jnp.asarray(x)
            ctx, Y = A.encode(enc, tenc, J(V.standardise(Xc)), J(V.standardise(Xb)), J(ci), J(cv), J(ti), vc)
            Pr, pooled, att, con = batch(ctx, J(ci), J(cv), J(ti), J(tv))
            if i == 0:
                ref = pr(ctx, J(ci), J(cv), J(ti), J(tv)); m_ = J(tv)[..., None]
                res.setdefault("instr_vs_model_rel", {})[name] = float(jnp.max(jnp.abs((ref - Pr) * m_)) / jnp.max(jnp.abs(ref * m_)))
            Pr, Yn = np.asarray(Pr, np.float64), np.asarray(Y, np.float64); m = tv[..., None].astype(np.float64)
            rows.append(A.per_image(Pr, Y, tv))
            Pp, Yp = (Pr * m) @ U, (Yn * m) @ U; Ep = Pp - Yp; vm = tv.astype(np.float64)
            comp += np.stack([vm.sum() * np.ones(vc.dim), Pp.sum((0, 1, 2)), Yp.sum((0, 1, 2)), (Pp ** 2).sum((0, 1, 2)), (Yp ** 2).sum((0, 1, 2)),
                              (Pp * Yp).sum((0, 1, 2)), (Ep ** 2).sum((0, 1, 2))])
            bandE.append(np.stack([(Ep[..., a:b] ** 2).sum((1, 2, 3)) for a, b in bands], 1))
            e2 = ((Pr - Yn) ** 2).sum(-1); y2 = (Yn ** 2).sum(-1)
            for dd, E_, Y_, N_ in ((nearest_ctx_dist(ci, cv, ti, tv, g), binE, binY, binN), (nearest_ctx_dist(ci0, cv0, ti, tv, g), binE0, binY0, None)):
                lo = 0.0; be, by, bn = [], [], []
                for hi in DBINS:
                    sel = (dd > lo) & (dd <= hi); be.append(np.where(sel, e2, 0).sum((1, 2))); by.append(np.where(sel, y2, 0).sum((1, 2))); bn.append(sel.sum((1, 2))); lo = hi
                E_.append(np.stack(be, 1)); Y_.append(np.stack(by, 1))
                if N_ is not None: N_.append(np.stack(bn, 1))
            attS.append(np.asarray(att, np.float64).mean(1)); conS.append(np.asarray(con, np.float64).mean(1))
            Pf = Pr[tv] - Pr[tv].mean(-1, keepdims=True); CP += Pf.T @ Pf; sP += Pf.sum(0); nn += len(Pf)
            if name == "clean":
                Sclean.append(np.asarray(pooled, np.float32)); Pclean.append((Pr * m).astype(np.float32)); ctx_clean.append(np.asarray(ctx))
            if name == "noise":                   # change of the prediction under noise; context shift in embedding space
                Pc = Pclean[i].astype(np.float64); dP = Pr * m - Pc
                cm = cv[..., None]; dc = np.sqrt((((np.asarray(ctx, np.float64) - ctx_clean[i]) * cm) ** 2).sum((1, 2)))
                noise_rows.append(np.stack([(dP ** 2).sum((1, 2, 3)), (Pc ** 2).sum((1, 2, 3)), (dP * Pc).sum((1, 2, 3)),
                                            (dP * (Pc - Yn * m)).sum((1, 2, 3)), dc], 1))
                res.setdefault("_dPband", []).append(np.stack([((dP @ U)[..., a:b] ** 2).sum((1, 2, 3)) for a, b in bands], 1))
        A_ = np.concatenate(rows); S = A_.sum(0); a_ = S[2] / S[3]
        bE = np.concatenate(bandE); dE, dY, dN, dE0, dY0 = (np.concatenate(x) for x in (binE, binY, binN, binE0, binY0))
        n_, sp, sy, spp, syy, spy, see = comp
        vy = syy / n_ - (sy / n_) ** 2; vp = spp / n_ - (sp / n_) ** 2; cpy = spy / n_ - sp * sy / n_ ** 2
        corr = cpy / np.sqrt(np.maximum(vp * vy, 1e-300)); slope = cpy / np.maximum(vy, 1e-300)
        bstat = []
        for a, b in bands:
            vyb, vpb, cb = vy[a:b].sum(), vp[a:b].sum(), cpy[a:b].sum()
            bstat.append({"comps": [a, b], "err_share": float(see[a:b].sum() / see.sum()), "err_over_target_var": float(see[a:b].sum() / n_[0] / max(vyb, 1e-300)),
                          "pred_var_over_target_var": float(vpb / max(vyb, 1e-300)), "corr": float(cb / math.sqrt(max(vpb * vyb, 1e-300))),
                          "slope": float(cb / max(vyb, 1e-300)), "target_var_share": float(vyb / vy.sum())})
        safe = lambda num, den: [float(x) if d > 0 else float("nan") for x, d in zip(num, den)]
        C = res["conds"][name] = {
            "nmse": float(S[0] / S[1]), "nmse_cal": float((S[3] * a_ ** 2 - 2 * a_ * S[2] + S[1]) / S[1]), "alpha_star": float(a_),
            "bands": bstat, "band_sum_check": float(bE.sum() / S[0]),
            "per_comp": {"err": (see / see.sum()).tolist(), "corr": corr.tolist(), "slope": slope.tolist()},
            "dist_nmse": safe(dE.sum(0) / np.maximum(dY.sum(0), 1e-300), dY.sum(0)), "dist_token_share": (dN.sum(0) / dN.sum()).tolist(),
            "dist_sum_check": float(dE.sum() / S[0]),
            "dist_nmse_by_clean_dist": safe(dE0.sum(0) / np.maximum(dY0.sum(0), 1e-300), dY0.sum(0)),
            "attn_layer_mean": np.concatenate(attS).mean((0, 2)).tolist(), "attn_heads_sorted_by_dist": np.sort(np.concatenate(attS).mean(0)[:, :, 1], -1).tolist(),
            "ctx_contribution": np.concatenate(conS).mean(0).tolist(),
            "spec_pred": spectrum(CP / nn - np.outer(sP / nn, sP / nn))}
        per[name] = {"err": A_[:, 0], "y2": A_[:, 1], "py": A_[:, 2], "p2": A_[:, 3], "band_err": bE, "dist_err": dE, "dist_y2": dY}
        if name == "noise":
            NR = np.concatenate(noise_rows); ac = res["conds"]["clean"]["alpha_star"]; Sc = per["clean"]
            Ecal_c = Sc["y2"] - 2 * ac * Sc["py"] + ac ** 2 * Sc["p2"]; Ecal_n = A_[:, 1] - 2 * ac * A_[:, 2] + ac ** 2 * A_[:, 3]
            # exact split: Ecal_n - Ecal_c = ac^2 ||dP||^2 + 2 ac dP.(ac Pc - Y), dP = Pn - Pc (both masked to valid targets)
            var_term, cross = noise_split(ac, NR)
            res.setdefault("_split_check", float(np.max(np.abs((Ecal_n - Ecal_c) - var_term - cross)) / max(float(np.max(np.abs(Ecal_n - Ecal_c))), 1e-30)))
            dPb = np.concatenate(res.pop("_dPband"))
            C["noise"] = {"nmse_cal_cleanalpha": float(Ecal_n.sum() / A_[:, 1].sum()), "clean_alpha": float(ac),
                          "delta_cal_nmse": float((Ecal_n - Ecal_c).sum() / A_[:, 1].sum()),
                          "delta_from_prediction_change": float(var_term.sum() / A_[:, 1].sum()), "delta_from_cross_term": float(cross.sum() / A_[:, 1].sum()),
                          "split_check_rel": res.pop("_split_check"),
                          "rel_change": float(NR[:, 0].sum() / NR[:, 1].sum()), "scale_med": float(np.median(NR[:, 2] / NR[:, 1] + 1)),
                          "change_band_share": (dPb.sum(0) / dPb.sum()).tolist(),
                          "noise_gain_med": float(np.median(np.sqrt(NR[:, 0]) / NR[:, 4])), "noise_gain_x_alpha_med": float(ac * np.median(np.sqrt(NR[:, 0]) / NR[:, 4]))}
            per["noise_cal"] = np.stack([Ecal_c, Ecal_n, var_term], 1); per["noise_dP2_P2_dc"] = NR[:, [0, 1, 4]]
    # encoder-fixed context interventions on the clean images
    r3 = np.random.default_rng(A.EVAL_SEED + 7); iv = {k: [] for k in ("near", "far", "random")}; nrm = []
    for i, (ci, cv, ti, tv) in enumerate(masks):
        cvs, n_rm = intervention_masks(r3, ci, cv, ti, tv, g); nrm.append(n_rm); c_ = jnp.asarray(ctx_clean[i])
        Yn = np.asarray(V.targets_of(tenc, jnp.asarray(V.standardise(Xev[i * bs:(i + 1) * bs])), jnp.asarray(ti), vc), np.float64)
        for k in iv:
            Pr = np.asarray(pr(c_, jnp.asarray(ci), jnp.asarray(cvs[k]), jnp.asarray(ti), jnp.asarray(tv)), np.float64)
            iv[k].append(A.per_image(Pr, Yn, tv))
    yc = per["clean"]["y2"].sum(); ec = per["clean"]["err"].sum()
    res["interventions"] = {k: {"nmse": float(np.concatenate(v)[:, 0].sum() / yc), "delta": float((np.concatenate(v)[:, 0].sum() - ec) / yc)} for k, v in iv.items()}
    NRM = np.concatenate(nrm); n_ctx = np.concatenate([m_[1].sum(1) for m_ in masks])
    res["interventions"].update(mean_removed_near_random=float(NRM[:, 0].mean()), mean_removed_far=float(NRM[:, 1].mean()),
                                mean_visible_ctx=float(n_ctx.mean()), frac_images_removed0=float(np.mean(NRM[:, 0] == 0)))
    for k, v in iv.items(): per[f"iv_{k}"] = np.concatenate(v)[:, 0]
    # sensitivity on all images and blocks (clean), scale-normalised by the clean calibration and by the prediction norm
    ac = res["conds"]["clean"]["alpha_star"]; SJ = []
    for i, (ci, cv, ti, tv) in enumerate(masks):
        c_ = jnp.asarray(ctx_clean[i]); keys = jax.random.split(jax.random.PRNGKey(100 + i), len(c_))
        SJ.append(np.stack([np.asarray(jt(c_, jnp.asarray(ci), jnp.asarray(cv), jnp.asarray(ti[:, k]), jnp.asarray(tv[:, k]), keys)) for k in range(ti.shape[1])], 1))
    SJ = np.concatenate(SJ)                       # (n, K, 4): sigma@20, sigma@40, ||P||, ||c||
    sig = SJ[..., 1]; el = sig * SJ[..., 3] / np.maximum(SJ[..., 2], 1e-30)
    ci, cv, ti, tv = masks[0]; c_ = jnp.asarray(ctx_clean[0]); ga = np.asarray(gn(c_, jnp.asarray(ci), jnp.asarray(cv), jnp.asarray(ti[:, 0]), jnp.asarray(tv[:, 0])))
    res["sensitivity"] = {"jac_top_med": float(np.median(sig)), "jac_top_x_alpha_med": float(ac * np.median(sig)), "elasticity_med": float(np.median(el)),
                          "power_iter_rel_gap_med": float(np.median(np.abs(SJ[..., 1] - SJ[..., 0]) / np.maximum(SJ[..., 1], 1e-30))),
                          "power_iter_rel_gap_p95": float(np.percentile(np.abs(SJ[..., 1] - SJ[..., 0]) / np.maximum(SJ[..., 1], 1e-30), 95)),
                          "gain_med": float(np.median(ga)), "gain_max": float(np.max(ga))}
    per["jac_top"] = sig; per["elasticity"] = el
    # hidden states: test set (clean) to disk for CKA; reference set (training images, same mask sequence) for probes
    Ste = np.concatenate(Sclean); np.save(states_path + ".tmp.npy", Ste); os.replace(states_path + ".tmp.npy", states_path)
    Str, Fk, Fev = [], [], np.concatenate([A.pooled(Pc_, tv_) for Pc_, (_, _, _, tv_) in zip([p.astype(np.float64) for p in Pclean], masks)])
    for i in range(0, len(Xknn), bs):
        ci, cv, ti, tv = masks[(i // bs) % len(masks)]; Xb = Xknn[i:i + bs]
        ctx, _ = A.encode(enc, tenc, jnp.asarray(V.standardise(Xb)), jnp.asarray(V.standardise(Xb)), jnp.asarray(ci), jnp.asarray(cv), jnp.asarray(ti), vc)
        Pr, pooled, _, _ = batch(ctx, jnp.asarray(ci), jnp.asarray(cv), jnp.asarray(ti), jnp.asarray(tv))
        Str.append(np.asarray(pooled, np.float32)); Fk.append(A.pooled(np.asarray(Pr, np.float64), tv))
    res["_probe_inputs"] = (Ste, np.concatenate(Str), np.concatenate(Fk), Fev)
    return res, per


def main():
    """Runs the full analysis pipeline per depth: loads runs, analyses each predictor, and saves results.
    Computes per-node CKA between runs on residualised states with a shuffled control."""
    root, outd = sys.argv[1], sys.argv[2]; depths = [int(x) for x in sys.argv[3:]] or [10, 6]
    st = os.path.join(outd, "states"); os.makedirs(st, exist_ok=True)
    vc, enc, tenc, _ = A.load_encoders(os.path.join(root, "Q1-PhaseA", A.V0_DIR))
    Xtr, ytr, Xte, yte = D.load_cifar10(root=os.path.join(root, "data"), n_train=-1, n_test=2000, seed=0)
    Xknn, yknn = Xtr[:5000], ytr[:5000]; masks = A.eval_masks(len(Xte), vc, 200)
    bf = os.path.join(outd, "target_basis.npz")
    if os.path.exists(bf):
        z = np.load(bf); U, ev, mu, bands = z["U"], z["ev"], z["mu"], [tuple(x) for x in z["bands"]]
    else:
        U, ev, mu, bands = target_basis(enc, tenc, vc, Xte, masks, 200); np.savez(bf, U=U, ev=ev, mu=mu, bands=np.array(bands))
    def ctx_feats(X):
        """Computes pooled clean context embeddings in batches, independent of the predictor.
        Serves as an input-information floor for the probes."""
        F = []
        for i in range(0, len(X), 200):
            ci, cv, ti, tv = masks[(i // 200) % len(masks)]; Xb = X[i:i + 200]
            ctx, _ = A.encode(enc, tenc, jnp.asarray(V.standardise(Xb)), jnp.asarray(V.standardise(Xb)), jnp.asarray(ci), jnp.asarray(cv), jnp.asarray(ti), vc)
            c = np.asarray(ctx, np.float64) * cv[..., None]; F.append(c.sum(1) / cv.sum(1, keepdims=True))
        return np.concatenate(F)
    cf = os.path.join(outd, "ctx_floor.json")
    if not readable(cf):
        Ft, Fr = ctx_feats(Xte), ctx_feats(Xknn); r = V.knn_both(Fr, yknn, Ft, yte); r["ridge"] = ridge_probe(Fr, yknn, Ft, yte); dump(r, cf)
    SA = os.path.join(root, "Q1-PhaseA/runs_vit/stageA")
    for depth in depths:
        bp = {}
        for d in glob.glob(f"{SA}/bp_d{depth}_lr*_e25_full_s*"):
            if os.path.exists(d + "/report.json") and os.path.exists(d + "/predictor.eqx"):
                c = json.load(open(d + "/report.json"))["cfg"]; bp[(c["lr"], c["seed"])] = os.path.basename(d)
        lr_sel = {lr for (lr, s) in bp if s > 0}
        assert len(lr_sel) == 1, f"BP seeds 1-4 at more than one lr: {lr_sel}"; lr_sel = lr_sel.pop()
        assert {s for (lr, s) in bp if lr == lr_sel} == {0, 1, 2, 3, 4}, f"d{depth}: BP seeds at lr {lr_sel}: {sorted(bp)}"
        runs = [(bp[(lr_sel, s)], "bp", s) for s in range(5)]; pc_lrs = set()
        for arm in PC_ARMS:
            ds = [d for d in glob.glob(f"{SA}/{arm}_d{depth}_lr*_e25_full_s0_*") if os.path.exists(d + "/report.json") and os.path.exists(d + "/predictor.eqx")]
            assert len(ds) <= 1, ds
            if ds:
                runs.append((os.path.basename(ds[0]), arm, 0)); pc_lrs.add(json.load(open(ds[0] + "/report.json"))["cfg"]["lr"])
            elif arm != "spc_u1":
                raise SystemExit(f"missing {arm} d{depth}")
        for lr in sorted(pc_lrs - {lr_sel}):     # lr-matched BP control (seed 0) for any PC arm trained at another lr
            if (lr, 0) in bp: runs.append((bp[(lr, 0)], "bp_lrmatch", 0))
        runs.append((f"init_d{depth}_s0", "init", 0))
        print("RUNS", depth, lr_sel, [r[0] for r in runs], flush=True)
        for name, arm, seed in runs:
            f = os.path.join(outd, f"latent_{name}.json"); old = readable(f)
            if old and old.get("sha") == SHA and os.path.exists(os.path.join(st, name + ".npy")): continue
            t0 = time.time()
            if arm == "init":
                a = A.ACfg(arm="bp", depth=depth, seed=0); pred = A.make_predictor(vc, a); rep = None
            else:
                rep = json.load(open(os.path.join(SA, name, "report.json"))); cfg = rep["cfg"]
                a = A.ACfg(**{k: v for k, v in cfg.items() if k in A.ACfg.__dataclass_fields__})
                pred = eqx.tree_deserialise_leaves(os.path.join(SA, name, "predictor.eqx"), A.make_predictor(vc, a))
            res, per = analyse(name, pred, enc, tenc, vc, Xte, Xknn, U, bands, os.path.join(st, name + ".npy"))
            Ste, Str, Fk, Fev = res.pop("_probe_inputs")
            res["knn_pred"] = V.knn_both(Fk, yknn, Fev, yte[:len(Fev)]); res["node_probes"] = node_probes(Ste, Str, yte, yknn)
            res.update(arm=arm, depth=depth, seed=seed, lr=(a.lr if arm != "init" else None), sha=SHA, sec=round(time.time() - t0), bands=[list(b) for b in bands])
            if rep:
                dev = {c: abs(res["conds"][c]["nmse"] - rep["eval"][c]["nmse"]) / rep["eval"][c]["nmse"] for c in CONDS}
                dev["noise_cal_cleanalpha"] = abs(res["conds"]["noise"]["noise"]["nmse_cal_cleanalpha"] - rep["eval"]["noise"]["nmse_cal_cleanalpha"]) / rep["eval"]["noise"]["nmse_cal_cleanalpha"]
                dev["knn_pred"] = abs(res["knn_pred"]["knn"] - rep["eval"]["knn_pred"]["knn"])
                res["report_dev"] = dev; res["reproduces_report"] = max(v for k, v in dev.items() if k != "knn_pred") <= 1e-4 and dev["knn_pred"] <= 2e-3
            flat = {}
            for c, d_ in per.items():
                if isinstance(d_, dict): flat.update({f"{c}__{kk}": vv for kk, vv in d_.items()})
                else: flat[c] = d_
            np.savez_compressed(os.path.join(outd, f"perimg_{name}.npz"), **flat)
            dump(res, f); print("DONE", name, res.get("reproduces_report"), res["instr_vs_model_rel"]["clean"], res["sec"], flush=True)
        # CKA per node >= 1 on position-residualised states, every pair; image-shuffled control
        cf2 = os.path.join(outd, f"cka_d{depth}.json"); ck = (readable(cf2) or {}).get("pairs", {})
        names = [r[0] for r in runs if os.path.exists(os.path.join(st, r[0] + ".npy"))]
        cache = {n: residualise(np.load(os.path.join(st, n + ".npy"))).reshape(-1, *np.load(os.path.join(st, n + ".npy"), mmap_mode="r").shape[2:]).astype(np.float32)
                 for n in names}                  # (rows, nodes, w); all state files are kept (later passes add spc_u1)
        for i1 in range(len(names)):
            for i2 in range(i1 + 1, len(names)):
                key = f"{names[i1]}|{names[i2]}"
                if key not in ck:
                    X, Y = cache[names[i1]], cache[names[i2]]; ck[key] = [linear_cka(X[:, j], Y[:, j]) for j in range(1, X.shape[1])]
        X, Y = cache[runs[0][0]], cache[runs[1][0]]; perm = np.random.default_rng(0).permutation(len(Y))
        dump({"pairs": ck, "nodes": "1..2L (node 0 = embedding, removed by residualisation)", "shuffled_control": [linear_cka(X[:, j], Y[perm, j]) for j in range(1, X.shape[1])],
              "rows": "(image, target block) states pooled over valid target tokens, minus the per-(mask shape, block) mean over images"}, cf2)
        print("CKA", depth, len(ck), flush=True); del cache
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
