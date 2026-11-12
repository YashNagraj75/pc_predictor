import glob, json, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp, equinox as eqx
jax.config.update("jax_default_matmul_precision", "highest")
import data as D, vit_ijepa as V, vit_epc as P, train_pred_vit as A
# Latent analysis of the stage-A ViT predictors (plan_latent_analysis.md, user-approved 2026-11-12).  Inference only, on the
# saved predictor.eqx of each 25-epoch run; the fixed stage-A test suite (same masks, corruptions, rng streams as evaluate).
# python latent_vit.py <root> <outdir> [depth ...]   -> <outdir>/latent_<run>.json, cka_d<D>.json, states in <outdir>/states/
BANDS = ((0, 8), (8, 32), (32, 192))             # principal-component bands of the target embeddings (dim 192)
DBINS = (1.01, 1.5, 2.1, 3.1, 99.0)              # distance (patch units) of a target patch to the nearest visible context patch
CONDS = {"clean": ("none", 0.0), "ctx_drop25": ("none", 0.25), "ctx_drop50": ("none", 0.5), "noise": ("noise", 0.0), "square": ("square", 0.0)}
ARMS_LA = ("bp", "epc_alm_k4", "epc_alm_blk4", "epc_k4", "spc_u1")


# Instrumented forward pass, identical to Predictor.__call__: node states h_0 (embedding) .. h_2L, the prediction, and per
# attention layer the attention of the valid target queries: mass on context keys, mean grid distance to context keys
# (attention renormalised over context), entropy over all valid keys.  One image, one target block.
def forward_instr(p, ctx, cidx, cvalid, tidx, tvalid, g):
    kvalid = jnp.concatenate([cvalid, tvalid]); NC, nt = cidx.shape[0], tidx.shape[0]
    pos = jnp.concatenate([cidx, tidx]); rc = jnp.stack([pos // g, pos % g], -1).astype(jnp.float32)
    dist = jnp.sqrt(jnp.sum((rc[:, None] - rc[None]) ** 2, -1))                        # (N, N) grid distance
    ctxkey = jnp.concatenate([cvalid, jnp.zeros(nt, bool)])
    h = p.embed_(ctx, cidx, tidx); hs, att = [h], []
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
            At = Aw[:, NC:]                                                             # target queries (H, nt, N)
            mc = jnp.sum(At * ctxkey[None, None], -1)
            dc = jnp.sum(At * ctxkey[None, None] * dist[NC:][None], -1) / jnp.maximum(mc, 1e-12)
            en = -jnp.sum(jnp.where(kvalid[None, None], At * jnp.log(jnp.maximum(At, 1e-30)), 0.0), -1)
            w = tvalid[None].astype(jnp.float32); nw = jnp.maximum(jnp.sum(w), 1.0)
            att.append(jnp.stack([jnp.sum(mc * w, -1) / nw, jnp.sum(dc * w, -1) / nw, jnp.sum(en * w, -1) / nw], -1))   # (H, 3)
        else:
            out = b.mlp(h)
        h = h + p.scale * out; hs.append(h)
    Pr = p.head(h, nt)
    tw = tvalid.astype(jnp.float32)[:, None]; nv = jnp.maximum(jnp.sum(tw), 1.0)
    pooled = jnp.stack([jnp.sum(x[NC:] * tw, 0) / nv for x in hs])                     # (n_nodes, w) over valid targets
    return Pr, pooled, jnp.stack(att)                                                   # att (L, H, 3)


def make_fns(pred, g):
    one = lambda c, ci, cv, ti, tv: forward_instr(pred, c, ci, cv, ti, tv, g)
    batch = eqx.filter_jit(jax.vmap(lambda c, ci, cv, ti, tv: jax.vmap(lambda t1, t2: one(c, ci, cv, t1, t2))(ti, tv)))

    def jac_top(c, ci, cv, ti, tv, iters=20):    # top singular value of d(valid predictions)/d(valid context rows)
        f = lambda z: pred(z, ci, cv, ti, tv) * tv[:, None]
        cm = cv[:, None].astype(c.dtype)
        v = jnp.broadcast_to(jnp.linspace(1.0, 2.0, c.shape[-1]), c.shape) * cm; v = v / jnp.linalg.norm(v)
        def it(_, v):
            _, Jv = jax.jvp(f, (c,), (v,)); w = jax.vjp(f, c)[1](Jv)[0] * cm
            return w / (jnp.linalg.norm(w) + 1e-30)
        v = jax.lax.fori_loop(0, iters, it, v)
        return jnp.linalg.norm(jax.jvp(f, (c,), (v,))[1])

    def rand_sens(c, ci, cv, ti, tv, key):       # ||P(c + d) - P(c)|| / ||d||, d = 1% of the rms context entry, valid rows
        f = lambda z: pred(z, ci, cv, ti, tv) * tv[:, None]
        cm = cv[:, None].astype(c.dtype); rms = jnp.sqrt(jnp.sum((c * cm) ** 2) / jnp.maximum(jnp.sum(cm) * c.shape[-1], 1.0))
        d = 0.01 * rms * jax.random.normal(key, c.shape) * cm
        return jnp.linalg.norm(f(c + d) - f(c)) / jnp.linalg.norm(d)
    jt = eqx.filter_jit(jax.vmap(jac_top)); rs = eqx.filter_jit(jax.vmap(rand_sens))
    gn = eqx.filter_jit(jax.vmap(lambda c, ci, cv, ti, tv: P.gain(pred, c, ci, cv, ti, tv)))
    return batch, jt, rs, gn


def nearest_ctx_dist(cidx, cvalid, tidx, tvalid, g):   # numpy: (B, K, NT) distance of each target patch to nearest valid ctx
    cr, cc = cidx // g, cidx % g; tr, tc = tidx // g, tidx % g
    d = np.sqrt((tr[..., None] - cr[:, None, None, :]) ** 2 + (tc[..., None] - cc[:, None, None, :]) ** 2)
    d = np.where(cvalid[:, None, None, :], d, np.inf).min(-1)
    return np.where(tvalid, d, np.nan)


def linear_cka(X, Y):                            # rows = samples; centred linear CKA
    X = X - X.mean(0); Y = Y - Y.mean(0)
    return float(np.linalg.norm(Y.T @ X) ** 2 / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)))


def spectrum(C):                                 # eigenvalues (desc) of a covariance, effective rank exp(H), participation ratio
    e = np.clip(np.linalg.eigvalsh(C)[::-1], 0, None); p = e / e.sum()
    return {"eff_rank": float(np.exp(-np.sum(p[p > 0] * np.log(p[p > 0])))), "part_ratio": float(e.sum() ** 2 / np.sum(e ** 2)),
            "top10_frac": [float(x) for x in p[:10]]}


def target_basis(tenc, vc, Xev, masks, bs):      # PCA basis of the (uncentred) second moment of the clean targets
    M, n = np.zeros((vc.dim, vc.dim)), 0
    for i, (ci, cv, ti, tv) in enumerate(masks):
        Xb = jnp.asarray(V.standardise(Xev[i * bs:(i + 1) * bs]))
        Y = np.asarray(V.targets_of(tenc, Xb, jnp.asarray(ti), vc), np.float64)[tv]
        M += Y.T @ Y; n += len(Y)
    e, U = np.linalg.eigh(M / n); return U[:, ::-1], e[::-1]


def analyse(run, pred, enc, tenc, vc, Xev, yev, Xknn, yknn, U, out_states, bs=200, n_jac=200):
    g = vc.grid; masks = A.eval_masks(len(Xev), vc, bs); batch, jt, rs, gn = make_fns(pred, g)
    res = {"run": run, "conds": {}}; per_img = {}
    for name, (kind, frac) in CONDS.items():
        r2 = np.random.default_rng(A.EVAL_SEED + 1)
        rows, bandE, bandY, binE, binY, attS, Pl, Pc = [], [], [], [], [], [], [], []
        CP, CY, sP, sY, nn = np.zeros((vc.dim, vc.dim)), np.zeros((vc.dim, vc.dim)), np.zeros(vc.dim), np.zeros(vc.dim), 0
        for i, (ci, cv, ti, tv) in enumerate(masks):
            Xb = Xev[i * bs:(i + 1) * bs]
            if frac > 0:
                ci, cv = A.drop_context(r2, ci, cv, frac)
            Xc = A.corrupt(r2, Xb, kind)
            ctx, Y = A.encode(enc, tenc, jnp.asarray(V.standardise(Xc)), jnp.asarray(V.standardise(Xb)), jnp.asarray(ci), jnp.asarray(cv), jnp.asarray(ti), vc)
            Pr, pooled, att = batch(ctx, jnp.asarray(ci), jnp.asarray(cv), jnp.asarray(ti), jnp.asarray(tv))
            if i == 0 and name == "clean":       # the instrumented pass must equal the model's own forward pass
                ref = A.predict(pred, ctx, jnp.asarray(ci), jnp.asarray(cv), jnp.asarray(ti), jnp.asarray(tv))
                res["instr_vs_model_maxabs"] = float(jnp.max(jnp.abs((ref - Pr) * jnp.asarray(tv)[..., None])))
            Pr, Yn = np.asarray(Pr, np.float64), np.asarray(Y, np.float64); m = tv[..., None].astype(np.float64)
            rows.append(A.per_image(Pr, Y, tv))
            Ep, Yp = ((Pr - Yn) * m) @ U, (Yn * m) @ U                                  # (B, K, NT, 192) in the PC basis
            bandE.append(np.stack([(Ep[..., a:b] ** 2).sum((1, 2, 3)) for a, b in BANDS], 1))
            bandY.append(np.stack([(Yp[..., a:b] ** 2).sum((1, 2, 3)) for a, b in BANDS], 1))
            dd = nearest_ctx_dist(ci, cv, ti, tv, g); e2 = ((Pr - Yn) ** 2).sum(-1); y2 = (Yn ** 2).sum(-1)
            lo = 0.0; be, by = [], []
            for hi in DBINS:
                sel = (dd > lo) & (dd <= hi); be.append(np.where(sel, e2, 0).sum((1, 2))); by.append(np.where(sel, y2, 0).sum((1, 2))); lo = hi
            binE.append(np.stack(be, 1)); binY.append(np.stack(by, 1))
            attS.append(np.asarray(att, np.float64).mean(1))                           # (B, L, H, 3), mean over blocks
            Pv, Yv = Pr[tv], Yn[tv]; CP += Pv.T @ Pv; CY += Yv.T @ Yv; sP += Pv.sum(0); sY += Yv.sum(0); nn += len(Pv)
            if name in ("clean", "noise"):
                Pl.append((Pr * m).astype(np.float32))
            if name == "clean":
                tw = tv.sum(-1).astype(np.float64); pz = np.asarray(pooled, np.float64)          # (B, K, nodes, w)
                Pc.append((pz * tw[..., None, None]).sum(1) / np.maximum(tw.sum(1), 1)[:, None, None])  # per image (B, nodes, w)
                np.save(os.path.join(out_states, f"{run}_b{i:02d}.npy"), pz.reshape(-1, pz.shape[2], pz.shape[3]).astype(np.float32))
        A_ = np.concatenate(rows); S = A_.sum(0)
        bE, bY, dE, dY = (np.concatenate(x) for x in (bandE, bandY, binE, binY))
        CPc, CYc = CP / nn - np.outer(sP / nn, sP / nn), CY / nn - np.outer(sY / nn, sY / nn)
        res["conds"][name] = {"nmse": float(S[0] / S[1]), "band_nmse": [float(x) for x in bE.sum(0) / bY.sum(0)],
                              "band_err_share": [float(x) for x in bE.sum(0) / bE.sum()],
                              "band_sum_check": float(bE.sum() / S[0]), "dist_nmse": [float(x) for x in dE.sum(0) / np.maximum(dY.sum(0), 1e-30)],
                              "dist_count_share": [float(x) for x in dY.sum(0) / dY.sum()], "dist_sum_check": float(dE.sum() / S[0]),
                              "attn": np.concatenate(attS).mean(0).tolist(), "spec_pred": spectrum(CPc), "spec_target": spectrum(CYc),
                              "pred_var_by_band": [float(np.trace(U[:, a:b].T @ CPc @ U[:, a:b]) / np.trace(U[:, a:b].T @ CYc @ U[:, a:b])) for a, b in BANDS]}
        per_img[name] = {"err": A_[:, 0], "y2": A_[:, 1], "band_err": bE, "band_y2": bY, "dist_err": dE, "dist_y2": dY}
        if name == "clean":
            Pclean, PCpool = np.concatenate(Pl), np.concatenate(Pc)
        if name == "noise":
            Pn = np.concatenate(Pl); dot = (Pn * Pclean).sum((1, 2, 3)); pc2 = (Pclean ** 2).sum((1, 2, 3))
            a = dot / pc2; orth = ((Pn - a[:, None, None, None] * Pclean) ** 2).sum((1, 2, 3))
            res["noise_response"] = {"rel_change": float(((Pn - Pclean) ** 2).sum() / pc2.sum()), "scale_med": float(np.median(a)),
                                     "orth_share": float(orth.sum() / ((Pn - Pclean) ** 2).sum()), "norm_ratio": float(np.sqrt((Pn ** 2).sum() / pc2.sum()))}
            per_img["noise_resp"] = ((Pn - Pclean) ** 2).sum((1, 2, 3)) / pc2
    # sensitivity on the first n_jac clean test images (target block 0)
    ci, cv, ti, tv = masks[0]; Xb = Xev[:n_jac]
    ctx, Y = A.encode(enc, tenc, jnp.asarray(V.standardise(Xb)), jnp.asarray(V.standardise(Xb)), jnp.asarray(ci), jnp.asarray(cv), jnp.asarray(ti), vc)
    a4 = (ctx, jnp.asarray(ci), jnp.asarray(cv), jnp.asarray(ti[:, 0]), jnp.asarray(tv[:, 0]))
    sj = np.asarray(jt(*a4)); sr = np.asarray(rs(*a4, jax.random.split(jax.random.PRNGKey(3), n_jac))); ga = np.asarray(gn(*a4))
    res["sensitivity"] = {"jac_top_med": float(np.median(sj)), "rand_sens_med": float(np.median(sr)), "gain_med": float(np.median(ga)),
                          "gain_max": float(np.max(ga)), "n_images": n_jac}
    per_img["jac_top"] = sj; per_img["rand_sens"] = sr
    # per-layer kNN of image-pooled target-position states; reference set from training images with the same mask sequence
    Fk = []
    for i in range(0, len(Xknn), bs):
        ci, cv, ti, tv = masks[(i // bs) % len(masks)]; Xb = Xknn[i:i + bs]; nb = len(Xb)
        ci, cv, ti, tv = ci[:nb], cv[:nb], ti[:nb], tv[:nb]
        ctx, _ = A.encode(enc, tenc, jnp.asarray(V.standardise(Xb)), jnp.asarray(V.standardise(Xb)), jnp.asarray(ci), jnp.asarray(cv), jnp.asarray(ti), vc)
        _, pooled, _ = batch(ctx, jnp.asarray(ci), jnp.asarray(cv), jnp.asarray(ti), jnp.asarray(tv))
        tw = tv.sum(-1).astype(np.float64); pz = np.asarray(pooled, np.float64)
        Fk.append((pz * tw[..., None, None]).sum(1) / np.maximum(tw.sum(1), 1)[:, None, None])
    Fk = np.concatenate(Fk)
    res["knn_by_node"] = [V.knn_both(Fk[:, j], yknn, PCpool[:, j], yev[:len(PCpool)]) for j in range(Fk.shape[1])]
    return res, per_img


def main():
    root, outd = sys.argv[1], sys.argv[2]; depths = [int(x) for x in sys.argv[3:]] or [6, 10]
    st = os.path.join(outd, "states"); os.makedirs(st, exist_ok=True)
    vc, enc, tenc, _ = A.load_encoders(os.path.join(root, "Q1-PhaseA", A.V0_DIR))
    Xtr, ytr, Xte, yte = D.load_cifar10(root=os.path.join(root, "data"), n_train=-1, n_test=2000, seed=0)
    Xknn, yknn = Xtr[:5000], ytr[:5000]
    U, ev = target_basis(tenc, vc, Xte, A.eval_masks(len(Xte), vc, 200), 200)
    json.dump({"eig": ev.tolist()}, open(os.path.join(outd, "target_basis.json"), "w"))
    SA = os.path.join(root, "Q1-PhaseA/runs_vit/stageA")
    for depth in depths:
        runs = []
        for arm in ARMS_LA:
            for d in sorted(glob.glob(f"{SA}/{arm}_d{depth}_lr*_e25_full_s*")):
                if os.path.isdir(d) and os.path.exists(d + "/report.json") and os.path.exists(d + "/predictor.eqx"):
                    rep = json.load(open(d + "/report.json")); cfg = rep["cfg"]
                    if arm == "bp" and cfg["lr"] != 3e-3: continue                       # BP: the selected lr (3e-3) only
                    runs.append((os.path.basename(d), arm, cfg, rep))
        print("RUNS", depth, [r[0] for r in runs], flush=True)
        for name, arm, cfg, rep in runs:
            f = os.path.join(outd, f"latent_{name}.json")
            if os.path.exists(f): continue
            t0 = time.time(); a = A.ACfg(**{k: v for k, v in cfg.items() if k in A.ACfg.__dataclass_fields__})
            pred = eqx.tree_deserialise_leaves(os.path.join(SA, name, "predictor.eqx"), A.make_predictor(vc, a))
            res, per = analyse(name, pred, enc, tenc, vc, Xte, yte, Xknn, yknn, U, st)
            res["reported"] = {c: rep["eval"][c]["nmse"] for c in CONDS}; res["arm"], res["depth"], res["seed"] = arm, depth, cfg["seed"]
            res["reproduces_report"] = all(abs(res["conds"][c]["nmse"] - res["reported"][c]) <= 1e-5 * res["reported"][c] for c in CONDS)
            res["sec"] = round(time.time() - t0)
            np.savez_compressed(os.path.join(outd, f"perimg_{name}.npz"), **{f"{k}__{kk}" if isinstance(v, dict) else k: (vv if isinstance(v, dict) else v)
                                                                             for k, v in per.items() for kk, vv in (v.items() if isinstance(v, dict) else [("", v)])})
            json.dump(res, open(f, "w")); print("DONE", name, res["reproduces_report"], res["instr_vs_model_maxabs"], res["sec"], flush=True)
        # CKA per node: every model against BP seed 0; BP seeds 1-4 give the "same solution" reference
        ref = [r[0] for r in runs if r[1] == "bp" and r[2]["seed"] == 0]
        if not ref: continue
        load = lambda n: np.concatenate([np.load(x) for x in sorted(glob.glob(os.path.join(st, f"{n}_b*.npy")))])
        R0 = load(ref[0]); cf = os.path.join(outd, f"cka_d{depth}.json")
        cka = json.load(open(cf))["cka"] if os.path.exists(cf) else {}        # a later pass (spc_u1) adds to the file
        for name, *_ in runs:
            if name == ref[0] or name in cka or not glob.glob(os.path.join(st, f"{name}_b*.npy")): continue
            X = load(name); cka[name] = [linear_cka(R0[:, j], X[:, j]) for j in range(R0.shape[1])]
        json.dump({"reference": ref[0], "cka": cka, "rows": "per (image, target block), pooled over the valid target tokens"},
                  open(os.path.join(outd, f"cka_d{depth}.json"), "w"))
        for name, *_ in runs:                    # state files are large (~0.1 GB per model); the BP seed-0 reference is kept
            if name != ref[0]:
                for x in glob.glob(os.path.join(st, f"{name}_b*.npy")): os.remove(x)
        print("CKA", depth, {k: round(float(np.mean(v)), 3) for k, v in cka.items()}, flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
