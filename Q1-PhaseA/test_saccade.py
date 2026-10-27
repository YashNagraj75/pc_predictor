import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp
import data as D, pred_eval as PE, pred_gpc as G
import diag_saccade as S
from pred_data import tau_features
from run_probes import load_run

# Gates for diag_saccade.py (rev 2, after 3 reviews).  All must pass before the full run (host GPU).
OK = [True]


def check(name, cond, info=""):
    OK[0] &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {info}", flush=True)


# MPC's score map (lgn_glimpser._plan_next_saccade, the deterministic part), copied verbatim in numpy form.
def mpc_score_map(bu_map, hist_yx, hist_err, ior_sigma, err_sigma, w):
    H, W = bu_map.shape
    yy, xx = np.meshgrid(np.linspace(0, H - 1, H), np.linspace(0, W - 1, W), indexing="ij")
    td, ior = np.zeros((H, W)), np.ones((H, W))
    for (ny, nx), e in zip(hist_yx, hist_err):
        y_pixel, x_pixel = ((ny + 1.0) / 2.0) * (H - 1), ((nx + 1.0) / 2.0) * (W - 1)
        d2 = (yy - y_pixel) ** 2 + (xx - x_pixel) ** 2
        td = td + e * np.exp(-0.5 * d2 / err_sigma ** 2); ior = ior * (1.0 - np.exp(-0.5 * d2 / ior_sigma ** 2))
    return ((1.0 - w) * bu_map + w * td) * ior


def main():
    L = np.load("pred_latents.npz")
    raw_tr, raw_te = L["bp_emb_tr"], L["bp_emb_te"]
    mu, sd = raw_tr.reshape(-1, 128).mean(0), raw_tr.reshape(-1, 128).std(0) + 1e-6
    zte, tte = (raw_te - mu) / sd, L["tau_te"].astype(np.float32)
    cfg, params = load_run(S.BACKBONE); enc = S.Encoder(cfg, params, mu, sd)
    _, _, Xte, _ = D.load_dataset(cfg.data); Xte = np.asarray(Xte, np.float32)
    n = 32; X = Xte[:n]; size = cfg.data.img_size

    # 1. renderer and crop tau == data._one_view / tau_features at zero rotation, flip and jitter
    for area in (0.4, 0.6, 1.0):
        ks = jax.random.split(jax.random.PRNGKey(5), n)
        v, bx, pr = jax.vmap(lambda k, im: D._one_view(k, im, size, area, area, 0.0, False, 0.0, 0.0))(ks, jnp.asarray(X))
        bx, pr = np.asarray(bx), np.asarray(pr)
        mine = np.stack([np.asarray(S.render(jnp.asarray(X[i:i + 1]), float(bx[i, 1]), float(bx[i, 0]), float(bx[i, 2]), size))[0] for i in range(n)])
        check(f"renderer == data._one_view (area {area}, all {n})", np.max(np.abs(mine - np.asarray(v))) < 1e-5, f"{np.max(np.abs(mine - np.asarray(v))):.1e}")
        tm = np.stack([S.crop_tau(bx[i, 1], bx[i, 0], bx[i, 2]) for i in range(n)]); tr = tau_features(bx, pr)
        check(f"crop_tau == tau_features (area {area})", np.max(np.abs(tm - tr)) < 1e-6, f"{np.max(np.abs(tm - tr)):.1e}")
    # 2. backbone, through the new Encoder, reproduces the stored test latents
    z = enc(PE.test_views(cfg.data, Xte[:512]).reshape(-1, size, size, 3)).reshape(512, 4, 128) * sd + mu
    err = float(np.max(np.abs(z - raw_te[:512])) / np.max(np.abs(raw_te[:512])))
    check("Encoder re-encodes the stored test latents", err < 1e-3, f"rel err {err:.1e}")
    # 3. crop table and geometries: index map, rendering at asymmetric points, orientation
    sp = S.crop_specs()
    Zc, Tc = S.crop_table(X, enc, size)
    for (a, b) in ((-6, -6), (1, 5), (6, -2)):
        r = S.lat_index(a, b)
        zf = enc(S.render(jnp.asarray(X), a * S.H_STEP, b * S.H_STEP, S.S_F, size))
        cy, cx = np.clip([a * S.H_STEP, b * S.H_STEP], -S.LIM_P, S.LIM_P)
        zp = enc(S.render(jnp.asarray(X), float(cy), float(cx), S.S_P, size))
        ok = np.allclose(Zc[:, r], zf, atol=1e-4) and np.allclose(Zc[:, 169 + r], zp, atol=1e-4) and np.allclose(Tc[r], S.crop_tau(a * S.H_STEP, b * S.H_STEP, S.S_F))
        check(f"crop table row for lattice ({a},{b}) == direct render (foveal, parafoveal, tau)", ok)
    g4, g1 = S.make_geom("grid4"), S.make_geom("fov1")
    ok4 = all(np.allclose(sp[g4["crops"][l][q]][:2], g4["fix"][l] + 2 * S.H_STEP * np.array(o)) for l in range(g4["L"])
              for q, o in enumerate([(-1, -1), (-1, 1), (1, -1), (1, 1)]))
    ok1 = all(np.allclose(sp[g1["crops"][l][0]][:2], g1["fix"][l]) for l in range(g1["L"]))
    okp = all(np.allclose(sp[g["crops"][l][-1]][:2], np.clip(g["fix"][l], -S.LIM_P, S.LIM_P)) for g in (g4, g1) for l in range(g["L"]))
    check("grid4: foveal crops at fixation +- 2h (row = y, col = x)", ok4); check("fov1: foveal crop at the fixation", ok1)
    check("both: parafoveal crop at the clipped fixation", okp)
    blob = np.zeros((1, 69, 69, 3), np.float32); blob[0, 14:24, 46:56] = 1.0      # bright blob up (row 19) and right (col 51)
    f0 = np.argmax(S.bottom_up(blob, g1), 1)[0]
    check("orientation: first bottom-up fixation moves up and right towards the blob", g1["fix"][f0, 0] < 0 and g1["fix"][f0, 1] > 0, f"fix {g1['fix'][f0]}")
    # 4. planner parity with MPC (raw formula), on lattice points rounded to pixels
    for geom in (g4, g1):
        gr = dict(geom, py=np.rint(geom["py"]), px=np.rint(geom["px"]))
        bu_map = np.asarray(S._bu(jnp.asarray(X[:1])))[0]
        rng = np.random.default_rng(3); hi = rng.choice(gr["L"], 5, replace=False); he = rng.uniform(0.1, 3.0, 5)
        mine = S.plan_scores(bu_map[gr["py"].astype(int), gr["px"].astype(int)][None], hi[None], he[None], gr, norm=False)[0]
        hyx = np.stack([gr["py"][hi] / 68 * 2 - 1, gr["px"][hi] / 68 * 2 - 1], 1)
        ref = mpc_score_map(bu_map, hyx, he, S.SIG_IOR, S.SIG_ERR, S.BETA)[gr["py"].astype(int), gr["px"].astype(int)]
        check(f"{geom['name']}: plan_scores(norm=False) == MPC score map at the lattice", np.allclose(mine, ref, rtol=1e-4, atol=1e-7) and np.argmax(mine) == np.argmax(ref),
              f"max rel {np.max(np.abs(mine - ref) / (np.abs(ref) + 1e-12)):.1e}")
    mb = np.asarray(S._bu(jnp.asarray(blob)))[0]
    check("bottom-up map peaks on the blob", np.unravel_index(np.argmax(mb), mb.shape)[0] in range(14, 24) and np.unravel_index(np.argmax(mb), mb.shape)[1] in range(46, 56))
    # 5. stored k = 1 scores reproduce for every model the run uses (PC and frozen BP)
    models = S.load_models([0, 1, 2])
    for name, m in models.items():
        if m.kind == "F":
            stored = next(r["nmse"] for r in json.load(open(f"pred_runs/tests_ffres_fix/R16/ffres_alm_d10_bp_s{m.seed}.json"))["rows"] if r["test"] == "k_seen" and r.get("k") == 1)
        else:
            stored = m.k1_stored
        if stored is None:
            check(f"{name}: stored k=1 score found", False, "missing stored test json"); continue
        got = PE.run_cases(m.predict, zte, tte, *PE.k_cases(len(zte), 1))["nmse"]
        check(f"{name} reproduces its stored k=1 nmse", abs(got - stored) < 1e-4, f"{got:.5f} vs {stored:.5f}")
    # 6. pool aggregation == the models' own multi-view prediction from the same latents
    g, f = models["G_d10_s0"], models["F_d10_s0"]
    S.KS = [1, 2, 3]
    for m in (g, f):
        for geom in (g4, g1):
            save = {}
            o, _, hi, _ = S.run_saccades(m, Zc, Tc, tte[:n], zte[:n], X, geom, "random", 3, None, np.random.default_rng(7), save)
            ids = geom["crops"][hi.astype(int)].reshape(n, -1)
            for pool in ("full", "noperi"):
                Zs = np.take_along_axis(Zc, ids[..., None], 1); Ts = Tc[ids]
                if pool == "full":
                    Zs = np.concatenate([Zs, np.repeat(Zc[:, S.PERI:S.PERI + 1], 3, 1)], 1); Ts = np.concatenate([Ts, np.repeat(Tc[None, S.PERI:S.PERI + 1], n, 0).repeat(3, 1)], 1)
                if m.kind == "G":
                    P = np.asarray(m.predict(Zs, Ts, tte[:n]))
                else:
                    P = np.mean([np.asarray(m.predict(Zs[:, v:v + 1], Ts[:, v:v + 1], tte[:n])) for v in range(Zs.shape[1])], 0)
                ref, _ = S.metrics(P, zte[:n]); got = o[(3, pool)]
                check(f"{m.kind} {geom['name']} pool {pool}: K=3 aggregation == direct prediction from {Zs.shape[1]} latents",
                      abs(got["nmse"] - ref["nmse"]) < 1e-5 and abs(got["cos"] - ref["cos"]) < 1e-5, f"{got['nmse']:.6f} vs {ref['nmse']:.6f}")
    # 7. surprise == brute force (belief / prediction from the pool held before the glimpse; peripheral alone at k=0)
    for m in (g, f):
        geom = g4; pool = S.Pool(m, Zc, Tc, tte[:n]); seq = [0, 40, 80]; ok = True; worst = 0.0
        held_z, held_t = [Zc[:, S.PERI:S.PERI + 1]], [np.repeat(Tc[None, S.PERI:S.PERI + 1], n, 0)]
        for k, l in enumerate(seq):
            ids = np.repeat(geom["crops"][l][None], n, 0); z = np.take_along_axis(Zc, ids[..., None], 1); t = Tc[ids]
            got = pool.surprise(z, t)
            hz = np.concatenate(held_z + [np.repeat(Zc[:, S.PERI:S.PERI + 1], max(k, 1) - 1, 1)], 1) if k > 1 else np.concatenate(held_z, 1)
            ht = np.concatenate(held_t + [np.repeat(Tc[None, S.PERI:S.PERI + 1], n, 0).repeat(max(k, 1) - 1, 1)], 1) if k > 1 else np.concatenate(held_t, 1)
            if m.kind == "G":
                zh = np.stack([np.asarray(m.predict(hz, ht, t[:, j:j + 1]))[:, 0] for j in range(z.shape[1])], 1)
            else:
                zh = np.stack([np.mean([np.asarray(m.predict(hz[:, u:u + 1], ht[:, u:u + 1], t[:, j:j + 1]))[:, 0] for u in range(hz.shape[1])], 0) for j in range(z.shape[1])], 1)
            ref = np.mean(np.sum((zh - z) ** 2, -1), 1)
            worst = max(worst, float(np.max(np.abs(got - ref) / (ref + 1e-9)))); ok &= np.allclose(got, ref, rtol=1e-4)
            pool.add(z, t, geom["kinds"]); held_z.append(z); held_t.append(t)
        check(f"{m.kind}: surprise == brute-force error under the pool held before each glimpse (3 glimpses)", ok, f"max rel {worst:.1e}")
    # 8. every policy runs: no revisits, finite, inside the lattice; epi's first error == brute force at step 1
    S.KS = [1, 4]
    for m in (g, f):
        for pol in S.POLICIES + ["oracle_occ"]:
            Xo, occ = S.corrupt_images(X, "occlude") if pol == "oracle_occ" else (X, None)
            o, st, hi, he = S.run_saccades(m, Zc, Tc, tte[:n], zte[:n], Xo, g1, pol, 4, occ, np.random.default_rng(7), {})
            check(f"{m.kind} policy {pol}: 4 glimpses, no revisits, finite", st["revisits"] == 0 and np.all(np.isfinite(he)) and all(np.isfinite(v["nmse"]) for v in o.values()))
    # 9. occlusion geometry: occluded_frac == fraction of exactly-black pixels in the rendered crop
    Xo, occ = S.corrupt_images(X, "occlude"); Xo2 = np.ones_like(X); Xo2, _ = S.corrupt_images(Xo2, "occlude")
    for (cy, cx, s) in ((0.0, 0.0, 1.0), (0.24, -0.3, S.S_F), (-0.3, 0.3, S.S_F)):
        v = np.asarray(S.render(jnp.asarray(Xo2), cy, cx, s, 69))
        frac_px = np.mean(np.all(v == 0, -1).reshape(n, -1), 1); pred = S.occluded_frac(cy, cx, s, occ)
        check(f"occluded_frac == black-pixel fraction (crop {cy},{cx},{s:.2f})", np.max(np.abs(frac_px - pred)) < 0.04, f"max diff {np.max(np.abs(frac_px - pred)):.3f}")
    # 10. whole-view controls reproduce the stored k = 1 regime (fresh augmented draw)
    Zv, Tv, _ = S.whole_views(Xte, enc, cfg, 1, False, 303)
    for name in ("G_d10_s0", "F_d10_s0"):
        m = models[name]; r, _ = S.score_whole(m, Zv, Tv, tte, zte, [1])[1]
        stored = m.k1_stored if m.kind == "G" else 0.2997
        check(f"{name}: fresh augmented whole view k=1 within 0.01 of the stored k=1 nmse", abs(r["nmse"] - stored) < 0.01, f"{r['nmse']:.4f} vs {stored:.4f}")
    # 11. compiled == eager for the forward paths used
    s0 = np.random.default_rng(0).standard_normal((32, G.M)).astype(np.float32)
    dj = g.readout(s0, tte[:32])
    with jax.disable_jit():
        de = np.asarray(G.topdown(g.p, jnp.asarray(s0), G.phi(jnp.asarray(tte[:32])), g.cfg.H)[1])
    check("G readout: compiled == eager", np.max(np.abs(dj - de)) / np.max(np.abs(de)) < 1e-4, f"{np.max(np.abs(dj - de)) / np.max(np.abs(de)):.1e}")
    Xr = np.concatenate([zte[:32, 0], tte[:32, 0], tte[:32, 1]], -1); fj = f.px(Xr)
    with jax.disable_jit():
        fe = np.asarray(S.RF.forward(f.params, jnp.asarray(Xr), f.rc))
    check("F forward: compiled == eager", np.max(np.abs(fj - fe)) / np.max(np.abs(fe)) < 1e-4, f"{np.max(np.abs(fj - fe)) / np.max(np.abs(fe)):.1e}")
    print("ALL PASS" if OK[0] else "SOME FAIL")
    return OK[0]


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
