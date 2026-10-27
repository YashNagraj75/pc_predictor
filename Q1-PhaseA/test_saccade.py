import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp
import data as D, pred_eval as PE, pred_gpc as G
import diag_saccade as S
from run_probes import load_run

# Gates for diag_saccade.py; all must pass before the full run (host GPU).
OK = [True]


def check(name, cond, info=""):
    OK[0] &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {info}", flush=True)


def main():
    L = np.load("pred_latents.npz")
    raw_tr, raw_te = L["bp_emb_tr"], L["bp_emb_te"]
    mu, sd = raw_tr.reshape(-1, 128).mean(0), raw_tr.reshape(-1, 128).std(0) + 1e-6
    zte, tte = (raw_te - mu) / sd, L["tau_te"].astype(np.float32)
    cfg, params = load_run(S.BACKBONE)
    _, _, Xte, _ = D.load_dataset(cfg.data); Xte = np.asarray(Xte, np.float32)
    n = 64; X = Xte[:n]

    # 1. renderer == data._one_view at zero rotation/flip/jitter, for random crops of area 0.4 and 1.0
    for area in (0.4, 0.6, 1.0):
        ks = jax.random.split(jax.random.PRNGKey(5), n)
        v, bx, _ = jax.vmap(lambda k, im: D._one_view(k, im, cfg.data.img_size, area, area, 0.0, False, 0.0, 0.0))(ks, jnp.asarray(X))
        mine = np.stack([np.asarray(S.render(jnp.asarray(X[i:i + 1]), float(bx[i, 1]), float(bx[i, 0]), float(bx[i, 2]), cfg.data.img_size))[0] for i in range(8)])
        check(f"renderer reproduces data._one_view (area {area})", np.max(np.abs(mine - np.asarray(v[:8]))) < 1e-5,
              f"max diff {np.max(np.abs(mine - np.asarray(v[:8]))):.1e}")
    # 2. backbone reproduces the stored test latents
    z = PE.encode(params, cfg.enc, PE.test_views(cfg.data, Xte[:512]))
    err = float(np.max(np.abs(z - raw_te[:512])) / np.max(np.abs(raw_te[:512])))
    check("backbone re-encodes the stored test latents", err < 1e-3, f"rel err {err:.1e}")
    # 3. loaded predictors reproduce their stored k = 1 scores
    g = S.GPred(10, 0)
    k1 = PE.run_cases(g.predict, zte, tte, *PE.k_cases(len(zte), 1))["nmse"]
    check("G d10 s0 reproduces its stored k=1 nmse", abs(k1 - g.k1_stored) < 1e-4, f"{k1:.5f} vs {g.k1_stored:.5f}")
    f = S.FPred(0)
    import json
    fs = next(r["nmse"] for r in json.load(open("pred_runs/tests_ffres_fix/R16/ffres_alm_d10_bp_s0.json"))["rows"] if r["test"] == "k_seen" and r.get("k") == 1)
    fk1 = PE.run_cases(f.predict, zte, tte, *PE.k_cases(len(zte), 1))["nmse"]
    check("F d10 s0 reproduces its stored k=1 nmse", abs(fk1 - fs) < 1e-4, f"{fk1:.5f} vs {fs:.5f}")
    # 4. sum-and-count aggregation == the models' own multi-view predict (G) / mean of single-view predictions (F)
    Lt = S.lattice(X, cfg, params, mu, sd)
    Zf, Zp, Zw, Tf, Tp, Tw = Lt
    S.KS = [1, 2, 3]
    for kind, m in (("G", g), ("F", f)):
        o, _, _ = S.run_saccades(m, kind, Lt, tte[:n], zte[:n], S.bottom_up(X), "random", 3, None, np.random.default_rng(7))
        perm = np.random.default_rng(7).permuted(np.tile(np.arange(81), (n, 1)), axis=1)[:, :3]
        zs, ts = [], []
        for k in range(3):
            zc, tc = S.gather(Zf, Zp, Tf, Tp, perm[:, k] // 9, perm[:, k] % 9)
            zs += [zc, Zw[:, None]]; ts += [tc, np.repeat(Tw[None, None], n, 0)]
        Zs, Ts = np.concatenate(zs, 1), np.concatenate(ts, 1)
        if kind == "G":
            P = np.asarray(g.predict(Zs, Ts, tte[:n]))
        else:
            P = np.mean([np.asarray(f.predict(Zs[:, v:v + 1], Ts[:, v:v + 1], tte[:n])) for v in range(Zs.shape[1])], 0)
        ref = S.metrics(P, zte[:n])["nmse"]
        check(f"{kind}: saccade aggregation at K=3 == direct prediction from all {Zs.shape[1]} crop latents",
              abs(o[3]["nmse"] - ref) < 1e-5, f"{o[3]['nmse']:.6f} vs {ref:.6f}")
    # 5. leave-one-crop-out belief == amortised belief of the other crops
    zc, tc = S.gather(Zf, Zp, Tf, Tp, perm[:, 0] // 9, perm[:, 0] % 9)
    Zs1, Ts1 = np.concatenate([zc, Zw[:, None]], 1), np.concatenate([tc, np.repeat(Tw[None, None], n, 0)], 1)
    a = g.amort(Zs1.reshape(-1, 128), Ts1.reshape(-1, 9)).reshape(n, 6, -1)
    s_loo = (a.sum(1) - a[:, 1]) / 5
    ref = np.asarray(G.start_s(g.p, g.am, jnp.asarray(np.delete(Zs1, 1, 1)), jnp.asarray(np.delete(Ts1, 1, 1)), g.cfg))
    check("G leave-one-out belief == start_s without that crop", np.max(np.abs(s_loo - ref)) < 1e-5, f"{np.max(np.abs(s_loo - ref)):.1e}")
    # 6. planner
    bu = S.bottom_up(X)
    first = S.plan(bu, np.zeros((n, 0), np.int64), np.zeros((n, 0)))
    check("planner step 1 == argmax of bottom-up salience", np.all(first == np.argmax(bu, 1)))
    hi, he = np.zeros((n, 0), np.int64), np.zeros((n, 0))
    for k in range(32):
        nx = S.plan(bu, hi, he); hi = np.concatenate([hi, nx[:, None]], 1); he = np.concatenate([he, np.ones((n, 1))], 1)
    rev = max(len(r) - len(set(r)) for r in hi)
    check("planner: inhibition of return prevents revisits over 32 steps", rev == 0, f"max revisits {rev}")
    check("planner: every fixation on the lattice", hi.min() >= 0 and hi.max() < 81)
    nx = S.plan(np.zeros_like(bu), np.full((n, 1), 40), np.ones((n, 1)), beta=1.0)
    d = np.hypot(nx // 9 - 4, nx % 9 - 4)
    check("planner, beta 1: never re-fixates the site just visited (IOR = 0 there)", np.all(d > 0),
          f"distances from it {sorted(set(d.round(2)))[:5]} (lattice units; sigmas exceed the fixation range)")
    # 7. compiled == eager for the two forward paths used here
    s0 = np.random.default_rng(0).standard_normal((32, G.M)).astype(np.float32)
    dj = g.decode(s0, tte[:32])
    with jax.disable_jit():
        de = np.asarray(G.topdown(g.p, jnp.asarray(s0), G.phi(jnp.asarray(tte[:32])), g.cfg.H)[1])
    check("G decode: compiled == eager", np.max(np.abs(dj - de)) / np.max(np.abs(de)) < 1e-4, f"{np.max(np.abs(dj - de)) / np.max(np.abs(de)):.1e}")
    Xr = np.concatenate([zte[:32, 0], tte[:32, 0], tte[:32, 1]], -1)
    fj = f.px(Xr)
    with jax.disable_jit():
        fe = np.asarray(S.RF.forward(f.params, jnp.asarray(Xr), S.RF.RCfg(arm="alm", depth=10, seed=0, T=400, alpha=1.0, lr=0.01, rho=16.0)))
    check("F forward: compiled == eager", np.max(np.abs(fj - fe)) / np.max(np.abs(fe)) < 1e-4, f"{np.max(np.abs(fj - fe)) / np.max(np.abs(fe)):.1e}")
    print("ALL PASS" if OK[0] else "SOME FAIL")
    return OK[0]


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
