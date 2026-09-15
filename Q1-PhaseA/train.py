import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # sandbox sets PYTHONSAFEPATH=1, which strips the script dir from sys.path

import os
import time
import json

import encoder as E
import equinox as eqx
import jax
import jax.numpy as jnp
import metrics as M
import numpy as np
import optax
import sigreg as S
from config import RunCfg

import data as D


def loss_from_views(params, Zin, A, cfg: RunCfg):
    """Shared objective.  Zin: (V, B, input_dim) -> (scalar, aux).

    Both arms optimise this.  Only the gradient mechanism differs.
    """
    Zv = jnp.stack([E.forward(params, Zin[v], cfg.enc)[0] for v in range(Zin.shape[0])])
    return S.lejepa_loss(Zv, A, cfg.data.n_global_views, cfg.sig.lambd,
                         cfg.use_sigreg, projector=cfg.projector,
                         t_max=cfg.sig.t_max,
                         n_points=cfg.sig.n_points,
                         scale_by_n=cfg.sig.scale_by_n)

@eqx.filter_jit
def bp_grads(params, Zin, A, cfg):
    """TODO -- reverse-mode gradient of `loss_from_views`.  One jax.grad call.
    This arm is the reference; it should be ~5 lines."""
    (loss, aux),g = eqx.filter_value_and_grad(loss_from_views, has_aux=True) (params,Zin,A,cfg)
    return g, {**aux, "loss": loss}


def optim(cfg): # Optimizer for both the arms
    """AdamW + linear warmup then cosine decay (LeJEPA MINIMAL.md
    L157-161: LinearLR(start_factor=0.01, 1 epoch) -> CosineAnnealingLR).
    Previously a bare constant LR with optax's default weight decay."""
    if cfg.opt.lr_schedule == "warmup_cosine":
        sched = optax.warmup_cosine_decay_schedule(
            init_value=cfg.opt.param_lr * 0.01,
            peak_value=cfg.opt.param_lr,
            warmup_steps=min(cfg.opt.warmup_steps, max(1, cfg.opt.n_steps // 10)),
            decay_steps=cfg.opt.n_steps,
            end_value=cfg.opt.param_lr * 1e-3,
        )
    else:
        sched = cfg.opt.param_lr
    return optax.chain(
        optax.clip_by_global_norm(cfg.opt.grad_clip),
        optax.adamw(sched, weight_decay=cfg.opt.weight_decay)
    )

@eqx.filter_jit
def grad_update(params,g, opt_state,opt):
    flt = eqx.filter(params,eqx.is_inexact_array)
    upd, opt_state = opt.update(g,opt_state,flt)
    return eqx.apply_updates(params,upd), opt_state

def pc_relax(params, Zin, A, cfg, key):
    V = Zin.shape[0]

    # init, z_0 is clamped to Zin[v]
    acts0 = []
    for v in range(V):
        z,hs = E.forward(params,Zin[v], cfg.enc)
        acts0.append([*hs, z])

    def loss_z(Zv):
        return S.lejepa_loss(Zv,A,cfg.data.n_global_views,cfg.sig.lambd,cfg.use_sigreg,projector=cfg.projector,t_max=cfg.sig.t_max,n_points=cfg.sig.n_points, scale_by_n=cfg.sig.scale_by_n)[0]


    def out_stack(acts):
        return jnp.stack([acts[v][-1] for v in range(V)])

    def F_int(acts):
        return sum(E.energy(params,acts[v],Zin[v],cfg.enc) for v in range(V))

    g0 = jax.grad(loss_z)(out_stack(acts0)) if cfg.pc.output_drive == "frozen" else None
    dF = jax.grad(F_int)
    dL = jax.grad(lambda a: loss_z(out_stack(a)))

    # 3. Eq. A.  The two variants differ only in the second term.
    def step(acts, _):
        g = dF(acts)
        if cfg.pc.output_drive == "coupled":
            gL = dL(acts)                                   # recomputed each t
            g = [[gi + gj for gi, gj in zip(g[v], gL[v])] for v in range(V)]
        else:
            g = [[*g[v][:-1], g[v][-1] + g0[v]] for v in range(V)]   # cached
        new = [[z - cfg.pc.activity_lr * gz for z, gz in zip(acts[v], g[v])]
               for v in range(V)]
        return new, (F_int(new), loss_z(out_stack(new)))

    acts_T, (F_hist, L_hist) = jax.lax.scan(step, acts0, None, length=cfg.pc.T)
    return acts_T, {"F":F_hist, "L": L_hist}




@eqx.filter_jit
def pc_grads(params, Zin, A, cfg, key):
    """TODO -- weight gradients from the relaxed activities.

    Once relaxed, each weight's gradient is LOCAL:
        dF/dW_l = -eps_l * dphi(z_{l-1})/dW_l
    i.e. the prediction error at layer l times the local input.  No
    backward pass through the network.
    """
    acts_T, hist = pc_relax(params,Zin,A,cfg,key)
    acts_T = jax.lax.stop_gradient(acts_T)
    V = Zin.shape[0]

    def F_tot(p):
        return sum(E.energy(p,acts_T[v], Zin[v],cfg.enc) for v in range(V))

    g = eqx.filter_grad(F_tot)(params)
    return g, {"F_final": hist["F"][-1],"L_final": hist["L"][-1],
               "F_hist": hist["F"], "L_hist": hist["L"]}


def _global_norm(g):
    leaves = jax.tree_util.tree_leaves(eqx.filter(g, eqx.is_inexact_array))
    return float(jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in leaves)))


def train(cfg: RunCfg, verbose=True):
    out = os.path.join(cfg.out_dir, cfg.name)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config.json"), "w") as f:
        f.write(cfg.to_json())

    # ---- data + init -------------------------------------------------
    Xtr, ytr, Xte, yte = D.load_dataset(cfg.data)
    Xtr_j = jnp.asarray(Xtr)
    n = Xtr_j.shape[0]

    probe_key = jax.random.PRNGKey(0)
    Zprobe, _ = D.make_views(probe_key, Xtr_j[:2], cfg.data)   # shape only
    input_dim = Zprobe.shape[-1]

    params = E.init(jax.random.PRNGKey(cfg.opt.seed), cfg.enc, input_dim)
    opt = optim(cfg)
    opt_state = opt.init(eqx.filter(params, eqx.is_inexact_array))

    log = []
    for step in range(cfg.opt.n_steps):
        # ---- ONE key stream, derived from seed and step ONLY ----------
        # cfg.arm is deliberately NOT mixed in: both arms must see the
        # same images, the same crops and the same SIGReg directions.
        skey = jax.random.fold_in(jax.random.PRNGKey(cfg.opt.seed + 1), step)
        kb, kv, ka, kg = jax.random.split(skey, 4)

        idx  = jax.random.choice(kb, n, (cfg.opt.batch_size,), replace=False)
        Zin, _ = D.make_views(kv, Xtr_j[idx], cfg.data)
        A      = S.sample_directions(ka, cfg.enc.embed_dim, cfg.sig.num_slices)

        # ---- the ONLY line that differs between arms ------------------
        if cfg.arm == "bp":
            g, aux = bp_grads(params, Zin, A, cfg)
        else:
            g, aux = pc_grads(params, Zin, A, cfg, kg)

        gnorm = _global_norm(g)                    # BEFORE clipping
        params, opt_state = grad_update(params, g, opt_state, opt)

        # common yardstick: the objective at the FEEDFORWARD embeddings,
        # identically defined for both arms.  PC's own L_final/F_final are
        # measured at the RELAXED activities and move with T -- they are
        # the right diagnostic for whether inference is converging, but
        # not comparable across arms or across T.  loss_ff is what makes
        # the two arms' training curves plottable on the same axes.
        L_ff, _ = loss_from_views(params, Zin, A, cfg)

        row = {"step": step, "grad_norm_preclip": gnorm, "loss_ff": float(L_ff)}
        row.update({k: float(v) for k, v in aux.items() if jnp.ndim(v) == 0})
        log.append(row)

        if step % cfg.opt.eval_every == 0 or step == cfg.opt.n_steps - 1:
            ev = evaluate(params, Xtr, ytr, Xte, yte, cfg)
            row["eval"] = ev
            with open(os.path.join(out, "log.json"), "w") as f:
                json.dump(log, f)
            if verbose:
                print(f"[{cfg.name}] step {step:5d}  |g|={gnorm:.3e}  "
                      f"{ {k: round(v,4) for k,v in row.items() if k!='eval'} }")

    with open(os.path.join(out, "log.json"), "w") as f:
        json.dump(log, f)
    eqx.tree_serialise_leaves(os.path.join(out, "params.eqx"), params)
    return params, log


# --------------------------------------------------------------- evaluation
def evaluate(params, Xtr, ytr, Xte, yte, cfg: RunCfg, n_probe=4000, seed=0):
    """COMPLETE -- the Q1 measurement pass.  No TODOs here.

    Runs the deterministic centre view (no augmentation) through the encoder,
    then applies the metric suite at every layer.  Returns a nested dict ready
    to json.dump.

    Both arms are evaluated by exactly this function, and at evaluation time
    there is no target attached -- so the zero-energy configuration is the
    feedforward pass for both.  The entire difference between arms is in the
    weights.
    """
    rng = np.random.default_rng(seed)
    itr = rng.choice(len(Xtr), min(n_probe, len(Xtr)), replace=False)
    ite = rng.choice(len(Xte), min(n_probe, len(Xte)), replace=False)

    def embed(X):
        xb = D.eval_batch(jnp.asarray(X), cfg.data)
        z, hs = E.forward(params, xb, cfg.enc)
        return np.asarray(z), [np.asarray(h) for h in hs]

    ztr, htr = embed(Xtr[itr])
    zte, hte = embed(Xte[ite])
    ytr_, yte_ = np.asarray(ytr)[itr], np.asarray(yte)[ite]

    out = {"embedding": {}, "hidden": []}

    # embedding layer: kNN and collapse are valid; spectrum only as a
    # constraint-satisfaction diagnostic (SIGReg regularises it directly).
    out["embedding"] = {
        "knn": M.knn_accuracy(ztr, ytr_, zte, yte_),
        "collapse": M.collapse_report(ztr),
        "twonn": M.twonn_dimension(ztr),
        "spectrum_DIAGNOSTIC_ONLY": M.spectrum_report(ztr),
    }
    # hidden layers: this is where the spectral metrics mean something.
    for li, (a, b) in enumerate(zip(htr, hte)):
        out["hidden"].append({
            "layer": li,
            "knn": M.knn_accuracy(a, ytr_, b, yte_),
            "spectrum": M.spectrum_report(a),
            "twonn": M.twonn_dimension(a),
            "collapse": M.collapse_report(a),
        })
    return out


def compare_arms(eval_a, params_a, eval_b, params_b, Xte, cfg, n=4000, seed=0):
    """COMPLETE -- cross-arm comparison: layer-wise CKA and the projected
    two-sample test.  Needs both arms' parameters and MATCHED inputs."""
    xb = D.eval_batch(jnp.asarray(Xte[:n]), cfg.data)
    za, hsa = E.forward(params_a, xb, cfg.enc)
    zb, hsb = E.forward(params_b, xb, cfg.enc)
    A = np.asarray(S.sample_directions(jax.random.PRNGKey(seed),
                                       za.shape[1], cfg.sig.num_slices))
    return {
        "cka_embedding": M.linear_cka(np.asarray(za), np.asarray(zb)),
        "cka_hidden": [M.linear_cka(np.asarray(a), np.asarray(b))
                       for a, b in zip(hsa, hsb)],
        "two_sample_embedding": M.projected_two_sample(np.asarray(za),
                                                       np.asarray(zb), A),
    }
