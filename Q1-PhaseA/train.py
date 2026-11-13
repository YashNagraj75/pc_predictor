# Training loop for the Q1-PhaseA encoder: BP (backprop) and PC-ALM arms share the same objective and
# optimiser, differing only in how the weight gradient is computed. Also holds the shared evaluation
# (metric suite over held-out embeddings) and cross-arm comparison (CKA, two-sample test) used by both.
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
import pcalm as PA
import optax
import sigreg as S
import sparse as SP
from config import RunCfg

import data as D


def loss_from_views(params, Zin, A, cfg: RunCfg):
    """Computes the shared LeJEPA objective from per-view inputs Zin (V, B, input_dim); both arms optimise this."""
    Zv = jnp.stack([SP.forward(params, Zin[v], cfg)[0] for v in range(Zin.shape[0])])
    return S.lejepa_loss(Zv, A, cfg.data.n_global_views, cfg.sig.lambd,
                         cfg.use_sigreg, projector=cfg.projector,
                         t_max=cfg.sig.t_max,
                         n_points=cfg.sig.n_points,
                         scale_by_n=cfg.sig.scale_by_n)

@eqx.filter_jit
def bp_grads(params, Zin, A, cfg):
    """Returns the backprop weight gradient and aux dict of bp_objective (the reference arm)."""
    (loss, aux),g = eqx.filter_value_and_grad(bp_objective, has_aux=True) (params,Zin,A,cfg)
    return g, {**aux, "loss": loss}


def bp_objective(params, Zin, A, cfg):
    """Computes the BP arm's objective: loss_from_views plus the deployed sparse-layer energy, if active.
    Equals loss_from_views exactly for sparse case 'none'."""
    loss, aux = loss_from_views(params, Zin, A, cfg)
    if SP.active(cfg):
        h = sum(SP.deployed_head_energy(params, Zin[v], cfg)
                for v in range(Zin.shape[0]))
        loss = loss + h
        aux = {**aux, "head_energy": h}
    return loss, aux


@eqx.filter_jit
def loss_ff_fn(params, Zin, A, cfg):
    """Computes the jitted task loss at the deployed (feedforward) network, for the per-step log."""
    return loss_from_views(params, Zin, A, cfg)[0]


def optim(cfg): # Optimizer for both the arms
    """Builds the AdamW optimiser (warmup + cosine decay) shared by both arms, with a separate unclipped
    group for the dictionary in sparse cases C/D."""
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
    def base():
        """Builds the clipped AdamW chain used for the encoder parameters."""
        return optax.chain(
            optax.clip_by_global_norm(cfg.opt.grad_clip),
            optax.adamw(sched, weight_decay=cfg.opt.weight_decay)
        )
    if cfg.sparse.case not in ("C", "D"):
        return base()
    # Case C (PC-C and C2 alike): the encoder keeps its usual clip; the dictionary is a separate group with
    # NO clip. At init |dF/dD| ~ 1.5 vs ~0.35 for the whole encoder, so one global clip would let D throttle
    # the encoder (the T=64 confound); D is unclipped since Adam is scale-invariant (found in review).
    dict_opt = optax.adamw(sched, weight_decay=cfg.opt.weight_decay)
    # case D: the head-layer means (params[3]) are EMA state, never trained
    return optax.multi_transform({"enc": base(), "dict": dict_opt, "frozen": optax.set_to_zero()},
                                 SP.param_labels)

@eqx.filter_jit
def grad_update(params,g, opt_state,opt):
    """Applies one optimiser update to params given gradient g; returns the updated params and optimiser state."""
    flt = eqx.filter(params,eqx.is_inexact_array)
    upd, opt_state = opt.update(g,opt_state,flt)
    return eqx.apply_updates(params,upd), opt_state

def pc_relax(params, Zin, A, cfg, key):
    """Runs pcalm.relax (PC is its alpha=0 case) and returns (acts, hist), dropping the duals."""
    acts, _duals, hist = PA.relax(params, Zin, A, cfg, key)
    return acts, hist


@eqx.filter_jit
def pc_grads(params, Zin, A, cfg, key):
    """Returns the local weight gradient from the relaxed primal-dual state (no backward pass).
    At alpha=0 this is the plain PC update."""
    return PA.grads(params, Zin, A, cfg, key)


def _global_norm(g):
    """Returns the global L2 norm of a gradient pytree's inexact-array leaves."""
    leaves = jax.tree_util.tree_leaves(eqx.filter(g, eqx.is_inexact_array))
    return float(jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in leaves)))


def train(cfg: RunCfg, verbose=True):
    """Trains the encoder under cfg (BP or PC-ALM arm), logging per-step diagnostics and periodic evaluation.
    Returns (final params, log list); stops early and marks DIVERGED on a non-finite loss or grad norm."""
    out = os.path.join(cfg.out_dir, cfg.name)
    os.makedirs(out, exist_ok=True)
    stale = os.path.join(out, "DIVERGED")      # from an earlier failed attempt
    if os.path.exists(stale):
        os.remove(stale)
    with open(os.path.join(out, "config.json"), "w") as f:
        f.write(cfg.to_json())

    # ---- data + init -------------------------------------------------
    Xtr, ytr, Xte, yte = D.load_dataset(cfg.data)
    Xtr_j = jnp.asarray(Xtr)
    n = Xtr_j.shape[0]

    probe_key = jax.random.PRNGKey(0)
    Zprobe, _ = D.make_views(probe_key, Xtr_j[:2], cfg.data)   # shape only
    input_dim = Zprobe.shape[-1]

    params = SP.init_params(jax.random.PRNGKey(cfg.opt.seed), cfg, input_dim)
    params = SP.init_mu(params, SP.mu_rows(Xtr, cfg), cfg)   # case D centring; no-op otherwise
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
        if cfg.sparse.case in ("C", "D"):          # encoder vs dictionary group
            gn_enc, gn_dict = _global_norm((g[0], g[1])), _global_norm(g[2])
            gn_G = _global_norm(g[4]) if len(g) == 6 else None     # case E generative dictionaries
        params, opt_state = grad_update(params, g, opt_state, opt)
        params = SP.project(params, cfg)          # case C: unit-norm D columns
        params = SP.update_mu(params, Zin, cfg)   # case D: EMA of head-layer means

        # common yardstick: the objective at the FEEDFORWARD embeddings, identically defined for both arms.
        # PC's own L_final/F_final are measured at the RELAXED activities and move with T -- useful for
        # inference convergence, but not comparable across arms/T; loss_ff makes the curves plottable together.
        L_ff = loss_ff_fn(params, Zin, A, cfg)

        row = {"step": step, "grad_norm_preclip": gnorm, "loss_ff": float(L_ff)}
        if not (np.isfinite(gnorm) and np.isfinite(float(L_ff))):
            # stop a diverged run instead of training on NaNs for hours; no
            # params.eqx is written, so downstream probes skip it loudly
            row["DIVERGED"] = True
            log.append(row)
            with open(os.path.join(out, "log.json"), "w") as f:
                json.dump(log, f)
            open(os.path.join(out, "DIVERGED"), "w").write(f"step {step}\n")
            print(f"[{cfg.name}] DIVERGED at step {step} -- stopping this run", flush=True)
            return params, log
        if cfg.sparse.case in ("C", "D"):
            row.update({"grad_norm_enc": gn_enc, "grad_norm_dict": gn_dict})
            if gn_G is not None:
                row["grad_norm_G"] = gn_G
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
    """Runs the deterministic centre-view pass through the encoder and applies the metric suite at every layer.
    Both arms use this exact function; returns a nested dict ready for json.dump."""
    rng = np.random.default_rng(seed)
    itr = rng.choice(len(Xtr), min(n_probe, len(Xtr)), replace=False)
    ite = rng.choice(len(Xte), min(n_probe, len(Xte)), replace=False)

    def embed(X):
        """Runs the deployed forward pass on X and returns (embedding, hidden activations) as numpy arrays."""
        xb = D.eval_batch(jnp.asarray(X), cfg.data)
        z, hs = SP.forward(params, xb, cfg)        # == E.forward for case none
        return np.asarray(z), [np.asarray(h) for h in hs]

    ztr, htr = embed(Xtr[itr])
    zte, hte = embed(Xte[ite])
    ytr_, yte_ = np.asarray(ytr)[itr], np.asarray(yte)[ite]

    out = {"embedding": {}, "hidden": []}
    if SP.active(cfg):     # rescaling cheat / sparsity / dictionary health
        out["sparse"] = SP.health(params, D.eval_batch(jnp.asarray(Xtr[itr[:1024]]),
                                                       cfg.data), cfg)

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


def compare_arms(eval_a, params_a, eval_b, params_b, Xte, cfg, n=4000, seed=0, cfg_b=None):
    """Compares two trained arms on matched inputs via layer-wise CKA and the projected two-sample test."""
    xb = D.eval_batch(jnp.asarray(Xte[:n]), cfg.data)
    # the DEPLOYED networks (E.forward would silently drop the threshold / ISTA
    # of sparse arms -- found in review).  cfg_b defaults to cfg.
    za, hsa = SP.forward(params_a, xb, cfg)
    zb, hsb = SP.forward(params_b, xb, cfg_b or cfg)
    A = np.asarray(S.sample_directions(jax.random.PRNGKey(seed),
                                       za.shape[1], cfg.sig.num_slices))
    return {
        "cka_embedding": M.linear_cka(np.asarray(za), np.asarray(zb)),
        "cka_hidden": [M.linear_cka(np.asarray(a), np.asarray(b))
                       for a, b in zip(hsa, hsb)],
        "two_sample_embedding": M.projected_two_sample(np.asarray(za),
                                                       np.asarray(zb), A),
    }
