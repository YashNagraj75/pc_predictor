# PC-trainable FC residual encoder (mu-PC style): builds the model/skip pair,
# runs the zero-energy feedforward pass, and computes the PC energy for it.

from typing import List, Tuple

import jax
import jax.numpy as jnp
import jpc
from config import EncoderCfg
import math

ACTS = {"relu": jax.nn.relu, "tanh": jnp.tanh, "gelu": jax.nn.gelu}


def init(key, cfg: EncoderCfg, input_dim: int):
    """Returns a PC-trainable FC residual net (model, skip), inspired by mu-PC."""
    L = cfg.depth + 2
    model = jpc.make_mlp(key,input_dim,cfg.width,L,cfg.embed_dim, cfg.act,use_bias=False,param_type=cfg.parameterisation)
    skip = jpc.make_skip_model(L)
    return (model, skip)


def gamma_for(cfg:EncoderCfg):
    """Returns the SiGReg output-scaling gamma for mu-PC configs, else None."""
    return 1.0 / math.sqrt(cfg.width) if cfg.parameterisation == "mupc" else None

def forward(params, x,cfg) :
    """Runs the zero-energy feedforward pass (no clamped target), returning (output, hidden activities).
    Eval arms differ only in weights, not in any test-time relaxation."""
    model, skip = params[0], params[1]   # case-C params carry a 3rd entry, D
    acts = jpc.init_activities_with_ffwd(model,x,skip_model=skip,param_type=cfg.parameterisation,gamma=gamma_for(cfg))
    return acts[-1], list(acts[:-1])



def energy(params, activities, x,cfg:EncoderCfg,y=None, out_scale=None) -> jnp.ndarray:
    """Computes the PC energy F = (1/2) sum_l ||z_l - f_l(z_{l-1})||^2 over free activities.
    x clamps z_0; y optionally clamps the output; no explicit output term (loss enters via train.py)."""
    model, skip = params[0], params[1]   # case-C params carry a 3rd entry, D
    return jpc.pc_energy_fn((model,skip),activities,activities[-1] if y is None else y, x = x,param_type=cfg.parameterisation,gamma=gamma_for(cfg),output_energy_scaling=out_scale)

