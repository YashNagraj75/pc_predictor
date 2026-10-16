from typing import List, Tuple

import jax
import jax.numpy as jnp
import jpc
from config import EncoderCfg
import math

ACTS = {"relu": jax.nn.relu, "tanh": jnp.tanh, "gelu": jax.nn.gelu}


# Returns a PC-trainable FC residual net, inspired by mu-PC
def init(key, cfg: EncoderCfg, input_dim: int):
    L = cfg.depth + 2
    model = jpc.make_mlp(key,input_dim,cfg.width,L,cfg.embed_dim, cfg.act,use_bias=False,param_type=cfg.parameterisation)
    skip = jpc.make_skip_model(L)
    return (model, skip)


# Gamma for scaling of output to match SiGReg input dim
def gamma_for(cfg:EncoderCfg):
    return 1.0 / math.sqrt(cfg.width) if cfg.parameterisation == "mupc" else None

# Returns (z, hs); hs enables layer-wise measurements (see module docstring).
# PC view: this is the ZERO-ENERGY config when no target is clamped --
# with nothing clamped at output, energy minimum = feedforward pass exactly.
# At eval, both arms run through this same function; arm difference is
# entirely in learned weights, not in any test-time relaxation.
def forward(params, x,cfg) :
    model, skip = params[0], params[1]   # case-C params carry a 3rd entry, D
    acts = jpc.init_activities_with_ffwd(model,x,skip_model=skip,param_type=cfg.parameterisation,gamma=gamma_for(cfg))
    return acts[-1], list(acts[:-1])



# PC energy F = (1/2) sum_l || z_l - f_l(z_{l-1}) ||^2.
# activities = free variables z_1..z_L, z_out; z_0 clamped to x, not in list.
# NO output term here; loss enters via output node's activity gradient (see train.py).
def energy(params, activities, x,cfg:EncoderCfg,y=None, out_scale=None) -> jnp.ndarray:
    model, skip = params[0], params[1]   # case-C params carry a 3rd entry, D
    return jpc.pc_energy_fn((model,skip),activities,activities[-1] if y is None else y, x = x,param_type=cfg.parameterisation,gamma=gamma_for(cfg),output_energy_scaling=out_scale)

