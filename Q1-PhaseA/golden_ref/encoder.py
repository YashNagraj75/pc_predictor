# Phase A encoder -- muPC-parameterised fully-connected residual net, frozen as the pre-sparsity
# golden reference (init/forward/energy wrap jpc's make_mlp/make_skip_model, init_activities_with_ffwd
# and pc_energy_fn). forward returns (z, hs) with hs = intermediate layer activations, low to high,
# which the layer-wise spectral/CKA metrics in metrics.py need. Used by _golden_pcalm.py to regenerate
# bit-identical outputs that test_sparse.py's case-none gate compares against.
from typing import List, Tuple

import jax
import jax.numpy as jnp
import jpc
from config import EncoderCfg
import math

ACTS = {"relu": jax.nn.relu, "tanh": jnp.tanh, "gelu": jax.nn.gelu}


def init(key, cfg: EncoderCfg, input_dim: int):
    """Builds the muPC-parameterised FC residual encoder and its skip-connection model, returning (model, skip)."""
    L = cfg.depth + 2
    model = jpc.make_mlp(key,input_dim,cfg.width,L,cfg.embed_dim, cfg.act,use_bias=False,param_type=cfg.parameterisation)
    skip = jpc.make_skip_model(L)
    return (model, skip)


def gamma_for(cfg:EncoderCfg):
    """Returns the SIGReg output-scaling gamma for the muPC parameterisation, or None otherwise."""
    return 1.0 / math.sqrt(cfg.width) if cfg.parameterisation == "mupc" else None

def forward(params, x,cfg) :
    """Runs the feedforward pass and returns (z, hs): the embedding and the intermediate layer activations."""
    model, skip = params
    acts = jpc.init_activities_with_ffwd(model,x,skip_model=skip,param_type=cfg.parameterisation,gamma=gamma_for(cfg))
    return acts[-1], list(acts[:-1])



def energy(params, activities, x,cfg:EncoderCfg,y=None, out_scale=None) -> jnp.ndarray:
    """Computes the PC energy of the free activities z_1..z_L, z_out against the clamped input x."""
    model, skip = params
    return jpc.pc_energy_fn((model,skip),activities,activities[-1] if y is None else y, x = x,param_type=cfg.parameterisation,gamma=gamma_for(cfg),output_energy_scaling=out_scale)

