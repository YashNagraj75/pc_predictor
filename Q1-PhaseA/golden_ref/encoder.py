"""Phase A encoder -- muPC-parameterised fully-connected residual net.

>>> THIS FILE IS THE PART WE IMPLEMENT TOGETHER.  The helpers in sigreg.py,
>>> metrics.py, data.py and config.py are complete and tested; this one and
>>> train.py are deliberately skeletons.

The contract the rest of the code depends on
--------------------------------------------
`init(key, cfg) -> params`
`forward(params, x) -> (z, hs)` where
    x  : (B, input_dim)
    z  : (B, embed_dim)   the embedding SIGReg acts on
    hs : list of (B, width) intermediate activations, LOW to HIGH

Returning `hs` is not optional decoration -- the whole Q1 measurement plan
rests on it.  SIGReg pins the embedding's marginal to N(0, I), so the
embedding-layer spectrum measures the regulariser, not the representation.
Every spectral metric has to be read at the hidden layers, and layer-wise CKA
needs the full stack from both arms.  An encoder that only returns `z` makes
Q1 unanswerable.

Architecture, following mu_pc.ipynb
-----------------------------------
    z_0            = x                                  (clamped input)
    z_l            = z_{l-1} + W_l phi(z_{l-1})          l = 1..L  (residual)
    z_out          = W_out z_L                           (readout)

Two parameterisations, selected by cfg.parameterisation:
  "sp"   -- standard: W ~ N(0, 1/fan_in)
  "mupc" -- the paper's depth-muP-style scaling.  This is the whole point of
            muPC: it is what makes learning rates transfer across depth and
            what stops the inference landscape's conditioning degrading with
            L.  Get the scaling right or the depth axis is meaningless.

OPEN ITEM carried over from reading mu_pc.ipynb: the notebook's `ScaledLinear`
docstring says hidden weights are scaled by (N_{l-1} * L)^{1/2}, but the code
as written replaces the weight with a standard normal draw and never divides
by that factor.  Either jpc applies it downstream or it is a real gap.
RESOLVE THIS FIRST -- everything about the depth sweep depends on it.
Cross-check against the muPC paper's Table of parameterisations before
trusting any depth-transfer result.

A second open item: mu_pc.ipynb contains TWO model definitions, a
jpc-constructed one (`make_mlp` / `make_skip_model`) and a hand-rolled
`FCResnet`.  Training was invoked on the jpc one; `FCResnet` was built but
never trained, and the cell that would have checked they are equivalent never
ran.  Decide which is the reference implementation here, once.
"""
from typing import List, Tuple

import jax
import jax.numpy as jnp
import jpc
from config import EncoderCfg
import math

ACTS = {"relu": jax.nn.relu, "tanh": jnp.tanh, "gelu": jax.nn.gelu}


def init(key, cfg: EncoderCfg, input_dim: int):
    """
    Return a PC trainable FC residual net, takes inspiration form mu-PC
    """
    L = cfg.depth + 2
    model = jpc.make_mlp(key,input_dim,cfg.width,L,cfg.embed_dim, cfg.act,use_bias=False,param_type=cfg.parameterisation)
    skip = jpc.make_skip_model(L)
    return (model, skip)


def gamma_for(cfg:EncoderCfg):
    """
        Gamma for scaling of output to match SiGReg input dim
    """
    return 1.0 / math.sqrt(cfg.width) if cfg.parameterisation == "mupc" else None

def forward(params, x,cfg) :
    """TODO -- feedforward pass.

    Must return (z, hs).  `hs` is what makes the layer-wise measurements
    possible; see the module docstring.

    In PC terms this is the ZERO-ENERGY configuration when no target is
    attached: with nothing clamped at the output, the energy minimum is
    exactly the feedforward pass.  So at evaluation time both arms are
    evaluated by this same function, and the entire difference between arms
    lives in the learned weights -- not in any relaxation at test time.
    """
    model, skip = params
    acts = jpc.init_activities_with_ffwd(model,x,skip_model=skip,param_type=cfg.parameterisation,gamma=gamma_for(cfg))
    return acts[-1], list(acts[:-1])



def energy(params, activities, x,cfg:EncoderCfg,y=None, out_scale=None) -> jnp.ndarray:
    """TODO -- PC energy F = (1/2) sum_l || z_l - f_l(z_{l-1}) ||^2.

    `activities` is the list of free variables z_1..z_L, z_out.
    z_0 is clamped to `x` and is not in the list.

    Note there is NO output term here.  The loss enters through the output
    node's activity gradient, not through the energy -- see train.py.
    """
    model, skip = params
    return jpc.pc_energy_fn((model,skip),activities,activities[-1] if y is None else y, x = x,param_type=cfg.parameterisation,gamma=gamma_for(cfg),output_energy_scaling=out_scale)

