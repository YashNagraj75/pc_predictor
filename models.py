"""Model definitions for mu-PC experiments.

Kept in a module (rather than a notebook cell) so the definitions survive
notebook save/restore cycles.
"""

import math
from typing import Callable, List

import equinox as eqx
import equinox.nn as nn
import jax.random as jr
import jpc


class ScaledLinear(eqx.Module):
    "Scaled Linear tranformations"
    linear: nn.Linear
    scaling: float = eqx.static_field()

    def __init__(self, in_features, out_features, *, key, scaling=1, param_type="sp", use_bias=True):
        keys = jr.split(key, 2)
        linear = nn.Linear(in_features, out_features, use_bias=use_bias, key=keys[0])
        if param_type == "mupc":
            W = jr.normal(keys[1], linear.weight.shape)  # Scale hidden weights by (Nl-1 * L) power 1/2
            linear = eqx.tree_at(lambda l: l.weight, linear, W)

        self.linear = linear
        self.scaling = scaling

    def __call__(self, x):
        return self.scaling * self.linear(x)


class ResNetBlock(eqx.Module):
    "Identity residual blaock scaled linear block"
    scaled_linear: eqx.Module
    act_fn: Callable = eqx.static_field()

    def __init__(self, in_features, out_features, *, key, scaling=1, param_type="sp", use_bias=True, act_fn="linear"):
        self.act_fn = act_fn
        self.scaled_linear = ScaledLinear(
            in_features,
            out_features,
            key=key,
            scaling=scaling,
            param_type=param_type,
            use_bias=use_bias
        )

    def __call__(self, x):
        res = x
        x = self.act_fn(x)
        return self.scaled_linear(x) + res


class Readout(eqx.Module):
    "Final layer which runs the act_fn and linear layer"
    scaled_linear: eqx.Module
    act_fn: Callable = eqx.static_field()

    def __init__(self, in_features, out_features, *, key, scaling=1, param_type="sp", use_bias=True, act_fn="linear"):
        self.act_fn = act_fn
        self.scaled_linear = ScaledLinear(
            in_features,
            out_features,
            key=key,
            scaling=scaling,
            param_type=param_type,
            use_bias=use_bias
        )

    def __call__(self, x):
        x = self.act_fn(x)
        return self.scaled_linear(x)


class FCResnet(eqx.Module):
    "Fully connected Resnet compatible with different parametrisations"
    layers: List[eqx.Module]

    def __init__(self, key, in_dim, out_dim, *, width, depth, act_fn="linear", use_bias=False, param_type="sp"):
        act_fn = jpc.get_act_fn(act_fn)
        if param_type == "sp":
            in_scaling = 1
            hidden_scaling = 1
            out_scaling = 1
        if param_type == "mupc":  # Exact parameterization from the paper
            in_scaling = 1 / math.sqrt(in_dim)
            hidden_scaling = 1 / math.sqrt(width * depth)
            out_scaling = 1 / width

        keys = jr.split(key, depth)  # Splitting seed for each layer for reproducaibility

        self.layers = [ScaledLinear(
            key=keys[0],
            in_features=in_dim,
            out_features=width,
            scaling=in_scaling,
            param_type=param_type,
            use_bias=use_bias
        )]

        for i in range(1, depth - 1):
            self.layers.append(ResNetBlock(
                width,
                width,
                key=keys[i],
                scaling=hidden_scaling,
                param_type=param_type,
                use_bias=use_bias,
                act_fn=act_fn
            ))

        self.layers.append(Readout(
            width,
            out_dim,
            key=keys[-1],
            scaling=out_scaling,
            param_type=param_type,
            use_bias=use_bias,
            act_fn=act_fn
        ))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, idx):
        return self.layers[idx]
