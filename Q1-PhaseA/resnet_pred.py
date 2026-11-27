import math
import jax, jax.numpy as jnp, equinox as eqx
import vit_ijepa as V

# Convolutional ResNet I-JEPA predictor (plan_resnet_predictor.md) with the transformer predictor's ePC interface:
# embed -> n residual blocks -> head, h_j = h_{j-1} + scale * f_j(h_{j-1}), one PC node per block, state = the full
# g x g grid (g*g, w).  Pre-activation blocks (GroupNorm, GELU, 3x3 conv) x 2, stride 1, no downsampling.  3x3 convs are
# written as one matrix multiply over the 9 shifted neighbourhoods, so every op runs on the GEMM path the GPU gate verifies.


class GN(eqx.Module):
    """GroupNorm over (positions, channels in group) of one sample (N, C), with affine gain and bias."""
    g: jax.Array
    b: jax.Array
    groups: int = eqx.field(static=True)

    def __init__(self, c, groups=8):
        self.g, self.b, self.groups = jnp.ones(c), jnp.zeros(c), groups

    def __call__(self, x):
        N, C = x.shape; z = x.reshape(N, self.groups, C // self.groups)
        m = z.mean((0, 2), keepdims=True); v = z.var((0, 2), keepdims=True)
        return ((z - m) / jnp.sqrt(v + 1e-5)).reshape(N, C) * self.g + self.b


def conv3(lin, h, g):
    """3x3 convolution, zero padding, of a (g*g, C) grid state as one GEMM: (g*g, 9C) @ W^T with lin = Lin(9C, C_out)."""
    x = jnp.pad(h.reshape(g, g, -1), ((1, 1), (1, 1), (0, 0)))
    cols = jnp.concatenate([x[dy:dy + g, dx:dx + g] for dy in range(3) for dx in range(3)], -1)
    return lin(cols.reshape(g * g, -1))


class ResBlock(eqx.Module):
    """Pre-activation residual branch: conv3(GELU(GN(conv3(GELU(GN(h))))))."""
    n1: GN
    c1: V.Lin
    n2: GN
    c2: V.Lin

    def __init__(self, key, w, groups):
        k = jax.random.split(key, 2)
        self.n1, self.c1, self.n2, self.c2 = GN(w, groups), V.Lin(k[0], 9 * w, w), GN(w, groups), V.Lin(k[1], 9 * w, w)

    def __call__(self, h, g):
        x = conv3(self.c1, jax.nn.gelu(self.n1(h), approximate=False), g)
        return conv3(self.c2, jax.nn.gelu(self.n2(x), approximate=False), g)


class ResPredictor(eqx.Module):
    """ResNet predictor over the patch grid; context and mask tokens scattered at their positions, zeros elsewhere."""
    inp: V.Lin
    out: V.Lin
    pos: jax.Array
    mask_tok: jax.Array
    blocks: list
    norm: GN
    scale: float = eqx.field(static=True)
    g: int = eqx.field(static=True)

    def __init__(self, key, cfg, depth, width=112, groups=8):
        k = jax.random.split(key, depth + 4)
        self.inp, self.out = V.Lin(k[0], cfg.dim, width), V.Lin(k[1], width, cfg.dim)
        self.pos = V.sincos_2d(cfg.grid, width)
        self.mask_tok = 0.02 * jax.random.truncated_normal(k[3], -2, 2, (width,))
        self.blocks = [ResBlock(k[4 + i], width, groups) for i in range(depth)]
        self.norm = GN(width, groups)
        self.scale = 1.0 / math.sqrt(depth)          # same rule as the transformer: 1 / sqrt(number of PC nodes)
        self.g = cfg.grid

    @property
    def n_sub(self):
        return len(self.blocks)

    def pc_masks(self, cvalid, tvalid):
        """(kv, row, tm): every grid position feeds the output through the convs, so each carries a PC error (review 2026-11-27)."""
        kv = jnp.ones(self.g * self.g, bool)
        return kv, kv[:, None].astype(jnp.float32), tvalid[:, None].astype(jnp.float32)

    def embed_m(self, ctx, cidx, cvalid, tidx, tvalid):
        """Grid input: (context embedding + pos) at context, (mask token + pos) at targets, zeros elsewhere."""
        pos = jax.lax.stop_gradient(self.pos[0])
        xc = (self.inp(ctx) + pos[cidx]) * cvalid[:, None]
        xt = (self.mask_tok[None] + pos[tidx]) * tvalid[:, None]
        z = jnp.zeros((self.g * self.g, pos.shape[-1]))
        return z.at[cidx].add(xc).at[tidx].add(xt)

    def f(self, j, h, kvalid):
        return self.blocks[j](h, self.g)

    def head_at(self, h, tidx):
        return self.out(self.norm(h)[tidx])

    def __call__(self, ctx, cidx, cvalid, tidx, tvalid):
        h = self.embed_m(ctx, cidx, cvalid, tidx, tvalid)
        for j in range(self.n_sub):
            h = h + self.scale * self.f(j, h, None)
        return self.head_at(h, tidx)
