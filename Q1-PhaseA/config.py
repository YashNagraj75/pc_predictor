"""Phase A configuration.  Every knob for both arms lives here; nothing else
   should hard-code a hyperparameter.

Defaults for the SIGReg block are taken from LeJEPA Algorithm 1 and Table 1(a).
Defaults for the encoder block mirror the mu_pc.ipynb notebook so the existing
muPC machinery transfers unchanged.
"""
from dataclasses import dataclass, field, asdict
from typing import Literal, Tuple
import json


@dataclass(frozen=True)
class DataCfg:
    dataset: Literal["galaxy10", "mnist"] = "galaxy10"
    root: str = "../data"
    img_size: int = 32              # all views resized to this; FC encoder needs fixed dim
    grayscale: bool = False
    n_train: int = -1               # -1 = all
    n_test: int = 2000
    seed: int = 0

    # ---- view generation (LeJEPA Sec 5) --------------------------------
    # Algorithm 2: "For non-ViT architectures (e.g., ResNet),
    #               set global_views = all_views."
    # Our encoder is a fully-connected residual net -> non-ViT -> V_g == V.
    n_views: int = 4
    n_global_views: int = 4         # keep == n_views for the FC encoder
    crop_area: Tuple[float, float] = (0.4, 1.0)
    max_rotation: float = 3.14159265   # radians; galaxies have no canonical
                                       # orientation, so full rotation is valid.
                                       # Set to 0.0 for MNIST (digits do).
    hflip: bool = True
    brightness: float = 0.2
    contrast: float = 0.2

    @property
    def input_dim(self) -> int:
        c = 1 if self.grayscale else 3
        return self.img_size * self.img_size * c


@dataclass(frozen=True)
class EncoderCfg:
    """muPC-parameterised fully-connected residual net (see mu_pc.ipynb)."""
    depth: int = 8                  # number of residual blocks
    width: int = 128
    embed_dim: int = 128            # K -- dimension SIGReg acts on
    act: Literal["relu", "tanh", "gelu"] = "relu"
    parameterisation: Literal["mupc", "sp"] = "mupc"


@dataclass(frozen=True)
class SIGRegCfg:
    """LeJEPA Definition 2 + Algorithm 1."""
    num_slices: int = 256           # |A|; Table 1(a) sweeps 512/2048, 256 is cheap
    t_max: float = 5.0              # integration range [-t_max, t_max]
    n_points: int = 17              # quadrature points; Table 1(a): 5/17/41 all fine
    scale_by_n: bool = True         # classical EP scales the integral by N
    resample_directions: bool = True  # paper: resampling each STEP is a free win.
                                      # NOTE: must stay FIXED within one PC
                                      # relaxation, else the energy is not stationary.
    lambd: float = 0.05             # weight on SIGReg vs the invariance term


@dataclass(frozen=True)
class PCCfg:
    """Predictive-coding inference (relaxation) settings."""
    T: int = 16                     # inference steps per weight update.
                                    # THE independent variable of this study.
    activity_lr: float = 0.5
    init: Literal["feedforward", "zeros"] = "feedforward"
    # How the output node is driven when the loss is not a per-sample target.
    #   "coupled" : eps_out = dL/dz_out recomputed every relaxation step from the
    #               current batch.  Exact, but couples samples -> the energy is
    #               no longer a sum of per-sample energies.
    #   "frozen"  : eps_out = dL/dz_out computed ONCE from the feedforward
    #               embeddings and held fixed for all T steps.  Factorises over
    #               the batch.  Direct analogue of the `fixed_prediction`
    #               assumption.  <-- build and test BOTH.
    output_drive: Literal["coupled", "frozen"] = "coupled"


@dataclass(frozen=True)
class OptCfg:
    param_lr: float = 1e-3
    batch_size: int = 256           # Table 1(c): 256-512 best
    n_steps: int = 2000
    eval_every: int = 200
    grad_clip: float = 1.0
    seed: int = 0
    # LeJEPA MINIMAL.md L154-161: AdamW(weight_decay=5e-2 ViT / 5e-4
    # ResNet) with LinearLR warmup 1 epoch then CosineAnnealingLR.
    # 11008 samples / bs 256 = 43 steps per epoch; 100 steps ~ 2.3 epochs.
    lr_schedule: Literal["warmup_cosine", "constant"] = "warmup_cosine"
    warmup_steps: int = 100
    weight_decay: float = 5e-4      # ResNet value; ours is a residual net


@dataclass(frozen=True)
class RunCfg:
    arm: Literal["bp", "pc"] = "pc"
    use_sigreg: bool = True         # False -> invariance only (collapse control)
    # "bn": parameterless BatchNorm over the pooled (V*B) axis before BOTH
    # loss terms, standing in for LeJEPA's BatchNorm1d projector.
    # "none" reproduces the original (collapsing) setup.
    projector: Literal["bn", "none"] = "bn"
    tag: str = ""
    out_dir: str = "runs"
    data: DataCfg = field(default_factory=DataCfg)
    enc: EncoderCfg = field(default_factory=EncoderCfg)
    sig: SIGRegCfg = field(default_factory=SIGRegCfg)
    pc: PCCfg = field(default_factory=PCCfg)
    opt: OptCfg = field(default_factory=OptCfg)

    @property
    def name(self) -> str:
        bits = [self.arm, "sigreg" if self.use_sigreg else "nosigreg"]
        if self.arm == "pc":
            bits += [f"T{self.pc.T}", self.pc.output_drive]
        if self.tag:
            bits.append(self.tag)
        return "_".join(bits)

    def to_json(self, path=None):
        s = json.dumps(asdict(self), indent=2, default=str)
        if path:
            open(path, "w").write(s)
        return s


# The 2x2 (x2) grid Q1 actually needs.
def q1_grid(**overrides):
    """arm x use_sigreg -> the four Phase A runs.  See README section 'The runs'."""
    out = []
    for arm in ("bp", "pc"):
        for use_sigreg in (True, False):
            out.append(RunCfg(arm=arm, use_sigreg=use_sigreg, **overrides))
    return out
