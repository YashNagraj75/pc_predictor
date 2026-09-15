"""Regenerate the numbers used in the progress report, from the notebook's
exact configuration. Writes report_data.json."""
import json
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import jpc
import optax
import equinox as eqx
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

SEED = 4329
INPUT_DIM, WIDTH, DEPTH, OUTPUT_DIM = 784, 128, 30, 10
ACT_FN = "relu"
ACTIVITY_LR, PARAM_LR, BATCH_SIZE = 5e-1, 1e-1, 64


class MNIST(datasets.MNIST):
    def __init__(self, train, save_dir="data"):
        super().__init__(save_dir, download=False, train=train,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize(0.1307, 0.3081)]))

    def __getitem__(self, i):
        img, t = super().__getitem__(i)
        return img.view(-1), torch.eye(10)[t]


torch.manual_seed(SEED)
np.random.seed(SEED)
train_loader = DataLoader(MNIST(True), batch_size=BATCH_SIZE, shuffle=True,
                          drop_last=True)
key = jr.PRNGKey(SEED)
img0 = np.asarray(next(iter(train_loader))[0])

out = {}

# ---- forward-pass activity norms at init, full depth profile
for pt in ("mupc", "sp"):
    m = jpc.make_mlp(key, INPUT_DIM, WIDTH, DEPTH, OUTPUT_DIM, ACT_FN,
                     param_type=pt)
    sk = jpc.make_skip_model(DEPTH)
    a = jpc.init_activities_with_ffwd(m, img0, skip_model=sk, param_type=pt)
    out["norms_" + pt] = [float(jnp.linalg.norm(z, axis=-1).mean()) for z in a]

# ---- SP control at depth 30: how many iterations before divergence
losses = []
m = jpc.make_mlp(key, INPUT_DIM, WIDTH, DEPTH, OUTPUT_DIM, ACT_FN,
                 param_type="sp")
sk = jpc.make_skip_model(DEPTH)
aopt = optax.sgd(ACTIVITY_LR)
popt = optax.adam(PARAM_LR)
pstate = popt.init((eqx.filter(m, eqx.is_array), sk))
for it, (img, lab) in enumerate(train_loader):
    img, lab = img.numpy(), lab.numpy()
    act = jpc.init_activities_with_ffwd(m, img, skip_model=sk, param_type="sp")
    astate = aopt.init(act)
    loss = float(jpc.mse_loss(act[-1], lab))
    losses.append(loss)
    if not np.isfinite(loss):
        break
    for t in range(DEPTH):
        r = jpc.update_pc_activities(params=(m, sk), activities=act, optim=aopt,
                                     opt_state=astate, output=lab, input=img,
                                     param_type="sp")
        act, astate = r["activities"], r["opt_state"]
    r = jpc.update_pc_params(params=(m, sk), activities=act, optim=popt,
                             opt_state=pstate, output=lab, input=img,
                             param_type="sp")
    m, sk, pstate = r["model"], r["skip_model"], r["opt_state"]
    if it > 40:
        break
out["sp_losses"] = losses
out["sp_diverge_iter"] = len(losses)

json.dump(out, open("report_data.json", "w"))
print("sp diverged at iter", out["sp_diverge_iter"])
print("sp loss trace", [float("%.4g" % v) for v in losses])
print("mupc norms  first/mid/last-hidden/out: %.3f %.3f %.3f %.3f"
      % (out["norms_mupc"][0], out["norms_mupc"][15], out["norms_mupc"][-2],
         out["norms_mupc"][-1]))
print("sp   norms  first/mid/last-hidden/out: %.3f %.3f %.3f %.3f"
      % (out["norms_sp"][0], out["norms_sp"][15], out["norms_sp"][-2],
         out["norms_sp"][-1]))
