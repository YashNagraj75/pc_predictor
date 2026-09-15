#!/bin/bash
set -x
cd /media/md_lab/hd_21/preprocessed/pc_predictor
# tear out the mismatched jax stack jpc's resolver installed
/snap/bin/uv pip uninstall --python .venv/bin/python   jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt jpc optax
# restore the working CUDA stack
/snap/bin/uv pip install --python .venv/bin/python "jax[cuda12]==0.10.2" "optax==0.2.8"
echo "=== jax alone ==="
.venv/bin/python -c "import jax; print(jax.__version__, jax.default_backend(), jax.devices())"
# add jpc WITHOUT letting it touch jax, mirroring the local install
/snap/bin/uv pip install --python .venv/bin/python --no-deps   "jpc @ git+https://github.com/thebuckleylab/jpc.git@7d6cc05742d28b543311e6c4d4ca140ba996f8b5"
echo "=== final ==="
.venv/bin/python -c "
import importlib.metadata as md
for p in ['jax','jaxlib','equinox','optax','jpc','numpy','h5py']:
    try: print(p, md.version(p))
    except Exception: print(p, 'MISSING')
import jax, jpc, equinox, optax
print('backend', jax.default_backend(), jax.devices())
import jax.numpy as jnp; x=jnp.ones((4096,4096)); print('matmul', float((x@x)[0,0]))
print('IMPORTS_OK')
"
echo "=== REPAIR_DONE ==="
