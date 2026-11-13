#!/bin/bash
set -x
ROOT=${PCP_ROOT:?set PCP_ROOT, see .env.example}
cd $ROOT
if [ ! -x .venv/bin/python ]; then
  "${UV:-uv}" venv --python 3.11 .venv
fi
"${UV:-uv}" pip install --python .venv/bin/python "jax[cuda12]" equinox optax numpy h5py
echo "=== GPU VERIFY ==="
.venv/bin/python -c "
import jax, jaxlib, equinox, optax, h5py
print('jax', jax.__version__, '| jaxlib', jaxlib.__version__)
print('equinox', equinox.__version__, '| optax', optax.__version__)
print('backend:', jax.default_backend())
print('devices:', jax.devices())
import jax.numpy as jnp
x = jnp.ones((8192,8192)); print('gpu matmul:', float((x@x)[0,0]))
"
echo "=== PROVISION_DONE rc=$? ==="
