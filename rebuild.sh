#!/bin/bash
set -x
ROOT=/media/md_lab/hd_21/preprocessed/pc_predictor
case "$ROOT" in *pc_predictor) ;; *) echo "GUARD: refusing, unexpected ROOT"; exit 1;; esac
cd $ROOT
export UV_LINK_MODE=copy
rm -rf "$ROOT/.venv" /tmp/probe3.13 /tmp/probe3.12
/snap/bin/uv venv --python 3.13 "$ROOT/.venv"
/snap/bin/uv pip install --python "$ROOT/.venv/bin/python"     "jax[cuda12]==0.11.0" "equinox==0.13.8" "optax==0.2.8" h5py
/snap/bin/uv pip install --python "$ROOT/.venv/bin/python" --no-deps     "jpc @ git+https://github.com/thebuckleylab/jpc.git@7d6cc05742d28b543311e6c4d4ca140ba996f8b5"
echo "=== FINAL VERIFY ==="
"$ROOT/.venv/bin/python" -c "
import importlib.metadata as md, sys
print('python', sys.version.split()[0])
for p in ['jax','jaxlib','equinox','optax','jpc','numpy','h5py','jaxtyping']:
    try: print(p, md.version(p))
    except Exception: print(p, 'MISSING')
import jax, jpc
print('backend', jax.default_backend(), jax.devices())
import jax.numpy as jnp; x = jnp.ones((8192,8192)); print('matmul', float((x@x)[0,0]))
print('IMPORTS_OK')
"
echo "=== REBUILD_DONE ==="
