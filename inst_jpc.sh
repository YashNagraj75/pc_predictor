#!/bin/bash
set -x
cd /media/md_lab/hd_21/preprocessed/pc_predictor
/snap/bin/uv pip install --python .venv/bin/python   "jpc @ git+https://github.com/thebuckleylab/jpc.git@7d6cc05742d28b543311e6c4d4ca140ba996f8b5"
echo "=== versions ==="
.venv/bin/python -c "
import importlib.metadata as md
for p in ['jax','jaxlib','equinox','optax','jpc','numpy','h5py']:
    try: print(p, md.version(p))
    except Exception: print(p, 'MISSING')
import jax; print('backend', jax.default_backend(), jax.devices())
import jpc; print('jpc import ok')
"
echo "=== JPC_DONE ==="
