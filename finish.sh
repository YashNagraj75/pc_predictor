#!/bin/bash
set -x
ROOT=/media/md_lab/hd_21/preprocessed/pc_predictor
cd $ROOT
export UV_LINK_MODE=copy
/snap/bin/uv pip install --python .venv/bin/python --no-deps    "diffrax==0.7.2" "lineax==0.1.1" "optimistix==0.1.0"    "wadler-lindig==0.1.7" "typing_extensions==4.16.0"
echo "=== VERIFY ==="
.venv/bin/python -c "
import importlib.metadata as md, sys
print('python', sys.version.split()[0])
for p in ['jax','jaxlib','equinox','optax','jpc','diffrax','lineax','optimistix','jaxtyping','numpy','h5py']:
    try: print(' ', p, md.version(p))
    except Exception: print(' ', p, 'MISSING')
import jax, jpc
print('backend', jax.default_backend(), jax.devices())
print('IMPORTS_OK')
" 2>&1 | grep -v SyntaxWarning | grep -v "^  with respect" | grep -v "^  \"\"\""
echo "=== SMOKE ==="
.venv/bin/python smoke_gpu.py 2>&1 | grep -v SyntaxWarning | grep -v "^  with respect"
echo "=== FINISH_DONE ==="
