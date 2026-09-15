#!/bin/bash
cd /media/md_lab/hd_21/preprocessed/pc_predictor
.venv/bin/python -c "import jax, jpc; print('jpc import OK'); print('backend', jax.default_backend(), jax.devices())" 2>&1 | grep -vE "SyntaxWarning|^  |^\s*\^|^\s*~"
echo "=== SMOKE ==="
.venv/bin/python smoke_gpu.py 2>&1 | grep -vE "SyntaxWarning|^  \\\$|^  with respect|^  \$"
echo "=== SMOKE2_DONE ==="
