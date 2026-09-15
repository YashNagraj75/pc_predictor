#!/bin/bash
cd /media/md_lab/hd_21/preprocessed/pc_predictor
.venv/bin/python timing.py 2>&1 | grep -vE "SyntaxWarning|^  |^\s*\^"
echo "=== TIMING_DONE ==="
