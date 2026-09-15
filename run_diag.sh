#!/bin/bash
cd /media/md_lab/hd_21/preprocessed/pc_predictor
.venv/bin/python diag_retrace.py 2>&1 | grep -vE "SyntaxWarning|^  with respect|^\s*\^"
echo "=== DIAG_DONE ==="
