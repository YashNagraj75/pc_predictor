#!/bin/bash
cd /media/md_lab/hd_21/preprocessed/pc_predictor
.venv/bin/python timing2.py 2>&1 | grep -vE "SyntaxWarning|^  with respect|^[[:space:]]*\^"
echo "=== TIMING2_END ==="
