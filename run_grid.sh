#!/bin/bash
cd /media/md_lab/hd_21/preprocessed/pc_predictor
.venv/bin/python run_q1grid.py 2>&1 | grep -vE "SyntaxWarning|^  with respect|^[[:space:]]*\^"
echo "=== SCRIPT_EXIT ==="
