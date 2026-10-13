#!/usr/bin/env bash
# Corrected PC arm (declared after the first grid, 2026-10-12): curvature-tracking activity step
# eta_h = 0.9 / (1 + 2 max_l ||W_l||^2), recomputed every step.  BP arm unchanged (reused).
cd "$(dirname "$0")"; export JAX_PLATFORMS=cpu
for bb in bp rand; do for s in 0 1 2; do
  f=pred_runs/ff_pcA_${bb}_emb_s$s.json; [ -f $f ] && continue
  ../.venv/bin/python -W ignore pred_ff.py train --data pred_latents.npz --backbone $bb --layer emb --arm pc --adaptive-eta --seed $s \
      --T 20 --steps 6000 --out $f 2>&1 | grep -E "step 6000|TEST|Error|Traceback"
done; done; echo GRID_DONE
