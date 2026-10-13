#!/usr/bin/env bash
# Phase P track 1 grid: {bp, pc} x {bp, rand} backbones x seeds {0,1,2}, layer emb (pre-registered),
# T=20, eta_h=0.05 (pre-registered 2026-10-12 from the real-batch stability sweep), 6000 steps.
cd "$(dirname "$0")"; export JAX_PLATFORMS=cpu; mkdir -p pred_runs
for arm in bp pc; do for bb in bp rand; do for s in 0 1 2; do
  f=pred_runs/ff_${arm}_${bb}_emb_s$s.json; [ -f $f ] && continue
  ../.venv/bin/python -W ignore pred_ff.py train --data pred_latents.npz --backbone $bb --layer emb --arm $arm --seed $s \
      --T 20 --eta-h 0.05 --steps 6000 --out $f 2>&1 | grep -E "step 6000|TEST|Error|Traceback"
done; done; done
echo GRID_DONE
