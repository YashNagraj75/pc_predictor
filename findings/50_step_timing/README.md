# 50 — Per-step training cost on one A4000 (2026-11-26)

`Q1-PhaseA/timing_vit.py` (commit a33a97e, rev 2 after 3 reviewers, no blocker), A4000 tmux job `_jobs/timing`, training
environment (Triton GEMMs off, TF32 off, highest matmul precision; asserted). Each arm's trained seed-0 predictor, one fixed
real batch of 256, final stage-A settings (T 20, 12 power iterations for ePC; 8 for standard PC). Every arm compiled first,
then 3 rounds of 10 timed steps in rotated order; seconds per step = encode + update + gain measurement / 50 (its schedule).
Only other GPU process: gnome-remote-desktop (161 MiB), present throughout.

| depth | backprop | gain-matched ePC | standard PC |
|---|---|---|---|
| 10 | 0.73 s | 14.6 s (20.0x) | 16.6 s (22.7x) |
| 6 | 0.46 s | 8.5 s (18.7x) | 9.7 s (21.2x) |

The update-only ratio is stable across rounds (ePC d10 21.3 / 20.9 / 20.9), so clock drift did not bias it.
Paper sentence: "On one A4000, a gain-matched ePC training step costs about 20x a backpropagation step (19-20x at depths 6
and 10); standard PC costs about 21-23x."
