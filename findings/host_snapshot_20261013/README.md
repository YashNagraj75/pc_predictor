# Host-only files recovered at the 2026-10-13 repo sync

When the GPU host checkout was reset to GitHub `main` (879e8ab), these 11 files on the host differed from every
version in git history. They are kept here with their original paths under `Q1-PhaseA/` rather than overwriting the
committed copies, because they are later re-runs, not corrections:

- `runs/*/probes.json`, `runs/*/probes_recon.png` (2026-09-16): re-probes of the e400 / T-sweep / random-init runs on the host.
- `probes_summary.csv` (2026-09-22): the summary written by the last host probe job (T=64 screen).
- `alm_selected.txt` (2026-09-22): `0.0625 2`, the PC-ALM selection at T=64 (the committed copy is an earlier selection).

The full set of 54 files the reset replaced is also on the host under `_sync_backup_20261013_1611/`.
