# findings

Figures, tables and write-ups from the PC-vs-BP encoder study, grouped by the
experiment that produced them. Folders are in chronological order, so reading
them in sequence follows how the question actually changed.

Assembled 2026-09-19. Every file here is a copy of a saved
artifact; the code that produced them is in `Q1-PhaseA/` and the run directories
with weights are in `Q1-PhaseA/runs/`.

## Read this first: what is still valid

`10_step_size_bug` invalidates the predictive-coding side of folders 04-09.

jpc's energy is a mean over the batch, so the activity gradient for one sample is
1/B of that sample's own gradient. The old `activity_lr = 0.5` at batch 256 therefore
realised a per-sample step of about 0.002. Measured objective reached by the
relaxation at each PER-SAMPLE step size (B=64 sweep; lower is better; our
bugged setting is the leftmost):

    0.0078125→0.794 | 0.03125→0.667 | 0.125→0.448 | 0.25→0.331 | 0.5→0.550 | 1→0.939 | 2→0.872

At the step we were using, 1 of 10 layers moved off the feedforward initialisation.
Credit never reached the deep layers, so the PC arm trained only its top few layers.

**Consequences.** The BP numbers in 04-09 stand -- backprop never uses the relaxation.
The PC numbers are real measurements of a mis-stepped relaxation, not of predictive
coding. In particular the folder-08 conclusion ('PC does not beat its own random
initialisation') is a correct observation whose stated cause was wrong. It is being
re-measured at the corrected step; until that lands, do not quote PC-vs-BP
comparisons from 04-09 as properties of predictive coding.

## Index

### `01_proposal_and_design/`

The original proposal and the design review that turned it into an experiment plan.

- `Towards Hierarchical Predictive Coding Predictors for Joint Embedding World Models.pdf` (995 KB)
- `pc_jepa_design_diagram.png` (500 KB)
- `pc_jepa_design_review.md` (17 KB)

### `02_graph_pc_exercises/`

Teaching phase: predictive coding on arbitrary computation graphs, including residual topologies. Established that PC reproduces backprop gradients at cosine 1.0 and that the error-wave front reaches a node at its shortest-path distance to the output.

- `exercise6_visualization.html` (21 KB)
- `graph_pc_results.png` (290 KB)
- `graph_pc_solutions.py` (18 KB)
- `graph_pc_walkthrough.ipynb` (55 KB)

### `03_collapse_diagnosis/`

Why the encoder collapsed before SIGReg was wired correctly, and how our setup differs from the published LeJEPA one.

- `lejepa_vs_ours_collapse_diagnosis.md` (7 KB)
- `sigreg_collapse_trap.png` (161 KB)

### `04_q1_fixed_grid/`

First PC-vs-BP grid at a fixed short budget.

- `q1_fixed_grid.png` (212 KB)
- `q1_fixed_grid_report.md` (4 KB)
- `q1_fixed_grid_results.csv` (0 KB)

### `05_q1_400epoch_grid/`

The same grid at the 400-epoch (17,200-step) budget.

- `e400.json` (40 KB)
- `q1_e400_grid.png` (227 KB)
- `q1_e400_report.md` (4 KB)
- `q1_e400_results.csv` (0 KB)

### `06_q1_inference_budget_sweep/`

Sweeping the PC inference budget T. More relaxation made the representation WORSE, which is what first pointed at the relaxation itself.

- `pc_inference_steps.png` (275 KB)
- `q1_tsweep.png` (211 KB)
- `q1_tsweep_report.md` (3 KB)
- `q1_tsweep_results.csv` (0 KB)
- `tsweep.json` (11 KB)

### `07_latent_quality_probes/`

The probe suite: what the two arms' embeddings actually contain. Scene-factor probes, class probes, decoder reconstructions and t-SNE.

- `q1_latent_probes.csv` (1 KB)
- `q1_latent_probes_by_target.csv` (0 KB)
- `q1_latent_probes_long.csv` (4 KB)
- `q1_latent_quality.png` (164 KB)
- `recon_bp_sigreg_e400.png` (63 KB)
- `recon_pc_sigreg_T16_coupled_e400.png` (63 KB)
- `recon_pc_sigreg_T64_coupled_Tsweep.png` (61 KB)
- `tsne_bp_sigreg_e400.png` (276 KB)
- `tsne_pc_sigreg_T16_coupled_e400.png` (283 KB)
- `tsne_pc_sigreg_T64_coupled_Tsweep.png` (278 KB)

### `08_phase0_controls/`

THE TURNING POINT. An untrained-encoder control (n=3) plus augmentation-invariance probes and a seed floor. PC-as-encoder scored INSIDE the untrained range on every probe, which retracted the earlier pro-PC reading.

- `ph0_augmentation_invariance.csv` (0 KB)
- `ph0_random_init_reference.csv` (1 KB)
- `ph0_scene_nmse.csv` (0 KB)
- `ph0_vs_random_init.png` (93 KB)

### `09_depth_starvation/`

Localising the failure by depth: per-layer weight displacement plus the probe trajectory. PC's updates reached only the top 3 of 10 layers, and every PC run's best kNN was at step 0.

- `q1_diag_learning.csv` (1 KB)
- `q1_pc_depth_starvation.png` (112 KB)

### `10_step_size_bug/`

ROOT CAUSE. jpc's energy is a batch MEAN, so activity_lr=0.5 at batch 256 realised a per-sample step of 0.002 against an optimum near 0.06-0.25. The relaxation never left its initialisation. This invalidates every PC number in folders 04-09.

- `q1_diag_inference.csv` (1 KB)

### `11_gradient_flow_diagrams/`

Explanatory diagrams: how credit flows in backprop vs predictive coding on a residual net, and the inference/learning decoupling.

- `diagram_bp_flow.png` (293 KB)
- `diagram_decoupling.png` (261 KB)
- `diagram_pc_flow.png` (388 KB)
- `pc_vs_bp_gradient_flow.html` (33 KB)

### `12_literature/`

Numbers transcribed from papers we read, for comparison against our own.

- `fig_mupc_results.png` (176 KB)
- `mpc_table1_vs_ijepa.csv` (0 KB)

### `13_reports/`

Assembled write-ups. Q1_PhaseA_report.pdf is the presentable one.

- `Q1_PhaseA_report.pdf` (1759 KB)
- `README.md` (15 KB)
- `pc_progress_report.pdf` (195 KB)
- `progress_report.md` (25 KB)

## Not in here

- Source papers we read (mu-PC, PC-ALM, LeJEPA, MPC, Whittington & Bogacz) -- they are
  inputs, not findings, and are in the artifact store.
- Trained weights (`params.eqx`) -- they live in `Q1-PhaseA/runs/<run>/` and are too
  large to duplicate.
- Raw per-step training logs -- `Q1-PhaseA/runs/<run>/log.json`.
