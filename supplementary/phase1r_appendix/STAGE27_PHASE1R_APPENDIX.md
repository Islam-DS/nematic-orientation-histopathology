# Supplementary: Synthetic Orientation-Tensor Validation (Phase 1R)

Recovered, not rerun. Every number below is read from the existing frozen artifacts of the project's earlier synthetic-benchmark phase (`results_v2/synthetic_replication/phase1r_results.json`, `results_v2/config/phase1r_config.json`, `results_v2/plots/phase1r_*.png`, checkpoints `results_v2/checkpoints/p8m_phase1r.pt` and `cnn_control_phase1r.pt`). No experiment was rerun for this appendix; nothing here is new evidence, only a documented recovery of pre-existing evidence into the manuscript's supplementary material.

## Purpose and scope

**This section demonstrates that the orientation-tensor learning framework (a typed, rotation-equivariant network trained with a masked Q-loss) can learn accurately under controlled, noise-free geometry: idealized rendered ellipses with a known analytic ground-truth orientation.**

**It does NOT validate the histopathology result.** The two settings differ in every respect that matters for the main manuscript's negative finding: the synthetic ellipses have an exact, noise-free analytic ground truth generated directly at a known angle; the GlaS anatomical reference (Q_anat) is instead a mask-derived, per-gland statistical summary with no analytic ground truth, embedded in real tissue with staining variation, annotation uncertainty and biological irregularity. A model succeeding on the synthetic task establishes only that the architecture and loss are capable of learning *some* rank-2 orientation signal when the target is exact and the input is clean; it does not and cannot establish that a meaningful anatomical orientation signal exists in H&E histology images, or that any correspondence should have been expected on GlaS. The main text's negative result on GlaS stands on its own and is not qualified by this appendix.

## S1. Synthetic dataset

Idealized single-ellipse grayscale-render images, 128 x 128 px, generated at 4x supersampling and downsampled with anti-aliasing (`skimage.transform.resize`). Each image has an exact analytic ground-truth order magnitude S and director angle phi used to render it.

| Parameter | Value |
|---|---|
| Image size | 128 x 128 px (rendered at 4x supersampling, then downsampled) |
| Order magnitude S range (train/val/test) | 0.20 - 0.95 |
| Order magnitude S range (calibration) | 0.30 - 0.90 |
| Mean radius | 18 - 34 px |
| Position offset | -14 to +14 px |
| Contrast | 0.6 - 1.0 |
| Brightness | 0.0 - 0.15 |
| Blur (sigma) | 0.0 - 1.2 px |
| Pixel noise (std) | 0.0 - 0.04 |
| Dataset sizes | train 900, val 200, test 300, calibration 120 |
| Seeds | calibration 2001, train 42, val 142, test 242, equivariance probe 342, model init 7 |

Generation code: `code_v2/synthetic_shapes.py`, `code_v2/synthetic_shapes_v2.py`; benchmark driver: `code_v2/run_phase1r_benchmark.py`.

## S2. Models compared

| Model | Class | Group / channels | Parameters |
|---|---|---|---|
| p8m (equivariant) | `NematicNet` (`code_v2/nematic_model.py`) | p8m, N=8, c1=8, c2=8 | 11,480 |
| Plain CNN (non-equivariant control) | `PlainCNN` (`code_v2/baseline_cnn.py`) | c1=64, c2=64 | 105,408 |

Training: Adam, lr = 0.002, batch size 64, up to 60 epochs, early stopping patience 6, MSE loss on (q1, q2). p8m ran 35 epochs (best val loss 0.01026, 229 s); the plain CNN ran 15 epochs (best val loss 0.01385, 23 s). Both on the same NVIDIA RTX 3050 6GB GPU used throughout the project.

Two non-learned baselines were also evaluated on the same held-out test set: a naive image-gradient estimator and a classical structure-tensor estimator (the same oracle used elsewhere in the project's calibration).

## S3. Held-out test-set results (n = 300)

| Method | Mean angular error (deg, mod pi) | Median angular error (deg) | Mean S error | corr(S_pred, S_true) |
|---|---|---|---|---|
| Ground-truth analytic (trivial reference) | 0 (by definition) | 0 | 0 | 1 |
| Naive gradient baseline | 45.95 | 45.39 | 0.583 | 0.163 |
| Structure-tensor baseline | 9.87 | 2.67 | 0.321 | 0.342 |
| **p8m (equivariant)** | **0.86** | 0.69 | **0.117** | **0.825** |
| Plain CNN (non-equivariant control) | 2.23 | 1.72 | 0.134 | 0.723 |

**Figure S1** (`results_v2/plots/phase1r_angular_mae_by_method.png`, reproduced below): held-out angular accuracy by method, with the pre-registered 10 deg threshold marked. The naive-gradient baseline is far above it; the structure-tensor oracle sits almost exactly at it; p8m and the plain CNN are both well below it, with p8m the more accurate of the two.

> Figure S1. Phase 1R held-out test set angular accuracy, four methods, dashed line at the pre-registered 10 deg threshold.

## S4. Rotation consistency (synthetic)

| Quantity | p8m | Plain CNN |
|---|---|---|
| Mean normalized equivariance error (E_model) | 0.0792 | 0.1119 |
| Rendering-noise rotation floor (E_render, calibration) | 0.5412 | 0.5412 |

**Figure S2** (`results_v2/plots/phase1r_equivariance_vs_floor.png`, reproduced below): both models' equivariance error sits far below the rendering-noise floor, with p8m again the lower of the two.

> Figure S2. Phase 1R equivariance error compared with the rendering-noise-only rotation floor.

Per-angle equivariance error (p8m / plain CNN), degrees of the nine tested non-zero rotation angles (same grid as the main manuscript's Criterion D, {15,30,37,45,60,90,123,150,173}):

| Angle | p8m | Plain CNN |
|---|---|---|
| 15 | 0.082 | 0.057 |
| 30 | 0.092 | 0.103 |
| 37 | 0.094 | 0.121 |
| 45 | 0.105 | 0.139 |
| 60 | 0.096 | 0.168 |
| 90 | 0.048 | 0.202 |
| 123 | 0.085 | 0.180 |
| 150 | 0.093 | 0.115 |
| 173 | 0.061 | 0.030 |

p8m's error is roughly flat across angles (no strong growth at large angles); the plain CNN's error grows through the middle of the range and falls again near 173 deg.

## S5. Pre-registered criteria and verdict (as originally recorded)

| Criterion | Result |
|---|---|
| Mean angular error < 10 deg | met |
| Mean S error < 0.15 | met |
| Substantially better than the naive-gradient baseline | met |
| Equivariance error < calibrated threshold | met |
| **Clearly better than the non-equivariant control** | **NOT met** |

**Recorded verdict: NO-GO.** Both p8m and the plain CNN performed well in absolute terms; the pre-registered criterion required the equivariant model to be *clearly* better than the non-equivariant control, and by the recorded protocol it was not (p8m was numerically better on every reported metric, but not by the required margin). This NO-GO on the "clear superiority over a plain CNN" question is itself a substantive negative finding from the synthetic phase, and it foreshadows the main manuscript's finding that equivariance alone did not confer a decisive practical advantage in the harder, real-tissue setting either — though, as stated above, the two settings are not directly comparable and this appendix draws no quantitative link between them.

## S6. Interpretation

1. Under controlled, noise-free synthetic geometry with an exact analytic ground truth, the rank-2 nematic Q-tensor representation and the typed equivariant architecture used in this study **can** be learned to high angular accuracy (well under 1 deg mean error for p8m).
2. This demonstrates that the framework's mathematics, loss and training procedure function correctly and are capable of solving the orientation-tensor regression problem when the signal is unambiguous.
3. It does **not** demonstrate that the same is true in real histopathology, where the target is not an exact analytic angle but a statistical summary derived from imperfect gland-instance annotations embedded in noisy, biologically variable tissue. The main manuscript's negative result on GlaS is a separate, real-data finding and is not offset, qualified, or contradicted by the synthetic result reported here.
4. The synthetic phase's own NO-GO on "clear superiority over a non-equivariant control" is reported for completeness and honesty, not omitted because it is unfavorable.

## Provenance

| Artifact | Path | MD5 |
|---|---|---|
| Results | `results_v2/synthetic_replication/phase1r_results.json` | `c19caa822edf97a8f14a9cb4492b4423` (verified byte-identical in Stage 2's audit, reports/STAGE2_SPLIT_RECONCILIATION.md §9) |
| Config | `results_v2/config/phase1r_config.json` | not previously hashed in a report; read-only in this stage |
| p8m checkpoint | `results_v2/checkpoints/p8m_phase1r.pt` | not modified in this stage |
| CNN control checkpoint | `results_v2/checkpoints/cnn_control_phase1r.pt` | not modified in this stage |
| Figures | `results_v2/plots/phase1r_angular_mae_by_method.png`, `phase1r_equivariance_vs_floor.png`, `phase1r_training_curves.png` | not modified in this stage |
| Benchmark driver | `code_v2/run_phase1r_benchmark.py` | not modified in this stage |
