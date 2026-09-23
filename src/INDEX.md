# `src/` File Index

72 scripts. This project was built as a sequence of gated stages, each of which read only its
predecessors' frozen outputs and wrote its own new ones; every script below is named for the stage that
produced it. That discipline is what makes the numbers traceable — but it also means the folder is not
self-explanatory from filenames alone. This index accounts for every file, grouped by what it actually
does, and marks which ones you need to run to reproduce the scientific result versus which ones only
rebuild the manuscript text or cross-check numbers that were already computed.

**If you only want to reproduce the reported numbers**, you need category **A** and, for the
robustness/secondary analyses, category **B**. Categories **C** and **D** consume already-computed
results and produce tables, figures, or manuscript text — running them changes no number, and skipping
them changes nothing about reproducibility.

## A. Core scientific pipeline (run these, in this order, to reproduce the primary result)

| # | File | Produces |
|---|---|---|
| 1 | `stage4_generate_targets.py` | The anatomical orientation-tensor target `Q_anat` from GlaS gland-instance masks |
| 2 | `anatomy_targets.py` | Shared target-construction utilities imported by (1) and by the CRAG target script |
| 3 | `stage9_train_baselines.py` | Trains the two baselines (plain CNN, Model 2) |
| 4 | `stage10_train_model3.py` | Trains the frozen, principal Model 3 checkpoint (seed 42) |
| 5 | `stage11_geometric_validation.py` | Evaluates Criteria A–C against the frozen thresholds |
| 6 | `stage12_rotation_validation.py` | Evaluates Criterion D (rotation consistency) |
| 7 | `model3_typed_equivariant.py`, `model3_loss.py`, `nematic_math.py`, `nematic_model.py`, `model2_posthoc_q.py`, `baseline_cnn.py`, `baseline_cnn_dualhead.py` | Model/loss/math definitions imported by (3)–(6); not run directly |
| 8 | `phase2_calibration.py`, `phase2_recalibration.py` | The one-time, already-frozen threshold calibration behind `configs/*_thresholds*.json` (Methods 3.4's "version 0/1/2" history); re-running is not needed to reproduce the primary result, only to re-derive the thresholds from scratch |

## B. Robustness, ablation, and cross-dataset extensions (needed for the secondary results reported in the manuscript)

| # | File | Produces |
|---|---|---|
| 9 | `stage13_permutation_null.py` | Model 2's channel-permutation null analysis |
| 10 | `stage14_train_ablation.py`, `stage14_evaluate.py`, `stage14_a6_spatial_reduction.py` | The five ablation conditions and the alternative rotation-criterion reductions |
| 11 | `stage15_pathology_analysis.py` | The four primary pathology-association tests (P1–P4) |
| 12 | `stage16_multiscale_analysis.py` | The six multi-scale reductions |
| 13 | `stage17_crag_targets_and_patches.py`, `stage17_crag_evaluate.py` | CRAG target construction and the frozen-checkpoint CRAG evaluation |
| 14 | `stage18_panda_compatibility_test.py` | The PANDA annotation-compatibility gate (Gleason-region-ratio finding) |
| 15 | `stage27_build_patch_cache.py`, `stage27_build_patient_disjoint_cache.py`, `stage27_patient_split_analysis.py`, `stage27_train_and_evaluate.py`, `stage27_compile_multiseed_results.py`, `stage27_rotation_metric_redesign.py` | The five-seed multi-seed study, the patient-disjoint sensitivity partition and its five-seed extension, and the five-metric rotation-consistency redesign |
| 16 | `stage28_s_distribution_diagnostics.py`, `stage28_gland_patch_size_assessment.py` | The S_DL/S_anat distribution diagnostic and the gland-vs-patch-size assessment |
| 17 | `oracle_baselines.py` | Deterministic (non-learned) orientation estimators used for the Phase 1R noise-floor calibration |

## C. Table, figure, and plot assembly (read already-computed results; produce no new statistic)

| File | What it renders |
|---|---|
| `stage14_master_table.py`, `stage14_plots.py` | Ablation table/figures |
| `stage15_master_table_and_plots.py` | Pathology table/figures |
| `stage16_tables_and_plots.py` | Multi-scale table/figures |
| `stage17_tables_and_plots.py` | CRAG table/figures |
| `stage18_panda_qc_figures.py` | PANDA QC figures |
| `stage20_figures_and_tables.py` | The publication figure/table package assembled from Stages 0–19 |
| `stage27_multiseed_plots.py` | Multi-seed training-loss/seed-variance figures |
| `stage28_s_distribution_plots.py` | S_DL/S_anat histogram and scatter figures |
| `stage28_build_summary_json.py` | The machine-readable Stage 28 summary (`analysis/stage28_summary.json`) |
| `generate_anatomy_diagnostics.py` | Gland/mask-quality diagnostic figures, no model involved |
| `stage5_distributions.py`, `stage5_validation.py` | Dataset-level distribution summaries and a sanity check of the Stage 4 targets |

## D. Manuscript assembly and internal number/claim audits (project-history tooling, not part of the scientific pipeline)

These scripts read a manuscript draft and a set of frozen result files and either (a) build the next
manuscript revision by inserting verified numbers into prose, or (b) independently re-check that a
manuscript's stated numbers match the frozen result files. Running them reproduces no scientific result
— they document and enforce internal consistency of the manuscript text itself, at each of the project's
26 manuscript-revision stages. Included for completeness and transparency, not because a reader needs to
re-run them.

`stage19_synthesis.py` · `stage21_numbers_and_claims.py` · `stage22_audit_checks.py` ·
`stage23_manuscript_audits.py` · `stage225_build_corrected_package.py` ·
`stage225_verify_corrected_package.py` · `stage24_cross_checks.py` · `stage24_review_checks.py` ·
`stage24_5_build_audits.py` · `stage24_5_build_manuscript.py` · `stage24_5_numbers_check.py` ·
`stage24_5_second_review.py` · `stage24_5_write_tables.py` · `stage24_6_admin_update.py` ·
`stage27_build_manuscript.py` · `stage28_build_manuscript.py`

## E. Early/superseded phases (kept for provenance; not part of the current model or result)

Phase 1 and Phase 1R were synthetic-data benchmarks run *before* the real-histology pipeline (A–B
above) was built, to sanity-check the equivariant Q-tensor architecture under controlled, noise-free
geometry. Both returned NO-GO verdicts on their own frozen criteria and are reported in the manuscript's
Supplementary Materials, not as evidence for the main result.

`run_synthetic_benchmark.py`, `synthetic_shapes.py`, `synthetic_shapes_v2.py`,
`make_synthetic_plots.py` (Phase 1) · `run_phase1r_benchmark.py`, `make_phase1r_plots.py` (Phase 1R) ·
`stage8_tiny_overfit.py`, `stage8_extended_tiny_overfit.py` (an 8-patch optimization-pacing engineering
check, not a scientific result — see manuscript Methods 3.4)

## F. Tests

`test_nematic_math.py` — unit tests for the tensor/angle math in `nematic_math.py`.

## Note on manuscript vs. this repository's numbering

The manuscript's own Methods section documents the reproduction order for categories A and B in prose
(this index is the same information, indexed by filename instead of by narrative). Section 6 of
`README.md` gives the short version for a first read; this file is the complete, file-by-file account.
