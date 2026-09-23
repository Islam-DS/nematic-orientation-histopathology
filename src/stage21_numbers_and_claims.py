"""
STAGE 21: builds 13_numbers_and_sources.csv and 12_claim_evidence_matrix.csv
for the manuscript result package. READ-ONLY with respect to every frozen
source: values are read from Stage 4/10/11-18 result files (or, for a few
facts that exist only in report text, transcribed with the report section
named as the source). No statistic is computed or re-estimated here.
"""
import os
import json
import csv

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(PROJECT_ROOT, "results_v2", "manuscript_stage21")
os.makedirs(OUT, exist_ok=True)


def load(p):
    with open(os.path.join(PROJECT_ROOT, p)) as f:
        return json.load(f)


F = {
    "s11": "results_v2/validation/stage11/stage11_results.json",
    "s12": "results_v2/validation/stage12/stage12_results.json",
    "s13": "results_v2/validation/stage13/stage13_results.json",
    "s14": "results_v2/validation/stage14/stage14_all_evaluations.json",
    "s14a6": "results_v2/validation/stage14/A6_spatial_reduction/a6_results.json",
    "s15": "results_v2/validation/stage15/stage15_results.json",
    "s16": "results_v2/validation/stage16/stage16_results.json",
    "s17": "results_v2/validation/stage17_crag/stage17_results.json",
    "s17t": "results_v2/validation/stage17_crag/crag_target_generation_summary.json",
    "s17c": "results_v2/validation/stage17_crag/crag_compatibility_report.json",
    "s10": "results_v2/model3/metrics/stage10_summary.json",
    "s4": "results_v2/anatomy_targets/stage4_target_generation_summary.json",
    "pre": "results_v2/phase2/preregistered_thresholds_v2.json",
}
D = {k: load(v) for k, v in F.items()}


def get(key, path):
    cur = D[key]
    for p in path:
        cur = cur[p]
    return cur


def rnd(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, float):
        return round(v, 4)
    return v


numbers = []


def N(label, unit, dataset, analysis, key, path, stage, interp, allowed, value=None, source_text=None):
    if path is not None:
        value = get(key, path)
        src_file, src_field = F[key], ".".join(str(p) for p in path)
    else:
        assert value is not None and source_text
        src_file, src_field = key, source_text
    numbers.append({"number": rnd(value), "label": label, "unit": unit, "dataset": dataset, "analysis": analysis,
                    "source_file": src_file, "source_field": src_field, "stage": stage,
                    "interpretation": interp, "allowed_use": allowed})


ALLOW_MAIN = "may be reported with the stated unit, dataset, and effect-size context"
ALLOW_CTX = "context/descriptive only; must not be used to override the pooled preregistered result"
P = ["pooled_testA_testB_80_images", "metrics"]

# ---------- Stage 4 / dataset
N("GlaS images total", "images", "GlaS", "Stage 4 target generation", "s4", ["all", "n_images"], "4", "canonical GlaS: train 85 / testA 60 / testB 20", ALLOW_MAIN)
N("GlaS train images", "images", "GlaS", "Stage 4", "s4", ["train", "n_images"], "4", "canonical train split", ALLOW_MAIN)
N("GlaS testA images", "images", "GlaS", "Stage 4", "s4", ["testA", "n_images"], "4", "held-out testA", ALLOW_MAIN)
N("GlaS testB images", "images", "GlaS", "Stage 4", "s4", ["testB", "n_images"], "4", "held-out testB", ALLOW_MAIN)
N("GlaS gland instances", "glands", "GlaS", "Stage 4", "s4", ["all", "n_gland_instances_total"], "4", "all 165 images", ALLOW_MAIN)
N("GlaS valid gland instances", "glands", "GlaS", "Stage 4", "s4", ["all", "validity_category_counts", "valid"], "4", "valid under unchanged Stage 4 rules", ALLOW_MAIN)
N("GlaS fragmented gland instances", "glands", "GlaS", "Stage 4", "s4", ["all", "n_fragmented_instances"], "4", "all in four testB images", ALLOW_MAIN)
N("Model 3 parameters", "parameters", "GlaS", "Stage 10 training", "s10", ["n_params"], "10", "typed D8.irrep(1,2) Model 3", ALLOW_MAIN)
N("Model 3 epochs run", "epochs", "GlaS", "Stage 10 training", "s10", ["n_epochs_run"], "10", "early stopping", ALLOW_MAIN)
N("Model 3 lowest development loss (selected checkpoint)", "loss", "GlaS", "Stage 10 training", "s10", ["best_dev_loss"], "10", "checkpoint selected on dev loss, not test", ALLOW_MAIN)
N("Stage 8 tiny-set L_final/L_initial (extended run)", "ratio", "GlaS (8 train patches)", "Stage 8 extended engineering run", "reports/STAGE8_EXTENDED_AUDIT_REPORT.md", None, "8", "engineering criterion <=0.10 met", "engineering/optimization statement only; not evidence of anatomical correspondence", value=0.0736, source_text="Section 4 table (ratio 0.07364)")
N("Stage 7 representation tests passed", "tests", "N/A (architecture)", "Stage 7 verification", "reports/STAGE7_AUDIT_REPORT.md", None, "7", "15/15 tests pass; irrep matches R(2a) to 0.00e+00", "mathematical/implementation validity statement", value=15, source_text="Section 9/tests table: '15/15 tests pass'")

# ---------- Preregistered thresholds
numbers.append({"number": 0.5, "label": "Criterion A threshold (Glass's delta)", "unit": "delta", "dataset": "GlaS", "analysis": "preregistration",
                "source_file": F["pre"], "source_field": "criteria.A_angular_correspondence.threshold_primary.rule (Glass_delta >= 0.5)", "stage": "0",
                "interpretation": "Cohen medium-effect convention", "allowed_use": "cite as frozen threshold"})
numbers.append({"number": get("pre", ["criteria", "B_order_magnitude_correspondence", "threshold", "r_gte"]), "label": "Criterion B threshold (r)", "unit": "r", "dataset": "GlaS", "analysis": "preregistration",
                "source_file": F["pre"], "source_field": "criteria.B_order_magnitude_correspondence.threshold.r_gte", "stage": "0",
                "interpretation": "Cohen medium-correlation convention (also p<0.01 and CI lower>0)", "allowed_use": "cite as frozen threshold"})
numbers.append({"number": 0.5, "label": "Criterion C threshold (Glass's delta)", "unit": "delta", "dataset": "GlaS", "analysis": "preregistration",
                "source_file": F["pre"], "source_field": "criteria.C_tensor_similarity.threshold_primary.rule_effect_size (Glass_delta >= 0.5, and permutation p < 0.01)", "stage": "0",
                "interpretation": "both conditions required", "allowed_use": "cite as frozen threshold"})
numbers.append({"number": 7, "label": "Criterion D required angles", "unit": "angles of 9", "dataset": "GlaS", "analysis": "preregistration",
                "source_file": F["pre"], "source_field": "criteria.D_rotation_consistency.threshold_primary.rule (>= 7 of 9 non-zero angles)", "stage": "0",
                "interpretation": "pass rule", "allowed_use": "cite as frozen threshold"})

# ---------- Stage 11 pooled
N("GlaS pooled patches (testA+testB)", "patches", "GlaS", "Stage 11", "s11", ["pooled_testA_testB_80_images", "n_patches_valid"], "11", "80 images", ALLOW_MAIN)
N("Criterion A mean angular error", "deg", "GlaS pooled", "Stage 11", "s11", P + ["A_angular_correspondence", "mean_deg"], "11", "context for Glass's delta", ALLOW_MAIN)
N("Criterion A Glass's delta", "delta", "GlaS pooled", "Stage 11", "s11", P + ["A_angular_correspondence", "glass_delta"], "11", "threshold 0.5 not met", ALLOW_MAIN)
N("Criterion A permutation p", "p", "GlaS pooled", "Stage 11", "s11", P + ["A_angular_correspondence", "permutation_p_value"], "11", "reportable ONLY with near-zero effect size", "report only alongside Glass's delta=0.041")
N("Criterion B Pearson r", "r", "GlaS pooled", "Stage 11", "s11", P + ["B_order_magnitude_correlation", "pearson_r"], "11", "threshold 0.30 not met", ALLOW_MAIN)
N("Criterion B p-value", "p", "GlaS pooled", "Stage 11", "s11", P + ["B_order_magnitude_correlation", "pearson_p"], "11", "not significant", ALLOW_MAIN)
N("Criterion B 95% CI lower", "r", "GlaS pooled", "Stage 11", "s11", P + ["B_order_magnitude_correlation", "bootstrap_95CI", 0], "11", "CI includes zero", ALLOW_MAIN)
N("Criterion B 95% CI upper", "r", "GlaS pooled", "Stage 11", "s11", P + ["B_order_magnitude_correlation", "bootstrap_95CI", 1], "11", "CI includes zero", ALLOW_MAIN)
N("Criterion C mean D_Q", "D_Q", "GlaS pooled", "Stage 11", "s11", P + ["C_tensor_similarity", "mean_D_Q"], "11", "tensor distance", ALLOW_MAIN)
N("Criterion C Glass's delta", "delta", "GlaS pooled", "Stage 11", "s11", P + ["C_tensor_similarity", "glass_delta"], "11", "threshold 0.5 not met", ALLOW_MAIN)
N("Criterion C permutation p", "p", "GlaS pooled", "Stage 11", "s11", P + ["C_tensor_similarity", "permutation_p_value"], "11", "p<0.01 alone is not agreement; effect near zero", "report only alongside Glass's delta=0.007")
N("testA Criterion A mean angular error", "deg", "GlaS testA", "Stage 11 per-split", "s11", ["testA", "metrics", "A_angular_correspondence", "mean_deg"], "11", "per-split descriptive", ALLOW_CTX)
N("testA order-magnitude r", "r", "GlaS testA", "Stage 11 per-split", "s11", ["testA", "metrics", "B_order_magnitude_correlation", "pearson_r"], "11", "per-split descriptive", ALLOW_CTX)
N("testB Criterion A mean angular error", "deg", "GlaS testB", "Stage 11 per-split", "s11", ["testB", "metrics", "A_angular_correspondence", "mean_deg"], "11", "per-split descriptive", ALLOW_CTX)
N("testB order-magnitude r", "r", "GlaS testB", "Stage 11 per-split", "s11", ["testB", "metrics", "B_order_magnitude_correlation", "pearson_r"], "11", "per-split; the criterion is defined on the pooled population and was not met there", ALLOW_CTX)

# ---------- Stage 12
for i, ang in enumerate([15, 30, 37, 45, 60, 90, 123, 150, 173], start=1):
    N(f"Rotation error at {ang} deg", "normalized error", "GlaS pooled", "Stage 12", "s12", ["per_angle", i, "model_error_mean"], "12", "empirical finite-grid pixel-domain error", ALLOW_MAIN)
    N(f"Frozen threshold at {ang} deg", "normalized error", "GlaS", "Stage 12", "s12", ["per_angle", i, "frozen_local_threshold"], "0/12", "preregistered per-angle threshold", ALLOW_MAIN)
N("Rotation angles passed", "angles of 9", "GlaS pooled", "Stage 12", "s12", ["criterion_D", "n_angles_passed"], "12", "required >= 7", ALLOW_MAIN)

# ---------- Stage 13 (Model 2 ONLY)
for split in ["testA", "testB"]:
    for fld, lab in [("z_score", "z"), ("observed_statistic_mean_S", "observed mean S"), ("null_mean_S", "null mean S"), ("null_std_S", "null SD"),
                     ("empirical_p_value_one_sided_null_le_observed", "one-sided p"), ("empirical_p_value_two_sided_secondary_diagnostic", "two-sided p"), ("n_permutations", "permutations")]:
        N(f"Model 2 permutation null {lab} ({split})", "as labeled", f"GlaS {split}", "Stage 13 (MODEL 2 only)", "s13", ["results", split, fld], "13",
          "descriptive; Model 2 post-hoc formulation only", "Model 2 descriptive statement only; never attribute to Model 3")

# ---------- Stage 14
EXPS = {"A0_reference": "A0", "A1_no_q": "A1", "A3_q_dominant": "A3", "A4_q_only": "A4", "A5_plain_cnn_control": "A5"}
for e, lab in EXPS.items():
    g = ["pooled_testA_testB", "geometric_metrics"]
    N(f"{lab} Glass's delta_A", "delta", "GlaS pooled", "Stage 14 ablation", "s14", [e] + g + ["A_angular_correspondence", "glass_delta"], "14", "diagnostic, not a new criterion verdict", "report all conditions together; no ranking")
    N(f"{lab} order-magnitude r", "r", "GlaS pooled", "Stage 14 ablation", "s14", [e] + g + ["B_order_magnitude_correlation", "pearson_r"], "14",
      "A1's Q head received no gradient (fixed random projection)" if lab == "A1" else "diagnostic", "report all conditions together; no ranking")
    N(f"{lab} mean D_Q", "D_Q", "GlaS pooled", "Stage 14 ablation", "s14", [e] + g + ["C_tensor_similarity", "mean_D_Q"], "14", "diagnostic", "report all conditions together; no ranking")
N("A6 R1 (full-field) rotation angles passed", "angles of 9", "GlaS pooled", "Stage 14 A6", "s14a6", ["reductions", "R1_full_spatial_mean", "n_angles_passed_diagnostic_only"], "14", "diagnostic vs original threshold", ALLOW_CTX)
N("A6 R2 (fixed mask) rotation angles passed", "angles of 9", "GlaS pooled", "Stage 14 A6", "s14a6", ["reductions", "R2_fixed_original_valid_mask_mean", "n_angles_passed_diagnostic_only"], "14", "diagnostic vs original threshold", ALLOW_CTX)

# ---------- Stage 15
for pid in ["P1", "P2", "P3", "P4"]:
    r = get("s15", ["primary_pooled", pid])
    key = "cohens_d" if "cohens_d" in r else "spearman_rho"
    N(f"{pid} effect size ({key})", "d or rho", "GlaS pooled", "Stage 15 pathology", "s15", ["primary_pooled", pid, key], "15", "image-level n=80", ALLOW_MAIN)
    N(f"{pid} raw p", "p", "GlaS pooled", "Stage 15 pathology", "s15", ["primary_pooled", pid, "p_value"], "15", "not significant", ALLOW_MAIN)
    N(f"{pid} Holm-adjusted p", "p", "GlaS pooled", "Stage 15 pathology", "s15", ["holm_bonferroni_adjusted_p", pid], "15", "correction across the 4 pre-specified tests", ALLOW_MAIN)
for pid in ["P2", "P4"]:
    N(f"{pid} rho 95% CI lower", "rho", "GlaS pooled", "Stage 15", "s15", ["primary_pooled", pid, "rho_95CI_bootstrap", 0], "15", "image-level bootstrap", ALLOW_MAIN)
    N(f"{pid} rho 95% CI upper", "rho", "GlaS pooled", "Stage 15", "s15", ["primary_pooled", pid, "rho_95CI_bootstrap", 1], "15", "image-level bootstrap", ALLOW_MAIN)
N("Pathology analysis images", "images", "GlaS testA+testB", "Stage 15", "s15", ["n_images_total"], "15", "unit of analysis is the image", ALLOW_MAIN)
N("Distinct patients in the 80 images", "patients", "GlaS testA+testB", "Stage 15", "s15", ["n_distinct_patients"], "15", "patient clustering limitation", ALLOW_MAIN)
N("Patients with mixed-grade images (R1)", "patients", "GlaS testA+testB", "Stage 15 R1", "s15", ["robustness_R1_patient_level", "patients_with_mixed_grade_label_across_images"], "15", "patient-level test not computed", ALLOW_MAIN)
N("M1 gland-area rho", "rho", "GlaS pooled", "Stage 15", "s15", ["morphology", "M1", "spearman_rho"], "15", "no association", ALLOW_MAIN)
N("M1 p", "p", "GlaS pooled", "Stage 15", "s15", ["morphology", "M1", "p_value"], "15", "no association", ALLOW_MAIN)
N("M2 gland-count rho", "rho", "GlaS pooled", "Stage 15", "s15", ["morphology", "M2", "spearman_rho"], "15", "no association", ALLOW_MAIN)
N("M2 p", "p", "GlaS pooled", "Stage 15", "s15", ["morphology", "M2", "p_value"], "15", "no association", ALLOW_MAIN)
N("C1 grade_label coefficient (adjusted for area)", "S_DL units", "GlaS pooled", "Stage 15", "s15", ["confound_adjustment_C1", "grade_label_coef"], "15", "consistent with unadjusted null", ALLOW_MAIN)
N("C1 grade_label p", "p", "GlaS pooled", "Stage 15", "s15", ["confound_adjustment_C1", "grade_label_p"], "15", "not significant", ALLOW_MAIN)
N("testB-only P2 rho (n=20)", "rho", "GlaS testB", "Stage 15 split-specific", "s15", ["primary_by_split", "testB", "P2", "spearman_rho"], "15", "non-primary, uncorrected, 4 benign images", "MUST NOT be presented as evidence; may be disclosed as a non-primary observation")
N("testB-only P2 raw p (n=20)", "p", "GlaS testB", "Stage 15 split-specific", "s15", ["primary_by_split", "testB", "P2", "p_value"], "15", "non-primary, uncorrected", "MUST NOT be presented as evidence; may be disclosed as a non-primary observation")

# ---------- Stage 16
for k, lab in [("1", "center"), ("3", "3x3"), ("5", "5x5"), ("9", "9x9"), ("17", "17x17"), ("full", "full-field")]:
    N(f"Multi-scale {lab} Glass's delta_A", "delta", "GlaS pooled", "Stage 16", "s16", ["anatomy_by_scale", k, "pooled", "metrics", "A_angular_correspondence", "glass_delta"], "16", "diagnostic", "report all six scales together")
    N(f"Multi-scale {lab} order-magnitude r", "r", "GlaS pooled", "Stage 16", "s16", ["anatomy_by_scale", k, "pooled", "metrics", "B_order_magnitude_correlation", "pearson_r"], "16", "diagnostic", "report all six scales together")
    N(f"Multi-scale {lab} rotation angles passed", "angles of 9", "GlaS pooled", "Stage 16", "s16", ["rotation_by_scale", k, "n_angles_passed_vs_original_threshold"], "16", "vs original Stage 12 threshold (descriptive)", "report all six scales together")

# ---------- Stage 17
N("CRAG images", "images", "CRAG", "Stage 17", "s17t", ["n_images"], "17", "173 train + 40 test", ALLOW_MAIN)
N("CRAG gland instances", "glands", "CRAG", "Stage 17", "s17t", ["n_gland_instances_total"], "17", "", ALLOW_MAIN)
N("CRAG valid patches (train)", "patches", "CRAG", "Stage 17", "s17t", ["n_patches_valid", "train"], "17", "", ALLOW_MAIN)
N("CRAG valid patches (test)", "patches", "CRAG", "Stage 17", "s17t", ["n_patches_valid", "test"], "17", "", ALLOW_MAIN)
N("CRAG images with a gland >30% of image area", "images", "CRAG", "Stage 17 compatibility", "s17c", ["crag_images_with_dominant_fused_gland_gt30pct"], "17", "fused-instance annotation convention", ALLOW_MAIN)
for pop in ["train", "test", "pooled"]:
    B = ["image_level_primary", pop]
    N(f"CRAG {pop} images with valid data", "images", "CRAG", "Stage 17 image-level", "s17", B + ["n_images"], "17", "image-level unit", ALLOW_MAIN)
    N(f"CRAG {pop} Glass's delta_A", "delta", "CRAG", "Stage 17 image-level", "s17", B + ["metrics", "A_angular_correspondence", "glass_delta"], "17", "external cross-dataset validation", "never pool with GlaS")
    N(f"CRAG {pop} order-magnitude r", "r", "CRAG", "Stage 17 image-level", "s17", B + ["metrics", "B_order_magnitude_correlation", "pearson_r"], "17", "external cross-dataset validation", "never pool with GlaS")
    N(f"CRAG {pop} order-magnitude p", "p", "CRAG", "Stage 17 image-level", "s17", B + ["metrics", "B_order_magnitude_correlation", "pearson_p"], "17", "", "never pool with GlaS")
    N(f"CRAG {pop} mean D_Q", "D_Q", "CRAG", "Stage 17 image-level", "s17", B + ["metrics", "C_tensor_similarity", "mean_D_Q"], "17", "", "never pool with GlaS")
N("CRAG pooled r 95% CI lower", "r", "CRAG", "Stage 17", "s17", ["image_level_primary", "pooled", "metrics", "B_order_magnitude_correlation", "bootstrap_95CI", 0], "17", "", "never pool with GlaS")
N("CRAG pooled r 95% CI upper", "r", "CRAG", "Stage 17", "s17", ["image_level_primary", "pooled", "metrics", "B_order_magnitude_correlation", "bootstrap_95CI", 1], "17", "", "never pool with GlaS")
N("CRAG patch-level diagnostic patches", "patches", "CRAG", "Stage 17 patch-level", "s17", ["patch_level_secondary_diagnostic", "n_patches_subsampled"], "17", "deterministic seed-42 subsample", ALLOW_CTX)
N("CRAG patch-level Glass's delta_A", "delta", "CRAG", "Stage 17 patch-level", "s17", ["patch_level_secondary_diagnostic", "metrics", "A_angular_correspondence", "glass_delta"], "17", "secondary diagnostic", ALLOW_CTX)
N("CRAG rotation angles passed", "angles of 9", "CRAG", "Stage 17", "s17", ["rotation_consistency", "n_angles_passed_vs_original_GlaS_threshold"], "17", "vs original GlaS threshold; descriptive, no CRAG-specific threshold", ALLOW_CTX)

# ---------- Stage 18 (report/summary-table sourced)
S18 = "results_v2/validation/stage18_panda/summary_table.csv"
for v, lab, src in [(10, "PANDA pilot cases (Radboud)", "radboud_pilot_manifest.csv rows"), (5160, "PANDA Radboud cases in train.csv", "data/PANDA/radboud_metadata.csv rows"),
                    (10616, "PANDA train.csv rows (Radboud + Karolinska)", "data/PANDA/train.csv rows")]:
    N(f"{lab}", "cases", "PANDA", "Stage 18 metadata", "data/PANDA/stage18_metadata_audit.md", None, "18", "Radboud only used for the pilot", "metadata description only", value=v, source_text=src)
for v, lab, col in [(49.3, "PANDA smallest Gleason-4 largest-component / benign-median ratio", "gleason_4_largest_vs_benign_median_ratio (case 00bbc148)"),
                    (6244.3, "PANDA largest Gleason-5 largest-component / benign-median ratio", "gleason_5_largest_vs_benign_median_ratio (case 00928370)")]:
    N(lab, "ratio", "PANDA Radboud pilot", "Stage 18 compatibility gate", S18, None, "18", "annotation-semantics finding, not a model result", "compatibility finding only; never a Model 3 result", value=v, source_text=col)
N("PANDA pilot cases with Gleason 4 and a computable benign reference", "cases", "PANDA Radboud pilot", "Stage 18 compatibility gate", S18, None, "18", "all 4 had a largest Gleason-4 region far larger than benign-gland scale", "compatibility finding only", value=4, source_text="rows with non-null gleason_4_largest_vs_benign_median_ratio")
N("PANDA pilot cases containing Gleason 4", "cases", "PANDA Radboud pilot", "Stage 18", S18, None, "18", "5 cases; one lacks a benign reference", "compatibility finding only", value=5, source_text="rows with gleason_4_pixel_fraction > 0")
for v, lab, col in [(136.8, "PANDA Gleason-4 ratio (case 018eabc8)", "gleason_4_largest_vs_benign_median_ratio"), (735.1, "PANDA Gleason-4 ratio (case 00928370)", "gleason_4_largest_vs_benign_median_ratio"), (1082.3, "PANDA Gleason-4 ratio (case 0018ae58)", "gleason_4_largest_vs_benign_median_ratio")]:
    N(lab, "ratio", "PANDA Radboud pilot", "Stage 18 compatibility gate", S18, None, "18", "annotation-semantics finding", "compatibility finding only", value=v, source_text=col)
N("PANDA benign-epithelium components per case, minimum (cases with benign)", "components", "PANDA Radboud pilot", "Stage 18", S18, None, "18", "gland-plausible components", "compatibility finding only", value=32, source_text="benign_epithelium_n_components (min over 9 cases with benign)")
N("PANDA benign-epithelium components per case, maximum", "components", "PANDA Radboud pilot", "Stage 18", S18, None, "18", "gland-plausible components", "compatibility finding only", value=498, source_text="benign_epithelium_n_components (max)")

# ---------- additional sourced values used in the narrative
N("CRAG valid patches (train+test total)", "patches", "CRAG", "Stage 17", "reports/STAGE17_AUDIT_REPORT.md", None, "17", "24,026 + 5,373", ALLOW_MAIN, value=29399, source_text="Section 9 table / Section 8 ('29,399 pooled valid patches')")
for split in ["testA", "testB"]:
    N(f"Model 2 permutation null patches ({split})", "patches", f"GlaS {split}", "Stage 13 (MODEL 2 only)", "s13", ["results", split, "n_patches"], "13", "", "Model 2 descriptive statement only; never attribute to Model 3")
N("Model 2 permutation null seed", "seed", "GlaS", "Stage 13 (MODEL 2 only)", "s13", ["results", "testA", "seed"], "13", "", "Model 2 descriptive statement only; never attribute to Model 3")
N("Oracle calibration Glass's delta (train-only)", "delta", "GlaS train", "Stage 0 recalibration (classical structure-tensor oracle, not Model 3)", "pre", ["calibration_summary_used_for_derivation", "A_oracle_glass_delta"], "0", "non-learned reference", "context for frozen criteria only; not a Model 3 result")
N("Oracle calibration Pearson r (train-only)", "r", "GlaS train", "Stage 0 recalibration (oracle)", "pre", ["calibration_summary_used_for_derivation", "B_oracle_pearson_r"], "0", "non-learned reference", "context for frozen criteria only; not a Model 3 result")
N("Oracle calibration r CI lower", "r", "GlaS train", "Stage 0 recalibration (oracle)", "pre", ["calibration_summary_used_for_derivation", "B_oracle_r_95CI", 0], "0", "", "context for frozen criteria only; not a Model 3 result")
N("Oracle calibration r CI upper", "r", "GlaS train", "Stage 0 recalibration (oracle)", "pre", ["calibration_summary_used_for_derivation", "B_oracle_r_95CI", 1], "0", "", "context for frozen criteria only; not a Model 3 result")
N("Bootstrap iterations (frozen)", "iterations", "GlaS", "preregistration", "pre", ["criteria", "B_order_magnitude_correspondence", "bootstrap_n"], "0", "", ALLOW_MAIN)
N("Bootstrap seed (frozen)", "seed", "GlaS", "preregistration", "pre", ["criteria", "B_order_magnitude_correspondence", "bootstrap_seed"], "0", "", ALLOW_MAIN)
N("Whole-network finite-grid residual, untrained Model 3 (percent, maximum)", "percent", "synthetic smooth input", "Stage 7 test L", "reports/STAGE7_AUDIT_REPORT.md", None, "7", "attributed to antialiased pooling / finite grid; not re-claimed as exact", "limitation statement", value=34, source_text="Section 20 / Stage 10 report Section 2 ('~34%')")
N("Benign images in testB (pathology split-specific)", "images", "GlaS testB", "Stage 15", "reports/STAGE15_AUDIT_REPORT.md", None, "15", "extreme class imbalance", "limitation statement", value=4, source_text="Section 6 ('only 4 benign images out of 20')")
N("Stage 14 A5 parameters", "parameters", "GlaS", "Stage 14 A5", "results_v2/validation/stage14/A5_plain_cnn_control/stage14_train_summary.json", None, "14", "matched to 11,674", ALLOW_MAIN, value=11676, source_text="n_params")
N("Stage 14 lambda_Q in A3", "weight", "GlaS", "Stage 14 A3", "configs/stage14_ablation_matrix.json", None, "14", "10x baseline weight", ALLOW_MAIN, value=10, source_text="experiments[A3].lambda_Q")
N("Stage 8 tiny-set patches", "patches", "GlaS train", "Stage 8", "reports/STAGE8_EXTENDED_AUDIT_REPORT.md", None, "8", "", ALLOW_MAIN, value=8, source_text="Section 3 table ('Samples: identical 8 canonical TRAIN patches')")
N("Stage 8 epoch limit extended from/to", "epochs", "GlaS train", "Stage 8", "reports/STAGE8_EXTENDED_AUDIT_REPORT.md", None, "8", "500 to 1000", ALLOW_MAIN, value=1000, source_text="Section 2 (MAX_EPOCHS: 500 -> 1000)")
N("Stage 8 original epoch limit", "epochs", "GlaS train", "Stage 8", "reports/STAGE8_EXTENDED_AUDIT_REPORT.md", None, "8", "", ALLOW_MAIN, value=500, source_text="Section 2 (MAX_EPOCHS: 500 -> 1000)")
N("CRAG fused-instance descriptive threshold (percent of image area)", "percent", "CRAG", "Stage 17 compatibility", F["s17c"], None, "17", "descriptive characterization threshold used in the Stage 17 audit; not a validity rule", ALLOW_MAIN, value=30, source_text="crag_images_with_dominant_fused_gland_gt30pct (images with a gland >30% of image area)")

N("GlaS patients total (Grade.csv)", "patients", "GlaS", "Stage 15 dataset audit", "reports/STAGE15_AUDIT_REPORT.md", None, "15", "patient ID column in Grade.csv", "limitation statement", value=16, source_text="Section 2 ('11 of 16 patients')")
N("GlaS patients spanning more than one canonical split", "patients", "GlaS", "Stage 15 dataset audit", "reports/STAGE15_AUDIT_REPORT.md", None, "15", "patient clustering across splits", "limitation statement", value=11, source_text="Section 2 ('11 of 16 patients')")
fields = ["number", "label", "unit", "dataset", "analysis", "source_file", "source_field", "stage", "interpretation", "allowed_use"]
with open(os.path.join(OUT, "13_numbers_and_sources.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in numbers:
        w.writerow({k: r[k] for k in fields})
print(f"13_numbers_and_sources.csv: {len(numbers)} rows")

# =========================================================== claim-evidence matrix
m = get("s11", P)
cl = []


def C(claim, src, ds, unit, n, stat, eff, p, ci, prereg, status, allowed, forbidden):
    cl.append({"claim": claim, "evidence_source": src, "dataset": ds, "sample_unit": unit, "sample_size": n, "statistic": stat,
               "effect_size": eff, "p_value": p, "confidence_interval": ci, "preregistered": prereg, "status": status,
               "allowed_manuscript_wording": allowed, "forbidden_overclaim": forbidden})


A_ = m["A_angular_correspondence"]; B_ = m["B_order_magnitude_correlation"]; C_ = m["C_tensor_similarity"]
C("The typed rank-2 (D8.irrep(1,2)) representation and Q construction are mathematically and numerically implemented as specified",
  "Stage 7 (15/15 tests); Stage 4 target generation", "N/A / GlaS", "architecture tests; 165 images", "15 tests; 165 images",
  "irrep matrix equals R(2a) to 0.00e+00", "NA", "NA", "NA", "engineering criterion", "SUPPORTED",
  "The Q construction and the typed representation were implemented and verified against the analytic transformation law.",
  "That the whole network is exactly equivariant on a finite pixel grid; that the representation is anatomically meaningful.")
C("Model 3 training is reproducible and the loss optimizes on a controlled tiny set",
  "Stage 8 extended run; Stage 10", "GlaS train", "patch (8 tiny-set patches); 1816/433 train/dev patches", "8; 1816/433",
  "L_final/L_initial = 0.0736 (Stage 8 extended); checkpoint reload bit-identical (Stage 10)", "NA", "NA", "NA", "engineering criterion", "SUPPORTED",
  "Optimization was stable and reproducible; the controlled tiny-set criterion was met after a single pre-authorized extension of the epoch limit.",
  "That optimization success implies anatomical correspondence or generalization.")
C("Model 3 predicted orientation corresponds to the anatomical orientation (Criterion A)", F["s11"], "GlaS testA+testB", "patch", 2083,
  f"Glass's delta = {A_['glass_delta']:.3f} (threshold >= 0.5); mean error {A_['mean_deg']:.2f} deg", f"{A_['glass_delta']:.3f}", f"perm p = {A_['permutation_p_value']} (reportable only with the near-zero effect)", "NA", "yes", "NOT DEMONSTRATED",
  "Criterion A was not met (Glass's delta 0.041 vs the preregistered 0.5).", "That a small permutation p indicates anatomical agreement; that the model 'partially learned' orientation.")
C("Model 3 predicted order magnitude corresponds to the anatomical order magnitude (Criterion B)", F["s11"], "GlaS testA+testB", "patch", 2083,
  f"Pearson r = {B_['pearson_r']:.3f} (threshold >= 0.30)", f"{B_['pearson_r']:.3f}", f"p = {B_['pearson_p']:.3f}", f"[{B_['bootstrap_95CI'][0]:.3f}, {B_['bootstrap_95CI'][1]:.3f}]", "yes", "NOT DEMONSTRATED",
  "Criterion B was not met (r = -0.034; 95% CI includes zero).", "Any positive-correlation language; per-split values (testB r = 0.308) as evidence.")
C("Model 3 predicted tensor is closer to the anatomical tensor than chance re-pairing (Criterion C)", F["s11"], "GlaS testA+testB", "patch", 2083,
  f"Glass's delta = {C_['glass_delta']:.3f} (threshold >= 0.5); mean D_Q {C_['mean_D_Q']:.3f}", f"{C_['glass_delta']:.3f}", f"perm p = {C_['permutation_p_value']}", "NA", "yes", "NOT DEMONSTRATED",
  "Criterion C was not met: the permutation condition was satisfied but the effect size (0.007) was negligible against the required 0.5.", "That 'the tensor test is significant'; that the permutation p supports correspondence.")
C("Model 3 shows the preregistered empirical rotation consistency (Criterion D)", F["s12"], "GlaS testA+testB", "patch", 2083,
  "4/9 non-zero angles passed (>= 7/9 required)", "NA", "NA", "NA", "yes", "NOT DEMONSTRATED",
  "Four of nine non-zero angles (90, 123, 150, 173 deg) passed; the criterion required seven.", "That the network is rotation-consistent; that passing large angles is meaningful; continuous equivariance of the whole network.")
C("The Model 2 channel-to-angle assignment differs from randomized assignments", F["s13"], "GlaS testA / testB", "patch", "1470 / 613",
  "z = -1.7106 (testA), -1.7044 (testB); n=100 permutations, seed 0", "NA", "one-sided 0.09; two-sided 0.14 (both splits)", "NA", "descriptive", "INCONCLUSIVE",
  "For Model 2 only, the observed order magnitude lay below the permutation-null mean in both splits, without reaching p < 0.05.", "Anti-nematic behaviour; any statement about Model 3; biological interpretation of the sign.")
C("The negative primary result is robust to Q-loss weighting, removal of Q supervision, Q-only training, and a parameter-matched non-equivariant control", F["s14"], "GlaS testA+testB", "patch", 2083,
  "Glass's delta_A across A0/A1/A3/A4/A5 ranges -0.044 to 0.046; r ranges -0.210 to 0.220", "see 13_numbers_and_sources.csv", "NA", "NA", "pre-frozen diagnostic matrix", "NOT DEMONSTRATED",
  "None of the tested design variations produced the preregistered anatomical correspondence; conditions were not ranked.", "That any ablation is 'better'; that equivariance is unnecessary or harmful; A1's r = 0.220 as learned correspondence (its Q head was untrained).")
C("Spatial aggregation scale modestly changes rotation pass counts but does not produce anatomical correspondence", F["s16"], "GlaS testA+testB", "patch", 2083,
  "rotation angles passed 4,4,4,4,5,6 (of 9) across six scales; Glass's delta_A -0.004 to 0.041", "NA", "NA", "NA", "pre-frozen diagnostic", "NOT DEMONSTRATED",
  "Across six pre-defined scales, rotation pass counts ranged from 4 to 6 of 9 and no scale met either preregistered criterion.", "That a 'correct' or 'optimal' scale exists; that full-field aggregation rescues the result.")
C("The learned S/orientation representation is associated with GlaS pathology grade", F["s15"], "GlaS testA+testB", "image", 80,
  "P1 d=-0.135; P2 rho=0.075; P3 d=-0.138; P4 rho=0.003 (all Holm-adjusted p = 1.0)", "see 13_numbers_and_sources.csv", "raw p 0.664, 0.509, 0.524, 0.979", "P2 [-0.136, 0.277]; P4 [-0.218, 0.231]", "yes (frozen matrix)", "NOT DEMONSTRATED",
  "No pathology association was demonstrated under the tested analyses.", "That the representation is unrelated to pathology in general; the testB-only P2 result as evidence.")
C("Patient-level sensitivity of the pathology result", F["s15"], "GlaS testA+testB", "patient", 12,
  "descriptive only; 3 of 12 patients have mixed-grade images", "NA", "NA", "NA", "yes (pre-specified robustness)", "INCONCLUSIVE",
  "Patient-level inference was not possible because of small n and mixed-grade patients; image-level results are subject to patient clustering.", "Any patient-level conclusion in either direction.")
C("CRAG provides external support for anatomical correspondence", F["s17"], "CRAG", "image", "171 / 40 / 211 (train/test/pooled)",
  "pooled Glass's delta_A = -0.080; r = -0.094; D_Q = 0.429; rotation 4/9", "-0.080", "r p = 0.175", "r 95% CI [-0.227, 0.049]", "descriptive (frozen Stage 17 matrix)", "NOT DEMONSTRATED",
  "In external cross-dataset validation on CRAG, correspondence was not demonstrated; results were consistent with the GlaS findings.", "Independent replication; combining GlaS and CRAG into one estimate; universal failure.")
C("PANDA Radboud masks can supply the frozen gland-instance anatomical target", "Stage 18 (reports/STAGE18_AUDIT_REPORT.md; compatibility_results.json)", "PANDA Radboud", "case (pilot)", 10,
  "4/4 pilot cases with Gleason 4 and a computable benign reference had a largest Gleason-4 region 49x-1082x the same-case benign-gland median (one case also Gleason 5: 6244x)", "NA", "NA", "NA", "yes (frozen design rule)", "INFEASIBLE",
  "The Radboud masks are semantic Gleason-pattern labels; under the frozen definition they do not provide gland-instance geometry, so no target was built and Model 3 was not run.", "That PANDA is a negative model result; that PANDA disproves the hypothesis; that PANDA is unsuitable for other tasks.")
C("A negative result here means nematic organization is absent from histopathology or that equivariant networks cannot learn orientation", "none (not supported by any stage)", "N/A", "N/A", "N/A", "N/A", "NA", "NA", "NA", "NA", "NOT SUPPORTED AS A CLAIM",
  "(No wording permitted; the study tested one frozen formulation, target, and protocol.)", "Any generalization from these results to the biological phenomenon or to equivariant architectures in general.")

cfields = ["claim", "evidence_source", "dataset", "sample_unit", "sample_size", "statistic", "effect_size", "p_value", "confidence_interval",
           "preregistered", "status", "allowed_manuscript_wording", "forbidden_overclaim"]
with open(os.path.join(OUT, "12_claim_evidence_matrix.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cfields)
    w.writeheader()
    for r in cl:
        w.writerow(r)
print(f"12_claim_evidence_matrix.csv: {len(cl)} rows")
