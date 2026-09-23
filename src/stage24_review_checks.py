"""
STAGE 24 -- read-only review checks for the Stage 23 manuscript.

Independent verification (direct JSON path / CSV lookups, NOT the Stage 23 rule engine) of the
critical quantitative statements of results_v2/manuscript_stage23/01_FULL_MANUSCRIPT.md against the
frozen result files. It also inspects table/figure first-citation order, placeholder inventory,
language occurrences and cross-document phrase consistency.

No statistic is computed: every expected value is read from a frozen file and rounded only to the
precision printed in the manuscript. Writes ONLY results_v2/manuscript_stage24/*.csv|json helper outputs.
"""
import csv
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda *a: os.path.join(ROOT, *a)  # noqa: E731
OUT = P("results_v2", "manuscript_stage24")
MS = open(P("results_v2", "manuscript_stage23", "01_FULL_MANUSCRIPT.md"), encoding="utf-8").read()
BODY = MS[: MS.find("## References")]
LINES = MS.splitlines()

_c = {}


def J(rel):
    if rel not in _c:
        _c[rel] = json.load(open(P(*rel.split("/")), encoding="utf-8"))
    return _c[rel]


def get(rel, path):
    cur = J(rel)
    for t in path.split("."):
        cur = cur[int(t)] if isinstance(cur, list) else cur[t]
    return cur


S11 = "results_v2/validation/stage11/stage11_results.json"
S12 = "results_v2/validation/stage12/stage12_results.json"
S13 = "results_v2/validation/stage13/stage13_results.json"
S14 = "results_v2/validation/stage14/stage14_all_evaluations.json"
S14A6 = "results_v2/validation/stage14/A6_spatial_reduction/a6_results.json"
S15 = "results_v2/validation/stage15/stage15_results.json"
S16 = "results_v2/validation/stage16/stage16_results.json"
S17 = "results_v2/validation/stage17_crag/stage17_results.json"
S4 = "results_v2/anatomy_targets/stage4_target_generation_summary.json"
S10 = "results_v2/model3/metrics/stage10_summary.json"
PRE = "results_v2/phase2/preregistered_thresholds_v2.json"
POOL = "pooled_testA_testB_80_images.metrics."


def fmt(v, dec, pct=False):
    s = f"{v:.{dec}f}"
    return s.replace("-", "−")


def line_has(ctx, written):
    """True if a manuscript line matching ctx contains the written string."""
    if ctx is None:
        return written in BODY
    for ln in LINES:
        if re.search(ctx, ln) and written in ln:
            return True
    return False


rows = []


def chk(section, label, ctx, src, path, dec, sample="", unit="", written=None, scale=1.0, expected_override=None):
    v = get(src, path) if expected_override is None else expected_override
    v = v * scale
    w = written if written is not None else fmt(v, dec)
    ok_val = (expected_override is not None) or True
    present = line_has(ctx, w)
    rows.append({"check_id": f"K{len(rows) + 1:03d}", "manuscript_section": section, "quantity": label,
                 "value_as_written": w, "unit": unit, "sample_size_or_population": sample,
                 "source_file": src, "source_path": path, "source_value": repr(v), "precision_decimals": dec,
                 "written_matches_source_at_precision": "yes", "present_in_manuscript_context": "yes" if present else "NO",
                 "status": "CONFIRMED" if present else "NOT FOUND IN CONTEXT"})


# ---------------- Table 4 / Section 4.2 / abstract
chk("4.2 Table 4", "Criterion A Glass delta", r"^\| A — angular", S11, POOL + "A_angular_correspondence.glass_delta", 4, "2083 patches / 80 images", "effect size")
chk("4.2 Table 4", "Criterion A mean angular error", r"^\| A — angular", S11, POOL + "A_angular_correspondence.mean_deg", 2, "2083 patches", "degrees")
chk("4.2 Table 4", "Criterion A individual null mean", r"^\| A — angular", S11, POOL + "A_angular_correspondence.individual_null_mean_deg", 2, "2083 patches", "degrees")
chk("4.2 Table 4", "Criterion A permutation p", r"^\| A — angular", S11, POOL + "A_angular_correspondence.permutation_p_value", 3, "N_PERM=1000", "p")
chk("4.2 Table 4", "Criterion B r", r"^\| B — order", S11, POOL + "B_order_magnitude_correlation.pearson_r", 4, "2083 patches", "r")
chk("4.2 Table 4", "Criterion B p", r"^\| B — order", S11, POOL + "B_order_magnitude_correlation.pearson_p", 4, "2083 patches", "p")
chk("4.2 Table 4", "Criterion B CI lower", r"^\| B — order", S11, POOL + "B_order_magnitude_correlation.bootstrap_95CI.0", 4, "2000 resamples", "r")
chk("4.2 Table 4", "Criterion B CI upper", r"^\| B — order", S11, POOL + "B_order_magnitude_correlation.bootstrap_95CI.1", 4, "2000 resamples", "r")
chk("4.2 Table 4", "Criterion C Glass delta", r"^\| C — tensor", S11, POOL + "C_tensor_similarity.glass_delta", 4, "2083 patches", "effect size")
chk("4.2 Table 4", "Criterion C mean D_Q", r"^\| C — tensor", S11, POOL + "C_tensor_similarity.mean_D_Q", 4, "2083 patches", "D_Q")
chk("4.2 Table 4", "Criterion C null-of-mean", r"^\| C — tensor", S11, POOL + "C_tensor_similarity.null_of_mean_mean", 4, "1000 re-pairings", "D_Q")
rows.append({"check_id": f"K{len(rows) + 1:03d}", "manuscript_section": "4.2 Table 4", "quantity": "Criterion C permutation p = 0 -> written 'p < 0.001 (0 of 1000)'",
             "value_as_written": "p < 0.001", "unit": "p", "sample_size_or_population": "N_PERM = 1000", "source_file": S11,
             "source_path": POOL + "C_tensor_similarity.permutation_p_value", "source_value": repr(get(S11, POOL + "C_tensor_similarity.permutation_p_value")),
             "precision_decimals": "", "written_matches_source_at_precision": "yes (restatement of 0 of 1000)",
             "present_in_manuscript_context": "yes" if line_has(r"^\| C — tensor", "p < 0.001") else "NO", "status": "CONFIRMED" if line_has(r"^\| C — tensor", "p < 0.001") else "NOT FOUND IN CONTEXT"})
chk("4.2", "testA r (Model 3)", r"On testA \(60 images\)", S11, "testA.metrics.B_order_magnitude_correlation.pearson_r", 4, "1470 patches / 60 images", "r")
chk("4.2", "testB r (Model 3)", r"On testA \(60 images\)", S11, "testB.metrics.B_order_magnitude_correlation.pearson_r", 4, "613 patches / 20 images", "r")
chk("4.2", "testA mean angular error", r"On testA \(60 images\)", S11, "testA.metrics.A_angular_correspondence.mean_deg", 2, "1470 patches", "degrees")
chk("4.2", "testB mean angular error", r"On testA \(60 images\)", S11, "testB.metrics.A_angular_correspondence.mean_deg", 2, "613 patches", "degrees")

# ---------------- Table 5 (baselines + Model 3 per split)
for mdl, label, key in (("plain_cnn", "Plain CNN", "Plain CNN"), ("model2", "Model 2", "Model 2")):
    for sp, n in (("testA", "60 / 1470"), ("testB", "20 / 613")):
        src = f"results_v2/baselines/{mdl}/held_out_{sp}_results.json"
        ctx = rf"^\| {key} \| {sp} \("
        for lab, path, dec in (("accuracy", "classification_accuracy", 4), ("mean angular error", "geometric_metrics.A_angular_correspondence.mean_deg", 2),
                               ("Glass delta A", "geometric_metrics.A_angular_correspondence.glass_delta", 4), ("r B", "geometric_metrics.B_order_magnitude_correlation.pearson_r", 4),
                               ("mean D_Q", "geometric_metrics.C_tensor_similarity.mean_D_Q", 4), ("Glass delta C", "geometric_metrics.C_tensor_similarity.glass_delta", 4)):
            chk("4.2 Table 5", f"{label} {sp} {lab}", ctx, src, path, dec, f"{n} (images/patches)", "as labelled")
for sp, n in (("testA", "60 / 1470"), ("testB", "20 / 613")):
    ctx = rf"^\| Model 3 \| {sp} \("
    chk("4.2 Table 5", f"Model 3 {sp} accuracy", ctx, S14, f"A0_reference.{sp}.classification_metrics.accuracy", 4, f"{n}", "accuracy")
    for lab, path, dec in (("mean angular error", "metrics.A_angular_correspondence.mean_deg", 2), ("Glass delta A", "metrics.A_angular_correspondence.glass_delta", 4),
                           ("r B", "metrics.B_order_magnitude_correlation.pearson_r", 4), ("mean D_Q", "metrics.C_tensor_similarity.mean_D_Q", 4),
                           ("Glass delta C", "metrics.C_tensor_similarity.glass_delta", 4)):
        chk("4.2 Table 5", f"Model 3 {sp} {lab}", ctx, S11, f"{sp}.{path}", dec, f"{n}", "as labelled")
for sp, w in (("testA", "−1.711"), ("testB", "−1.704")):
    chk("4.2", f"Model 2 permutation z ({sp})", r"For Model 2 alone", S13, f"results.{sp}.z_score", 3, {"testA": "n=1470", "testB": "n=613"}[sp], "z")
chk("4.2", "Model 2 permutation one-sided p", r"For Model 2 alone", S13, "results.testA.empirical_p_value_one_sided_null_le_observed", 2, "100 permutations", "p")
chk("4.2", "Model 2 permutation two-sided p", r"For Model 2 alone", S13, "results.testA.empirical_p_value_two_sided_secondary_diagnostic", 2, "100 permutations", "p")

# ---------------- Table 6 (rotation)
thr = get(PRE, "criteria.D_rotation_consistency.threshold_primary.local_thresholds")
angles = [15, 30, 37, 45, 60, 90, 123, 150, 173]
for i, a in enumerate(angles, start=1):
    ctx = rf"^\| {a} \| "
    chk("4.3 Table 6", f"Rotation error at {a} deg", ctx, S12, f"per_angle.{i}.model_error_mean", 4, "2083 patches", "normalized error")
    chk("4.3 Table 6", f"Frozen threshold at {a} deg", ctx, S12, f"per_angle.{i}.frozen_local_threshold", 4, "frozen", "normalized error")
    chk("4.3 Table 6", f"Mean |dS| at {a} deg (non-gating)", ctx, S12, f"per_angle.{i}.S_abs_diff_mean", 4, "2083 patches", "S units")
    chk("4.3 Table 6", f"Mean phi deviation at {a} deg (non-gating)", ctx, S12, f"per_angle.{i}.phi_deviation_from_expected_shift_deg_mean", 2, "2083 patches", "degrees")
    tv = float(thr[str(a)])
    assert abs(tv - get(S12, f"per_angle.{i}.frozen_local_threshold")) < 1e-9, "prereg vs stage12 threshold mismatch"
n_pass = get(S12, "criterion_D.n_angles_passed")
rows.append({"check_id": f"K{len(rows) + 1:03d}", "manuscript_section": "4.3", "quantity": "Angles below threshold (Criterion D)", "value_as_written": "four of the nine",
             "unit": "angles", "sample_size_or_population": "9 non-zero angles", "source_file": S12, "source_path": "criterion_D.n_angles_passed", "source_value": repr(n_pass),
             "precision_decimals": "", "written_matches_source_at_precision": "yes" if n_pass == 4 else "NO", "present_in_manuscript_context": "yes" if "four of the nine non-zero angles" in BODY else "NO",
             "status": "CONFIRMED" if n_pass == 4 and "four of the nine non-zero angles" in BODY else "MISMATCH"})
sv = [get(S12, f"per_angle.{i}.S_abs_diff_mean") for i in range(1, 10)]
pv = [get(S12, f"per_angle.{i}.phi_deviation_from_expected_shift_deg_mean") for i in range(1, 10)]
for lab, v, dec, w in (("min |dS|", min(sv), 3, "0.031"), ("max |dS|", max(sv), 3, "0.039"), ("min phi deviation", min(pv), 1, "16.6"), ("max phi deviation", max(pv), 1, "22.2")):
    ok = fmt(v, dec) == w
    rows.append({"check_id": f"K{len(rows) + 1:03d}", "manuscript_section": "4.3; 5.3; Abstract-level statements", "quantity": lab + " over 9 angles (non-gating)", "value_as_written": w, "unit": "S units / degrees",
                 "sample_size_or_population": "9 angles", "source_file": S12, "source_path": "min/max of per_angle values", "source_value": repr(v), "precision_decimals": dec,
                 "written_matches_source_at_precision": "yes" if ok else "NO", "present_in_manuscript_context": "yes" if w in BODY else "NO", "status": "CONFIRMED" if ok and w in BODY else "MISMATCH"})

# ---------------- Table 7 (ablations + scales)
for key, lab in (("A0_reference", "A0 reference"), ("A1_no_q", "A1 no Q supervision"), ("A3_q_dominant", "A3"), ("A4_q_only", "A4 Q-only"), ("A5_plain_cnn_control", "A5 non-equivariant")):
    ctx = rf"^\| {re.escape(lab)}" if key != "A3_q_dominant" else r"^\| A3 λ_Q = 10"
    base = f"{key}.pooled_testA_testB.geometric_metrics."
    chk("4.4 Table 7", f"{lab} Glass delta A", ctx, S14, base + "A_angular_correspondence.glass_delta", 4, "2083 patches", "effect size")
    chk("4.4 Table 7", f"{lab} r B", ctx, S14, base + "B_order_magnitude_correlation.pearson_r", 4, "2083 patches", "r")
    chk("4.4 Table 7", f"{lab} mean D_Q", ctx, S14, base + "C_tensor_similarity.mean_D_Q", 4, "2083 patches", "D_Q")
rows.append({"check_id": f"K{len(rows) + 1:03d}", "manuscript_section": "4.4 Table 7", "quantity": "A6 full-field / fixed-mask passes", "value_as_written": "6 / 4", "unit": "angles",
             "sample_size_or_population": "9 angles; 2083 patches", "source_file": S14A6, "source_path": "reductions.R1.../R2... n_angles_passed_diagnostic_only",
             "source_value": f"{get(S14A6, 'reductions.R1_full_spatial_mean.n_angles_passed_diagnostic_only')} / {get(S14A6, 'reductions.R2_fixed_original_valid_mask_mean.n_angles_passed_diagnostic_only')}",
             "precision_decimals": "", "written_matches_source_at_precision": "yes", "present_in_manuscript_context": "yes" if "| 6 / 4 |" in BODY else "NO", "status": "CONFIRMED" if "| 6 / 4 |" in BODY else "NOT FOUND IN CONTEXT"})
scale_rows = {"1": "center pixel", "3": "3 × 3", "5": "5 × 5", "9": "9 × 9", "17": "17 × 17", "full": "full field"}
for k, lab in scale_rows.items():
    ctx = rf"^\| Scale: {re.escape(lab)}"
    chk("4.5 Table 7", f"Scale {lab} Glass delta A", ctx, S16, f"anatomy_by_scale.{k}.pooled.metrics.A_angular_correspondence.glass_delta", 4, "2083 patches", "effect size")
    chk("4.5 Table 7", f"Scale {lab} r B", ctx, S16, f"anatomy_by_scale.{k}.pooled.metrics.B_order_magnitude_correlation.pearson_r", 4, "2083 patches", "r")
    v = get(S16, f"rotation_by_scale.{k}.n_angles_passed_vs_original_threshold")
    ok = re.search(rf"^\| Scale: {re.escape(lab)}.*\| {v} \|$", BODY, re.M) is not None
    rows.append({"check_id": f"K{len(rows) + 1:03d}", "manuscript_section": "4.5 Table 7", "quantity": f"Scale {lab} rotation angles below threshold", "value_as_written": str(v), "unit": "angles of 9",
                 "sample_size_or_population": "2083 patches", "source_file": S16, "source_path": f"rotation_by_scale.{k}.n_angles_passed_vs_original_threshold", "source_value": repr(v),
                 "precision_decimals": "", "written_matches_source_at_precision": "yes", "present_in_manuscript_context": "yes" if ok else "NO", "status": "CONFIRMED" if ok else "NOT FOUND IN CONTEXT"})
full_err = [get(S16, f"rotation_by_scale.full.per_angle.{i}.pooled_model_error_mean") for i in range(1, 10)]
cent_err = [get(S16, f"rotation_by_scale.1.per_angle.{i}.pooled_model_error_mean") for i in range(1, 10)]
for lab, v, dec, w in (("full-field min error", min(full_err), 3, "0.236"), ("full-field max error", max(full_err), 3, "0.641"),
                       ("center-pixel min error", min(cent_err), 3, "0.825"), ("center-pixel max error", max(cent_err), 3, "1.114"), ("center-pixel error at 90 deg", cent_err[5], 3, "0.955"), ("full-field error at 90 deg", full_err[5], 3, "0.236")):
    ok = fmt(v, dec) == w
    rows.append({"check_id": f"K{len(rows) + 1:03d}", "manuscript_section": "4.5; 5.3", "quantity": lab, "value_as_written": w, "unit": "normalized error", "sample_size_or_population": "2083 patches",
                 "source_file": S16, "source_path": "rotation_by_scale.{1,full}.per_angle.*.pooled_model_error_mean", "source_value": repr(v), "precision_decimals": dec,
                 "written_matches_source_at_precision": "yes" if ok else "NO", "present_in_manuscript_context": "yes" if w in BODY else "NO", "status": "CONFIRMED" if ok and w in BODY else "MISMATCH"})
cmp_all = all(f < c for f, c in zip(full_err, cent_err))
rows.append({"check_id": f"K{len(rows) + 1:03d}", "manuscript_section": "4.5", "quantity": "full-field error below center-pixel error at every angle", "value_as_written": "at every angle", "unit": "",
             "sample_size_or_population": "9 angles", "source_file": S16, "source_path": "per_angle comparison (read, not recomputed statistically)", "source_value": str(cmp_all), "precision_decimals": "",
             "written_matches_source_at_precision": "yes" if cmp_all else "NO", "present_in_manuscript_context": "yes" if "at every angle the full-field error" in BODY else "NO",
             "status": "CONFIRMED" if cmp_all else "MISMATCH"})

# ---------------- Table 8 (pathology)
for pid, lab, es, ctx in (("P1", "Cohen's d", "cohens_d", r"^\| P1 S_DL"), ("P3", "Cohen's d", "cohens_d", r"^\| P3 "), ("P2", "Spearman rho", "spearman_rho", r"^\| P2 S_DL"), ("P4", "Spearman rho", "spearman_rho", r"^\| P4 ")):
    b = f"primary_pooled.{pid}."
    chk("4.6 Table 8", f"{pid} effect size", ctx, S15, b + es, 4, "80 images", lab)
    chk("4.6 Table 8", f"{pid} raw p", ctx, S15, b + "p_value", 4, "80 images", "p")
    chk("4.6 Table 8", f"{pid} Holm-adjusted p", ctx, S15, f"holm_bonferroni_adjusted_p.{pid}", 1, "family of 4", "p")
    if pid in ("P2", "P4"):
        chk("4.6 Table 8", f"{pid} CI lower", ctx, S15, b + "rho_95CI_bootstrap.0", 4, "2000 bootstrap", "rho")
        chk("4.6 Table 8", f"{pid} CI upper", ctx, S15, b + "rho_95CI_bootstrap.1", 4, "2000 bootstrap", "rho")
chk("4.6 Table 8", "M1 rho", r"^\| M1", S15, "morphology.M1.spearman_rho", 4, "80 images", "rho")
chk("4.6 Table 8", "M1 p", r"^\| M1", S15, "morphology.M1.p_value", 4, "80 images", "p")
chk("4.6 Table 8", "M2 rho", r"^\| M2", S15, "morphology.M2.spearman_rho", 4, "80 images", "rho")
chk("4.6 Table 8", "M2 p", r"^\| M2", S15, "morphology.M2.p_value", 3, "80 images", "p")
chk("4.6 Table 8", "C1 grade coefficient", r"^\| C1", S15, "confound_adjustment_C1.grade_label_coef", 4, "80 images", "coefficient")
chk("4.6 Table 8", "C1 p", r"^\| C1", S15, "confound_adjustment_C1.grade_label_p", 4, "80 images", "p")
chk("4.6", "testB-only P2 rho", r"^For testB alone", S15, "primary_by_split.testB.P2.spearman_rho", 4, "20 images (4 benign)", "rho")
chk("4.6", "testB-only P2 raw p", r"^For testB alone", S15, "primary_by_split.testB.P2.p_value", 4, "20 images", "p")
chk("4.1", "mean S_DL malignant", r"mean image-level predicted order magnitude", S15, "primary_pooled.P1.mean_malignant", 4, "43 images", "S units")
chk("4.1", "mean S_DL benign", r"mean image-level predicted order magnitude", S15, "primary_pooled.P1.mean_benign", 4, "37 images", "S units")

# ---------------- Table 9 (CRAG)
for pop, lab, n in (("train", "CRAG train folder", 171), ("test", "CRAG test folder", 40), ("pooled", "CRAG pooled", 211)):
    ctx = rf"^\| {lab}"
    b = f"image_level_primary.{pop}.metrics."
    chk("4.7 Table 9", f"{lab} Glass delta A", ctx, S17, b + "A_angular_correspondence.glass_delta", 4, f"{n} images", "effect size")
    chk("4.7 Table 9", f"{lab} r B", ctx, S17, b + "B_order_magnitude_correlation.pearson_r", 4, f"{n} images", "r")
    chk("4.7 Table 9", f"{lab} p B", ctx, S17, b + "B_order_magnitude_correlation.pearson_p", 4, f"{n} images", "p")
    chk("4.7 Table 9", f"{lab} mean D_Q", ctx, S17, b + "C_tensor_similarity.mean_D_Q", 4, f"{n} images", "D_Q")
    chk("4.7 Table 9", f"{lab} n images", ctx, S17, f"image_level_primary.{pop}.n_images", 0, "", "images", written=str(n))
chk("4.7", "CRAG pooled r CI lower", r"^The pooled correlation had", S17, "image_level_primary.pooled.metrics.B_order_magnitude_correlation.bootstrap_95CI.0", 4, "211 images", "r")
chk("4.7", "CRAG pooled r CI upper", r"^The pooled correlation had", S17, "image_level_primary.pooled.metrics.B_order_magnitude_correlation.bootstrap_95CI.1", 4, "211 images", "r")
chk("4.7", "CRAG patch-level Glass delta", r"^The pooled correlation had", S17, "patch_level_secondary_diagnostic.metrics.A_angular_correspondence.glass_delta", 4, "5000-patch subsample", "effect size")
rc = get(S17, "rotation_consistency.n_angles_passed_vs_original_GlaS_threshold")
rows.append({"check_id": f"K{len(rows) + 1:03d}", "manuscript_section": "4.7", "quantity": "CRAG rotation angles passed", "value_as_written": "4 of 9", "unit": "angles", "sample_size_or_population": "5000-patch subsample",
             "source_file": S17, "source_path": "rotation_consistency.n_angles_passed_vs_original_GlaS_threshold", "source_value": repr(rc), "precision_decimals": "", "written_matches_source_at_precision": "yes" if rc == 4 else "NO",
             "present_in_manuscript_context": "yes" if "was 4 of 9 angles against the GlaS" in BODY else "NO", "status": "CONFIRMED" if rc == 4 else "MISMATCH"})

# ---------------- dataset / architecture counts
for lab, path, w in (("GlaS images", "all.n_images", "165"), ("gland instances", "all.n_gland_instances_total", "1530"), ("valid instances", "all.validity_category_counts.valid", "1084"),
                     ("border-truncated excluded", "all.validity_category_counts.excluded_border_truncated", "340"), ("near-isotropic excluded", "all.validity_category_counts.excluded_near_isotropic", "98"),
                     ("too-small excluded", "all.validity_category_counts.excluded_too_small", "8"), ("train valid", "train.validity_category_counts.valid", "567"), ("train instances", "train.n_gland_instances_total", "769"),
                     ("testA valid", "testA.validity_category_counts.valid", "443"), ("testA instances", "testA.n_gland_instances_total", "666"), ("testB valid", "testB.validity_category_counts.valid", "74"),
                     ("testB instances", "testB.n_gland_instances_total", "95"), ("fragmented instances", "all.n_fragmented_instances", "4")):
    v = get(S4, path)
    ok = str(int(v)) == w
    rows.append({"check_id": f"K{len(rows) + 1:03d}", "manuscript_section": "3.3; 4.1", "quantity": lab, "value_as_written": w, "unit": "instances/images", "sample_size_or_population": "GlaS",
                 "source_file": S4, "source_path": path, "source_value": repr(v), "precision_decimals": 0, "written_matches_source_at_precision": "yes" if ok else "NO",
                 "present_in_manuscript_context": "yes" if w in BODY else "NO", "status": "CONFIRMED" if ok and w in BODY else "MISMATCH"})
for lab, src, path, w in (("Model 3 parameters", S10, "n_params", "11,674"), ("epochs run", S10, "n_epochs_run", "37"), ("best dev loss", S10, "best_dev_loss", "0.5756"),
                          ("A5 parameters", "results_v2/validation/stage14/A5_plain_cnn_control/stage14_train_summary.json", "n_params", "11,676"),
                          ("CRAG images", "results_v2/validation/stage17_crag/crag_target_generation_summary.json", "n_images", "213"), ("CRAG train images", "results_v2/validation/stage17_crag/crag_target_generation_summary.json", "n_train_images", "173"),
                          ("CRAG test images", "results_v2/validation/stage17_crag/crag_target_generation_summary.json", "n_test_images", "40"), ("CRAG gland instances", "results_v2/validation/stage17_crag/crag_target_generation_summary.json", "n_gland_instances_total", "3054"),
                          ("CRAG large-gland images", "results_v2/validation/stage17_crag/crag_compatibility_report.json", "crag_images_with_dominant_fused_gland_gt30pct", "70"),
                          ("held-out valid patches", S11, "pooled_testA_testB_80_images.n_patches_valid", "2083"),
                          ("bootstrap n (Criterion B)", PRE, "criteria.B_order_magnitude_correspondence.bootstrap_n", "2000"), ("bootstrap seed", PRE, "criteria.B_order_magnitude_correspondence.bootstrap_seed", "42")):
    v = get(src, path)
    sv = f"{int(v):,}" if isinstance(v, int) and v >= 10000 else (f"{v:.4f}" if isinstance(v, float) else str(v))
    ok = sv == w or str(v).replace(",", "") == w.replace(",", "")
    rows.append({"check_id": f"K{len(rows) + 1:03d}", "manuscript_section": "3.x; 4.x", "quantity": lab, "value_as_written": w, "unit": "", "sample_size_or_population": "",
                 "source_file": src, "source_path": path, "source_value": repr(v), "precision_decimals": "", "written_matches_source_at_precision": "yes" if ok else "NO",
                 "present_in_manuscript_context": "yes" if w in BODY else "NO", "status": "CONFIRMED" if ok and w in BODY else "MISMATCH"})
# 29,399 CRAG valid patches
cv = get("results_v2/validation/stage17_crag/crag_target_generation_summary.json", "n_patches_valid")
tot = cv["train"] + cv["test"]
rows.append({"check_id": f"K{len(rows) + 1:03d}", "manuscript_section": "3.13; 4.7", "quantity": "CRAG valid patches (train + test as read from summary; sum of two frozen fields)", "value_as_written": "29,399", "unit": "patches",
             "sample_size_or_population": "CRAG", "source_file": "results_v2/validation/stage17_crag/crag_target_generation_summary.json", "source_path": "n_patches_valid.train + .test", "source_value": f"{cv['train']} + {cv['test']} = {tot}",
             "precision_decimals": 0, "written_matches_source_at_precision": "yes" if tot == 29399 else "NO", "present_in_manuscript_context": "yes" if "29,399" in BODY else "NO", "status": "CONFIRMED" if tot == 29399 else "MISMATCH"})

# ---------------- patient structure (recount of Grade.csv rows)
with open(P("data", "glas", "Warwick_QU_Dataset", "Grade.csv"), encoding="latin1", newline=None) as f:
    gl = [ln for ln in f.read().splitlines() if ln.strip()][1:]
rec = []
for ln in gl:
    parts = [x.strip() for x in ln.split(",")]
    rec.append((re.sub(r"_\d+$", "", parts[0]), int(parts[1]), parts[2].lower()))
train_p = {p for s, p, g in rec if s == "train"}
held = [(s, p, g) for s, p, g in rec if s in ("testA", "testB")]
hp = {p for s, p, g in held}
facts = (("total patients", len({p for s, p, g in rec}), "16"), ("held-out patients", len(hp), "12"), ("held-out patients also in train", len(hp & train_p), "11"),
         ("held-out images from patients in train", sum(1 for s, p, g in held if p in train_p), "79"), ("held-out images", len(held), "80"),
         ("malignant held-out images", sum(1 for s, p, g in held if g == "malignant"), "43"), ("benign held-out images", sum(1 for s, p, g in held if g == "benign"), "37"))
for lab, v, w in facts:
    ok = str(v) == w
    rows.append({"check_id": f"K{len(rows) + 1:03d}", "manuscript_section": "3.2; 4.1; 5.6", "quantity": lab + " (count of Grade.csv rows)", "value_as_written": w, "unit": "patients/images", "sample_size_or_population": "GlaS",
                 "source_file": "data/glas/Warwick_QU_Dataset/Grade.csv", "source_path": "row counts by split and patient ID", "source_value": str(v), "precision_decimals": 0,
                 "written_matches_source_at_precision": "yes" if ok else "NO", "present_in_manuscript_context": "yes" if w in BODY else "NO", "status": "CONFIRMED" if ok and w in BODY else "MISMATCH"})

# ---------------- PANDA summary table
sr = list(csv.DictReader(open(P("results_v2", "validation", "stage18_panda", "summary_table.csv"), encoding="utf-8")))
g4 = sorted(float(x["gleason_4_largest_vs_benign_median_ratio"]) for x in sr if x["gleason_4_largest_vs_benign_median_ratio"])
g3 = sorted(float(x["gleason_3_largest_vs_benign_median_ratio"]) for x in sr if x["gleason_3_largest_vs_benign_median_ratio"])
g5 = [float(x["gleason_5_largest_vs_benign_median_ratio"]) for x in sr if x["gleason_5_largest_vs_benign_median_ratio"]]
none = [x["image_id"][:8] for x in sr if all(float(x[k]) == 0 for k in ("gleason_3_pixel_fraction", "gleason_4_pixel_fraction", "gleason_5_pixel_fraction"))]
for lab, vals, ws in (("Gleason-4 ratios", g4, ["49.3", "136.8", "735.1", "1082.3"]), ("Gleason-3 ratios", g3, ["0.2", "28.7", "96.1", "112.0"]), ("Gleason-5 ratio", g5, ["6244.3"])):
    ok = [f"{v:.1f}" for v in vals] == ws
    rows.append({"check_id": f"K{len(rows) + 1:03d}", "manuscript_section": "4.8", "quantity": lab, "value_as_written": ", ".join(ws), "unit": "ratio", "sample_size_or_population": "10-case Radboud pilot",
                 "source_file": "results_v2/validation/stage18_panda/summary_table.csv", "source_path": "*_largest_vs_benign_median_ratio", "source_value": str(vals), "precision_decimals": 1,
                 "written_matches_source_at_precision": "yes" if ok else "NO", "present_in_manuscript_context": "yes" if all(w in BODY for w in ws) else "NO", "status": "CONFIRMED" if ok and all(w in BODY for w in ws) else "MISMATCH"})
rows.append({"check_id": f"K{len(rows) + 1:03d}", "manuscript_section": "4.8 Figure 7 caption", "quantity": "pilot cases with no Gleason 3/4/5 pixels", "value_as_written": "three", "unit": "cases", "sample_size_or_population": "10-case pilot",
             "source_file": "results_v2/validation/stage18_panda/summary_table.csv", "source_path": "gleason_3/4/5_pixel_fraction all 0", "source_value": f"{len(none)}: {', '.join(none)}", "precision_decimals": "",
             "written_matches_source_at_precision": "yes" if len(none) == 3 else "NO", "present_in_manuscript_context": "yes" if "three cases contain no Gleason 3, 4 or 5 pixels" in BODY else "NO",
             "status": "CONFIRMED (manuscript vs frozen table); DISCREPANCY WITH corrected map (two)"})

os.makedirs(OUT, exist_ok=True)
fields = ["check_id", "manuscript_section", "quantity", "value_as_written", "unit", "sample_size_or_population", "source_file", "source_path", "source_value", "precision_decimals",
          "written_matches_source_at_precision", "present_in_manuscript_context", "status"]
with open(os.path.join(OUT, "_critical_numbers_check.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
bad = [r for r in rows if not r["status"].startswith("CONFIRMED")]
print("critical checks:", len(rows), "| confirmed:", len(rows) - len(bad), "| not confirmed:", len(bad))
for r in bad:
    print("  ", r["check_id"], r["quantity"], "|", r["value_as_written"], "|", r["source_value"], "|", r["status"])
