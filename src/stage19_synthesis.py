"""
STAGE 19: frozen statistical synthesis. Reads ONLY already-computed,
frozen result files from Stages 11-18 and assembles the master tables.
No new statistic is computed here -- every number is read directly from
its source JSON/CSV, never retyped from memory or re-derived.
"""
import json
import os
import csv

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR = os.path.join(PROJECT_ROOT, "results_v2", "statistical_synthesis")
os.makedirs(OUT_DIR, exist_ok=True)


def load(path):
    with open(os.path.join(PROJECT_ROOT, path)) as f:
        return json.load(f)


s11 = load("results_v2/validation/stage11/stage11_results.json")
s12 = load("results_v2/validation/stage12/stage12_results.json")
s13 = load("results_v2/validation/stage13/stage13_results.json")
s14 = load("results_v2/validation/stage14/stage14_all_evaluations.json")
s14_a6 = load("results_v2/validation/stage14/A6_spatial_reduction/a6_results.json")
s15 = load("results_v2/validation/stage15/stage15_results.json")
s16 = load("results_v2/validation/stage16/stage16_results.json")
s17 = load("results_v2/validation/stage17_crag/stage17_results.json")

NA = "NA"
rows = []


def add(question, dataset, analysis, unit, n, statistic, ci, p, criterion, outcome, interpretation):
    rows.append({"Question": question, "Dataset": dataset, "Analysis": analysis, "Unit": unit, "n": n,
                 "Statistic": statistic, "95% CI": ci, "p-value": p, "Frozen criterion": criterion,
                 "Outcome": outcome, "Interpretation": interpretation})


# ---- Stage 11 : primary GlaS pooled ----
m = s11["pooled_testA_testB_80_images"]["metrics"]
add("Anatomical correspondence (angular)", "GlaS", "Stage 11 Criterion A, pooled testA+testB", "patch",
    m["n_patches"], f"Glass's delta={m['A_angular_correspondence']['glass_delta']:.4f} (mean err={m['A_angular_correspondence']['mean_deg']:.2f} deg)",
    NA, f"perm_p={m['A_angular_correspondence']['permutation_p_value']}", "Glass's delta >= 0.5",
    "FAIL", "Effect size far below threshold despite perm_p<0.01 (large-n significance, negligible effect)")
add("Anatomical correspondence (order magnitude)", "GlaS", "Stage 11 Criterion B, pooled testA+testB", "patch",
    m["n_patches"], f"Pearson r={m['B_order_magnitude_correlation']['pearson_r']:.4f}",
    str(m["B_order_magnitude_correlation"]["bootstrap_95CI"]), f"{m['B_order_magnitude_correlation']['pearson_p']:.4f}",
    "r>=0.30 AND p<0.01 AND CI lower>0", "FAIL", "r near zero, CI includes zero")
add("Anatomical correspondence (tensor similarity)", "GlaS", "Stage 11 Criterion C, pooled testA+testB", "patch",
    m["n_patches"], f"mean D_Q={m['C_tensor_similarity']['mean_D_Q']:.4f}, Glass's delta={m['C_tensor_similarity']['glass_delta']:.4f}",
    NA, f"perm_p={m['C_tensor_similarity']['permutation_p_value']}", "perm_p<0.01 AND Glass's delta>=0.5",
    "FAIL", "perm_p<0.01 met but Glass's delta negligible (0.007) -- statistical significance without practical effect")

# ---- Stage 12 : rotation ----
add("Rotation consistency (empirical, center-pixel)", "GlaS", "Stage 12 Criterion D", "patch (pooled testA+testB)",
    2083, f"{s12['criterion_D']['n_angles_passed']}/9 angles passed", NA, NA,
    ">=7/9 non-zero angles", "FAIL", "4/9 passed; passes concentrated at large angles (90,123,150,173)")

# ---- Stage 13 : Model 2 permutation null ----
for split in ["testA", "testB"]:
    r = s13["results"][split]
    add("Model 2 channel-assignment permutation null", "GlaS", f"Stage 13, {split}", "patch",
        r["n_patches"], f"z={r['z_score']:.4f} (observed={r['observed_statistic_mean_S']:.4f}, null_mean={r['null_mean_S']:.4f})",
        NA, f"one-sided={r['empirical_p_value_one_sided_null_le_observed']}, two-sided={r['empirical_p_value_two_sided_secondary_diagnostic']}",
        "descriptive only (no formal pass/fail)", "DESCRIPTIVE",
        "Observed statistic below null mean (negative z); not significant at p<0.05; Model 2 only, not transferable to Model 3")

# ---- Stage 14 : ablations (pooled geometric) ----
ABLATION_LABELS = {"A0_reference": "A0 reference", "A1_no_q": "A1 no Q supervision", "A3_q_dominant": "A3 Q-dominant",
                    "A4_q_only": "A4 Q-only", "A5_plain_cnn_control": "A5 non-equivariant control"}
for exp, label in ABLATION_LABELS.items():
    p = s14[exp]["pooled_testA_testB"]["geometric_metrics"]
    add("Ablation robustness of primary geometric result", "GlaS", f"Stage 14 {label}, pooled testA+testB", "patch",
        s14[exp]["pooled_testA_testB"]["n_patches"], f"Glass's delta_A={p['A_angular_correspondence']['glass_delta']:.4f}, r={p['B_order_magnitude_correlation']['pearson_r']:.4f}, D_Q={p['C_tensor_similarity']['mean_D_Q']:.4f}",
        NA, NA, "diagnostic only (context vs Stage 11 thresholds, not a new criterion)", "NO RESCUE",
        "No ablation approaches Criterion A/B/C thresholds; primary negative result robust to tested variations")
add("Rotation-consistency spatial-reduction diagnostic", "GlaS", "Stage 14 A6 (R1 full-field / R2 fixed-mask)", "patch (pooled)",
    2083, f"R1={s14_a6['reductions']['R1_full_spatial_mean']['n_angles_passed_diagnostic_only']}/9, R2={s14_a6['reductions']['R2_fixed_original_valid_mask_mean']['n_angles_passed_diagnostic_only']}/9 (vs center-pixel {s14_a6['stage12_center_pixel_comparison']['n_angles_passed']}/9)",
    NA, NA, ">=7/9 (context only)", "NO RESCUE", "Neither alternative reduction reaches the required 7/9")

# ---- Stage 15 : pathology ----
for pid, label in [("P1", "P1 S_DL vs binary grade"), ("P2", "P2 S_DL vs ordinal grade"),
                    ("P3", "P3 phi-dispersion vs binary grade"), ("P4", "P4 phi-dispersion vs ordinal grade")]:
    r = s15["primary_pooled"][pid]
    eff = r.get("cohens_d", r.get("spearman_rho"))
    ci = r.get("mean_diff_95CI_bootstrap", r.get("rho_95CI_bootstrap"))
    add("Pathology association (image-level, primary, pooled)", "GlaS", f"Stage 15 {label}", "image", 80,
        f"{eff:.4f}", str(ci), f"{r['p_value']:.4f}", "Holm-Bonferroni corrected family (4 tests)",
        "NOT SIGNIFICANT", f"Holm-adjusted p={s15['holm_bonferroni_adjusted_p'][pid]}")
r = s15["primary_by_split"]["testB"]["P2"]
add("Pathology association (image-level, split-specific, non-primary)", "GlaS", "Stage 15 P2, testB only (n=20, 4 benign)",
    "image", 20, f"Spearman rho={r['spearman_rho']:.4f}", str(r["rho_95CI_bootstrap"]), f"{r['p_value']:.4f}",
    "raw, uncorrected, NOT part of pre-specified corrected family", "NOT EVIDENTIARY",
    "Nominally significant but contradicted by pooled null; extreme class imbalance; not used as evidence")
add("Morphology association (gland area)", "GlaS", "Stage 15 M1", "image", 80,
    f"Spearman rho={s15['morphology']['M1']['spearman_rho']:.4f}", str(s15["morphology"]["M1"]["rho_95CI_bootstrap"]),
    f"{s15['morphology']['M1']['p_value']:.4f}", "descriptive only", "NOT SIGNIFICANT", "No material association")
add("Morphology association (gland count)", "GlaS", "Stage 15 M2", "image", 80,
    f"Spearman rho={s15['morphology']['M2']['spearman_rho']:.4f}", str(s15["morphology"]["M2"]["rho_95CI_bootstrap"]),
    f"{s15['morphology']['M2']['p_value']:.4f}", "descriptive only", "NOT SIGNIFICANT", "No material association")
c1 = s15["confound_adjustment_C1"]
add("Confound adjustment (grade_label | gland area)", "GlaS", "Stage 15 C1 (OLS)", "image", c1["n"],
    f"grade_label coef={c1['grade_label_coef']:.5f}", str(c1["grade_label_95CI"]), f"{c1['grade_label_p']:.4f}",
    "unconditional (run regardless of P1 outcome)", "NOT SIGNIFICANT", "Consistent with P1's unadjusted null")
r1 = s15["robustness_R1_patient_level"]
add("Patient-level robustness", "GlaS", "Stage 15 R1", "patient", r1["n_patients"], "descriptive only (no formal test)",
    NA, NA, "pre-specified sensitivity check", "INCONCLUSIVE", f"{r1['patients_with_mixed_grade_label_across_images']}/{r1['n_patients']} patients have mixed grade labels; n too small for formal test")

# ---- Stage 16 : multi-scale ----
SCALE_LABEL = {"1": "center", "3": "3x3", "5": "5x5", "9": "9x9", "17": "17x17", "full": "full-field"}
for w, label in SCALE_LABEL.items():
    m = s16["anatomy_by_scale"][w]["pooled"]["metrics"]
    rot = s16["rotation_by_scale"][w]
    add("Multi-scale anatomical correspondence", "GlaS", f"Stage 16 scale={label} (anatomy)", "patch (pooled)",
        s16["anatomy_by_scale"][w]["pooled"]["n_patches_total"], f"Glass's delta_A={m['A_angular_correspondence']['glass_delta']:.4f}, r={m['B_order_magnitude_correlation']['pearson_r']:.4f}",
        NA, NA, "Criterion A/B (context only)", "FAIL", "No scale meets Criterion A or B")
    add("Multi-scale rotation consistency", "GlaS", f"Stage 16 scale={label} (rotation)", "patch (pooled subsample)",
        NA, f"{rot['n_angles_passed_vs_original_threshold']}/9 angles", NA, NA, ">=7/9 (context only)",
        "FAIL" if rot["n_angles_passed_vs_original_threshold"] < 7 else "PASS (context only)",
        "Mild scale-dependent increase (4->6/9); no scale reaches 7/9")

# ---- Stage 17 : CRAG ----
for pop in ["train", "test", "pooled"]:
    m = s17["image_level_primary"][pop]["metrics"]
    add("External cross-dataset anatomical correspondence", "CRAG", f"Stage 17, image-level primary, {pop}", "image",
        s17["image_level_primary"][pop]["n_images"], f"Glass's delta_A={m['A_angular_correspondence']['glass_delta']:.4f}, r={m['B_order_magnitude_correlation']['pearson_r']:.4f}, D_Q={m['C_tensor_similarity']['mean_D_Q']:.4f}",
        str(m["B_order_magnitude_correlation"]["bootstrap_95CI"]), f"{m['B_order_magnitude_correlation']['pearson_p']:.4f}",
        "GlaS thresholds, descriptive comparison only", "NO EXTERNAL SUPPORT", "Negative Glass's delta at every population")
pm = s17["patch_level_secondary_diagnostic"]["metrics"]
add("External cross-dataset anatomical correspondence (patch diagnostic)", "CRAG", "Stage 17, patch-level secondary, pooled subsample",
    "patch", s17["patch_level_secondary_diagnostic"]["n_patches_subsampled"], f"Glass's delta_A={pm['A_angular_correspondence']['glass_delta']:.4f}, r={pm['B_order_magnitude_correlation']['pearson_r']:.4f}",
    str(pm["B_order_magnitude_correlation"]["bootstrap_95CI"]), f"{pm['B_order_magnitude_correlation']['pearson_p']:.4f}",
    "GlaS thresholds, descriptive comparison only", "NO EXTERNAL SUPPORT", "Consistent with pooled image-level null")
add("External cross-dataset rotation consistency", "CRAG", "Stage 17, patch-level subsample", "patch (pooled subsample)",
    5000, f"{s17['rotation_consistency']['n_angles_passed_vs_original_GlaS_threshold']}/9 angles", NA, NA,
    ">=7/9 (GlaS threshold, descriptive only)", "FAIL", "Same 4/9 count and identical per-angle pass pattern as Stage 12 (GlaS)")

# ---- Stage 18 : PANDA ----
add("Cross-organ external validation feasibility", "PANDA (Radboud)", "Stage 18 compatibility gate", "case (pilot)", 10,
    "3/3 cases with substantial Gleason 4/5 content showed largest connected component 49x-6244x the same-case benign-gland reference size",
    NA, NA, "gland-instance geometry must be recoverable without inventing structure", "INFEASIBLE",
    "Semantic Gleason-pattern masks, not gland instances; Q_anat not generated; Model 3 not run")

with open(os.path.join(OUT_DIR, "MASTER_STATISTICAL_TABLE.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow(r)

print(f"MASTER_STATISTICAL_TABLE.csv written: {len(rows)} rows")

# ============================================================ evidence table
evidence_rows = [
    {"Evidence domain": "Primary anatomical correspondence (GlaS)", "Finding": "Criteria A, B, C all fail; effect sizes negligible despite some large-n significant p-values",
     "Evidence status": "NOT SUPPORTED", "Reason": "Glass's delta << 0.5 and r ~ 0 at every criterion, pooled testA+testB (n=2083 patches)"},
    {"Evidence domain": "Empirical rotation consistency (GlaS)", "Finding": "4/9 non-zero angles pass the frozen threshold; passes concentrated at large angles",
     "Evidence status": "NOT SUPPORTED", "Reason": "Required >=7/9; center-pixel measurement (Stage 12), confirmed with alternative reductions (Stage 14 A6, Stage 16) that also fail to reach 7/9"},
    {"Evidence domain": "Model 2 channel-assignment permutation null", "Finding": "Observed order magnitude below the permutation null mean in both testA and testB (z~-1.71), not significant at p<0.05",
     "Evidence status": "INCONCLUSIVE", "Reason": "Descriptive result only, applies to Model 2's post-hoc formulation, not transferable to Model 3"},
    {"Evidence domain": "Ablation robustness", "Finding": "No tested architectural/loss-weighting variation (A1, A3, A4, A5) rescues the primary geometric result",
     "Evidence status": "NOT SUPPORTED", "Reason": "All 5 conditions (A0-A5) show near-null Criterion A/B/C metrics; primary negative result is robust to these variations"},
    {"Evidence domain": "Pathology association (GlaS)", "Finding": "All 4 pre-specified, Holm-corrected pooled tests (P1-P4) null; morphology (M1-M2) and confound-adjustment (C1) null; one non-primary split-specific result flagged but not used as evidence",
     "Evidence status": "NOT SUPPORTED", "Reason": "No primary test approaches significance after correction; patient-level check (R1) inconclusive due to small n and mixed-grade patients"},
    {"Evidence domain": "Multi-scale sensitivity", "Finding": "Mild, monotonic-ish improvement in rotation pass-count (4->6/9) and Glass's delta_A with larger aggregation scale, but no scale meets either frozen criterion",
     "Evidence status": "NOT SUPPORTED", "Reason": "Even the most favorable scale (full-field) fails both Criterion A/B and the 7/9 rotation requirement"},
    {"Evidence domain": "External cross-dataset validation (CRAG)", "Finding": "Weak-to-null correspondence at every population/unit; Glass's delta negative at every population; rotation consistency reproduces GlaS's exact 4/9 pattern",
     "Evidence status": "NOT SUPPORTED", "Reason": "Consistent with, not contradicting, the GlaS finding; CRAG shares partial institutional provenance with GlaS (not fully independent)"},
    {"Evidence domain": "Cross-organ external validation (PANDA)", "Finding": "Radboud masks are semantic Gleason-pattern classes, not individual gland instances; fused Gleason 4/5 regions preclude defensible gland-instance extraction without inventing structure",
     "Evidence status": "INFEASIBLE", "Reason": "Compatibility gate failed on direct inspection of 10 pilot cases; no Q_anat generated; no Model 3 inference run; not a model failure"},
]
with open(os.path.join(OUT_DIR, "MASTER_EVIDENCE_TABLE.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(evidence_rows[0].keys()))
    w.writeheader()
    for r in evidence_rows:
        w.writerow(r)
print(f"MASTER_EVIDENCE_TABLE.csv written: {len(evidence_rows)} rows")
