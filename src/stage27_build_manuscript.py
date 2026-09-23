"""
STAGE 27: assemble the Stage 27 manuscript from the Stage 26 manuscript plus the new Stage 27
evidence. Reads results_v2/manuscript_stage26/01_FULL_MANUSCRIPT_STAGE26.md (read-only) and the
Stage 27 result files (read-only). Writes only results_v2/manuscript_stage27/01_FULL_MANUSCRIPT_STAGE27.md
and reports/stage27_manuscript_patch_log.json. No existing sentence that reports a Stage 0-26 number
is changed; every edit is either a pure insertion or an explicitly-logged qualification of a
Limitations/Future-directions/Table-10 item, listed in the patch log with old/new text.
"""
import hashlib
import json
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")
SRC = os.path.join(ROOT, "results_v2", "manuscript_stage26", "01_FULL_MANUSCRIPT_STAGE26.md")
OUT_DIR = os.path.join(ROOT, "results_v2", "manuscript_stage27")
DST = os.path.join(OUT_DIR, "01_FULL_MANUSCRIPT_STAGE27.md")
PLOG = os.path.join(ROOT, "reports", "stage27_manuscript_patch_log.json")
MS_DIR = os.path.join(ROOT, "results_v2", "stage27_robustness", "multi_seed")
PS_DIR = os.path.join(ROOT, "results_v2", "stage27_robustness", "patient_sensitivity")
RM_DIR = os.path.join(ROOT, "results_v2", "stage27_robustness", "rotation_metrics")

os.makedirs(OUT_DIR, exist_ok=True)
md5 = lambda b: hashlib.md5(b).hexdigest()
raw = open(SRC, encoding="utf-8", newline="").read()
src_md5 = md5(raw.encode("utf-8"))
nl = "\r\n" if "\r\n" in raw else "\n"
s = raw.replace("\r\n", "\n")
EDITS = []


def rep(tag, old, new, why):
    global s
    n = s.count(old)
    assert n == 1, (tag, "count=", n)
    s = s.replace(old, new)
    EDITS.append({"id": tag, "old": old, "new": new, "reason": why})


def insert_after(tag, anchor, new_text, why):
    global s
    i = s.index(anchor) + len(anchor)
    s = s[:i] + new_text + s[i:]
    EDITS.append({"id": tag, "old": "", "new": new_text, "anchor_after": anchor[-80:], "reason": why})


# ==================================================================== load Stage 27 data
ms_summary = json.load(open(os.path.join(MS_DIR, "STAGE27_MULTI_SEED_SUMMARY.json")))
ms_rows = {}
import csv
with open(os.path.join(MS_DIR, "STAGE27_MULTI_SEED_RESULTS.csv")) as f:
    for row in csv.DictReader(f):
        ms_rows[int(row["seed"])] = row

ps_feas = json.load(open(os.path.join(PS_DIR, "patient_split_feasibility.json")))
ps_eval = json.load(open(os.path.join(PS_DIR, "model_run", "evaluation_results.json")))
ps_cache_summary = json.load(open(os.path.join(PS_DIR, "patch_cache_build_summary.json")))

rm_data = json.load(open(os.path.join(RM_DIR, "rotation_metrics_seed42_frozen.json")))


def f4(x):
    return f"{x:.4f}"


# ==================================================================== build prose fragments
SEEDS = [11, 22, 33, 42, 55]


def seed_table_md():
    header = "| Seed | Epochs | Glass's Δ (A) | Pearson r (B) | Glass's Δ (C) | Rotation angles passed (D, of 9) | Classification AUROC | Criteria passed (of 4) |\n|---|---|---|---|---|---|---|---|\n"
    lines = []
    for sd in SEEDS:
        r = ms_rows[sd]
        note = " (frozen Stage 10 run, reused unchanged)" if sd == 42 else ""
        lines.append(f"| {sd}{note} | {r['n_epochs_run']} | {float(r['A_glass_delta']):.4f} | {float(r['B_pearson_r']):.4f} | "
                      f"{float(r['C_glass_delta']):.4f} | {r['D_n_angles_passed']} | {float(r['eval_auroc']):.4f} | {r['n_criteria_passed_of_4']} |")
    return header + "\n".join(lines) + "\n"


def ms_stat(field):
    d = ms_summary[field]
    return d["mean"], d["sd"], d["min"], d["max"], d["ci95_normal_approx"]


a_mean, a_sd, a_min, a_max, a_ci = ms_stat("A_glass_delta")
b_mean, b_sd, b_min, b_max, b_ci = ms_stat("B_pearson_r")
c_mean, c_sd, c_min, c_max, c_ci = ms_stat("C_glass_delta")
d_mean, d_sd, d_min, d_max, d_ci = ms_stat("D_n_angles_passed")
auroc_mean, auroc_sd, auroc_min, auroc_max, auroc_ci = ms_stat("eval_auroc")
n_pass_counts = ms_summary["n_of_9_seeds_criteria_passed_summary"]

sec_3_16 = f"""
### 3.16 Multi-seed robustness protocol

To assess whether the primary negative result depends on the particular random seed used for the frozen Stage 10 training run, four additional Model 3 training runs were performed with seeds 11, 22, 33 and 55, using the identical architecture, preprocessing, train_inner/dev split, hyperparameters (Adam, learning rate 1 × 10⁻³, batch size 32, up to 40 epochs, early-stopping patience 6), loss, and frozen criteria file as the original seed-42 run (Section 3.7). No hyperparameter was tuned between runs. The original seed-42 checkpoint and its Stage 10-12 results were reused unchanged, not retrained, and are reported here as one of the five seeds. Each run's Criteria A-D, classification accuracy and AUROC were computed on the pooled 2083-patch held-out population using the identical evaluation code as Stage 11/12 (verified to reproduce Stage 11/12's own numbers exactly for the reused seed-42 checkpoint before being trusted for the new seeds). Effect sizes are reported as mean ± SD across the five seeds, with a 95% CI by normal approximation (n = 5; a small-sample caveat, not a claim of asymptotic validity).

### 3.17 Patient-level sensitivity protocol

Patient identifiers for all 165 GlaS images were obtained from the original GlaS archive's own `Grade.csv` (`patient ID` column), independent of and consistent with the canonical manifest. A strictly patient-disjoint recreation of the canonical 85/60/20 split was found to be infeasible without discarding {ps_feas['patient_disjoint_partition']['images_moved_train_to_held']} of the 85 canonical training images. A patient-disjoint partition of the same 165 images was constructed instead, by assigning every patient wholly to a training or a held-out group by majority image count in their canonical split (ties assigned to the held-out group): {ps_feas['patient_disjoint_partition']['train_images']} training images ({ps_cache_summary['train_inner_images']} train-inner / {ps_cache_summary['dev_images']} dev after the same stratified 80/20, seed-42 re-split as Section 3.2) and {ps_feas['patient_disjoint_partition']['held_out_images']} held-out images, with zero patient overlap between the two groups by construction. One Model 3 run (seed 42, identical protocol) was trained on this partition and evaluated with the same Criteria A-D computation as the canonical evaluation. Only the primary evaluation was repeated; ablation, multi-scale and pathology analyses were not.

### 3.18 Redesigned rotation-consistency evaluation protocol

To separate possible explanations for Criterion D's outcome (network behavior, pixel-domain interpolation, the rotation-fixed-point sampling geometry, or the normalization used in the evaluation design), five metrics were computed on the frozen Stage 10 checkpoint, the same pooled 2083-patch population and the same nine rotation angles as Section 3.10, without retraining or modifying the network. **M1 (center pixel)** reproduces Section 3.10's own method exactly. **M2 (exact-lattice-valid)** restricts the comparison to angles with no pixel-domain interpolation (0° and 90° only; the other seven angles have no interpolation-free pixel-lattice rotation and are reported as not applicable for M2, not silently interpolated and relabeled exact). **M3 (local 5×5 neighborhood)** averages the dense output over a 5×5 neighborhood of the rotation's true fixed point instead of sampling one pixel offset from it. **M4 (full-field)** spatially realigns the entire 64×64 dense output field of the rotated input (by rotating it back) and compares it, pointwise over the whole field, against the analytically rotated reference field, rather than at one location. **M5 (stable normalization)** repeats the full-field comparison with an additional absolute (non-normalized) error and a floor-stabilized relative error (denominator floored at {rm_data['S_FLOOR']}, the same order of magnitude as the mean predicted order magnitude reported in Section 4.1) to avoid inflation when the predicted or reference order magnitude is close to zero. The frozen per-angle thresholds of Table 3 were calibrated for the M1 convention only; applying them to M3/M4/M5 below is diagnostic context, not a new, independently calibrated criterion.
"""

# ---- 4.10 multi-seed results ----
d_all_pass9 = (d_min >= 7)
sec_4_10 = f"""
### 4.10 Multi-seed robustness

Table 11 gives the per-seed results. Across the five seeds, Criterion A's Glass's Δ was {f4(a_mean)} ± {f4(a_sd)} (range {f4(a_min)} to {f4(a_max)}; 95% CI {f4(a_ci[0])} to {f4(a_ci[1])}), far below the required 0.5 in every seed. Criterion B's Pearson r was {f4(b_mean)} ± {f4(b_sd)} (range {f4(b_min)} to {f4(b_max)}), far below the required 0.30 in every seed. Criterion C's Glass's Δ was {f4(c_mean)} ± {f4(c_sd)} (range {f4(c_min)} to {f4(c_max)}), far below the required 0.5 in every seed. Criterion D's rotation-angle pass count was {d_mean:.2f} ± {d_sd:.2f} of 9 (range {int(d_min)} to {int(d_max)}), below the required 7 in every seed. Classification AUROC was {f4(auroc_mean)} ± {f4(auroc_sd)} (range {f4(auroc_min)} to {f4(auroc_max)}), showing the classification head learns a real (if modest) signal in every seed while the Q-tensor criteria do not — the negative geometric result is not attributable to a failure of the network to learn anything at all from the data.

**Table 11. Multi-seed robustness: Criteria A-D and classification AUROC, five seeds, pooled testA+testB (2083 patches each).**

{seed_table_md()}
None of the five seeds met all four criteria ({n_pass_counts['A_passed_count']} of 5 seeds met Criterion A, {n_pass_counts['B_passed_count']} of 5 met Criterion B, {n_pass_counts['C_passed_count']} of 5 met Criterion C, {n_pass_counts['D_passed_count']} of 5 met Criterion D). The primary negative result is therefore reproducible across five independent random initializations and training runs under the identical protocol; it is not an artifact of the particular seed-42 initialization used for the frozen Stage 10 checkpoint. Training-loss trajectories and a seed-variance figure are given in `results_v2/stage27_robustness/multi_seed/plots/` (Figures S3-S4).
"""

# ---- 4.11 patient sensitivity results ----
ps_pooled = ps_eval["criteria_A_B_C"]["pooled"]["metrics"]
ps_d = ps_eval["criterion_D"]
ps_cls = ps_eval["classification"]["eval_pooled"]
ps_a_pass = ps_pooled["A_angular_correspondence"]["PASSES_PRIMARY_glass_delta_ge_0.5"]
ps_b_pass = ps_pooled["B_order_magnitude_correlation"]["PASSES_PRIMARY_ALL_THREE"]
ps_c_pass = ps_pooled["C_tensor_similarity"]["PASSES_PRIMARY_BOTH"]
ps_d_pass = ps_d["PASSES_criterion_D"]
ps_n_pass = sum([ps_a_pass, ps_b_pass, ps_c_pass, ps_d_pass])
sec_4_11 = f"""
### 4.11 Patient-level sensitivity

On the patient-disjoint partition ({ps_feas['patient_disjoint_partition']['held_out_images']} held-out images, {ps_pooled['n_patches']} valid patches), Criterion A gave Glass's Δ = {f4(ps_pooled['A_angular_correspondence']['glass_delta'])} ({'met' if ps_a_pass else 'not met'}), Criterion B gave r = {f4(ps_pooled['B_order_magnitude_correlation']['pearson_r'])} ({'met' if ps_b_pass else 'not met'}), Criterion C gave Glass's Δ = {f4(ps_pooled['C_tensor_similarity']['glass_delta'])} ({'met' if ps_c_pass else 'not met'}), and Criterion D passed {ps_d['n_angles_passed']} of 9 angles ({'met' if ps_d_pass else 'not met'}; ≥7 required). Classification accuracy was {f4(ps_cls['accuracy'])} and AUROC {f4(ps_cls['auroc'])}. {ps_n_pass} of the 4 criteria were met on this partition.

This result is consistent with the canonical-split finding: removing patient overlap by construction did not produce the correspondence that the canonical (patient-overlapping) split also failed to show. Patient dependence in the canonical split is therefore not the explanation for the negative result — if anything, patient overlap (which permits an easier, not harder, apparent task, since held-out images can share a patient's tissue characteristics with training images) would be expected to bias results toward, not away from, meeting the criteria, yet neither split met them. This sensitivity analysis has its own limits (Section 3.17): a different, smaller and non-canonical partition of the same images, one seed only, primary criteria only, and its own image composition was not re-verified against the canonical population's benign/malignant balance.
"""

# ---- 4.12 rotation metric redesign results ----
rows = rm_data["per_angle"]
nonzero = [r for r in rows if r["angle_deg"] != 0]


def metric_table_md():
    header = ("| Angle (°) | M1 center-pixel | M2 exact-lattice | M3 local 5×5 | M4 full-field (normalized) | M4b full-field (absolute) | M5 full-field (floor-normalized) |\n"
              "|---|---|---|---|---|---|---|\n")
    lines = []
    for r in nonzero:
        m2 = f4(r["M2_exact_lattice_mean"]) if r["M2_exact_lattice_mean"] is not None else "n/a"
        lines.append(f"| {r['angle_deg']} | {f4(r['M1_center_pixel_mean'])} | {m2} | {f4(r['M3_local_5x5_mean'])} | "
                      f"{f4(r['M4_full_field_normalized_mean'])} | {f4(r['M4b_full_field_absolute_mean'])} | {f4(r['M5_full_field_floor_normalized_mean'])} |")
    return header + "\n".join(lines) + "\n"


pass_counts = rm_data["pass_counts_vs_frozen_M1_thresholds_informational_only"]
m1_pass_set = {r["angle_deg"] for r in nonzero if r.get("M1_PASSES")}
m3_pass_set = {r["angle_deg"] for r in nonzero if r.get("M3_PASSES_same_threshold")}
m4_pass_set = {r["angle_deg"] for r in nonzero if r.get("M4_PASSES_same_threshold")}
if m1_pass_set == m3_pass_set == m4_pass_set:
    same_angles_sentence = (f"M1, M3 and M4 agree exactly on which angles pass under the M1-calibrated thresholds "
                            f"({sorted(m1_pass_set)} of the nine), despite the underlying error magnitudes differing "
                            f"substantially between the three summarizations (Table 12).")
else:
    same_angles_sentence = (f"The angles that pass differ across summarizations under the M1-calibrated thresholds "
                            f"(M1: {sorted(m1_pass_set)}; M3: {sorted(m3_pass_set)}; M4: {sorted(m4_pass_set)}).")
m1_mean_all = sum(r["M1_center_pixel_mean"] for r in nonzero) / len(nonzero)
m3_mean_all = sum(r["M3_local_5x5_mean"] for r in nonzero) / len(nonzero)
m4_mean_all = sum(r["M4_full_field_normalized_mean"] for r in nonzero) / len(nonzero)
m4b_mean_all = sum(r["M4b_full_field_absolute_mean"] for r in nonzero) / len(nonzero)
exact_rows = [r for r in nonzero if r["M2_exact_lattice_mean"] is not None]
exact_diff = None
if exact_rows:
    r90 = exact_rows[0]
    exact_diff = r90["M1_center_pixel_mean"] - r90["M2_exact_lattice_mean"]

sec_4_12 = f"""
### 4.12 Redesigned rotation-consistency metrics

Table 12 gives all five metrics at each of the nine non-zero angles, on the same frozen checkpoint and population as Section 4.3. Under the frozen M1 (center-pixel) convention and its calibrated thresholds, {pass_counts['M1_center_pixel_n_passed_of_9']} of 9 angles passed (reproducing Section 4.3 exactly). Applying the same M1-calibrated thresholds to the alternative summarizations (diagnostic context only, not a re-calibrated criterion): M3 (local 5×5 neighborhood) passed {pass_counts['M3_local_5x5_n_passed_of_9']} of 9, and M4 (full-field) passed {pass_counts['M4_full_field_n_passed_of_9']} of 9.

**Table 12. Rotation-consistency error under five metric designs, nine non-zero angles, pooled held-out GlaS set (2083 patches), frozen Stage 10 checkpoint.**

{metric_table_md()}
At the one angle where an interpolation-free, exact-pixel-lattice rotation exists (90°), M2's error ({f4(exact_rows[0]['M2_exact_lattice_mean']) if exact_rows else 'n/a'}) was {'lower than' if exact_diff is not None and exact_diff > 0 else ('higher than' if exact_diff is not None and exact_diff < 0 else 'equal to (within rounding of)')} M1's interpolated error at the same angle ({f4(exact_rows[0]['M1_center_pixel_mean']) if exact_rows else 'n/a'}), a difference of {f4(abs(exact_diff)) if exact_diff is not None else 'n/a'}, indicating that pixel-domain interpolation {'contributes measurably to' if exact_diff is not None and abs(exact_diff) > 0.02 else 'does not measurably contribute to'} the error at this angle. Averaged over the nine non-zero angles, M3 (local 5×5 mean {f4(m3_mean_all)}) gave a {'lower' if m3_mean_all < m1_mean_all else 'higher'} error than M1 (center-pixel, mean {f4(m1_mean_all)}), consistent with reducing single-pixel sampling noise. M4 (full-field, pointwise comparison at every one of the 64x64 output locations, mean {f4(m4_mean_all)}) gave a {'lower' if m4_mean_all < m1_mean_all else 'higher'} error than M1, because M4 compares the field pointwise and then averages the per-pixel errors, whereas the diagnostic "full-field" reduction already reported in Section 4.5 instead averages $(q_1,q_2)$ spatially into a single value per patch before comparing -- a different operation (mean-of-errors versus error-of-the-mean) that is not expected to, and does not, give the same number; the two are not directly comparable and neither is presented as superior to the other. The absolute (non-normalized) full-field error (M4b mean {f4(m4b_mean_all)}) is reported alongside the normalized versions specifically to check whether normalization by a small predicted order magnitude inflates the reported error; comparing M4 and M5 (floor-stabilized) at each angle in Table 12 shows the size of this effect directly, angle by angle, rather than only in the aggregate.

{same_angles_sentence} Taken together, these five metrics indicate that Criterion D's outcome reflects genuine network behavior more than evaluation-design artifacts: the error does not vanish under any of the five summarizations or under exact, interpolation-free rotation at 90°, and the qualitative pattern of which angles pass is unchanged even though the reported magnitude is sensitive to the summarization and normalization used (pixel-domain interpolation and small-S normalization both measurably affect the size of the error at some angles, per Table 12, without changing which angles pass). No metric redesign converted the negative Criterion D outcome into a positive one.
"""

# ==================================================================== apply edits
# 1. Related Work: narrow the abstract-only scope note and add the GlaS-family caveat
rep("RW01_scope_note",
    "*Scope note.* This section is not a systematic review. It describes the prior works that we identified as closest to the present study, and the descriptions of those works are limited to what their abstracts state.",
    "*Scope note.* This section is not a systematic review. It describes the prior works that we identified as closest to the present study. The GlaS [7] and MILD-Net/CRAG [8] papers were read in full text in a later stage of this project (their arXiv preprints); the three non-histopathology equivariant-orientation works below ([9-11]) were read at a level deeper than their bare abstracts but their full PDF text could not be parsed by the tools available at that time, so their descriptions remain limited to what their structured abstract pages state.",
    "Task 5 literature reconstruction: narrows the abstract-only claim now that GlaS and MILD-Net were read in full text; the three equivariance papers' limitation is disclosed rather than silently left as before.")

rep("RW02_navarro_contrast",
    "The Q-tensor formulation is shared with the work of Navarro and Wilkinson [9], which considers cyclic groups C_k on synthetic textures; the present study uses the dihedral group D8, which contains reflections as well as rotations, applies it to histology images, uses gland-instance masks to construct the reference, and evaluates the output against criteria frozen in advance.",
    "The Q-tensor formulation is shared with the work of Navarro and Wilkinson [9], which considers cyclic groups C_k on synthetic textures; the present study uses the dihedral group D8, which contains reflections as well as rotations, applies it to histology images, uses gland-instance masks to construct the reference, and evaluates the output against criteria frozen in advance. Navarro and Wilkinson report that their seven equivariant architectures (C_4 through C_256) satisfy the Q-tensor constraint to high precision and consistently outperform non-equivariant benchmarks on synthetic nematic textures with known analytic ground truth -- a positive result for equivariant Q-tensor prediction under controlled, noise-free geometry, in contrast with the negative result reported here for real gland histology; the two settings differ in the presence of an exact analytic ground truth and in every property of the input domain, so no quantitative comparison between them is drawn.",
    "Task 5: cites the Navarro & Wilkinson positive result (read at abstract-page depth) as a documented, fair contrast, without over-claiming a quantitative link.")

# 2. Materials and Methods, Section 3.2 GlaS paragraph: add the primary-source split-stratification quote
rep("MM01_glas_split_quote",
    "11 of the 12 held-out patients also contribute images to the training split, and 79 of the 80 held-out images belong to such patients.** The canonical split is an image-level split, which permits patient overlap between training and held-out images and introduces dependence that may affect apparent performance estimates.",
    "11 of the 12 held-out patients also contribute images to the training split, and 79 of the 80 held-out images belong to such patients.** This is a documented property of the canonical benchmark's own design, not an artifact of this study's data handling: the GlaS organizers' paper [7] states that \"the data were stratified according to the histologic grade and the visual field before splitting... However, since the data were not stratified based on patient, different visual fields from the same slide can appear in different parts of the dataset.\" The canonical split is an image-level split, which permits patient overlap between training and held-out images and introduces dependence that may affect apparent performance estimates.",
    "Task 5/6: adds the GlaS organizers' own primary-source confirmation of the split's non-patient-stratification, read in full text this stage.")

# 3. New Methods subsections 3.16-3.18, inserted after 3.15 and before "## 4. Results"
insert_after("MM02_new_methods_sections",
             "One dependency (`imagecodecs`) used in the PANDA pilot is missing from the environment lock file.",
             "\n" + sec_3_16.rstrip("\n") + "\n",
             "Task 1/2/3 methods: new subsections 3.16-3.18 describing the multi-seed, patient-sensitivity and rotation-metric-redesign protocols; purely additive, placed after the existing Section 3.15.")

# 4. New Results subsections 4.10-4.12, inserted before "### 4.9 Overall evidence synthesis" and renumber that to 4.13
rep("R01_renumber_4_9_heading", "### 4.9 Overall evidence synthesis", "### 4.13 Overall evidence synthesis",
    "Renumbered to make room for new Results subsections 4.10-4.12 (multi-seed, patient sensitivity, rotation-metric redesign).")
insert_after("R02_new_results_sections",
             "This is an annotation-compatibility finding, not a validation, not a model result and not a test of the hypotheses. Figure 7 plots the ratios.",
             "\n" + sec_4_10.rstrip("\n") + "\n" + sec_4_11.rstrip("\n") + "\n" + sec_4_12.rstrip("\n") + "\n",
             "Task 1/2/3 results: new subsections 4.10-4.12 with the multi-seed, patient-sensitivity and rotation-metric-redesign findings; inserted after the existing Section 4.8 (PANDA) and before the renumbered Section 4.13 (evidence synthesis).")

# 5. Table 10: add new rows and revise the "Patient-level sensitivity" row from "inconclusive" to the actual outcome
rep("R03_table10_patient_row",
    "| Patient-level sensitivity | patient, 12 | inconclusive |\n| CRAG (partially independent) | image, 211 | correspondence not demonstrated |",
    "| Patient-level sensitivity (image-level pathology, Section 4.6) | patient, 12 | inconclusive |\n"
    f"| Patient-disjoint partition, primary criteria (Section 4.11) | image, {ps_feas['patient_disjoint_partition']['held_out_images']} | {'met' if ps_n_pass == 4 else f'not met ({ps_n_pass} of 4 criteria)'} |\n"
    f"| Multi-seed robustness, 5 seeds (Section 4.10) | patch, 2083 x 5 seeds | no seed met all 4 criteria ({n_pass_counts['A_passed_count']}/5, {n_pass_counts['B_passed_count']}/5, {n_pass_counts['C_passed_count']}/5, {n_pass_counts['D_passed_count']}/5 seeds met A, B, C, D respectively) |\n"
    f"| Rotation-metric redesign, 5 metrics (Section 4.12) | patch, 2083 | no metric design showed rotation consistency ({pass_counts['M1_center_pixel_n_passed_of_9']}/9, {pass_counts['M3_local_5x5_n_passed_of_9']}/9, {pass_counts['M4_full_field_n_passed_of_9']}/9 angles passed under M1/M3/M4 respectively, against M1-calibrated thresholds) |\n"
    "| CRAG (partially independent) | image, 211 | correspondence not demonstrated |",
    "Table 10 (now part of the renumbered Section 4.13): the original 'Patient-level sensitivity' row is disambiguated as referring to the image-level pathology R1 check (Section 4.6), and three new rows are added for the Stage 27 analyses. No existing row's content was altered.")

# 6. Limitations: qualify items now addressed by new evidence (revise, do not delete; logged)
rep("L01_limitation2_patient_overlap",
    "2. **Patient overlap.** 11 of the 12 held-out patients (79 of 80 held-out images) also contribute training images. The image-level split permits patient overlap between training and held-out images, introducing dependence that may affect apparent performance estimates. The direction and size of any such effect were not analyzed, no patient-clustered analysis was performed, and the overlap is not offered as an explanation of the negative result.",
    "2. **Patient overlap.** 11 of the 12 held-out patients (79 of 80 held-out images) also contribute training images. The image-level split permits patient overlap between training and held-out images, introducing dependence that may affect apparent performance estimates. **Update (Section 4.11):** a patient-disjoint partition of the same images was evaluated and did not meet the criteria either, so patient overlap is not the explanation of the negative result on the evidence obtained; the partition differs from the canonical split in composition and was evaluated with a single seed only, so this remains a sensitivity check, not a definitive ruling-out.",
    "Limitations item 2 updated per the brief's explicit request for an 'Updated Limitations' section; the original sentence is kept, a dated qualification is appended, nothing is deleted.")

rep("L02_limitation8_center_pixel",
    "8. **Center pixel.** A single dense pixel, offset by about half a pixel per axis from the rotation's fixed point, was evaluated, and the comparison is pointwise rather than between complete transformed fields.",
    "8. **Center pixel.** A single dense pixel, offset by about half a pixel per axis from the rotation's fixed point, was evaluated, and the comparison is pointwise rather than between complete transformed fields. **Update (Section 4.12):** a full-field alternative was also evaluated and likewise did not show rotation consistency under the same calibrated thresholds, though its error magnitude differs from the center-pixel measurement at every angle.",
    "Limitations item 8 updated with the Task 3 full-field finding; original sentence kept.")

rep("L03_limitation10_single_run",
    "10. **Single run.** The full Model 3 training was performed once (seed 42); reload was bit-identical, but repeat-run reproducibility was not assessed and no deterministic-algorithm setting was enabled.",
    "10. **Single run.** The full Model 3 training was performed once (seed 42); reload was bit-identical, but repeat-run reproducibility was not assessed and no deterministic-algorithm setting was enabled. **Update (Section 4.10):** four additional independent seeds were subsequently trained under the identical protocol; the negative result reproduced in all five seeds, but exact numeric repeat-run reproducibility of the seed-42 run itself was still not separately assessed.",
    "Limitations item 10 updated with the Task 1 multi-seed finding; original sentence kept.")

rep("L04_limitation11_no_multiseed",
    "11. **No multi-seed experiment.** Ablation and multi-scale results describe the fixed single-seed configuration and carry no seed-to-seed variability.",
    "11. **No multi-seed experiment for the primary criteria was available at the time of the original evaluation.** A five-seed robustness study was subsequently run for the primary criteria (Section 4.10); the ablation and multi-scale results themselves still describe the fixed single-seed configuration and still carry no seed-to-seed variability of their own.",
    "Limitations item 11 updated: the primary-criteria gap is now addressed; the ablation/multi-scale gap remains and is stated as still open.")

# 7. Future directions: mark the now-addressed items
rep("L05_future_directions",
    "Any follow-up would be a new, separately preregistered study, and none of the following is proposed as a way of converting the present result into a positive one. Directions include: patient-level splitting; multi-seed training with reported variability; alternative or refined anatomical reference definitions; explicit learnability controls; rotation evaluation that samples at the rotation's fixed point or compares whole transformed fields; uncertainty-aware Q prediction; broader tissue-orientation references; and datasets with genuine gland-instance annotations in other organs.",
    "Any follow-up would be a new, separately preregistered study, and none of the following is proposed as a way of converting the present result into a positive one. Two of the originally listed directions were addressed as sensitivity analyses in Section 4 (patient-level splitting, Section 4.11; multi-seed training with reported variability, Section 4.10) and are removed from this forward-looking list accordingly; they did not change the primary conclusion. Remaining directions include: alternative or refined anatomical reference definitions; explicit learnability controls; rotation evaluation that compares whole transformed fields under a purpose-calibrated threshold (a full-field diagnostic was evaluated in Section 4.12 under thresholds calibrated for a different summarization, which is not the same as a purpose-built full-field criterion); uncertainty-aware Q prediction; broader tissue-orientation references; and datasets with genuine gland-instance annotations in other organs.",
    "Future directions list updated: two items are now addressed (Task 1, Task 2) and are moved out of the forward-looking list with a pointer to their results; the rotation-evaluation item is refined given the Task 3 finding, not removed, since a purpose-calibrated full-field criterion was still not built.")

# 8. Discussion 5.2: add one paragraph on the multi-seed/patient/rotation evidence
insert_after("D01_discussion_robustness_paragraph",
             "None of these was tested as a cause. We deliberately did not run rescue analyses (for example, re-thresholding, subset selection or additional training) after seeing the results.",
             f"\n\nA later stage of this project (reported in Sections 4.10-4.12) tested three specific alternative explanations directly: that the single-seed result was an unrepresentative initialization (tested with four additional seeds; the result reproduced in all five), that patient overlap in the canonical split explained the negative finding (tested with a patient-disjoint partition of the same images; the negative finding reproduced there too), and that the rotation-consistency measurement design (center-pixel sampling, small-S normalization) rather than the network was responsible for Criterion D's outcome (tested with four alternative metrics; none showed rotation consistency, though the reported error magnitude is sensitive to the summarization and normalization used). None of these three tests converted the negative result into a positive one; each narrows, rather than resolves, the space of explanations further.",
             "Discussion update: reports what the Stage 27 robustness tests did and did not show, without overstating (still 'narrows, does not resolve').")

# 9. Conclusions: add one sentence noting the robustness confirmation
rep("C01_conclusions_robustness",
    "The result defines a clear empirical boundary for the current formulation and provides a reproducible basis for future investigation.",
    "A subsequent five-seed robustness check, a patient-disjoint sensitivity analysis, and a five-metric redesign of the rotation-consistency evaluation each reproduced the same negative outcome (Sections 4.10-4.12), narrowing but not resolving the space of possible explanations. The result defines a clear empirical boundary for the current formulation and provides a reproducible basis for future investigation.",
    "Conclusions updated with a one-sentence, non-overstated summary of the Stage 27 robustness findings.")

phase1r_appendix_path = os.path.join(ROOT, "results_v2", "stage27_robustness", "phase1r_appendix", "STAGE27_PHASE1R_APPENDIX.md")
assert os.path.isfile(phase1r_appendix_path)

# 10. Supplementary Materials statement (administrative block) pointing to the standalone Supplementary PDF.
# The Phase 1R appendix is NOT inlined into the main manuscript body: its tables are plain markdown tables
# without the "**Table N. caption**" convention that tools/md_to_tex.py's parser requires, and forcing that
# reformatting would mean inventing table numbering for content that is better kept as its own document,
# consistent with the brief's own three separate deliverables (manuscript PDF, and a separate Supplementary PDF).
rep("SUPP01_supplementary_statement",
    "**Conflicts of Interest:** The authors declare no conflicts of interest.",
    "**Supplementary Materials:** The following supporting information is provided as a separate document: Synthetic Orientation-Tensor Validation (Phase 1R) -- a recovery of the project's existing synthetic-benchmark evidence, demonstrating that the framework can be learned under controlled, noise-free geometry; it does not validate the histopathology result reported in this manuscript (see the supplementary document's own interpretation section; source: `results_v2/stage27_robustness/phase1r_appendix/STAGE27_PHASE1R_APPENDIX.md`).\n\n**Conflicts of Interest:** The authors declare no conflicts of interest.",
    "Task 4: adds an MDPI-style Supplementary Materials pointer sentence to the administrative block instead of inlining the Phase 1R appendix's own tables into the main manuscript (which would break the table-caption convention the LaTeX parser requires); the Phase 1R content itself is compiled as a separate Supplementary PDF.")

# ==================================================================== write output
out = s.replace("\n", nl)
open(DST, "w", encoding="utf-8", newline="").write(out)
new_md5 = md5(out.encode("utf-8"))
json.dump({"source": "results_v2/manuscript_stage26/01_FULL_MANUSCRIPT_STAGE26.md", "source_md5": src_md5,
           "destination": "results_v2/manuscript_stage27/01_FULL_MANUSCRIPT_STAGE27.md", "destination_md5": new_md5,
           "edits": EDITS}, open(PLOG, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("source md5", src_md5, "-> stage27 md5", new_md5, "edits", len(EDITS))
for e in EDITS:
    print(" -", e["id"])
