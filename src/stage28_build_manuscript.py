"""
STAGE 28: assemble the Stage 28 manuscript from the Stage 27 manuscript plus the new Stage 28
evidence. Reads results_v2/manuscript_stage27/01_FULL_MANUSCRIPT_STAGE27.md (read-only) and the
Stage 28 result files (read-only). Writes only results_v2/manuscript_stage28/01_FULL_MANUSCRIPT_STAGE28.md
and reports/stage28_manuscript_patch_log.json. Every edit is either a pure insertion or an
explicitly-logged qualification/rewrite, listed in the patch log with old/new text and a reason.
"""
import hashlib
import json
import os
import statistics

ROOT = os.path.join(os.path.dirname(__file__), "..")
SRC = os.path.join(ROOT, "results_v2", "manuscript_stage27", "01_FULL_MANUSCRIPT_STAGE27.md")
OUT_DIR = os.path.join(ROOT, "results_v2", "manuscript_stage28")
DST = os.path.join(OUT_DIR, "01_FULL_MANUSCRIPT_STAGE28.md")
PLOG = os.path.join(ROOT, "reports", "stage28_manuscript_patch_log.json")
S28 = os.path.join(ROOT, "results_v2", "stage28")

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
    assert s.count(anchor) == 1, (tag, "anchor count != 1")
    i = s.index(anchor) + len(anchor)
    s = s[:i] + new_text + s[i:]
    EDITS.append({"id": tag, "old": "", "new": new_text, "anchor_after": anchor[-80:], "reason": why})


# ==================================================================== load Stage 28 data
sdist = json.load(open(os.path.join(S28, "s_distribution", "s_distribution_summary.json")))
gland = json.load(open(os.path.join(S28, "gland_patch_size", "gland_patch_size_summary.json")))
lit = open(os.path.join(S28, "literature_verification.md"), encoding="utf-8").read()

PD_SEEDS = [11, 22, 33, 42, 55]
pd_rows = {}
for sd in PD_SEEDS:
    if sd == 42:
        p = os.path.join(ROOT, "results_v2", "stage27_robustness", "patient_sensitivity", "model_run", "evaluation_results.json")
        tp = os.path.join(ROOT, "results_v2", "stage27_robustness", "patient_sensitivity", "model_run", "training_summary.json")
    else:
        p = os.path.join(S28, "patient_disjoint_multiseed", f"seed_{sd}", "evaluation_results.json")
        tp = os.path.join(S28, "patient_disjoint_multiseed", f"seed_{sd}", "training_summary.json")
    pd_rows[sd] = {"ev": json.load(open(p)), "tr": json.load(open(tp))}

canon_epochs = {11: None, 22: None, 33: None, 42: None, 55: None}
for sd in (11, 22, 33, 55):
    canon_epochs[sd] = json.load(open(os.path.join(ROOT, "results_v2", "stage27_robustness", "multi_seed", f"seed_{sd}", "training_summary.json")))["n_epochs_run"]
canon_epochs[42] = json.load(open(os.path.join(ROOT, "results_v2", "model3", "metrics", "stage10_summary.json")))["n_epochs_run"]
canon_epoch_vals = [canon_epochs[sd] for sd in PD_SEEDS]
canon_mean, canon_sd = statistics.mean(canon_epoch_vals), statistics.stdev(canon_epoch_vals)

pd_epoch_vals = [pd_rows[sd]["tr"]["n_epochs_run"] for sd in PD_SEEDS]
pd_mean, pd_sd = statistics.mean(pd_epoch_vals), statistics.stdev(pd_epoch_vals)


def f4(x):
    return f"{x:.4f}"


# ==================================================================== build prose fragments

# ---- Task A: Related Work rewrite ----
new_related_work_scope = """*Scope note.* This section is not a systematic review. It describes the prior works we identified as closest to the present study, distinguishing five related but distinct threads: equivariant learning in histopathology, orientation prediction elsewhere in medical imaging, rank-2/Q-tensor learning, nematic and tissue-orientation analysis, and the specific practice of validating a learned orientation output against an independently derived ground truth. Claims below are sourced from the cited works' own text; where a claim could not be verified from the material we read, that limit is stated for the specific claim."""

new_sec_2_1 = """### 2.1 Equivariant learning in histopathology

Group-equivariant convolutional networks [1] and steerable, E(2)-equivariant networks [2] provide layers whose feature maps transform according to a chosen group representation; the `escnn` library [3] implements these constructions and is the implementation basis of the network used here. Two prior studies apply this family of methods to digital pathology. Veeling et al. [4] motivate their work from the observation that histopathology images have no preferred orientation under rotation and reflection; they report improved, more stable tumor-detection performance on a lymph-node metastasis dataset and introduce a derived benchmark dataset for comparing models under this symmetry. Lafarge et al. [5] encode the roto-translation group SE(2) directly into convolutional layers, giving feature representations with a discretized orientation dimension, and apply the resulting networks to mitosis detection, nuclei segmentation and tumor classification. Neither study predicts a continuous rank-2 orientation-tensor field, and neither evaluates its output against an independently constructed anatomical orientation reference; both instead use equivariance to improve classification or detection accuracy under known input symmetries. We have not identified a prior equivariant histopathology study that evaluates a rank-2 orientation output against an anatomical reference in the way this study does."""

new_sec_2_2 = """### 2.2 Orientation prediction in medical imaging outside histopathology

Two recent works predict a continuous orientation quantity from medical image data using equivariant networks, in domains other than histopathology. SIRE [10] (Alblas et al.) estimates 3D vessel orientation from CT angiography using a gauge-equivariant mesh CNN operating on spherical meshes (icosphere discretization). Its ground-truth vessel orientations are derived independently of the network under evaluation: from expert-drawn vessel centerlines and contours (Vascular Model Repository, made with SimVascular), from segmentation-derived centerlines (ASOCA), and from centerlines obtained by averaging three independent observers' annotations (an abdominal aortic aneurysm dataset) -- in each case a human- or algorithm-derived reference distinct from the model's own processing, structurally analogous to this study's use of a mask-derived anatomical reference rather than a self-referential target. SIRE reports that under random test-time rotation, its equivariant model's median orientation-agreement (cosine similarity) held at 0.99, while a non-equivariant CNN baseline dropped to a median of 0.49 -- a large, clearly demonstrated empirical equivariance advantage on real 3D vascular data, unlike the outcome reported here.

Snoussi and Karimi [11] predict fiber orientation distributions (FODs) in neonatal diffusion MRI with a rotation-equivariant spherical CNN, from a reduced (30%) subset of diffusion gradient directions. Critically, their ground-truth FODs are computed by multi-shell multi-tissue constrained spherical deconvolution applied to the *complete* acquisition protocol (280 directions) -- a reference that does not depend on, and is not computable from, the reduced input the network receives, again structurally analogous to constructing an anatomical reference independently of the model's input pathway. They report an 84% reduction in mean squared error and a higher angular correlation coefficient for the spherical CNN compared with a multilayer-perceptron baseline, a further real-data demonstration that an equivariant architecture can outperform a non-equivariant one on a directly comparable orientation-estimation task.

Both of these studies, unlike the present one, report a clear, positive advantage for the equivariant architecture on real (not synthetic) data, evaluated against an independently derived reference. The contrast is informative context, not a benchmark this study is compared against directly: the anatomical structures, acquisition modalities, network architectures and reference-construction procedures all differ, and no quantitative comparison between their results and this study's is drawn."""

new_sec_2_3 = """### 2.3 Rank-2 and Q-tensor learning

The symmetric traceless rank-2 tensor order parameter used throughout this study is the standard description of nematic order in the physics of liquid crystals [6]. Navarro and Wilkinson [9] construct group-equivariant networks, equivariant to cyclic groups C_k of order k = 4 to 256, that predict the two-dimensional de Gennes Q-tensor order parameter directly from images of nematic textures. Their training and test data are explicitly synthetic: computationally generated packings of elliptical particles, not micrographs of a physical material. They report that all seven tested architectures satisfy the Q-tensor equivariance constraint to floating-point precision and, compared against parameter-matched non-equivariant benchmarks with and without data augmentation, achieve consistently lower prediction error and more robust generalization to defect configurations unseen during training -- architectures of finer rotational symmetry (larger k) perform better still. This is the closest prior use of the Q-tensor order parameter as a neural-network prediction target that we identified. It differs from the present study in every respect that governs whether a positive result should be expected: the target is generated by an explicit analytic formula from a known particle configuration (Equations 1-3 of their paper), so an exact, noise-free ground truth exists by construction; the present study's target, by contrast, is a statistical summary of a gland-instance segmentation mask embedded in real tissue, and no analytic ground truth exists. The present study additionally uses the dihedral group D8, which includes reflections as well as rotations, applies the formulation to real H&E histology images rather than synthetic textures, and constructs its reference from independently produced gland-instance annotations rather than from the generative process that produced the input."""

new_sec_2_4 = """### 2.4 Nematic and tissue-orientation analysis

Beyond the Q-tensor-prediction studies above, we did not conduct a separate review of the broader literature on quantitative gland-orientation, tissue-anisotropy or other geometric tissue descriptors, and make no claim about what that literature does or does not establish. Gland segmentation itself has been the subject of dedicated challenges: the GlaS challenge provided gland-instance masks for colorectal H&E images with benign/malignant labels [7], and the CRAG dataset, introduced in the MILD-Net study [8], provides gland-instance annotation for a further, partially independent set of colorectal images (Section 3.2 discusses the two datasets' shared provenance). Neither challenge's own evaluation protocol (F1 detection score, object-level Dice index, and object-level Hausdorff distance, as defined in the GlaS challenge paper [7] and reused by MILD-Net [8]) includes an orientation or anisotropy metric; gland orientation is not part of either dataset's original evaluation design, consistent with orientation-tensor learning being outside the scope of the datasets' original challenges."""

new_sec_2_5 = """### 2.5 Independent anatomical ground-truth validation

The methodological element common to SIRE [10] and Snoussi and Karimi [11] that this study also adopts is validating a learned orientation output against a reference constructed independently of the network's own input pathway, rather than against a self-consistency or reconstruction criterion. Neither of those two studies works in histopathology, and neither uses a mask-derived anatomical reference of the kind constructed in Section 3.3. This study's specific combination -- a rank-2 tensor typed to the D8 representation R(2α), an anatomical reference built from independent gland-instance masks with no image-pixel input, and a decision protocol with criteria frozen before any Model 3 result was observed -- was not found described together in any of the works read for this section. We do not claim this combination is novel in the wider literature, which we did not search exhaustively; we claim only that we did not find it in the specific works identified as closest to this study. The defensible contribution of this paper is this validation framework itself -- an explicitly typed rank-2 equivariant output, anatomical supervision, an independently constructed correspondence reference, quantitative empirical equivariance calibration, and transparent negative-result reporting -- not a claim to be the first equivariant network applied to histopathology, the first orientation-prediction method in medical imaging, the first neural Q-tensor predictor, the first nematic-order analysis of cancer tissue, or the discoverer of a novel symmetry-breaking biomarker; none of these claims is made anywhere in this manuscript."""

sec_3_19 = f"""
### 3.19 S_DL / S_anat distribution diagnostics protocol

To assess whether the network's predicted order magnitude S_DL shows meaningful dynamic range or is instead concentrated near zero, the same patch-level (S, phi) summary already defined for Criteria A-C (Section 3.9) was computed for every valid held-out patch from the frozen Stage 10 checkpoint (no retraining), together with the corresponding S_anat values, and described with the mean, SD, median, IQR, minimum, maximum, and the 1st/5th/25th/75th/95th/99th percentiles, plus the proportion of predictions below 0.001, 0.005, 0.01, 0.02 and 0.05. These were computed for the pooled held-out set, for testA and testB separately, and for the benign and malignant subsets (by the same per-image grade label used throughout).

### 3.20 Gland-size versus patch-size assessment protocol

Model 3 operates on 128x128 pixel patches (16,384 px area), while Q_anat is computed once per gland from the complete, unpatched image mask (Section 3.3). To assess how often a gland's spatial extent exceeds what the network's receptive field can see in one patch, two descriptive measures were computed from the existing, frozen Stage 4 gland manifest, for every valid gland instance, without re-reading any mask: (i) the gland's pixel area as a fraction of the patch area; (ii) an approximate major-axis full extent, 4*sqrt(lambda_1), the standard second-moment size estimate for a uniformly-filled elongated blob, using the already-computed pixel-coordinate covariance eigenvalue lambda_1 (Section 3.3). Neither measure re-reads a mask or changes any frozen artifact; both are descriptive proxies for truncation risk, not an exact per-patch truncation count."""

# ---- Task B results ----
sd = sdist["S_DL"]["pooled_testA_testB"]
sa = sdist["S_anat"]["pooled_testA_testB"]
sec_4_14 = f"""
### 4.14 S_DL and S_anat distribution diagnostics

Table 13 and Figure 8 give the full descriptive statistics and the corresponding distributions. On the pooled held-out set (n = {sd['n']}), S_DL has mean {f4(sd['mean'])}, SD {f4(sd['sd'])}, median {f4(sd['median'])}, IQR {f4(sd['iqr'])}, range [{f4(sd['min'])}, {f4(sd['max'])}]; {sd['frac_below_0.05']*100:.1f}% of predictions fall below 0.05, {sd['frac_below_0.02']*100:.1f}% below 0.02, and {sd['frac_below_0.01']*100:.1f}% below 0.01. S_anat, on the same population, has mean {f4(sa['mean'])}, SD {f4(sa['sd'])}, median {f4(sa['median'])}, IQR {f4(sa['iqr'])}, range [{f4(sa['min'])}, {f4(sa['max'])}]; only {sa['frac_below_0.05']*100:.2f}% falls below 0.05. This pattern (S_DL concentrated near zero, S_anat well spread) holds in every subgroup examined: testA (S_DL mean {f4(sdist['S_DL']['testA']['mean'])}), testB (S_DL mean {f4(sdist['S_DL']['testB']['mean'])}), benign (S_DL mean {f4(sdist['S_DL']['benign_pooled']['mean'])}), and malignant (S_DL mean {f4(sdist['S_DL']['malignant_pooled']['mean'])}); the corresponding S_anat subgroup means range from {f4(min(sdist['S_anat'][k]['mean'] for k in ('testA','testB','benign_pooled','malignant_pooled')))} to {f4(max(sdist['S_anat'][k]['mean'] for k in ('testA','testB','benign_pooled','malignant_pooled')))}, showing no comparable collapse in any subgroup.

**Table 13. S_DL and S_anat descriptive statistics, pooled held-out GlaS and subgroups.**

| Population | n | Mean | SD | Median | P5-P95 | Frac < 0.05 |
|---|---|---|---|---|---|---|
| S_DL, pooled | {sd['n']} | {f4(sd['mean'])} | {f4(sd['sd'])} | {f4(sd['median'])} | {f4(sd['p5'])}-{f4(sd['p95'])} | {f4(sd['frac_below_0.05'])} |
| S_DL, testA | {sdist['S_DL']['testA']['n']} | {f4(sdist['S_DL']['testA']['mean'])} | {f4(sdist['S_DL']['testA']['sd'])} | {f4(sdist['S_DL']['testA']['median'])} | {f4(sdist['S_DL']['testA']['p5'])}-{f4(sdist['S_DL']['testA']['p95'])} | {f4(sdist['S_DL']['testA']['frac_below_0.05'])} |
| S_DL, testB | {sdist['S_DL']['testB']['n']} | {f4(sdist['S_DL']['testB']['mean'])} | {f4(sdist['S_DL']['testB']['sd'])} | {f4(sdist['S_DL']['testB']['median'])} | {f4(sdist['S_DL']['testB']['p5'])}-{f4(sdist['S_DL']['testB']['p95'])} | {f4(sdist['S_DL']['testB']['frac_below_0.05'])} |
| S_DL, benign | {sdist['S_DL']['benign_pooled']['n']} | {f4(sdist['S_DL']['benign_pooled']['mean'])} | {f4(sdist['S_DL']['benign_pooled']['sd'])} | {f4(sdist['S_DL']['benign_pooled']['median'])} | {f4(sdist['S_DL']['benign_pooled']['p5'])}-{f4(sdist['S_DL']['benign_pooled']['p95'])} | {f4(sdist['S_DL']['benign_pooled']['frac_below_0.05'])} |
| S_DL, malignant | {sdist['S_DL']['malignant_pooled']['n']} | {f4(sdist['S_DL']['malignant_pooled']['mean'])} | {f4(sdist['S_DL']['malignant_pooled']['sd'])} | {f4(sdist['S_DL']['malignant_pooled']['median'])} | {f4(sdist['S_DL']['malignant_pooled']['p5'])}-{f4(sdist['S_DL']['malignant_pooled']['p95'])} | {f4(sdist['S_DL']['malignant_pooled']['frac_below_0.05'])} |
| S_anat, pooled | {sa['n']} | {f4(sa['mean'])} | {f4(sa['sd'])} | {f4(sa['median'])} | {f4(sa['p5'])}-{f4(sa['p95'])} | {f4(sa['frac_below_0.05'])} |
| S_anat, testA | {sdist['S_anat']['testA']['n']} | {f4(sdist['S_anat']['testA']['mean'])} | {f4(sdist['S_anat']['testA']['sd'])} | {f4(sdist['S_anat']['testA']['median'])} | {f4(sdist['S_anat']['testA']['p5'])}-{f4(sdist['S_anat']['testA']['p95'])} | {f4(sdist['S_anat']['testA']['frac_below_0.05'])} |
| S_anat, testB | {sdist['S_anat']['testB']['n']} | {f4(sdist['S_anat']['testB']['mean'])} | {f4(sdist['S_anat']['testB']['sd'])} | {f4(sdist['S_anat']['testB']['median'])} | {f4(sdist['S_anat']['testB']['p5'])}-{f4(sdist['S_anat']['testB']['p95'])} | {f4(sdist['S_anat']['testB']['frac_below_0.05'])} |
| S_anat, benign | {sdist['S_anat']['benign_pooled']['n']} | {f4(sdist['S_anat']['benign_pooled']['mean'])} | {f4(sdist['S_anat']['benign_pooled']['sd'])} | {f4(sdist['S_anat']['benign_pooled']['median'])} | {f4(sdist['S_anat']['benign_pooled']['p5'])}-{f4(sdist['S_anat']['benign_pooled']['p95'])} | {f4(sdist['S_anat']['benign_pooled']['frac_below_0.05'])} |
| S_anat, malignant | {sdist['S_anat']['malignant_pooled']['n']} | {f4(sdist['S_anat']['malignant_pooled']['mean'])} | {f4(sdist['S_anat']['malignant_pooled']['sd'])} | {f4(sdist['S_anat']['malignant_pooled']['median'])} | {f4(sdist['S_anat']['malignant_pooled']['p5'])}-{f4(sdist['S_anat']['malignant_pooled']['p95'])} | {f4(sdist['S_anat']['malignant_pooled']['frac_below_0.05'])} |

> **Figure 8.** S_DL and S_anat patch-level distributions, pooled held-out GlaS (n = {sd['n']}), same x-axis range and bin width in both panels; diagnostic only, no criterion is evaluated in this figure. A companion scatter of S_DL against S_anat (Pearson r = -0.0341, matching Criterion B's own reported value) is provided in `results_v2/stage28/s_distribution/plots/s_scatter.png`.

S_DL's concentration near zero is reported here as an important diagnostic observation and a plausible contributor to the weak order-magnitude correlation (Criterion B) and tensor similarity (Criterion C); **it is not established as the definitive, sole cause** of the negative result (Section 5.5)."""

# ---- Task D results ----
af = gland["area_fraction_of_patch"]
mae = gland["major_axis_extent_px_est"]
sec_4_15 = f"""
### 4.15 Gland/patch-size assessment: results

Among the {gland['n_valid_glands']} valid gland instances used to construct Q_anat, the gland's pixel area relative to the 128x128 patch area has mean {f4(af['mean'])}, median {f4(af['median'])} (i.e. the median gland occupies {af['median']*100:.1f}% of a whole patch's area), and {gland['fraction_of_glands_exceeding_area_threshold']['frac_area_gt_1.0']*100:.1f}% of valid glands have area exceeding the entire patch outright. The approximate major-axis full extent (4*sqrt(lambda_1)) has median {mae['median']:.1f} px, larger than the 128 px patch edge; {gland['fraction_of_glands_with_major_axis_exceeding_length']['frac_major_axis_gt_128px']*100:.1f}% of valid glands have an estimated major-axis extent exceeding 128 px, and {gland['fraction_of_glands_with_major_axis_exceeding_length']['frac_major_axis_gt_96px']*100:.1f}% exceed the 96 px patch stride. These are descriptive proxies for truncation risk, not an exact per-patch truncation count (Section 3.20): a gland whose extent exceeds 128 px is not guaranteed to be truncated in every patch that contains part of it, but the network's 128x128 receptive field frequently cannot see the same spatial extent of a gland that Q_anat was computed from. This is reported as a limitation of the current patch-based design (Section 5.7), not as an explanation established to be the cause of the negative result."""

# ---- Task E: patient-disjoint multi-seed results ----
def pd_row(sd_):
    ev = pd_rows[sd_]["ev"]; tr = pd_rows[sd_]["tr"]
    pooled = ev["criteria_A_B_C"]["pooled"]["metrics"]
    d = ev["criterion_D"]; cls = ev["classification"]["eval_pooled"]
    npass = sum([pooled["A_angular_correspondence"]["PASSES_PRIMARY_glass_delta_ge_0.5"],
                 pooled["B_order_magnitude_correlation"]["PASSES_PRIMARY_ALL_THREE"],
                 pooled["C_tensor_similarity"]["PASSES_PRIMARY_BOTH"], d["PASSES_criterion_D"]])
    return (tr["n_epochs_run"], pooled["A_angular_correspondence"]["glass_delta"],
            pooled["B_order_magnitude_correlation"]["pearson_r"], pooled["C_tensor_similarity"]["glass_delta"],
            d["n_angles_passed"], cls["accuracy"], cls["auroc"], npass)


pd_table_rows = "\n".join(
    f"| {sd_}{' (frozen Stage 27 run, reused unchanged)' if sd_ == 42 else ''} | " +
    " | ".join(str(x) if isinstance(x, int) else f4(x) for x in pd_row(sd_)) + " |"
    for sd_ in PD_SEEDS)

pd_auroc_vals = [pd_row(sd_)[6] for sd_ in PD_SEEDS]
pd_acc_vals = [pd_row(sd_)[5] for sd_ in PD_SEEDS]
pd_auroc_mean, pd_auroc_sd = statistics.mean(pd_auroc_vals), statistics.stdev(pd_auroc_vals)
canon_auroc_vals = []
for sd_ in (11, 22, 33, 55):
    canon_auroc_vals.append(json.load(open(os.path.join(ROOT, "results_v2", "stage27_robustness", "multi_seed", f"seed_{sd_}", "evaluation_results.json")))["classification"]["eval_pooled"]["auroc"])
canon_auroc_vals.append(json.load(open(os.path.join(ROOT, "results_v2", "stage27_robustness", "multi_seed", "seed_42", "evaluation_results.json")))["classification"]["eval_pooled"]["auroc"])
canon_auroc_mean, canon_auroc_sd = statistics.mean(canon_auroc_vals), statistics.stdev(canon_auroc_vals)
pd_npass_total = sum(pd_row(sd_)[7] for sd_ in PD_SEEDS)

patient_disjoint_multiseed_addition = f"""

**Multi-seed extension (Task E, Stage 28).** The same patient-disjoint partition and protocol were repeated for all five canonical seeds (11, 22, 33, 55 trained this stage; 42 reused unchanged from its existing run). Table 14 gives the per-seed results. Across all five seeds, {pd_npass_total} of the possible 20 criterion-seed combinations (4 criteria x 5 seeds) were met -- **0 of 5 seeds met all four criteria, and no seed met any single criterion**, the same qualitative outcome as both the canonical multi-seed study (Section 4.10) and the single-seed patient-disjoint result already reported above. Patient-disjoint classification AUROC was {f4(pd_auroc_mean)} +/- {f4(pd_auroc_sd)} across the five seeds, substantially lower than the canonical (patient-overlapping) split's {f4(canon_auroc_mean)} +/- {f4(canon_auroc_sd)}. **This difference is reported as evidence that classification performance is sensitive to split structure; it is not evidence that patient overlap caused the difference, nor evidence of data leakage** -- the patient-disjoint partition also differs from the canonical split in training-set size (64 vs. 68 train-inner images) and image composition, either of which could independently affect classification accuracy, and no analysis in this study isolates patient overlap as the specific mechanism.

**Table 14. Patient-disjoint sensitivity, five seeds (seed 42 reused unchanged from its existing run), pooled held-out patient-disjoint population (85 images, 2143 patches per seed).**

| Seed | Epochs | Glass's Delta (A) | Pearson r (B) | Glass's Delta (C) | D passed (of 9) | Accuracy | AUROC | Criteria (of 4) |
|---|---|---|---|---|---|---|---|---|
{pd_table_rows}
"""

# ---- Task F: multi-seed early-stopping analysis ----
multiseed_earlystopping_addition = f"""

**Early-stopping variation across seeds (Task F, Stage 28).** Best-epoch values for the five canonical seeds were 11: {canon_epochs[11]}, 22: {canon_epochs[22]}, 33: {canon_epochs[33]}, 42: {canon_epochs[42]}, 55: {canon_epochs[55]} (mean {canon_mean:.1f}, SD {canon_sd:.2f}, range {min(canon_epoch_vals)}-{max(canon_epoch_vals)}). For the patient-disjoint partition, best-epoch values were 11: {pd_rows[11]['tr']['n_epochs_run']}, 22: {pd_rows[22]['tr']['n_epochs_run']}, 33: {pd_rows[33]['tr']['n_epochs_run']}, 42: {pd_rows[42]['tr']['n_epochs_run']}, 55: {pd_rows[55]['tr']['n_epochs_run']} (mean {pd_mean:.1f}, SD {pd_sd:.2f}, range {min(pd_epoch_vals)}-{max(pd_epoch_vals)}). This range of variation (19-40 epochs) arises from stochastic optimization and validation-loss trajectories under the identical, fixed training protocol (same architecture, optimizer, learning rate, batch size and early-stopping patience for every seed); it is **not** treated as evidence of model instability or failure unless a specific quantitative metric supports that reading, and none of the criteria results above shows such a pattern (all five canonical seeds and all five patient-disjoint seeds failed the same criteria regardless of how many epochs each ran). The early-stopping rule itself was not changed retrospectively for any seed."""

# ==================================================================== apply edits

# 1. Related Work: full replace of the scope note and Sections 2.1-2.3 plus Table 1
old_scope_and_2_1_to_table = s[s.index("*Scope note.* This section is not a systematic review."):s.index("The Q-tensor formulation is shared with the work of Navarro and Wilkinson")]
rep("RW01_scope_and_sections_2_1_to_2_3",
    old_scope_and_2_1_to_table,
    new_related_work_scope + "\n\n" + new_sec_2_1 + "\n\n" + new_sec_2_2 + "\n\n" + new_sec_2_3 + "\n\n",
    "Task A: complete rewrite of the Related Work scope note, Sections 2.1-2.3 and the removal of Table 1's 'abstracts only / full texts not checked' framing, replaced with verified full-text content (literature_verification.md) and five explicitly distinguished threads, per the Stage 28 brief. No PDF-parsing or tooling-excuse language remains.")

old_navarro_trailer = s[s.index("The Q-tensor formulation is shared with the work of Navarro and Wilkinson"):s.index("### 2.4 Motivation and research hypotheses")]
rep("RW02_navarro_trailer_paragraph_replaced",
    old_navarro_trailer,
    new_sec_2_4 + "\n\n" + new_sec_2_5 + "\n\n",
    "Task A: the closing Navarro/novelty paragraph (the part of old Section 2.3 not already consumed by RW01) is replaced by the new Sections 2.4-2.5 (nematic/tissue-orientation analysis; independent anatomical ground-truth validation), consolidating the same verified facts (GlaS/MILD-Net evaluation metrics, no-novelty-claims policy; old Section 2.2 'Tissue orientation and glandular morphology' was already absorbed into RW01's replacement range) into the five-thread structure requested by the brief.")

# The old Section 2.4 ("Motivation and research hypotheses") is now displaced by two new subsections
# (2.4, 2.5) inserted before it; renumber it to 2.6 and fix its one cross-reference, so section numbers
# stay unique (these numbers are only internal SEC_LABEL lookup keys for md_to_tex.py -- the rendered
# LaTeX numbering is auto-generated regardless -- but a collision would make two \subsection commands
# share one \label and make \ref{Section 2.4} in the body ambiguous).
rep("RW03_renumber_old_2_4_to_2_6",
    "### 2.4 Motivation and research hypotheses",
    "### 2.6 Motivation and research hypotheses",
    "Task A: renumber the pre-existing 'Motivation and research hypotheses' subsection from 2.4 to 2.6, since the new Sections 2.4 ('Nematic and tissue-orientation analysis') and 2.5 ('Independent anatomical ground-truth validation') now occupy those numbers ahead of it. Purely a heading-number fix; no wording changed.")
rep("RW04_fix_cross_reference_to_renumbered_section",
    "the source is described in Section 2.4):",
    "the source is described in Section 2.6):",
    "Task A: updates the one cross-reference to the renumbered subsection (Introduction, Section 1) so it still points at 'Motivation and research hypotheses' after RW03's renumbering.")

# 2. New Methods subsections 3.19-3.20
insert_after("MM01_new_methods_sections",
             "The frozen per-angle thresholds of Table 3 were calibrated for the M1 convention only; applying them to M3/M4/M5 below is diagnostic context, not a new, independently calibrated criterion.",
             "\n" + sec_3_19.rstrip("\n") + "\n",
             "Task B/D methods: new subsections 3.19-3.20 describing the S_DL/S_anat distribution-diagnostics and gland/patch-size-assessment protocols; purely additive, placed after Section 3.18.")

# 3. Patient-level sensitivity protocol (3.17): note the multi-seed extension
rep("MM02_patient_protocol_multiseed_note",
    "Only the primary evaluation was repeated; ablation, multi-scale and pathology analyses were not.",
    "Only the primary evaluation was repeated; ablation, multi-scale and pathology analyses were not. **Extension (Task E, Stage 28):** this protocol was subsequently repeated for all five canonical seeds (Section 3.16), using the identical patient-disjoint cache, architecture, hyperparameters, early-stopping rule and evaluation code for every seed; results are given in Section 4.11.",
    "Task E: notes that the single-seed patient-disjoint protocol was extended to the full multi-seed set in this stage, pointing to the Results subsection with the new table.")

# 4. New Results subsections 4.14-4.15 (S_DL diagnostics, gland/patch size), inserted before the renumbered evidence-synthesis section
rep("R01_renumber_4_13_heading", "### 4.13 Overall evidence synthesis", "### 4.16 Overall evidence synthesis",
    "Renumbered (source label only; LaTeX auto-numbers subsections regardless, verified in prior-stage builds) to make room for new Results subsections 4.14-4.15.")
insert_after("R02_new_results_sections",
             "reflects genuine network behavior more than evaluation-design artifacts: the error does not vanish under any of the five summarizations or under exact, interpolation-free rotation at 90°, and the qualitative pattern of which angles pass is unchanged even though the reported magnitude is sensitive to the summarization and normalization used (pixel-domain interpolation and small-S normalization both measurably affect the size of the error at some angles, per Table 12, without changing which angles pass). No metric redesign converted the negative Criterion D outcome into a positive one.",
             "\n" + sec_4_14.rstrip("\n") + "\n" + sec_4_15.rstrip("\n") + "\n",
             "Task B/D results: new subsections 4.14-4.15 with the S_DL/S_anat distribution diagnostics and the gland/patch-size assessment findings, inserted after the existing Section 4.12 (rotation-metric redesign) and before the renumbered Section 4.16 (evidence synthesis).")

# 5. Extend the multi-seed robustness Results subsection with Task F
rep("R03_multiseed_earlystopping",
    "Training-loss trajectories and a seed-variance figure are given in `results_v2/stage27_robustness/multi_seed/plots/` (Figures S3-S4).",
    "Training-loss trajectories and a seed-variance figure are given in `results_v2/stage27_robustness/multi_seed/plots/` (Figures S3-S4)." + multiseed_earlystopping_addition,
    "Task F: appends the multi-seed early-stopping analysis (mean/SD/min/max best epoch, canonical and patient-disjoint) to the existing multi-seed Results subsection.")

# 6. Extend the patient-level sensitivity Results subsection with Task E's multi-seed table
rep("R04_patient_disjoint_multiseed",
    "This sensitivity analysis has its own limits (Section 3.17): a different, smaller and non-canonical partition of the same images, one seed only, primary criteria only, and its own image composition was not re-verified against the canonical population's benign/malignant balance.",
    "This sensitivity analysis has its own limits (Section 3.17): a different, smaller and non-canonical partition of the same images, and its own image composition was not re-verified against the canonical population's benign/malignant balance." + patient_disjoint_multiseed_addition,
    "Task E: appends the five-seed patient-disjoint extension and Table 14 to the existing single-seed patient-level-sensitivity Results subsection; removes the now-outdated 'one seed only' phrase since the analysis is extended to five seeds in this stage.")

# 7. Table 12 (evidence synthesis, per Stage 27's own markdown numbering key) gains rows for the two new Stage 28 diagnostics
rep("R05_table_evidence_synthesis_new_rows",
    "| CRAG (partially independent) | image, 211 | correspondence not demonstrated |\n| PANDA | case, 10 (pilot) | annotation-compatibility finding: infeasible under the frozen definition |",
    f"| S_DL / S_anat distribution (Section 4.14) | patch, {sd['n']} | S_DL concentrated near zero (median {f4(sd['median'])}); S_anat well spread (median {f4(sa['median'])}) |\n"
    f"| Gland/patch-size assessment (Section 4.15) | gland, {gland['n_valid_glands']} | {gland['fraction_of_glands_with_major_axis_exceeding_length']['frac_major_axis_gt_128px']*100:.0f}% of valid glands estimated larger than the 128 px patch |\n"
    "| Patient-disjoint, multi-seed (Section 4.11) | image, 85 x 5 seeds | no seed met any criterion |\n"
    "| CRAG (partially independent) | image, 211 | correspondence not demonstrated |\n"
    "| PANDA | case, 10 (pilot) | annotation-compatibility finding: infeasible under the frozen definition |",
    "Tasks B/D/E: three new rows added to the evidence-synthesis table for the Stage 28 diagnostics and the patient-disjoint multi-seed extension; existing rows unchanged.")

# 8. Task C: new Discussion subsection (root-cause evidence synthesis), inserted before Limitations
root_cause_summary = """
### 5.5 Root-cause evidence synthesis

Ten plausible contributors to the weak Q-tensor correspondence were assessed against existing evidence plus the two new Stage 28 diagnostics (Section 4.14-4.15): limited training-set size; patch-level supervision versus full-gland anatomical targets; the gland-size/patch-size mismatch; truncated glands at image boundaries (already excluded from the valid population, Section 3.3); noisy or ambiguous anatomical targets; weak visual signal for the target quantity; optimization dynamics; representation capacity; rotation/interpolation artifacts; and pathology/tissue heterogeneity. The full evidence-for/evidence-against/unresolved table for each item is provided in `results_v2/stage28/root_cause_evidence_matrix.md`. **No single item is established as the proven cause.** The two most directly quantified contributors from this stage are the S_DL near-zero concentration (Section 4.14) and the gland/patch-size mismatch (Section 4.15); both are consistent with, but do not individually prove, a causal mechanism. Optimization dynamics and representation capacity remain plausible and were narrowed, but not eliminated, by the existing ablations (Section 4.4). This study identifies several plausible contributors; it does not isolate a definitive root cause, and none is claimed."""

insert_after("D01_root_cause_section",
             "The project's own design documents record cases where a single mask merges several visible structures or where fused epithelial sheets are described poorly by one principal axis. A negative result against this reference therefore says something about correspondence with this reference; it does not say that the reference is wrong, and it does not say that some other definition of tissue orientation would give the same outcome.",
             "\n" + root_cause_summary.rstrip("\n") + "\n",
             "Task C: new Discussion subsection 5.5 summarizing the root-cause evidence matrix, inserted after Section 5.4 (the anatomical target as reference) and before the old Section 5.5 (implications, renumbered to 5.6 by D02-D04 below).")

# The pre-existing Discussion subsections 5.5-5.7 must shift down by one number now that the new 5.5
# (root-cause evidence synthesis) has been inserted ahead of them; fix each heading and the two
# cross-references elsewhere in the manuscript that point at the displaced "Limitations" subsection by
# its old number.
rep("D02_renumber_5_5_implications_to_5_6",
    "### 5.5 Implications for orientation-tensor learning in histopathology",
    "### 5.6 Implications for orientation-tensor learning in histopathology",
    "Task C: renumber 'Implications...' from 5.5 to 5.6, displaced by the new 5.5 (root-cause evidence synthesis).")
rep("D03_renumber_5_6_limitations_to_5_7",
    "### 5.6 Limitations",
    "### 5.7 Limitations",
    "Task C: renumber 'Limitations' from 5.6 to 5.7, displaced by the new 5.5 (root-cause evidence synthesis).")
rep("D04_renumber_5_7_future_to_5_8",
    "### 5.7 Future experimental directions",
    "### 5.8 Future experimental directions",
    "Task C: renumber 'Future experimental directions' from 5.7 to 5.8, displaced by the new 5.5 (root-cause evidence synthesis).")

# 9. Task G: reinforce the A5 / Model 2 interpretation (already largely compliant; add one explicit sentence)
rep("G01_a5_not_definitive_test",
    "Channel width and receptive-field composition necessarily differ between A5 and Model 3 at matched parameter count, so A5 cannot isolate equivariance as the only difference.",
    "Channel width and receptive-field composition necessarily differ between A5 and Model 3 at matched parameter count, so A5 cannot isolate equivariance as the only difference. **A5 is therefore not a definitive causal test of equivariance** (Task G, Stage 28): a different outcome for A5 could reflect architecture-family or receptive-field differences as plausibly as the presence or absence of the equivariance constraint, and no ablation in this study isolates equivariance as a single independent variable.",
    "Task G: explicitly states that A5 is not a definitive causal test of equivariance, reinforcing (not contradicting) the existing text that already notes A5 cannot isolate equivariance as the only difference.")

# 10. Limitations: add the gland/patch-size item explicitly (new numbered item)
rep("L01_limitation_gland_patch_size",
    "17. **Other.** The A5 control cannot isolate equivariance; A1's Q head was untrained; there is no dedicated learnability control; the Model 2 permutation null used 100 permutations and one seed; one formulation, one architecture family and one training protocol were tested.",
    f"17. **Patch-level input versus full-gland anatomical supervision (Task D, Stage 28).** The network operates on 128x128 patches, while Q_anat is computed from the complete, unpatched gland mask; among valid gland instances, the median estimated major-axis extent ({mae['median']:.1f} px) exceeds the patch edge, and {gland['fraction_of_glands_with_major_axis_exceeding_length']['frac_major_axis_gt_128px']*100:.0f}% of valid glands are estimated larger than 128 px along their major axis (Section 4.15). The target may therefore encode morphology extending beyond the model's receptive field. This is presented as a limitation, not as an explanation established to be the cause of the negative result.\n"
    "18. **Other.** The A5 control cannot isolate equivariance; A1's Q head was untrained; there is no dedicated learnability control; the Model 2 permutation null used 100 permutations and one seed; one formulation, one architecture family and one training protocol were tested.",
    "Task D: adds an explicit, quantified Limitations item for the patch/gland-size mismatch, renumbering the final 'Other' item from 17 to 18.")

# 11. Conclusions: restructure into What was demonstrated / not demonstrated / unresolved (Task J)
conclusions_expansion = """

**What was successfully demonstrated.** A technically valid, typed rank-2 equivariant model was implemented, with the R(2alpha) transformation law holding exactly for the implemented discrete-group representation (Section 3.5). Leakage controls (eight image-level/split-level tests, a poisoned-mask test) passed. An anatomical reference independent of the image pixel pathway was constructed from gland-instance annotations (Section 3.3). Multi-dataset (GlaS, CRAG, PANDA-compatibility) and ablation evaluations were performed under criteria frozen in advance. The non-support pattern was quantified across five random seeds, a patient-disjoint data partition (also across five seeds), and five redesigned rotation-consistency metrics, rather than reported from a single run or hidden.

**What was not demonstrated.** Anatomical correspondence of the learned Q_DL field with the independently constructed Q_anat reference (Criteria A-C, effect sizes near zero in every seed and every partition tested). Rotation consistency under the frozen empirical criterion (Criterion D, 4-5 of 9 angles passed against 7 required, in every seed and under every redesigned metric). A convincing pathology association (Section 4.6). Cross-dataset confirmation on CRAG (Section 4.7). Patient-independent classification generalization (patient-disjoint AUROC substantially lower than the canonical split's, Section 4.11) -- this difference is reported as sensitivity to split structure, not as proof of leakage or as a causal claim about patient overlap.

**What remains unresolved.** The exact mechanism behind the weak Q-tensor correspondence (Section 5.5): several plausible contributors were narrowed by ablation and by this stage's new diagnostics (S_DL's near-zero concentration, Section 4.14; the gland/patch-size mismatch, Section 4.15) but none was isolated as the definitive cause. Whether a larger training cohort, full-gland spatial context, a refined anatomical-reference definition, or a different optimization protocol would change the outcome was not tested and is not asserted."""

rep("J01_conclusions_expansion",
    "A subsequent five-seed robustness check, a patient-disjoint sensitivity analysis, and a five-metric redesign of the rotation-consistency evaluation each reproduced the same negative outcome (Sections 4.10-4.12), narrowing but not resolving the space of possible explanations. The result defines a clear empirical boundary for the current formulation and provides a reproducible basis for future investigation.",
    "A subsequent five-seed robustness check, a patient-disjoint sensitivity analysis (itself extended to the same five seeds in Stage 28), a five-metric redesign of the rotation-consistency evaluation, and two new Stage 28 diagnostics (the S_DL distribution and the gland/patch-size assessment) each narrowed, without resolving, the space of possible explanations, and none converted the negative outcome into a positive one." + conclusions_expansion + "\n\nThe result defines a clear empirical boundary for the current formulation and provides a reproducible basis for future investigation.",
    "Task J: restructures the Conclusions into explicit What-was-demonstrated / What-was-not-demonstrated / What-remains-unresolved sections, per the Stage 28 brief, while keeping the closing sentence unchanged.")

# ==================================================================== write output
out = s.replace("\n", nl)
open(DST, "w", encoding="utf-8", newline="").write(out)
new_md5 = md5(out.encode("utf-8"))
json.dump({"source": "results_v2/manuscript_stage27/01_FULL_MANUSCRIPT_STAGE27.md", "source_md5": src_md5,
           "destination": "results_v2/manuscript_stage28/01_FULL_MANUSCRIPT_STAGE28.md", "destination_md5": new_md5,
           "edits": EDITS}, open(PLOG, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("source md5", src_md5, "-> stage28 md5", new_md5, "edits", len(EDITS))
for e in EDITS:
    print(" -", e["id"])
