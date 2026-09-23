"""Stage 24.5 -- manuscript repair builder (text edits only; no evidence is generated).
Reads results_v2/manuscript_stage23/01_FULL_MANUSCRIPT.md (never modified) and writes
results_v2/manuscript_stage24_5/01_FULL_MANUSCRIPT_REPAIRED.md (optional: --log PATH writes a JSON edit log).
Every replaced string is asserted to match exactly once; no number, table value or threshold is changed.
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "results_v2", "manuscript_stage23", "01_FULL_MANUSCRIPT.md")
OUTDIR = os.path.join(ROOT, "results_v2", "manuscript_stage24_5")
os.makedirs(OUTDIR, exist_ok=True)

T = open(SRC, encoding="utf-8").read().replace("\r\n", "\n")
LOG = []


def rep(old, new, issue, desc, count=1):
    global T
    n = T.count(old)
    assert n == count, f"[{issue}] expected {count} occurrence(s), found {n}: {old[:90]!r}"
    T = T.replace(old, new)
    LOG.append({"issue": issue, "description": desc, "old": old, "new": new})


def rep_re(pattern, new, issue, desc, flags=re.M | re.S, count=1):
    global T
    ms = list(re.finditer(pattern, T, flags))
    assert len(ms) == count, f"[{issue}] regex expected {count}, found {len(ms)}: {pattern[:80]!r}"
    old = ms[0].group(0)
    T = re.sub(pattern, lambda m: new, T, flags=flags)
    LOG.append({"issue": issue, "description": desc, "old": old, "new": new})


def rep_between(start, end, new, issue, desc, keep_end=True):
    """Replace text from `start` (inclusive) up to `end` (exclusive if keep_end)."""
    global T
    i = T.index(start)
    j = T.index(end, i + len(start))
    old = T[i:j]
    T = T[:i] + new + (T[j:] if keep_end else T[j + len(end):])
    LOG.append({"issue": issue, "description": desc, "old": old, "new": new})


# ------------------------------------------------------------------ header (M08, H06)
rep("**Authors:** [AUTHOR NAMES AND AFFILIATIONS — TO BE PROVIDED BY THE AUTHORS]",
    "**Authors:** [USER TO COMPLETE — author names, affiliations and ORCID iDs]",
    "S24-H06", "administrative placeholder in the standard form")
rep("**Corresponding author:** [TO BE PROVIDED]", "**Corresponding author:** [USER TO COMPLETE]",
    "S24-H06", "administrative placeholder in the standard form")
rep_re(r"^\*Draft prepared for submission to MDPI.*?\*\n\n---\n\n", "---\n\n", "S24-M08",
       "draft/status note with internal stage reference removed")

# ------------------------------------------------------------------ abstract (H04)
ABSTRACT = (
    "Glandular orientation in histopathology is defined up to a head–tail flip, which motivates a rank-2 nematic tensor "
    "output and rotation-equivariant networks. We tested whether a D8-equivariant network whose Q-tensor head is typed as "
    "`D8.irrep(1,2)` (11,674 parameters) can learn a mask-derived gland-elongation orientation reference on GlaS "
    "(85 training images). Four prospectively frozen internal criteria were evaluated on 80 held-out images "
    "(2083 overlapping patches; canonical image-level split, not patient-independent). None was met: angular "
    "correspondence Glass's Δ = 0.0412 (≥ 0.5 required); order-magnitude correlation r = −0.0341 (≥ 0.30); tensor "
    "similarity Glass's Δ = 0.0073 (≥ 0.5); empirical center-pixel, pixel-domain rotation consistency 4 of 9 angles "
    "(7 required). Single-seed ablations and six aggregation scales did not change this outcome. No pathology "
    "association was demonstrated (Holm-adjusted p = 1.0). An external cross-dataset evaluation on CRAG (partially "
    "independent of GlaS) gave no supporting evidence (pooled Δ = −0.0797, r = −0.0937), and the PANDA masks, which "
    "encode semantic Gleason-pattern regions, were incompatible with the gland-instance construction. Under the "
    "tested protocol the formulation did not demonstrate the intended learning objective; the result does not show "
    "that orientation information is absent from histology or that equivariant learning cannot address it."
)
KEYWORDS = ("**Keywords:** rotation equivariance; nematic order; rank-2 orientation tensor; Q-tensor; histopathology; "
            "gland morphology; steerable convolutional networks; negative result")
rep_between("## Abstract\n", "---\n\n## 1. Introduction",
            "## Abstract\n\n" + ABSTRACT + "\n\n" + KEYWORDS + "\n\n", "S24-H04",
            "347-word four-paragraph abstract replaced by the single-paragraph abstract; journal length rule not verified")

# ------------------------------------------------------------------ introduction (M10, L01)
rep("judged against an independently constructed reference derived from expert gland segmentations?",
    "judged against an independently constructed, mask-derived reference computed from gland-instance annotations?",
    "S24-M10", "'expert gland segmentations' had no source in the manuscript")
rep("We therefore evaluated one specific formulation under a pre-specified protocol.",
    "We therefore evaluated one specific formulation under a protocol whose evaluation criteria were frozen internally in advance.",
    "S24-M10", "terminology: prospectively frozen internal criteria")
rep('**H2 — Equivariance validity.** "When the input image is rotated by α, the predicted Q tensor should transform '
    'according to: Q_DL(R_α x) ≈ R(α) Q_DL(x) R(α)^T. The scalar order magnitude should remain invariant."',
    '**H2 — Equivariance validity.** "When the input image is rotated by α, the predicted Q tensor should transform '
    'according to: Q_DL(R_α x) ≈ R(α) Q_DL(x) R(α)^T [paragraph break in the original] The scalar order magnitude '
    'should remain invariant."',
    "S24-L01", "period inserted after '^T' removed; paragraph break of the recorded original marked instead")
rep("(Whitespace and line breaks of the original blocks are normalized here; the wording is unchanged.)",
    "(Line breaks of the original blocks are normalized here, and the one paragraph break inside H2 is marked in "
    "brackets; wording and punctuation are unchanged.)",
    "S24-L01", "note on normalisation of the quoted hypotheses")
rep("the systematic, pre-specified validation framework", "the systematic evaluation framework with prospectively frozen internal criteria",
    "S24-M10", "terminology")

# ------------------------------------------------------------------ section 2 (H02, M11)
SEC2 = """## 2. Related Work

*Scope note.* This section is not a systematic review. It describes the prior works that we identified as closest to the present study, and the descriptions of those works are limited to what their abstracts state.

### 2.1 Equivariant learning in medical imaging and histopathology

Group-equivariant convolutional networks [1] and steerable, E(2)-equivariant networks [2] provide layers whose feature maps transform according to a chosen group representation; the `escnn` library [3] implements these constructions and is the implementation basis of the network used here. Rotation-equivariant networks have been applied to digital pathology, where rotation and reflection of tissue are label-preserving [4,5]: Veeling et al. [4] report improved tumor detection on a lymph-node metastasis dataset, and Lafarge et al. [5] apply roto-translation-equivariant networks to mitosis detection, nuclei segmentation and tumor classification. We have not established whether any equivariant histopathology study has evaluated a rank-2 orientation output against an anatomical reference, and we make no claim on that point.

### 2.2 Tissue orientation and glandular morphology

Gland segmentation has been the subject of dedicated challenges. The GlaS challenge provided gland-instance masks for colorectal H&E images with benign/malignant labels [7]; the CRAG dataset, described in the MILD-Net study [8], provides gland-instance annotation for a further set of colorectal images. In this study, gland size, count and elongation are computed from such masks (Sections 3.3 and 3.12). The literature on quantitative gland-orientation or tissue-anisotropy descriptors, and on other geometric tissue descriptors, was not reviewed for this manuscript, and we make no statement about it.

### 2.3 Nematic order and rank-2 orientation tensors

The symmetric traceless tensor order parameter is the standard description of nematic order [6]. Three recent works are, in our reading of their abstracts, conceptually closest to the present study:

- Navarro and Wilkinson [9] (arXiv preprint) develop group-equivariant networks for predicting the two-dimensional Q-tensor order parameter of nematic liquid crystals, with architectures equivariant to cyclic groups C_k (k = 4, 8, …, 256), and evaluate them on synthetic microscopic textures.
- SIRE [10] (Alblas et al.; arXiv preprint) estimates artery orientations in three-dimensional medical images with SO(3)-equivariant gauge-equivariant mesh networks and approximate scale invariance, and evaluates them on three datasets.
- Snoussi and Karimi [11] (arXiv preprint) use a rotation-equivariant spherical CNN to estimate fiber orientation distributions in neonatal diffusion MRI from a reduced number of gradient directions, and compare it with a multilayer-perceptron baseline.

Table 1 lists what the abstracts of these works, and of the two histopathology studies of Section 2.1, report. Entries that an abstract does not state are marked "not reported"; the full texts were not checked.

**Table 1. Closest prior work as reported in the abstracts ("not reported" = not stated in the abstract; full texts not checked).**

| Prior work | Equivariance / symmetry | Predicted quantity or task | Data or domain | Evaluation as stated in the abstract |
|---|---|---|---|---|
| Navarro & Wilkinson [9] | cyclic groups C_k, k = 4 to 256 | two-dimensional Q-tensor order parameter | synthetic microscopic nematic textures | compared with non-equivariant benchmark models; reference type not reported |
| SIRE [10] | SO(3) rotation equivariance with approximate scale invariance | artery orientation | 3D medical images (VMR, ASOCA and AAA datasets) | evaluated on three datasets, including cross-domain generalization; reference type not reported |
| Snoussi & Karimi [11] | rotation equivariance (spherical CNN) | fiber orientation distribution | neonatal diffusion MRI (43 dHCP datasets) | compared with a multilayer-perceptron baseline (mean squared error, angular correlation coefficient); reference type not reported |
| Veeling et al. [4] | rotation equivariance | tumor detection | lymph-node metastasis histopathology | not reported |
| Lafarge et al. [5] | SE(2) roto-translation equivariance | mitosis detection, nuclei segmentation, tumor classification | histopathology | not reported |

The Q-tensor formulation is shared with the work of Navarro and Wilkinson [9], which considers cyclic groups C_k on synthetic textures; the present study uses the dihedral group D8, which contains reflections as well as rotations, applies it to histology images, uses gland-instance masks to construct the reference, and evaluates the output against criteria frozen in advance. We did not establish how the reference orientations in [10] and [11] were obtained, and we draw no distinction on that basis. We do not claim to be the first to use equivariant networks in histopathology, to predict orientation in medical imaging, to predict a Q-tensor with deep learning, or to analyze orientation in cancer, and we do not claim that the combination studied here is absent from the wider literature, which was not searched exhaustively.

### 2.4 Motivation and research hypotheses

The motivation is to test, under conditions that make a negative result interpretable, whether the typed representation together with an anatomical reference yields learnable and anatomically meaningful orientation output. The three hypotheses H1–H3 quoted in Section 1 are reproduced verbatim from the original project specification (its "Core hypotheses" section). That specification is a logged instruction rather than a versioned repository file, and it is not checksummed; only the H2 text block has a recorded hash (SHA-256 `9010f172…d5d220`, 207 characters). H2 contains two clauses: the tensor transformation law and the invariance of the scalar order magnitude. The recorded hypotheses do not explicitly assign individual criteria to individual hypotheses.

---

"""
rep_between("## 2. Related Work", "## 3. Materials and Methods", SEC2, "S24-H02/M11",
            "Section 2 rewritten to source-supported statements; Table 1 rebuilt from verified abstract content; placeholders removed")

# ------------------------------------------------------------------ 3.1 (M05, M07)
rep("The study was run as a sequence of stages, each defined and checksummed before it was executed. Every stage read its predecessors' outputs but never modified them.",
    "The study was run as a sequence of stages. Every stage read its predecessors' outputs but never modified them; relevant frozen artifacts (the criteria file, the analysis matrices, the Model 3 checkpoint and the result files) were checksum-verified during the reproducibility audit.",
    "S24-M05", "checksum language restricted to what the audit supports")

FIG = {}
FIG[1] = ("> **Figure 1.** Schematic of the framework; no data are plotted. Top row, left to right: an H&E image; the "
          "gland-instance mask; the principal orientation θ and anisotropy S of each gland, computed from the mask by "
          "principal component analysis; the reference tensor Q_anat; the D8-equivariant network (Model 3); its predicted "
          "tensor Q_DL; and the evaluation of Q_DL against Q_anat (Section 4; the final box is labelled with internal stage "
          "numbers in the graphic). The arrows show the order in which the quantities are derived, not the inputs of the "
          "network: the network receives only the H&E image, and Q_anat serves as the supervision target on the training "
          "images and as the comparison reference on the held-out images. The lower panel shows the tensor representation "
          "and its transformation law under rotation. The rotation-consistency test, the loss and the frozen criteria are "
          "not drawn.")
FIG[2] = ("> **Figure 2.** Construction of the anatomical reference for one illustrative GlaS image (`testA_1`, a held-out "
          "image shown only to illustrate the construction; no model output is shown): H&E image, gland-instance mask, "
          "principal orientation of valid glands (line length proportional to S), and the dense order-magnitude field on "
          "valid pixels. Illustration of the construction; not evidence for any criterion.")
FIG[3] = ("> **Figure 3.** Frozen effect sizes for Criteria A–C on the pooled held-out GlaS set (80 images; 2083 overlapping "
          "patches), with the frozen thresholds (dashed lines). Primary results. The vertical axis shows Glass's Δ for panels "
          "A and C and Pearson's r for panel B, each against its own threshold. The permutation p-values in the panel titles "
          "are consistent with a large sample and are not evidence of agreement; panel C's title prints the permutation p as zero, which "
          "corresponds to p < 0.001 (0 of 1000 re-pairings). The word \"preregistered\" in the graphic's title is internal "
          "wording: the criteria were prospectively frozen internal criteria and were not entered in an external registry "
          "(Section 3.9).")
FIG[4] = ("> **Figure 4.** Normalized rotation error (bars) and frozen per-angle thresholds (hatched) at the nine non-zero "
          "angles, pooled held-out GlaS set (n = 2083 patches). Primary, gating criterion; center-pixel, pixel-domain "
          "measurement (see text). Error bars are green where the error is below the threshold and red where it is not; "
          "the same status is given in Table 6.")
FIG[5] = ("> **Figure 5.** Ablation conditions (left, center) and multi-scale reductions (right), pooled held-out GlaS set "
          "(2083 patches): Glass's Δ and r with the frozen thresholds as context (dashed lines), and rotation pass counts "
          "by scale. Diagnostic; no condition or scale is ranked or preferred; one seed per ablation condition; A2 was not "
          "run. In the right panel Glass's Δ (grey circles) is read on the left axis and the number of angles below "
          "threshold (blue squares) on the right axis; the dotted horizontal line marks the required 7 of 9 angles on the "
          "right axis. Rotation thresholds were calibrated for the center-pixel convention.")
for k in (1, 2, 3, 4, 5):
    rep_re(rf"^> \*\*Figure {k}\.\*\*[^\n]*$", FIG[k], "S24-M07/M08/L02/L03" if k in (1, 4, 5) else "S24-M08",
           f"Figure {k} caption made to describe the figure; internal source path removed")

# ------------------------------------------------------------------ 3.2 (H05, M03, M09, M10)
rep("### 3.2 Datasets\n\n**Table 2.", "### 3.2 Datasets\n\nThree public datasets were used; Table 2 lists their roles.\n\n**Table 2.",
    "S24-H05", "Table 2 cited in the text before its placement")
rep("per-file SHA-256 hashes were recorded in the project's manifest.",
    "per-file SHA-256 hashes were recorded in the project's data manifest. The license and terms of use of the data as distributed by the mirror are [DATASET LICENSE TO VERIFY].",
    "S24-M09", "licence not stated unless verified")
rep("The canonical split is an image-level split. The held-out set is therefore",
    "The canonical split is an image-level split, which permits patient overlap between training and held-out images and introduces dependence that may affect apparent performance estimates. The held-out set is therefore",
    "S24-M03", "neutral patient-overlap statement")
rep("**CRAG.** CRAG (version 2, released with the MILD-Net study [8]) contains 213 colorectal images (173 in its training folder and 40 in its test folder, labelled `valid` in the archive) with 3054 gland instances. It provides no pathology labels and no patient metadata. It was obtained by the project owner from an OpenDataLab mirror; no archive-level checksum was recorded. CRAG and GlaS were acquired at the same hospital (University Hospitals Coventry and Warwickshire, Coventry) and curated by the same group (University of Warwick Tissue Image Analytics laboratory). Independence of CRAG from GlaS is therefore partial.",
    "**CRAG.** CRAG, a colorectal adenocarcinoma gland dataset described in the MILD-Net study [8], contains 213 colorectal images (173 in its training folder and 40 in its test folder, labelled `valid` in the archive) with 3054 gland instances. The archive obtained here provides no pathology labels and no patient metadata; the MILD-Net study [8] reports that the images derive from 38 whole-slide images of different patients, and describes the gland boundaries in GlaS and CRAG as annotated by an expert pathologist. The archive was obtained from an OpenDataLab mirror; no archive-level checksum was recorded, and its license and terms of use are [DATASET LICENSE TO VERIFY]. According to [8], both datasets were obtained from the University Hospitals Coventry and Warwickshire (UHCW) NHS Trust, Coventry, and both are associated with the same research group (University of Warwick, Tissue Image Analytics laboratory). Independence of CRAG from GlaS is therefore partial.",
    "S24-H02/M09", "CRAG description aligned with the primary source [8]; 'version 2' and 'project owner' removed")
rep("The data licence (CC BY-SA-NC 4.0) requires citation of the source study.",
    "The license and terms of use of the PANDA data are [DATASET LICENSE TO VERIFY].",
    "S24-M09", "unverified licence string removed")

# ------------------------------------------------------------------ 3.3 (M06, H05)
rep("and not a universal biological truth.\n",
    "and not a universal biological truth. Figure 2 illustrates the construction for one image. Patch-level summaries of the reference, as used by Criteria A–C, are defined in Section 3.9; the order magnitude of such a summary is not the per-gland anisotropy S defined above.\n",
    "S24-H03/H05", "pointer to the patch-level rule; Figure 2 cited")

# ------------------------------------------------------------------ 3.6 (M06)
rep("Patches with no valid cell contribute zero to L_Q.",
    "Patches with no valid cell contribute zero to L_Q. The training objective therefore operates on the dense field at the level of individual cells; the patch-level summaries used by Criteria A–C are computed only for evaluation (Section 3.9).",
    "S24-M06", "training unit vs evaluation unit stated")

# ------------------------------------------------------------------ 3.9 (H03, M01, M08, M10, H05)
rep("### 3.9 Experimental validation criteria", "### 3.9 Evaluation criteria", "S24-M10", "heading: 'validation' implied a positive outcome")
rep("The criteria are defined in `results_v2/phase2/preregistered_thresholds_v2.json` (MD5 `4d68f3877d2cc0ec50aac19629182740`).",
    "The criteria are defined in a frozen criteria file (version 2; MD5 `4d68f3877d2cc0ec50aac19629182740`; available in the project repository).",
    "S24-M08", "internal file path (and a file name containing 'preregistered') removed")
rep("was frozen before any Phase 2 (Model 3) training", "was frozen before any Model 3 training", "S24-M08", "undefined internal 'Phase 2' removed")
rep("\n\n**Table 3. Frozen criteria.**", "\n\nThe four frozen criteria are listed in Table 3.\n\n**Table 3. Frozen criteria.**",
    "S24-H05", "Table 3 cited before its placement")
rep("The train-only oracle calibration gave Glass's Δ = 0.152 and r = 0.0115 (95% CI −0.029 to 0.054), i.e. even a non-learned estimator showed little correspondence with the reference; these are calibration reference points, not Model 3 results.",
    "For the internal threshold calibration, a classical structure-tensor estimator was evaluated on the 2249 training patches and gave Glass's Δ = 0.152 and r = 0.0115 (95% CI −0.029 to 0.054). These calibration values were obtained on training data, were used only to derive the frozen thresholds, and are not held-out model results; they are not directly comparable with the primary Model 3 results as estimates of model performance, and no comparison between them is drawn.",
    "S24-M01", "context for the oracle calibration values")
rep("The recorded hypotheses do not explicitly assign individual criteria to individual hypotheses, and no assignment is made here.\n\n### 3.10",
    "The recorded hypotheses do not explicitly assign individual criteria to individual hypotheses, and no assignment is made here.\n\n"
    "**Patch-level summaries.** Criteria A–C are evaluated on one pair (S, φ) per patch, for the predicted field and for the reference. In the frozen implementation, the valid cells of a patch are the cells of the 64 × 64 output grid whose reference is valid (Section 3.6); the same cells are used for the prediction and for the reference. For the field in question, the spatial means q̄₁ = mean(q₁) and q̄₂ = mean(q₂) over the valid cells are computed, and then S = √(q̄₁² + q̄₂²) and φ = ½ atan2(q̄₂, q̄₁) (mod π). The patch-level order magnitude of the reference is therefore the magnitude of the mean tensor of the patch and not the per-gland anisotropy of Section 3.3: it is at most the cell-weighted mean of the per-gland order magnitudes, and it is smaller whenever the glands within a patch differ in orientation. The training objective (Section 3.6) operates on the dense field at the level of individual cells, whereas Criteria A–C evaluate these patch-level summaries; overlapping patches from the same image are not independent observations. The summarization rule is part of the frozen implementation. It was not modified, and it is not claimed to be optimal or to correspond to a biological unit of observation.\n\n### 3.10",
    "S24-H03/M06", "explicit patch-level S and phi rule added (frozen implementation, no new formula)")

# ------------------------------------------------------------------ 3.10 (M04)
rep("both computed at the same pixel.\n\n### 3.11",
    "both computed at the same pixel.\n\n"
    "The comparison is pointwise: the predicted tensor at one dense location of the rotated patch is compared with the R(2α)-rotated tensor at the same dense index of the unrotated patch. This equals the field-level statement Q_DL(R_α x) ≈ R(α) Q_DL(x) R(α)ᵀ only at the rotation's fixed point, which is why the offset noted above matters. Sign convention: the input is rotated with `skimage.transform.rotate` using a positive angle argument α (the convention established for this model in the project's earlier synthetic benchmark, and reused unchanged), coordinates are x = column and y = row, and the expected output is (q₁′, q₂′) = (q₁ cos 2α − q₂ sin 2α, q₁ sin 2α + q₂ cos 2α). The classical structure-tensor estimator used to derive the thresholds was rotated with the opposite sign of the angle argument in its own calibration; the two conventions were kept separate, and the model's convention was used for Model 3.\n\n### 3.11",
    "S24-M04", "pointwise vs field-level statement and the documented rotation sign convention")

# ------------------------------------------------------------------ 3.11 (M08, L-A7)
rep("**A0** is the reference (the frozen Stage 10 checkpoint, not retrained).", "**A0** is the reference (the frozen Model 3 checkpoint, not retrained).",
    "S24-M08", "internal stage label removed")
rep("**A7** reuses existing evidence about target learnability.",
    "**A7** reused two existing artifacts as joint, imperfect evidence about target learnability, without new training: the predefined plain CNN baseline (64-channel blocks, non-equivariant, trained jointly on classification and Q; Section 3.8) and the A4 condition (Q head alone).",
    "S24-L10", "A7 evidence named")

# ------------------------------------------------------------------ 3.14 (M08)
rep("the frozen Stage 4 construction would be applied", "the frozen gland-instance construction of Section 3.3 would be applied",
    "S24-M08", "internal stage label removed")

# ------------------------------------------------------------------ 3.15 (M05, L09)
rep("No clustered or patient-level reanalysis was performed.",
    "No clustered or patient-level reanalysis was performed. Holm–Bonferroni correction was applied only to the four primary pathology tests P1–P4; the ablation, multi-scale, morphology, per-split and CRAG analyses are descriptive or diagnostic, and no inference is drawn from their p-values.",
    "S24-L09", "multiplicity statement")
rep("Frozen artifacts (criteria file, matrices, checkpoint, result files) carry recorded checksums that were re-verified after each subsequent stage.",
    "Relevant frozen artifacts (criteria file, matrices, checkpoint, result files) were checksum-verified during the reproducibility audit.",
    "S24-M05", "checksum language restricted to what the audit supports")

# ------------------------------------------------------------------ 4 (H05, H08, M08, M10, L05, L07, L06)
rep("All values are read from frozen result files (see the numbers audit).", "All values are read from frozen result files.",
    "S24-M08", "internal audit reference removed")
rep("### 4.2 Primary geometric validation on GlaS", "### 4.2 Primary geometric evaluation on GlaS", "S24-M10", "heading wording")
rep("the effect-size results are the primary basis for interpretation.\n\n**Table 4.",
    "the effect-size results are the primary basis for interpretation. The primary results are given in Table 4 and Figure 3.\n\n**Table 4.",
    "S24-H05", "Table 4 and Figure 3 cited")
rep("it indicates that a difference of about one degree is detectable with 2083 patches, not that the model and reference agree.",
    "with 2083 overlapping patches it is consistent with the sample size, and the difference between the two mean errors is about 1.1°; it does not indicate that the model and the reference agree.",
    "S24-L05", "removed a detectability claim")
rep("Model 2 is a post-hoc, un-typed formulation evaluated under a different output construction, and the split-wise values differ in sign and size between splits for all three models.",
    "Model 2 is a post-hoc, un-typed formulation evaluated under a different output construction. The split-wise values differ in size between testA and testB for all three models; the signs of Glass's Δ (A), of r and of Glass's Δ (C) are the same in both splits for the plain CNN and for Model 2, and differ between the splits only for Model 3.",
    "S24-H08", "sign statement corrected (checked against frozen baseline values)")
rep("**Table 5. Descriptive per-split values (no frozen criteria applied; models not pooled or ranked).**",
    "**Table 5. Descriptive per-split values (no frozen criteria applied; models not pooled or ranked; rows are in a fixed order that carries no ranking).**",
    "S24-H08", "Table 5 caption: unranked")
rep("Empirical center-pixel, pixel-domain rotation consistency did not meet the frozen criterion: four of the nine non-zero angles (90°, 123°, 150°, 173°) were below their frozen thresholds, whereas seven were required (Table 6).",
    "Empirical center-pixel, pixel-domain rotation consistency did not meet the frozen criterion: four of the nine non-zero angles (90°, 123°, 150°, 173°) were below their frozen thresholds, whereas seven were required (Table 6 and Figure 4).",
    "S24-H05", "Figure 4 cited")
rep("no condition approached the frozen thresholds (Table 7).", "no condition approached the frozen thresholds (Table 7 and Figure 5).",
    "S24-H05", "Figure 5 cited")
rep("The A7 evidence on target learnability is suggestive only; no dedicated classification-free control was run.",
    "A7 relied on the existing plain CNN baseline and on A4 (Section 3.11); neither is a dedicated, classification-free target-learnability control, and no such control was run, so A7 is descriptive only.",
    "S24-L10", "A7 evidence named and limited")
rep("| Scale: full field | 0.0412 | −0.0341 | — | 6 |\n",
    "| Scale: full field | 0.0412 | −0.0341 | — | 6 |\n\nA dash means that the quantity was not reported for that row; for A0 the rotation result is given in Table 6.\n",
    "S24-L07", "dash meaning explained")
rep("every adjusted p-value was 1.0 (Table 8).", "every adjusted p-value was 1.0 (Table 8 and Figure 6a).", "S24-H05", "Figure 6 cited")
rep("so the image-level analysis contains 211 images (171 + 40).", "so the image-level analysis contains 211 images (171 + 40). Table 9 and Figure 6b give the image-level results.",
    "S24-H05", "Table 9 and Figure 6b cited")
rep("This is consistent with the grading definitions (pattern 4 is fused or cribriform glandular growth, and pattern 5 lacks gland formation) [CITATION TO BE ADDED], and applying the GlaS gland-instance construction would have required inventing boundaries the annotation does not contain.",
    "Applying the GlaS gland-instance construction would have required inventing boundaries the annotation does not contain.",
    "S24-M11", "unsourced statement about grading definitions removed")
rep("This is an annotation-compatibility finding, not a validation, not a model result and not a test of the hypotheses.\n",
    "This is an annotation-compatibility finding, not a validation, not a model result and not a test of the hypotheses. Figure 7 plots the ratios.\n",
    "S24-H05", "Figure 7 cited")
rep("The figure title refers to the 10-case pilot; three cases contain no Gleason 3, 4 or 5 pixels, and one Gleason-4 case has no benign reference and is omitted.",
    "Of the 10 pilot cases, three (00743313, 004dd32d and 01642d24) contain no Gleason 3, 4 or 5 pixels, and one Gleason-4 case has no benign reference and is omitted; the remaining six cases give the nine bars.",
    "S24-H07", "three cases without Gleason 3/4/5 pixels, named as in the frozen summary table")
rep("### 4.9 Overall evidence synthesis\n\n**Table 10.", "### 4.9 Overall evidence synthesis\n\nTable 10 summarizes the outcome of each analysis.\n\n**Table 10.",
    "S24-H05", "Table 10 cited")
rep("| PANDA | case, 10 (pilot) | annotation-compatibility finding: infeasible under the frozen definition |\n",
    "| PANDA | case, 10 (pilot) | annotation-compatibility finding: infeasible under the frozen definition |\n\nThe labels are descriptive: \"not met\" and \"not demonstrated\" both mean that the analysis did not provide supporting evidence (supplementary evidence tables, if submitted, use the label \"NOT SUPPORTED\" for these outcomes).\n",
    "S24-L06", "outcome vocabulary aligned with the supplementary evidence table")
rep_re(r" Source: `results_v2/figures/stage20/figure[67][^`]*`\.", "", "S24-M08", "internal source paths removed from captions", count=2)

# ------------------------------------------------------------------ 5 (M01, M02, M03, M06, M10, L05)
rep("First, the effect sizes are not merely below threshold but close to zero (0.0412, −0.0341, 0.0073), and they do not vary materially across the tested ablation conditions or aggregation scales. Second, the calibration showed that even a non-learned structure-tensor estimator had little correspondence with the reference on training patches (Δ = 0.152; r = 0.0115), so the reference is demanding relative to what a simple image-gradient estimator recovers. Third, the small p-values that accompany Criteria A and C are a property of the sample size and the overlapping patches; they are not evidence of agreement and were not treated as such.",
    "First, the primary effect sizes are not merely below threshold but close to zero (0.0412, −0.0341, 0.0073). Across the single-seed ablation conditions and the aggregation scales, Glass's Δ stayed between −0.0441 and 0.0461, far below the required 0.5. Correlation estimates varied across the scales (−0.0341 to 0.0147) and remained near zero overall; they varied more widely across the ablation conditions (−0.2097 to 0.2203) without any reaching the required 0.30, and the largest value belongs to A1, a condition whose Q head received no gradient and which is not read as learned correspondence (Section 4.4). Second, for the internal threshold calibration, a classical structure-tensor estimator evaluated on training patches also gave an effect size well below the requirement (Δ = 0.152; r = 0.0115); these calibration values are not held-out model results and are not directly comparable with the Model 3 results, and they are mentioned only as context for the frozen thresholds. Third, the small p-values that accompany Criteria A and C are consistent with the large number of overlapping patches; they are not evidence of agreement and were not treated as such.",
    "S24-M01/M02", "M02: 'do not vary materially' replaced by the actual ranges; M01: calibration context")
rep("against an independent reference with criteria fixed in advance,", "against an independently constructed, mask-derived reference with criteria frozen internally in advance,",
    "S24-M10", "terminology")
rep_between("### 5.6 Limitations", "### 5.7 Future experimental directions", """### 5.6 Limitations

1. **Split.** The GlaS held-out set is the canonical image-level split; it is not patient-independent.
2. **Patient overlap.** 11 of the 12 held-out patients (79 of 80 held-out images) also contribute training images. The image-level split permits patient overlap between training and held-out images, introducing dependence that may affect apparent performance estimates. The direction and size of any such effect were not analyzed, no patient-clustered analysis was performed, and the overlap is not offered as an explanation of the negative result.
3. **Patch dependence.** The 2083 evaluated patches overlap (128 px, stride 96) and are not independent biological units; patch-level p-values and intervals are descriptive.
4. **Reference.** The mask-derived reference is a reference, not absolute ground truth, and it is not independent of the training objective.
5. **Scope of the reference.** It describes gland elongation, not necessarily broader tissue orientation.
6. **Unit of evaluation.** Training operates on dense (q₁, q₂) fields at the level of individual cells, whereas Criteria A–C evaluate patch-level masked-mean summaries (Section 3.9). The summarization rule is part of the frozen implementation and was not shown to be optimal or to reflect a biological unit of observation.
7. **Rotation test.** It is a finite-grid, pixel-domain measurement with interpolation.
8. **Center pixel.** A single dense pixel, offset by about half a pixel per axis from the rotation's fixed point, was evaluated, and the comparison is pointwise rather than between complete transformed fields.
9. **Exact angles.** Only 90° of the nine tested angles is an exact pixel-lattice rotation.
10. **Single run.** The full Model 3 training was performed once (seed 42); reload was bit-identical, but repeat-run reproducibility was not assessed and no deterministic-algorithm setting was enabled.
11. **No multi-seed experiment.** Ablation and multi-scale results describe the fixed single-seed configuration and carry no seed-to-seed variability.
12. **CRAG independence.** CRAG shares hospital and research group with GlaS; independence is partial, and no archive checksum was recorded.
13. **CRAG annotation.** In 70 of 213 CRAG images one gland covers more than 30% of the image, so gland-level orientation is a coarser quantity there.
14. **PANDA.** The Radboud annotations are semantic, so the planned target could not be constructed; only 10 cases were examined, and no restricted construction was attempted.
15. **Pathology structure.** Only two image-level labels exist, patient clustering is present, and three of the 12 held-out patients have images of both grades.
16. **Few held-out patients.** Only 12 distinct patients contribute the 80 held-out images.
17. **Other.** The A5 control cannot isolate equivariance; A1's Q head was untrained; there is no dedicated learnability control; the Model 2 permutation null used 100 permutations and one seed; one formulation, one architecture family and one training protocol were tested.

""", "S24-M03/M06", "limitations rewritten: neutral patient-overlap wording; unit of evaluation; pointwise comparison")

rep("The multi-scale analysis applied six pre-frozen reductions", "The multi-scale analysis applied six reductions frozen in advance", "S24-M10", "terminology")
rep("Six pre-frozen reductions of the same dense prediction were evaluated", "Six reductions of the same dense prediction, frozen in advance, were evaluated", "S24-M10", "terminology")
rep("across the four pre-specified tests, none was significant", "across the four tests frozen in advance, none was significant", "S24-M10", "terminology")

# ------------------------------------------------------------------ 6 (L05)
rep("This study developed and rigorously evaluated", "This study developed and evaluated", "S24-L05", "evaluative adverb removed")

# ------------------------------------------------------------------ availability / administrative (H06, M09)
rep_between("## Data Availability Statement", "---\n\n## References", """## Data Availability Statement

The GlaS data are the public challenge dataset [7], obtained from a Kaggle mirror of the challenge release (no version tag or published checksum manifest; per-file SHA-256 hashes are in the project's data manifest). CRAG [8] was obtained from an OpenDataLab mirror. PANDA [12] was obtained through the Kaggle competition. The licenses and terms of use of the three datasets are [DATASET LICENSE TO VERIFY]. Derived artifacts (manifests, frozen result files, tables) are held in the project repository [REPOSITORY URL OR DOI — USER TO COMPLETE]; data-sharing terms for the third-party datasets are [USER TO COMPLETE].

## Code Availability Statement

Analysis code and frozen configuration files are held in the project repository [PUBLIC REPOSITORY URL, RELEASE TAG AND LICENSE — USER TO COMPLETE].

## Author Contributions

[USER TO COMPLETE — CRediT statement]

## Funding

[USER TO COMPLETE]

## Institutional Review Board Statement and Informed Consent Statement

[USER TO COMPLETE — statement on the secondary use of public, previously published datasets; journal requirement to be verified]

## Use of Generative AI

[USER TO COMPLETE — disclosure, if required by the journal]

## Conflicts of Interest

[USER TO COMPLETE]

""", "S24-H06/M09", "administrative placeholders in the standard [USER TO COMPLETE] form; licence placeholders; no invented information")

# ------------------------------------------------------------------ references (H01)
REFS = """## References

1. Cohen, T.S.; Welling, M. Group Equivariant Convolutional Networks. In Proceedings of the 33rd International Conference on Machine Learning (ICML 2016); Proceedings of Machine Learning Research, Volume 48; PMLR: 2016; pp. 2990–2999.
2. Weiler, M.; Cesa, G. General E(2)-Equivariant Steerable CNNs. In Advances in Neural Information Processing Systems 32 (NeurIPS 2019); 2019; arXiv:1911.08251.
3. Cesa, G.; Lang, L.; Weiler, M. A Program to Build E(N)-Equivariant Steerable CNNs. In Proceedings of the International Conference on Learning Representations (ICLR 2022); 2022. Software: escnn, https://github.com/QUVA-Lab/escnn (version 1.0.11 used here).
4. Veeling, B.S.; Linmans, J.; Winkens, J.; Cohen, T.; Welling, M. Rotation Equivariant CNNs for Digital Pathology. In Medical Image Computing and Computer Assisted Intervention – MICCAI 2018; Lecture Notes in Computer Science; Springer: 2018; pp. 210–218. https://doi.org/10.1007/978-3-030-00934-2_24.
5. Lafarge, M.W.; Bekkers, E.J.; Pluim, J.P.W.; Duits, R.; Veta, M. Roto-Translation Equivariant Convolutional Networks: Application to Histopathology Image Analysis. Med. Image Anal. 2021, 68, 101849. https://doi.org/10.1016/j.media.2020.101849.
6. de Gennes, P.G.; Prost, J. The Physics of Liquid Crystals, 2nd ed.; Oxford University Press: Oxford, UK, 1993. https://doi.org/10.1093/oso/9780198520245.001.0001.
7. Sirinukunwattana, K.; Pluim, J.P.W.; Chen, H.; Qi, X.; Heng, P.-A.; Guo, Y.B.; Wang, L.Y.; Matuszewski, B.J.; Bruni, E.; Sanchez, U.; Böhm, A.; Ronneberger, O.; Ben Cheikh, B.; Racoceanu, D.; Kainz, P.; Pfeiffer, M.; Urschler, M.; Snead, D.R.J.; Rajpoot, N.M. Gland Segmentation in Colon Histology Images: The GlaS Challenge Contest. Med. Image Anal. 2017, 35, 489–502. https://doi.org/10.1016/j.media.2016.08.008.
8. Graham, S.; Chen, H.; Gamper, J.; Dou, Q.; Heng, P.-A.; Snead, D.; Tsang, Y.W.; Rajpoot, N. MILD-Net: Minimal Information Loss Dilated Network for Gland Instance Segmentation in Colon Histology Images. Med. Image Anal. 2019, 52, 199–211. https://doi.org/10.1016/j.media.2018.12.001.
9. Navarro, J.; Wilkinson, M. On the Equivariant Learning of the Q-tensor Order Parameter. arXiv 2026, arXiv:2605.27679 (preprint).
10. Alblas, D.; Suk, J.; Brune, C.; Yeung, K.K.; Wolterink, J.M. SIRE: Scale-Invariant, Rotation-Equivariant Estimation of Artery Orientations Using Graph Neural Networks. arXiv 2023, arXiv:2311.05400 (preprint).
11. Snoussi, H.; Karimi, D. Equivariant Spherical CNNs for Accurate Fiber Orientation Distribution Estimation in Neonatal Diffusion MRI with Reduced Acquisition Time. arXiv 2025, arXiv:2504.01925 (preprint).
12. Bulten, W.; et al. Artificial Intelligence for Diagnosis and Gleason Grading of Prostate Cancer: The PANDA Challenge. Nat. Med. 2022, 28, 154–163. https://doi.org/10.1038/s41591-021-01620-2. [AUTHOR LIST TO COMPLETE PER JOURNAL STYLE]
13. Cohen, J. Statistical Power Analysis for the Behavioral Sciences, 2nd ed.; Lawrence Erlbaum Associates: Hillsdale, NJ, USA, 1988.
"""
i = T.index("## References")
LOG.append({"issue": "S24-H01", "description": "reference list rewritten from verified sources (see 05_REFERENCE_VERIFICATION_FINAL.csv); drafter note removed",
            "old": T[i:], "new": REFS})
T = T[:i] + REFS

# ------------------------------------------------------------------ spelling (C02): American spelling throughout
SPELL = [("fibre", "fiber"), ("Fibre", "Fiber"), ("analysed", "analyzed"), ("analyse", "analyze"), ("centre", "center"),
         ("behaviour", "behavior"), ("licence", "license"), ("colour", "color"), ("artefact", "artifact")]
for a, b in SPELL:
    T, n = re.subn(rf"\b{a}\b", b, T)
    if n:
        LOG.append({"issue": "S24-C02", "description": f"spelling '{a}' -> '{b}' ({n}x)", "old": a, "new": b})

open(os.path.join(OUTDIR, "01_FULL_MANUSCRIPT_REPAIRED.md"), "w", encoding="utf-8", newline="\n").write(T)
if "--log" in sys.argv:
    json.dump(LOG, open(sys.argv[sys.argv.index("--log") + 1], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("edits:", len(LOG), "| chars:", len(T))
