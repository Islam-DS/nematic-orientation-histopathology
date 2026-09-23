# Stage 28 Task C — Root-cause evidence matrix

Date: 2026-09-23. Purpose: for each of the ten plausible contributors to the weak Q-tensor correspondence listed in the Stage 28 brief, state the evidence for, evidence against, and what remains unresolved, using only existing frozen results (Stages 0–27) plus the two new Stage 28 diagnostics (S_DL distribution, gland/patch-size assessment). **No single cause is identified as proven.** This is a narrowing exercise, not a determination.

## 1. Limited training-set size (85 images, 1816 training-inner patches)

- **Evidence for:** 85 images is small by deep-learning standards; the network has 11,674 parameters, a small model, but even small models typically need more than ~1800 patches of a highly variable natural signal (H&E tissue) to learn a subtle geometric regression target. The tiny-set optimization check (Section 3.7 of the manuscript) needed the epoch limit doubled (500→1000) before even an 8-patch batch reached its internal convergence criterion, suggesting the optimization landscape for this target is not trivially easy even at negligible scale.
- **Evidence against:** The classification head, trained on the exact same 1816 patches with the exact same optimizer and epoch budget, reaches AUROC 0.87–0.89 on the canonical held-out set (Section 4.1/Stage 27 multi-seed results) — a real, non-trivial signal learned from the same amount of data. This shows the training-set size is not so small that the network cannot learn *anything* discriminative from it; it specifically did not learn the geometric Q-tensor correspondence to the required degree.
- **Unresolved:** Whether a larger GlaS-scale training set (not available; GlaS provides only 165 images total) would improve Q correspondence was not tested and cannot be tested without new data.

## 2. Patch-level supervision versus full-gland anatomical targets

- **Evidence for:** Task D's new descriptive analysis (`results_v2/stage28/gland_patch_size/gland_patch_size_summary.json`) found the median valid gland instance occupies 88% of a 128×128 patch's area, and an estimated 79.3% of valid glands have a major-axis extent exceeding 128 px — i.e., most glands are comparable in size to, or larger than, the patch itself. Since Q_anat is computed once from the complete, unpatched mask (Section 3.3), while the network only ever sees a 128×128 window, the network frequently cannot see the same spatial extent of a gland that the reference orientation was computed from.
- **Evidence against:** None directly rules this out; it is a structural property of the pipeline, not itself falsifiable from existing data. Ablation A4 (Q-head trained alone, no classification objective) also failed to produce correspondence, which does not by itself indicate the patch/gland mismatch is *not* the cause — a smaller receptive field would still be a binding constraint under A4 too.
- **Unresolved:** Whether a larger patch size, a full-image architecture, or a gland-instance-cropped input (all of which would require a new experimental design, out of scope for this stage) would resolve the mismatch was not tested.

## 3. Gland-size / patch-size mismatch (truncation at patch boundaries)

- **Evidence for:** Same as item 2; additionally, 45.0% of valid gland instances have area exceeding the full 16,384 px patch area outright (`frac_area_gt_1.0` = 0.4502), meaning nearly half of all valid glands could not fit inside a single patch under any positioning.
- **Evidence against:** Patches were retained only if ≥10% of their pixels had a valid anatomical reference (Section 3.2), so every evaluated patch does contain *some* genuine gland signal; the network is not being asked to predict from pure background.
- **Unresolved:** This is the same underlying issue as item 2, examined at the population level rather than architecturally; the two are not independent contributors and should be read together, not summed as separate causes.

## 4. Truncated glands at patch boundaries (border-truncation specifically)

- **Evidence for:** Within the frozen Stage 4 gland manifest, 340 of the 1530 total gland instances (22.2%) are excluded from the valid population specifically because at least 2% of their own pixels lie on the *image* border (`excluded_border_truncated`, Section 3.3) — a different, coarser truncation (image edge, not patch edge) already controlled for in the frozen pipeline.
- **Evidence against:** Image-border truncation is already excluded from the valid population by design, so it cannot explain the residual weak correspondence among the 1084 *valid* instances used for supervision and evaluation.
- **Unresolved:** Patch-boundary truncation (as opposed to image-boundary truncation) was not separately quantified in this stage beyond the area/extent proxies in item 2–3; a literal per-patch bounding-box overlap calculation was not performed (would require re-reading masks, out of scope per the brief's "do not introduce a new experimental design" instruction).

## 5. Noisy or ambiguous anatomical orientation targets

- **Evidence for:** The project's own design documents (cited in Discussion 5.4) record cases where a single mask merges several visible structures or where fused epithelial sheets are described poorly by one principal axis. S_anat's own distribution (Stage 28 Task B) is wide and well-spread (mean 0.51, IQR 0.34, only 0.05% below 0.05) — i.e., the reference itself is not degenerate or collapsed, arguing against "the target itself is uninformative noise" as a full explanation, though it does not rule out that the target is *biologically* ambiguous even while being *numerically* well-behaved.
- **Evidence against:** A well-spread reference distribution suggests Q_anat carries real geometric information; if the target were pure noise, no amount of correct network behavior could recover a correlation, but the target's own internal structure (elongated glands have well-defined principal axes by construction, via eigendecomposition of a real covariance matrix) is mathematically well-defined, not noisy by construction.
- **Unresolved:** Whether the *specific* value of S_anat/phi_anat for a given gland reflects genuine, biologically meaningful elongation (as opposed to an artifact of annotation style) was not and cannot be tested from the existing data without a second, independent annotation of the same glands.

## 6. Weak visual signal for the desired tensor representation

- **Evidence for:** The classical structure-tensor oracle (used for calibration, Section 3.9) achieved Δ = 0.152 and r = 0.0115 against the same train-only reference — a weak but non-zero classical signal, far below the required thresholds. If even a hand-designed, non-learned image-texture estimator finds only a weak correspondence, the H&E pixel signal for gland-elongation orientation may itself be weak in this dataset, independent of the learning algorithm used.
- **Evidence against:** The oracle and the network are not directly comparable estimators (different features, different scale of operation), and a weak *classical* signal does not prove a weak *learnable* signal is impossible — deep networks routinely exceed simple statistics-based baselines on texture tasks elsewhere.
- **Unresolved:** No stronger classical or learned baseline was tested to bound how much orientation signal is actually recoverable from GlaS H&E images by any method.

## 7. Optimization dynamics

- **Evidence for:** The Q-loss plateaued well above zero in the full-data run, while a tiny-set run only reached the internal convergence criterion after the epoch limit was doubled (Section 5.2) — some evidence that the loss landscape for this specific target is harder to optimize than the classification loss trained alongside it. Stage 27's multi-seed study shows dev-loss trajectories that are noisy and non-monotonic across all seeds (`results_v2/stage27_robustness/multi_seed/plots/training_trajectories.png`), and best-epoch varies from 19 to 40 across seeds (Task F, this stage) — consistent with a training signal that does not converge cleanly to a stable optimum.
- **Evidence against:** No systematic learning-rate or optimizer sweep was run to test whether different optimization hyperparameters change the outcome; the ablation matrix (A1, A3) varied only the loss-weighting term λ_Q, not the optimizer itself, and neither changed the qualitative result.
- **Unresolved:** Whether a different optimizer, learning-rate schedule, or longer training budget would materially change the Q correspondence was not tested (would constitute a new experimental design).

## 8. Representation capacity

- **Evidence for:** Model 3 has only 11,674 parameters, a genuinely small network by modern standards, with a Q-head that is a single 3×3 convolution layer. A5 (parameter-matched, non-equivariant control) also failed to produce correspondence at a similar parameter count (11,676), which is at least consistent with — though does not prove — a capacity ceiling rather than an equivariance-specific problem.
- **Evidence against:** A5's architecture necessarily differs from Model 3 in channel width and receptive-field composition even at matched parameter count (Discussion 5.3/manuscript Limitations item 17; Task G of this stage discusses this further), so A5 cannot cleanly isolate "capacity" from "architecture family" as separate variables.
- **Unresolved:** No larger-capacity variant of the typed equivariant architecture was tested; whether more channels or additional equivariant layers would change the outcome is unknown.

## 9. Rotation/interpolation artifacts

- **Evidence for:** Stage 27 Task 3's five-metric rotation redesign found the reported error magnitude is measurably sensitive to pixel-domain interpolation, sampling location and small-S normalization (`results_v2/stage27_robustness/rotation_metrics/`); at 90° (the one angle with no interpolation), the error did not meaningfully differ from the interpolated measurement, but at other angles the contribution of interpolation and normalization to the reported number was not separately zero.
- **Evidence against:** All five metrics — differing in interpolation exposure, spatial pooling, and normalization — agreed exactly on which four of nine angles pass, arguing that the qualitative rotation-consistency shortfall is a property of the trained network's behavior, not an artifact of any one measurement choice (Stage 27 Task 3 conclusion, reproduced in the Stage 27 manuscript Section 4.11).
- **Unresolved:** This item pertains specifically to Criterion D (rotation consistency), not to Criteria A–C (static anatomical correspondence, evaluated without any rotation), so it cannot explain the near-zero static correspondence (Glass's Δ ≈ 0.04, r ≈ −0.03) even if it partially explains Criterion D's specific pass count.

## 10. Pathology and tissue heterogeneity

- **Evidence for:** GlaS spans a range of histologic grades and tissue architectures (benign to malignant, multiple growth patterns per Section 3.2/Table 2); the pathology association analysis (Section 4.6/manuscript) found S_DL's benign/malignant split (Stage 28 Task B: benign mean 0.0328, malignant mean 0.0304) shows no material difference, and neither did S_anat's own benign/malignant split (0.4918 vs 0.5265) — both are small differences relative to their spread, suggesting tissue heterogeneity across grades is not obviously driving the near-zero correspondence in a grade-specific way.
- **Evidence against:** The heterogeneity hypothesis was not tested at a finer granularity (e.g., by histologic subtype beyond the binary benign/malignant label, which was the only label available), so a subtler heterogeneity effect cannot be ruled out.
- **Unresolved:** Whether specific morphological subtypes (e.g., well- vs. poorly-differentiated glands) show different correspondence was not examined; GlaS's available labels do not support this without additional annotation.

## Summary judgment (per the brief: no single cause chosen)

The strongest, most directly *quantified* new evidence from this stage points to two related, structural contributors that are not mutually exclusive with each other or with several of the others: (a) **S_DL's distribution is heavily concentrated near zero** (Task B: 82.9% of predictions below 0.05, against a healthy, well-spread S_anat), which is consistent with — but does not, on its own, prove — a form of representation collapse in the Q-head's magnitude channel; and (b) **the patch size is frequently smaller than the gland the target was computed from** (Task D: median gland ≈ patch area, 79% of glands wider than the patch). Both are consistent with, but do not individually establish, a causal mechanism. Optimization dynamics (item 7) and representation capacity (item 8) remain plausible contributing factors that were narrowed by ablation but not eliminated. No experiment in this project isolates a single, definitive root cause, and none should be claimed.
