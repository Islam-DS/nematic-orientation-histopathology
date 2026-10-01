# From Equivariance to Nematic Order: Anatomically Supervised Orientation-Tensor Learning in Histopathology

**Authors:**  
Md. Mahbubul Islam¹˒²,* ([ORCID: 0009-0009-1910-1980](https://orcid.org/0009-0009-1910-1980))  
Ainur Yerkos¹˒² ([ORCID: 0000-0001-5949-6942](https://orcid.org/0000-0001-5949-6942))  
Nadezhda Kunicina¹˒²,* ([ORCID: 0000-0002-0980-0958](https://orcid.org/0000-0002-0980-0958))

**Affiliations:**  
¹ Al-Farabi Kazakh National University, Almaty, Kazakhstan  
² Institute of Industrial Electronics, Electrical Engineering and Energy, Riga Technical University, Riga, Latvia

**Correspondence:**  
Md. Mahbubul Islam, islam_md_mahbubul@live.kaznu.kz; Nadezhda Kunicina, Nadezda.Kunicina@rtu.lv  
\* Corresponding authors.

---

## Abstract

Glandular orientation in histopathology is defined up to a head–tail flip, which motivates a rank-2 nematic tensor output and rotation-equivariant networks. We tested whether a D8-equivariant network whose Q-tensor head is typed as `D8.irrep(1,2)` (11,674 parameters) can learn a mask-derived gland-elongation orientation reference on GlaS (85 training images). Four prospectively frozen internal criteria were evaluated on 80 held-out images (2083 overlapping patches; canonical image-level split, not patient-independent). None was met: angular correspondence Glass's Δ = 0.0412 (≥ 0.5 required); order-magnitude correlation r = −0.0341 (≥ 0.30); tensor similarity Glass's Δ = 0.0073 (≥ 0.5); empirical center-pixel, pixel-domain rotation consistency 4 of 9 angles (7 required). Single-seed ablations and six aggregation scales did not change this outcome. No pathology association was demonstrated (Holm-adjusted p = 1.0). An external cross-dataset evaluation on CRAG (partially independent of GlaS) gave no supporting evidence (pooled Δ = −0.0797, r = −0.0937), and the PANDA masks, which encode semantic Gleason-pattern regions, were incompatible with the gland-instance construction. The negative result reproduced across five random seeds and a patient-disjoint data partition: 0 of 20 canonical-split and 0 of 20 patient-disjoint seed-criterion combinations met. Under the tested protocol the formulation did not demonstrate the intended learning objective; the result does not show that orientation information is absent from histology or that equivariant learning cannot address it.

**Keywords:** rotation equivariance; nematic order; rank-2 orientation tensor; Q-tensor; histopathology; gland morphology; steerable convolutional networks; negative result

---

## 1. Introduction

Histopathology images contain structured spatial organization. The elongation and alignment of glands, the arrangement of stroma, and the regularity of tubular architecture are geometric properties that pathologists already use, and that can in principle be quantified. For glandular tissue, the orientation of an individual gland is an axis: a gland elongated along direction *n* is equally elongated along −*n*. A vector cannot describe an axis, because an angle θ and θ + π would then describe the same orientation without the representation reflecting it. The standard device for this head–tail symmetry, taken from the physics of nematic liquid crystals [6], is a symmetric traceless rank-2 tensor. In two dimensions, in the convention used throughout this work,

Q = S [[cos 2θ, sin 2θ], [sin 2θ, −cos 2θ]],

with S ≥ 0 the order magnitude and θ the director angle. Under a rotation of the plane by α the tensor transforms as Q′ = R(α) Q R(α)ᵀ, equivalently the pair (q₁, q₂) = S(cos 2θ, sin 2θ) rotates by 2α. This tensor form and transformation law are established mathematics, not a contribution of this work.

Standard convolutional networks do not encode how their outputs should change when the input is transformed; group-equivariant and steerable networks impose such structure by construction, so that a transformation of the input produces a specified transformation of the output [1–3], and have been applied to histopathology, where tissue has no preferred orientation on the slide [4,5]. This motivates an empirical question: if a network's output is *typed* as a rank-2 representation of the rotation–reflection group, so that the transformation law above holds by construction for the group elements, can the output be trained to reproduce an anatomically meaningful orientation field, judged against an independently constructed, mask-derived reference computed from gland-instance annotations? Answering this requires a reference not derived from the network's own inputs, evaluation criteria fixed before results are seen, controls for the obvious alternative explanations, and an honest accounting of what a negative outcome does and does not imply.

We evaluated one specific formulation under a protocol whose four evaluation criteria were frozen internally in advance (Section 3.4), against three hypotheses recorded verbatim, before any result was observed, in the original project specification:

**H1: Geometric validity.** The learned Q_DL field should correspond to an independently derived anatomical orientation field Q_anat.

**H2: Equivariance validity.** When the input image is rotated by α, the predicted Q tensor should transform as Q_DL(R_α x) ≈ R(α) Q_DL(x) R(α)ᵀ, and the scalar order magnitude should remain invariant.

**H3: Pathological relevance.** The learned orientation/order representation may contain information associated with pathological state or tissue morphology beyond conventional features.

The recorded hypotheses do not explicitly assign individual criteria to individual hypotheses, and none is inferred here. The principal outcome is negative: none of the four frozen criteria was met, and secondary analyses (ablations, spatial aggregation, pathology, an external dataset, five-seed and patient-disjoint robustness checks) did not alter that conclusion. We report the design, the results, and the limitations in full, together with the measurement details that qualify how the rotation criterion may be read. The contribution of this paper is the experimental formulation, the explicit anatomical supervision, the choice of a typed rank-2 representation, and the systematic evaluation framework with prospectively frozen internal criteria, not a demonstration of successful orientation-tensor learning.

---

## 2. Related Work

This section is not a systematic review; it describes the prior work identified as closest to the present study. Claims below are sourced from the cited works' own text.

### 2.1 Equivariant learning in histopathology

Group-equivariant convolutional networks [1] and steerable, E(2)-equivariant networks [2] provide layers whose feature maps transform according to a chosen group representation; the `escnn` library [3] implements these constructions and is the implementation basis of the network used here. Veeling et al. [4] motivate rotation-equivariant CNNs from the observation that histopathology has no preferred orientation, reporting improved, more stable tumor-detection performance on a lymph-node metastasis dataset. Lafarge et al. [5] encode the roto-translation group SE(2) directly into convolutional layers and apply the resulting networks to mitosis detection, nuclei segmentation, and tumor classification. Neither study predicts a continuous rank-2 orientation-tensor field or evaluates its output against an independently constructed anatomical reference; both instead use equivariance to improve classification or detection accuracy under known input symmetries. We have not identified a prior equivariant histopathology study that evaluates a rank-2 orientation output against an anatomical reference in the way this study does.

### 2.2 Orientation prediction in medical imaging

Two recent works predict a continuous orientation quantity from medical images with equivariant networks outside histopathology. SIRE [10] estimates 3D vessel orientation from CT angiography with a gauge-equivariant mesh CNN, against ground truth derived independently of the network (expert-drawn centerlines, segmentation-derived centerlines, or multi-observer-averaged centerlines across three datasets); this is structurally analogous to this study's use of a mask-derived reference rather than a self-referential target. Under random test-time rotation, SIRE's equivariant model held a median orientation-agreement of 0.99 against 0.49 for a non-equivariant baseline. Snoussi and Karimi [11] predict fiber orientation distributions in neonatal diffusion MRI with a rotation-equivariant spherical CNN from a reduced (30%) subset of diffusion directions, evaluated against references computed from the complete acquisition (again a reference independent of the reduced input the network receives), and report an 84% mean-squared-error reduction over a multilayer-perceptron baseline. Both studies, unlike the present one, report a clear positive equivariance advantage on real data against an independently derived reference; the anatomical structures, modalities, architectures, and reference-construction procedures all differ from this study, and no quantitative comparison is drawn.

### 2.3 Q-tensors and nematic order

The symmetric traceless rank-2 tensor order parameter is the standard description of nematic order in liquid-crystal physics [6], and the same formalism quantifies orientational order in biological tissue, including nematic cell polarity in three-dimensional epithelia [17]. Navarro and Wilkinson [9] construct networks equivariant to cyclic groups C_k (k = 4 to 256) that predict the two-dimensional de Gennes Q-tensor directly from synthetic images of computationally generated elliptical-particle packings, with an exact, noise-free analytic ground truth by construction; all seven tested architectures satisfy the Q-tensor equivariance constraint to floating-point precision and outperform parameter-matched non-equivariant benchmarks. This is the closest prior use of the Q-tensor as a neural-network prediction target identified, and differs from the present study in the respect that most governs whether a positive result should be expected: their target is generated by an explicit analytic formula, whereas this study's target is a statistical summary of a gland-instance segmentation mask embedded in real tissue, with no analytic ground truth. This study additionally uses the dihedral group D8 (rotations and reflections), applies the formulation to real H&E histology, and constructs its reference from independently produced gland-instance annotations rather than from the generative process that produced the input.

Quantitative gland-orientation and tissue-anisotropy descriptors have independent precedent in digital pathology: gland orientation and shape, extracted by computational pathology, distinguish aggressive from indolent early colon carcinoma [18], and co-occurring gland-angularity features in localized subgraphs predict biochemical recurrence in prostate cancer [19]. Classical structure-tensor orientation estimation [15], used here only to calibrate the frozen thresholds (Section 3.4), and tumor-associated collagen alignment as a stromal orientation biomarker [16] are further, longer-standing examples of orientation as a pathology-relevant quantity. None of this literature evaluates a *neural, rank-2-typed* orientation output against an anatomical reference; it establishes that gland- and stroma-orientation features are independently pathology-relevant, motivating the present formulation without predicting how a network trained on real tissue would fare against it.

### 2.4 Contribution and scope

The element common to SIRE [10] and Snoussi and Karimi [11] that this study adopts is validating a learned orientation output against a reference constructed independently of the network's own input pathway. Neither of those works is in histopathology, and neither uses a mask-derived anatomical reference of the kind constructed in Section 3.2. This study's specific combination (a rank-2 tensor typed to the D8 representation R(2α), an anatomical reference built from independent gland-instance masks with no image-pixel input, and a decision protocol with criteria frozen before any result was observed) was not found described together in the works read for this section. We do not claim this combination is novel in the wider literature, which was not searched exhaustively; we claim only that we did not find it in the works identified as closest to this study. The defensible contribution of this paper is the validation framework itself: an explicitly typed rank-2 equivariant output, anatomical supervision, an independently constructed correspondence reference, quantitative empirical equivariance calibration, and transparent negative-result reporting, not a claim to be the first equivariant network applied to histopathology, the first orientation-prediction method in medical imaging, the first neural Q-tensor predictor, the first nematic-order analysis of cancer tissue, or the discoverer of a novel symmetry-breaking biomarker; none of these claims is made anywhere in this manuscript.

---

## 3. Materials and Methods

### 3.1 Datasets

Three public datasets were used (Table 1). **GlaS** [7,12] (obtained via a Kaggle mirror; research-use-only license, commercial use not permitted, citation of [7] and [12] required) provides 165 colorectal H&E images with 1530 gland-instance masks and benign/malignant labels, in the canonical split of 85 training, 60 testA, and 20 testB images. The 80 held-out images come from 12 patients; 11 of these also contribute training images, and 79 of the 80 held-out images belong to such a patient, a documented property of the canonical benchmark's own design (the GlaS organizers stratified by grade and visual field but not by patient [7]), not an artifact of this study's data handling. The canonical split therefore permits patient overlap between training and held-out data and is neither patient-independent nor an independent patient-level test set; the eight pre-development leakage tests were image-level and split-level, not patient-level. Images were cut into 128 × 128 pixel patches with stride 96 (overlapping), retained if ≥10% of pixels had a valid anatomical reference: the 85 training images gave 1816 training-inner and 433 development patches (68/17 image split, stratified 80/20, seed 42); the held-out images gave 1470 (testA) and 613 (testB) patches, 2083 in total. **CRAG** [8], described in the MILD-Net study, contains 213 colorectal images (173 train-folder, 40 test-folder) with 3054 gland instances, derived from 38 whole-slide images and annotated by an expert pathologist, obtained via an OpenDataLab mirror; the source page required sign-on, so its terms of use were not independently verified and are not assumed to be free of restriction. Both GlaS and CRAG derive from the University Hospitals Coventry and Warwickshire NHS Trust and the same research group, so their independence is partial; no exact duplicates were found by content-hash comparison. CRAG was used only to evaluate the frozen model checkpoint; nothing was retrained. **PANDA** (Radboud subset) [13] (CC BY-SA-NC 4.0) provided a 10-case pilot, examined only for annotation-compatibility (Section 3.4); its masks encode semantic Gleason-pattern regions rather than gland instances and were not used for model inference.

**Table 1. Datasets and their roles.**

| Dataset | Role | Images | Split | Organ | Annotation |
|---|---|---|---|---|---|
| GlaS | primary; training and frozen evaluation | 165 | canonical: train 85 / testA 60 / testB 20 (image-level) | colorectal | gland-instance masks; benign/malignant label and 5-level ordinal grade |
| CRAG | external evaluation of the frozen model; partially independent of GlaS | 213 | dataset's own split: 173 train / 40 test; none used for training | colorectal | gland-instance masks; no pathology labels or patient metadata |
| PANDA (Radboud) | annotation-compatibility assessment only; no inference | 10 (pilot) of 5,160 Radboud slides | not applicable | prostate | semantic tissue/Gleason-pattern masks |

### 3.2 Anatomical orientation reference

The reference, Q_anat, is computed from GlaS gland-instance masks only; no H&E pixel value enters its computation, and the image and mask pathways share no function signature (verified by static inspection and by a poisoned-mask test in which replacing a mask with random labels left the image tensor byte-identical). For each gland instance, computed once from its complete mask in the unpatched image, the pixel coordinates define a 2 × 2 covariance matrix with eigenvalues λ₁ ≥ λ₂; the director angle is φ = atan2(v₁ᵧ, v₁ₓ) mod π (v₁ the principal eigenvector; mod π encodes the head–tail symmetry), the order magnitude is S = (λ₁ − λ₂)/(λ₁ + λ₂), and the target is (q₁, q₂) = S(cos 2φ, sin 2φ). Every pixel takes the (q₁, q₂) of its gland instance if that instance is valid (≥ 20 pixels, < 2% of its pixels on the image border, S ≥ 0.15); every other pixel is marked invalid. Of 1530 instances, 1084 were valid (567/769 training, 443/666 testA, 74/95 testB); four instances (all in testB) had more than one connected component and were retained. Q_anat is independent of image pixel values but not of the target definition, the mask annotation, or the training objective, since the same construction supplies training supervision. It describes the elongation of individual glands, not a continuous tissue-fiber field or a universal biological truth: the project's design record notes cases where a single mask merges several visible structures or where fused epithelial sheets are described poorly by one principal axis. A negative result against this reference says something about correspondence with this specific reference, not that the reference is wrong or that another definition of tissue orientation would give the same outcome. Figure 1 and Figure 2 illustrate the framework and the reference construction.

> **Figure 1.** Schematic of the framework; no data are plotted. Top row, left to right: an H&E image; the gland-instance mask; the principal orientation θ and anisotropy S of each gland, computed from the mask by principal component analysis; the reference tensor Q_anat; the D8-equivariant network; its predicted tensor Q_DL; and the evaluation of Q_DL against Q_anat. The network receives only the H&E image; Q_anat is the supervision target on training images and the comparison reference on held-out images. The lower panel shows the tensor representation and its transformation law under rotation.

> **Figure 2.** Conceptual schematic of the anatomical-reference construction (idealized, not a dataset example). (a) H&E tissue with glands; (b) gland-instance segmentation; (c) principal axis of one gland (eigenvector v₁, eigenvalues λ₁ ≥ λ₂) and orientation angle θ, defined modulo π; (d) order magnitude S = (λ₁ − λ₂)/(λ₁ + λ₂) and rank-2 tensor Q with (q₁, q₂) = S(cos 2θ, sin 2θ). Every pixel of a valid gland takes the (q₁, q₂) of its gland.

### 3.3 Rank-2 equivariant model

The network output at each location is the pair (q₁, q₂), interpreted as Q = [[q₁, q₂], [q₂, −q₁]] with S = √(q₁² + q₂²) ≥ 0 and θ = ½ atan2(q₂, q₁) mod π, so the head–tail symmetry is built into the representation. The group is D8, the dihedral group of order 16, implemented as `escnn.gspaces.flipRot2dOnR2(N=8)`; `D8.irrep(1,2)` is a two-dimensional irreducible representation in which rotation by α acts as R(2α), verified to agree with the analytic matrix to 0.00e+00 at α = 45°. This guarantee is algebraic and confined to the 16 elements of the implemented discrete group, not a claim of continuous SO(2) equivariance; on the finite pixel grid, antialiased pooling makes the whole network only approximately equivariant (in the untrained network, median interior transformation error 0.00% at the identity and one reflection, up to 33.95% at other group elements, tested at all 16 elements with a smooth synthetic input). This residual is a property of the finite-grid implementation, distinct from the algebraic representation result.

Model 3, the principal model, takes a 3 × 128 × 128 RGB patch through two equivariant convolution blocks (`R2Conv`, 5 × 5 kernels, `InnerBatchNorm`, ReLU; eight regular-representation fields each) separated by antialiased average pooling. Two heads share the second block: a classification head (group pooling, global average pooling, linear layer) gives per-patch grade logits; the Q head is a 3 × 3 `R2Conv` typed to output field `[D8.irrep(1,2)]`, giving a dense (q₁, q₂) map of size 2 × 64 × 64. The network has 11,674 parameters; its Q output is physically typed, not two of eight arbitrary scalar channels assigned meaning after training. The loss is L = L_class + λ_Q L_Q (λ_Q = 1.0, fixed); L_Q is a masked mean-squared error on (q₁, q₂), evaluated only where the (2×2-pooled) reference is valid. Training therefore operates on the dense field at the level of individual cells; the patch-level summaries used by Criteria A–C (mean (q₁, q₂) over a patch's valid cells, giving patch-level S and φ) are computed only for evaluation.

Two predefined baselines were trained under the identical split, loss, optimizer, and schedule: a **plain CNN** (two unconstrained 5 × 5 convolution blocks, no equivariance, plain 2-channel Q output), and **Model 2**, a historical post-hoc formulation (a p4m/D4-equivariant network whose Q value is obtained after training by a second-harmonic projection of eight fixed-angle orientation channels; un-typed output). Model 2 was also used for a channel-permutation null analysis (100 permutations, seed 0). A classical structure-tensor estimator [15] was used only to calibrate the frozen thresholds. The frozen criteria were not applied to either baseline; their values are reported descriptively (Section 4.1), never pooled with Model 3, and no baseline finding is transferred to it.

### 3.4 Training and evaluation criteria

Model 3 was trained on the 1816 training-inner patches (Adam, learning rate 1 × 10⁻³, batch size 32, up to 40 epochs, early stopping patience 6 on development-set loss, no augmentation, FP32, one NVIDIA RTX 3050 6 GB GPU, seed 42); training ran 37 epochs, selected checkpoint at epoch 30 (development loss 0.5756). The full training was performed once with seed 42; checkpoint reload was bit-identical, and a controlled tiny-set optimization check (8 patches, single batch) reproduced its first 500 epochs numerically identically between an original and extended run, meeting the pre-specified L_final/L_initial ≤ 0.10 criterion only after a single pre-authorized extension to 1000 epochs (0.0736, versus 0.139 at 500 epochs), an optimization-pacing check, not evidence of generalization. No deterministic-algorithm setting was enabled, and multi-run reproducibility of the full training was separately assessed by the multi-seed protocol below.

Four criteria were frozen in a versioned criteria file (version 2, checksum-verified for this manuscript) before any Model 3 training and have not been edited since (Table 2). An earlier version (v1) was invalidated because its calibration images had been drawn from canonical testA, held-out data; version 2 was recalibrated from scratch using only the 85 canonical training images (2249 patches). The decision rule, verbatim: "GO only if criteria A, B, C, and D ALL pass under their PRIMARY rules above, evaluated on the held-out TEST set (testA + testB, 80 images) only. No partial-credit weighting." These are prospectively frozen internal criteria, not entered in an external registry.

**Table 2. Frozen criteria.**

| Criterion | Metric | Frozen requirement |
|---|---|---|
| A: angular correspondence | Glass's Δ = (null mean − model mean angular error)/null SD, against the exact individual-level cross-pairing null of the model's own predicted angles against the reference angles | Δ ≥ 0.5 (Cohen's medium-effect convention [14]). Secondary, non-binding: permutation p < 0.01; mean error < 40° |
| B: order-magnitude correspondence | Pearson r(S_DL, S_anat) | r ≥ 0.30 **and** p < 0.01 **and** bootstrap 95% CI lower bound > 0 (2000 resamples, seed 42) |
| C: tensor similarity | D_Q = ‖Q_DL − Q_anat‖_F | one-sided permutation p < 0.01 (1000 re-pairings) **and** Glass's Δ ≥ 0.5 |
| D: empirical rotation consistency | normalized equivariance error per angle vs a per-angle threshold | error below threshold at ≥ 7 of 9 non-zero angles |

Criterion D's thresholds are the mean plus three SD of a rotation floor measured with the classical structure-tensor estimator [15] on real training patches; the same estimator, evaluated on the 2249 training patches, gave Glass's Δ = 0.152 and r = 0.0115 for the calibration itself, a training-data value used only to derive the thresholds, not a held-out model result, and not directly comparable with the Model 3 results reported in Section 4. Criteria A–C are evaluated on one (S, φ) pair per patch (the spatial mean of the field over the patch's valid cells for the prediction and for the reference); this patch-level order magnitude is the magnitude of the mean tensor of the patch, not the per-gland anisotropy of Section 3.2, and overlapping patches from the same image are not independent observations.

For rotation consistency (Criterion D), each held-out patch was rotated by α ∈ {15°, 30°, 37°, 45°, 60°, 90°, 123°, 150°, 173°} (`skimage.transform.rotate`, bilinear interpolation); the dense Q map of the rotated patch was compared, at the dense center pixel, with R(2α) applied to the unrotated patch's Q value there. Because the 128-pixel input's rotation center falls at approximately (31.5, 31.5) in the 64 × 64 dense grid, the sampled pixel lies about half a dense pixel per axis from the true fixed point; the frozen calibration used the same convention. Only 90° is an exact pixel-lattice rotation; the other angles involve interpolation. The error is normalized by the predicted magnitude, so it is inflated for patches with small predicted S. Two non-gating checks were also computed: S(R_α x) ≈ S(x) and φ(R_α x) ≈ φ(x) + α (mod π).

To separate network behavior from possible measurement-design artifacts, four alternative rotation-consistency metrics were computed post hoc on the same frozen checkpoint and population, without retraining: an exact-lattice-only comparison (0°/90° only, avoiding interpolation entirely); a local 5×5-neighborhood average around the true fixed point (reducing single-pixel sampling noise); a full-field comparison (the entire 64×64 field, pointwise, rather than one location); and a floor-stabilized variant of the full-field error (denominator floored at 0.05, to check whether normalization by a small predicted magnitude inflates the reported error). The frozen per-angle thresholds were calibrated for the original (center-pixel) convention only; applying them to the alternatives is diagnostic context, not a re-calibrated criterion.

Five ablation conditions were frozen before training (single seed each, diagnostic only, no new criterion): no Q supervision; λ_Q × 10; Q-head-only (no classification loss); a parameter-matched non-equivariant control (11,676 vs. 11,674 parameters; channel width and receptive-field composition still differ between this control and Model 3, so it cannot isolate equivariance as the only difference and is not a definitive causal test of it); and two alternative spatial reductions of the rotation criterion. Six multi-scale reductions of the same dense prediction (center pixel; 3×3, 5×5, 9×9, 17×17 windows; full field) were evaluated similarly. Pathology association was tested on the 80 held-out images (patch-level S_DL and φ-dispersion aggregated per image) against binary and ordinal grade, as four tests frozen as a family under Holm–Bonferroni correction (Mann–Whitney U / Cohen's d for the binary comparisons, Spearman ρ with bootstrap 95% CI for the ordinal comparisons); morphology and confound-adjusted checks were outside this corrected family. CRAG evaluation applied the frozen checkpoint and algorithm, unmodified, with the image as the primary unit (patch-level diagnostics used a deterministic 5000-patch subsample). For PANDA, a compatibility gate was frozen before any target generation: if individual gland instances could be recovered from the semantic masks without inventing boundaries the annotation does not provide, the Section 3.2 construction would be applied; otherwise the assessment would be declared infeasible and no reference or inference produced.

Five additional Model 3 training runs (seeds 11, 22, 33, 55, plus the original seed 42) assessed whether the primary result depends on the particular random initialization, under the identical architecture, split, hyperparameters, and evaluation code. A patient-disjoint sensitivity partition was separately constructed from patient identifiers in the original GlaS archive's `Grade.csv`: a strictly patient-disjoint recreation of the canonical 85/60/20 split was infeasible without discarding 26 of 85 training images, so instead every patient was assigned wholly to a training or held-out group by majority image count in their canonical split (80 training images [64 train-inner/16 dev] and 85 held-out images, zero patient overlap by construction); this protocol was run for all five seeds. Two further diagnostic analyses used only the frozen checkpoint, without retraining: the full percentile/threshold distribution of predicted S_DL versus S_anat across the pooled held-out patches (testing for meaningful dynamic range versus collapse toward zero), and a gland-versus-patch-size assessment from the frozen gland manifest (gland pixel area as a fraction of the 128×128 patch area, and an approximate major-axis extent 4*sqrt(lambda_1) from the already-computed covariance eigenvalue, as descriptive proxies for how often a gland's spatial extent exceeds what one patch's receptive field can see).

### 3.5 Statistical analysis and reproducibility

Effect sizes are reported before p-values throughout. Patch-level p-values and bootstrap intervals are descriptive: the 2083 evaluated patches are overlapping image patches, not independent biological units, so such p-values can be anti-conservative, and effect-size results are the primary basis for interpretation. Holm–Bonferroni correction was applied only to the four primary pathology tests; ablation, multi-scale, morphology, per-split, and CRAG analyses are descriptive or diagnostic, and no inference is drawn from their p-values. Software: Python 3.11.8, PyTorch 2.6.0 (CUDA 12.4), escnn 1.0.11. Seeds were 42 for training, the development split, bootstrap resampling, and the CRAG subsample, and 0 for the Model 2 permutation null; the multi-seed and patient-disjoint protocols additionally used seeds 11, 22, 33, 55. Relevant frozen artifacts (criteria file, result files, model checkpoint) were checksum-verified. No determinism claim is made beyond checkpoint-reload bit-identity, verified for the seed-42 run only.

---

## 4. Results

All values are read from frozen result files; effect sizes are reported before p-values, and criteria are those of Table 2.

### 4.1 Primary anatomical correspondence

The reference construction was numerically stable for all 165 GlaS images (1530 gland instances, 1084 valid); the implementation passed all architecture and leakage tests, which are execution-integrity results and do not bear on correspondence. None of the three correspondence criteria was met on the pooled 80-image, 2083-patch held-out set (Table 3, Figure 3). Criterion A's mean angular error (43.38°) was close to its own cross-pairing null (44.50°), giving Glass's Δ = 0.0412 against the required 0.5; the secondary permutation p = 0.002 reflects the large overlapping-patch sample and is not evidence of agreement. Criterion B was not significant (r = −0.0341, p = 0.1196, 95% CI −0.0757 to 0.0091). For Criterion C, 0 of 1000 re-pairings produced a mean D_Q as small as the observed 0.5092 (p < 0.001), but the null-of-mean was 0.5108 and the effect size only 0.0073 against the required 0.5, illustrating why the criterion requires both a significance and an effect-size condition. Per-split values (testA r = −0.1577, mean error 46.90°; testB r = 0.3077, mean error 34.96°) are descriptive, of opposite sign between splits, and do not override the pooled result.

**Table 3. Primary frozen criteria on the pooled held-out GlaS set (80 images; 2083 patches).**

| Criterion | Effect size (observed) | Frozen requirement | Other reported values | Status |
|---|---|---|---|---|
| A: angular | Glass's Δ = 0.0412 | Δ ≥ 0.5 | mean error 43.38° (null mean 44.50°); permutation p = 0.002 (secondary) | Not met |
| B: order magnitude | r = −0.0341 | r ≥ 0.30, p < 0.01, CI lower bound > 0 | p = 0.1196; 95% CI −0.0757 to 0.0091 | Not met |
| C: tensor | Glass's Δ = 0.0073 | permutation p < 0.01 and Δ ≥ 0.5 | mean D_Q = 0.5092 (null-of-mean 0.5108); permutation p < 0.001 | Not met (significance met, effect size not) |
| D: rotation | 4 of 9 angles | ≥ 7 of 9 | Section 4.2 | Not met |

> **Figure 3.** Frozen effect sizes for Criteria A–C on the pooled held-out GlaS set, with frozen thresholds (dashed lines). The permutation p-values are consistent with a large overlapping-patch sample and are not evidence of agreement.

For context, the two baselines were evaluated descriptively per split (Table 4); the frozen criteria were not applied to them, and no ranking among models is implied. Model 2's permutation null gave observed mean S below the null mean in both splits (testA z = −1.711; testB z = −1.704; two-sided p = 0.14 in each, 100 permutations), a descriptive, inconclusive result not attributable to Model 3.

**Table 4. Descriptive per-split values (no frozen criteria applied; not pooled or ranked).**

| Model | Split (images / patches) | Classification accuracy | Mean angular error (°) | Glass's Δ (A) | r (B) | Mean D_Q | Glass's Δ (C) |
|---|---|---|---|---|---|---|---|
| Plain CNN | testA (60 / 1470) | 0.5463 | 49.43 | 0.0327 | −0.1562 | 0.5499 | 0.0340 |
| Plain CNN | testB (20 / 613) | 0.7732 | 34.78 | 0.0112 | −0.3495 | 0.4753 | 0.0174 |
| Model 2 | testA (60 / 1470) | 0.7327 | 36.03 | 0.3233 | 0.1526 | 0.5069 | 0.0184 |
| Model 2 | testB (20 / 613) | 0.8646 | 39.94 | 0.1371 | 0.0341 | 0.5070 | 0.0104 |
| Model 3 | testA (60 / 1470) | 0.7449 | 46.90 | 0.0248 | −0.1577 | 0.5133 | 0.0034 |
| Model 3 | testB (20 / 613) | 0.8793 | 34.96 | −0.0255 | 0.3077 | 0.4992 | −0.0030 |

### 4.2 Rotation consistency

Empirical rotation consistency did not meet the frozen criterion: four of nine non-zero angles (90°, 123°, 150°, 173°) were below threshold, against seven required (Table 5, Figure 4). Model error was roughly flat across angles (0.82–1.11), and the four passing angles were simply those with the largest thresholds; this pattern was not used to reinterpret the criterion. Three things are distinguished: the transformation law is exact for the typed output under the discrete group (irrep matrix error 0.00e+00); the finite-grid implementation is only approximately equivariant (up to 33.95% median residual in the untrained network); and the frozen empirical test measured pixel-domain consistency of the *trained* network at nine discrete angles. Criterion D not being met is a statement about the third; it neither contradicts the first nor settles the second, and nothing in these results supports the claim that the whole network is, or is not, equivariant in general. The non-gating S- and φ-invariance checks gave mean |ΔS| of 0.031–0.039 (the same order as the ≈0.03 mean predicted S itself) and mean φ-deviation of 16.6°–22.2° across the nine angles.

**Table 5. Rotation consistency, pooled held-out GlaS set (2083 patches): Criterion D and non-gating additional checks.**

| Angle (°) | Model error | Frozen threshold | Below threshold | Mean \|ΔS\| | Mean φ deviation (°) |
|---|---|---|---|---|---|
| 15 | 1.0182 | 0.3503 | No | 0.0346 | 19.18 |
| 30 | 0.9812 | 0.5339 | No | 0.0361 | 20.39 |
| 37 | 0.9000 | 0.5809 | No | 0.0335 | 18.27 |
| 45 | 0.8246 | 0.6527 | No | 0.0310 | 16.61 |
| 60 | 1.0489 | 0.8228 | No | 0.0363 | 20.57 |
| 90 | 0.9552 | 1.0362 | Yes | 0.0322 | 16.94 |
| 123 | 1.0226 | 1.2350 | Yes | 0.0367 | 20.82 |
| 150 | 1.1144 | 1.2877 | Yes | 0.0393 | 22.21 |
| 173 | 1.0731 | 1.2556 | Yes | 0.0375 | 21.52 |

> **Figure 4.** Normalized rotation error (bars) and frozen per-angle thresholds (hatched), pooled held-out GlaS set. Green: below threshold; red: not.

The four post-hoc alternative metrics (Section 3.4, Table 6) agreed with the original result. The exact-lattice comparison at 90° gave an error (0.9552) identical to the interpolated original to four decimal places, indicating interpolation does not measurably contribute to the error there. The local 5×5-neighborhood average gave a lower mean error (0.8029 vs. 0.9931) than the center-pixel convention, consistent with reduced sampling noise, while the full-field comparison gave a higher mean error (mean 1.1265), because it compares the field pointwise rather than comparing means of the field, a different operation, not a contradiction; the floor-stabilized and absolute (non-normalized) full-field variants (mean 0.0697 absolute) were reported alongside it specifically to check whether normalization by a small predicted magnitude inflates the error, which Table 6 shows angle by angle. All three alternative summarizations that could be evaluated against the frozen thresholds (exact-lattice being restricted to two angles) passed the same four angles (90°, 123°, 150°, 173°) as the original measurement, and no alternative converted the negative outcome into a positive one; Criterion D's outcome therefore reflects genuine network behavior more than evaluation-design artifacts, even though the reported error magnitude is sensitive to the summarization and normalization used.

**Table 6. Rotation-consistency error under five metric designs, nine non-zero angles, pooled held-out GlaS set (2083 patches).**

| Angle (°) | M1 center-pixel | M2 exact-lattice | M3 local 5×5 | M4 full-field (normalized) | M4b full-field (absolute) | M5 full-field (floor-normalized) |
|---|---|---|---|---|---|---|
| 15 | 1.0182 | n/a | 0.8150 | 1.1035 | 0.0686 | 0.8023 |
| 30 | 0.9812 | n/a | 0.8273 | 1.1806 | 0.0738 | 0.8590 |
| 37 | 0.9000 | n/a | 0.7681 | 1.1326 | 0.0710 | 0.8240 |
| 45 | 0.8246 | n/a | 0.7177 | 1.0948 | 0.0688 | 0.7967 |
| 60 | 1.0489 | n/a | 0.8496 | 1.1975 | 0.0747 | 0.8701 |
| 90 | 0.9552 | 0.9552 | 0.7368 | 0.9144 | 0.0530 | 0.6465 |
| 123 | 1.0226 | n/a | 0.8248 | 1.2133 | 0.0756 | 0.8805 |
| 150 | 1.1144 | n/a | 0.8675 | 1.2429 | 0.0773 | 0.9013 |
| 173 | 1.0731 | n/a | 0.8197 | 1.0585 | 0.0648 | 0.7640 |

### 4.3 Ablation analysis

Under the fixed single-seed configuration, no ablation condition approached the frozen thresholds (Table 7, Figure 5): Glass's Δ for angular correspondence ranged from −0.0441 to 0.0461, and r from −0.2097 to 0.2203. The largest correlation (r = 0.2203, no-Q-supervision condition) must not be read as learned correspondence: that condition's Q head received no gradient and is a fixed random projection. Raising λ_Q tenfold, training the Q head alone, and replacing the equivariant network with a parameter-matched plain CNN did not produce correspondence; the parameter-matched control's architectural differences beyond equivariance mean it is not a definitive causal test of equivariance's contribution. Alternative spatial reductions of the rotation criterion gave 6 of 9 and 4 of 9 angles below threshold; neither reached seven. The six multi-scale reductions of the same dense prediction (Table 7) likewise ranged from −0.0039 to 0.0412 (Δ) and −0.0341 to 0.0147 (r), with rotation pass counts of 4, 4, 4, 4, 5, and 6 of 9; no scale met any criterion, though at every angle the full-field error (0.236–0.641) was lower than the center-pixel error (0.825–1.114); this is reported as an observation about a diagnostic reduction, not an explanation.

**Table 7. Ablation conditions (single seed) and multi-scale reductions; pooled held-out GlaS set, 2083 patches; diagnostic only.**

| Analysis | Glass's Δ (A) | r (B) | Mean D_Q (C) | Rotation angles below threshold (of 9) |
|---|---|---|---|---|
| Reference (frozen checkpoint) | 0.0412 | −0.0341 | 0.5092 | n/a |
| No Q supervision | −0.0441 | 0.2203 | 0.5157 | n/a |
| λ_Q = 10 | 0.0403 | −0.1195 | 0.5110 | n/a |
| Q-head only | 0.0112 | −0.2097 | 0.5111 | n/a |
| Non-equivariant, parameter-matched | 0.0461 | −0.0251 | 0.5436 | n/a |
| Full-field / fixed-mask reduction | n/a | n/a | n/a | 6 / 4 |
| Scale: center pixel | −0.0039 | 0.0147 | n/a | 4 |
| Scale: 3 × 3 | −0.0007 | 0.0125 | n/a | 4 |
| Scale: 5 × 5 | 0.0165 | 0.0057 | n/a | 4 |
| Scale: 9 × 9 | 0.0268 | 0.0067 | n/a | 4 |
| Scale: 17 × 17 | 0.0392 | 0.0098 | n/a | 5 |
| Scale: full field | 0.0412 | −0.0341 | n/a | 6 |

> **Figure 5.** Ablation conditions (left, center) and multi-scale reductions (right), with frozen thresholds as context. No condition or scale is ranked or preferred.

### 4.4 Multiscale diagnostics: S_DL distribution and gland/patch-size mismatch

The predicted order magnitude S_DL is concentrated near zero, in contrast with a well-spread S_anat (Table 8, Figure 7). On the pooled held-out set (n = 2083), S_DL has mean 0.0315 (SD 0.0197, median 0.0283, IQR 0.0271, range [0.0003, 0.1101]); 82.9% of predictions fall below 0.05 and 32.1% below 0.02. S_anat, on the same population, has mean 0.5104 (SD 0.2137, median 0.4986); only 0.05% falls below 0.05. This pattern holds in every subgroup examined (testA, testB, benign, malignant); S_anat subgroup means range only from 0.4918 to 0.5265, showing no comparable collapse. This is reported as an important diagnostic observation and a plausible contributor to the weak order-magnitude correlation and tensor similarity (Criteria B and C), not as an established, sole cause of the negative result (Section 5.2).

**Table 8. S_DL and S_anat descriptive statistics, pooled held-out GlaS and subgroups.**

| Population | n | Mean | SD | Median | P5–P95 | Frac < 0.05 |
|---|---|---|---|---|---|---|
| S_DL, pooled | 2083 | 0.0315 | 0.0197 | 0.0283 | 0.0061–0.0681 | 0.8286 |
| S_DL, testA | 1470 | 0.0287 | 0.0192 | 0.0248 | 0.0052–0.0660 | 0.8619 |
| S_DL, testB | 613 | 0.0384 | 0.0190 | 0.0365 | 0.0109–0.0726 | 0.7488 |
| S_DL, benign | 964 | 0.0328 | 0.0195 | 0.0297 | 0.0072–0.0695 | 0.8154 |
| S_DL, malignant | 1119 | 0.0304 | 0.0197 | 0.0271 | 0.0053–0.0665 | 0.8400 |
| S_anat, pooled | 2083 | 0.5104 | 0.2137 | 0.4986 | 0.1905–0.8550 | 0.0005 |
| S_anat, testA | 1470 | 0.5109 | 0.2215 | 0.5059 | 0.1817–0.8524 | 0.0007 |
| S_anat, testB | 613 | 0.5094 | 0.1939 | 0.4680 | 0.2561–0.8616 | 0.0000 |
| S_anat, benign | 964 | 0.4918 | 0.2256 | 0.4773 | 0.1678–0.8612 | 0.0000 |
| S_anat, malignant | 1119 | 0.5265 | 0.2017 | 0.5091 | 0.2319–0.8467 | 0.0009 |

Separately, among the 1084 valid gland instances used to construct Q_anat, gland pixel area relative to the 128 × 128 patch area has median 0.8828 (the median gland occupies 88.3% of a whole patch), and 45.0% of valid glands exceed the entire patch area outright; the approximate major-axis extent (4*sqrt(lambda_1)) has median 179.9 px, exceeding the 128 px patch edge for 79.3% of valid glands and the 96 px patch stride for 96.0%. These are descriptive truncation-risk proxies, not an exact per-patch truncation count (a gland whose extent exceeds 128 px is not guaranteed to be truncated in every patch containing part of it), but they indicate the network's 128 × 128 receptive field frequently cannot see the same spatial extent of a gland that Q_anat was computed from, a limitation of the current patch-based design (Section 5.3) rather than an established cause of the negative result.

> **Figure 7.** S_DL and S_anat patch-level distributions, pooled held-out GlaS (n = 2083), same axis range and bin width in both panels; diagnostic only.

### 4.5 Cross-dataset validation on CRAG and PANDA compatibility

The frozen checkpoint was applied to 211 valid CRAG images (171 train-folder, 40 test-folder; two train-folder images had no valid target pixels), 29,399 valid patches (Table 9, Figure 6b). Correspondence was not demonstrated: pooled Glass's Δ = −0.0797, r = −0.0937 (bootstrap 95% CI −0.2274 to 0.0495), mean D_Q = 0.4288. A secondary patch-level diagnostic on a deterministic 5000-patch subsample gave Glass's Δ = −0.0021, and rotation consistency on that subsample was 4 of 9 angles, with the same passing angles as Section 4.2. These results are consistent with, and not a separate confirmation of, the GlaS findings; CRAG and GlaS were never pooled. In 70 of 213 CRAG images a single gland instance covers more than 30% of the image, so gland-level orientation is a coarser quantity there.

**Table 9. CRAG image-level results (frozen model; GlaS thresholds for description only).**

| Population | n images | Glass's Δ (A) | Pearson r (B) | p (B) | Mean D_Q |
|---|---|---|---|---|---|
| CRAG train folder | 171 | −0.0974 | −0.0862 | 0.2623 | 0.4228 |
| CRAG test folder | 40 | −0.0166 | −0.1389 | 0.3928 | 0.4544 |
| CRAG pooled (within CRAG only) | 211 | −0.0797 | −0.0937 | 0.1752 | 0.4288 |

PANDA was assessed as a potential external validation dataset but was not used because its available Radboud masks encode semantic Gleason-pattern regions rather than individual gland instances (in the 10-case pilot, benign epithelium separated into 32–498 gland-sized connected components per case, and the largest same-class Gleason region ranged from 0.2 to 6244.3 times the same case's median benign-component area). Constructing gland boundaries from these fused regions would require assumptions the annotation protocol does not support; PANDA was therefore excluded from the present anatomical validation under the frozen compatibility gate (Figure S1, Supplementary Materials, plots the component-size ratios). This is a data/annotation-compatibility finding, not a model result.

### 4.6 Pathology association and robustness sensitivity analyses

Under Holm–Bonferroni correction across the four pathology tests frozen in advance, none was significant (all adjusted p = 1.0; Table 10, Figure 6a): Cohen's d = −0.1348 (S_DL vs. binary grade, raw p = 0.6641) and −0.1378 (φ-dispersion vs. binary grade, p = 0.5242); Spearman ρ = 0.0750 (S_DL vs. ordinal grade, p = 0.5086) and 0.0030 (φ-dispersion vs. ordinal grade, p = 0.9787). Morphology and an area-adjusted model were also null. A testB-only subgroup result (ρ = −0.6008, p = 0.0051, n = 20, only 4 benign) was not treated as evidentiary, being split-specific and contradicted by the pooled analysis. The 80 images derive from 12 patients (three contributing both grades), so a formal patient-level pathology test was not possible and is inconclusive.

**Table 10. Image-level pathology and morphology analyses (GlaS, n = 80 images).**

| Analysis | Effect size | 95% CI | Raw p | Holm-adjusted p |
|---|---|---|---|---|
| S_DL vs. binary grade | Cohen's d = −0.1348 | not computed for d | 0.6641 | 1.0 |
| S_DL vs. ordinal grade | Spearman ρ = 0.0750 | −0.1361 to 0.2768 | 0.5086 | 1.0 |
| φ-dispersion vs. binary grade | Cohen's d = −0.1378 | not computed for d | 0.5242 | 1.0 |
| φ-dispersion vs. ordinal grade | Spearman ρ = 0.0030 | −0.2179 to 0.2314 | 0.9787 | 1.0 |
| S_DL vs. gland area | ρ = 0.0311 | n/a | 0.7845 | not in family |
| S_DL vs. gland count | ρ = 0.1743 | n/a | 0.122 | not in family |

> **Figure 6.** (a) Image-level pathology tests on GlaS; Holm-adjusted p = 1.0 for all. (b) CRAG image-level Glass's Δ by population, with the GlaS threshold as context; datasets shown separately, never pooled.

Two robustness analyses tested whether the primary result depends on a single training run. **Five-seed robustness** (Table 11): Criterion A's Glass's Δ was 0.0512 ± 0.0134 across seeds (range 0.0349–0.0690), Criterion B's r was −0.0277 ± 0.0335 (range −0.0745 to 0.0081), Criterion C's Glass's Δ was 0.0039 ± 0.0030 (range 0.0002–0.0073), and Criterion D's pass count was 4.60 ± 0.55 of 9 (range 4–5); classification AUROC was 0.8572 ± 0.0250 (range 0.8317–0.8921), confirming the classification head learns a real signal in every seed while the Q-tensor criteria do not, so the negative geometric result is not attributable to a general failure to learn from the data. None of the five seeds met any criterion (0 of the 20 possible seed-criterion combinations met). Best-epoch values ranged from 19 to 40 (mean 31.2, SD 8.23) under the identical, fixed protocol for every seed; this reflects stochastic optimization variation, not evidence of instability, since every seed failed the same criteria regardless of epoch count.

**Table 11. Five-seed robustness: Criteria A–D and classification AUROC, pooled testA+testB (2083 patches each).**

| Seed | Epochs | Glass's Δ (A) | r (B) | Glass's Δ (C) | Rotation passed (D, of 9) | AUROC | Criteria met (of 4) |
|---|---|---|---|---|---|---|---|
| 11 | 28 | 0.0548 | 0.0012 | 0.0034 | 5 | 0.8365 | 0 |
| 22 | 40 | 0.0562 | −0.0745 | 0.0063 | 5 | 0.8921 | 0 |
| 33 | 32 | 0.0690 | −0.0395 | 0.0021 | 5 | 0.8545 | 0 |
| 42 (frozen checkpoint) | 37 | 0.0412 | −0.0341 | 0.0073 | 4 | 0.8711 | 0 |
| 55 | 19 | 0.0349 | 0.0081 | 0.0002 | 4 | 0.8317 | 0 |

**Patient-disjoint sensitivity** (Table 12): on the single-seed patient-disjoint partition (85 held-out images, 2143 patches), 0 of 4 criteria were met (Glass's Δ = 0.0603, r = 0.1150, Glass's Δ = 0.0028, 4 of 9 angles), consistent with the canonical-split finding: removing patient overlap by construction did not produce the correspondence the overlapping canonical split also failed to show, so patient overlap is not the explanation for the negative geometric result. Extending this protocol to all five seeds gave 0 of 20 possible criterion–seed combinations met; no seed met all four criteria, and no seed met any single one. Patient-disjoint classification AUROC was 0.6425 ± 0.0156 across the five seeds, substantially lower than the canonical split's 0.8572 ± 0.0250. This difference is reported as evidence that classification performance is sensitive to split structure; it is not evidence that patient overlap caused the difference, nor evidence of data leakage: the patient-disjoint partition also differs from the canonical split in training-set size and image composition, and no analysis here isolates patient overlap as the specific mechanism.

**Table 12. Patient-disjoint sensitivity, five seeds, pooled held-out patient-disjoint population (85 images, 2143 patches per seed).**

| Seed | Epochs | Glass's Δ (A) | r (B) | Glass's Δ (C) | D passed (of 9) | Accuracy | AUROC | Criteria (of 4) |
|---|---|---|---|---|---|---|---|---|
| 11 | 40 | 0.0349 | 0.1057 | 0.0012 | 5 | 0.5441 | 0.6222 | 0 |
| 22 | 40 | 0.0231 | 0.0817 | 0.0016 | 6 | 0.5329 | 0.6496 | 0 |
| 33 | 40 | 0.0559 | 0.0189 | 0.0025 | 5 | 0.5478 | 0.6640 | 0 |
| 42 | 40 | 0.0603 | 0.1150 | 0.0028 | 4 | 0.5604 | 0.6360 | 0 |
| 55 | 40 | 0.0332 | 0.0147 | 0.0015 | 6 | 0.5520 | 0.6408 | 0 |

### 4.7 Overall evidence synthesis

Table 13 summarizes every analysis. The same pattern holds throughout: effect sizes near zero for anatomical correspondence, rotation consistency below the frozen requirement in every configuration tested, and no supporting evidence from pathology or from an external dataset. "Not met" and "not demonstrated" both mean an analysis did not provide supporting evidence for the tested criterion or association.

**Table 13. Evidence summary by analysis (descriptive labels; no composite score).**

| Analysis | Unit, n | Outcome |
|---|---|---|
| Criteria A, B, C (GlaS) | patch, 2083 (80 images) | not met |
| Criterion D (GlaS) | patch, 2083 | not met (4 of 9, in every alternative metric tested) |
| Model 2 permutation null | patch, 1470/613 | inconclusive; Model 2 only |
| Ablation conditions | patch, 2083 | no condition met the criteria |
| Multi-scale reductions | patch, 2083 | no scale met the criteria |
| Pathology association | image, 80 | not demonstrated |
| Five-seed robustness | patch, 2083 × 5 seeds | no seed met any criterion |
| Patient-disjoint, single + five-seed | image, 85 (× 5 seeds) | not met (0 of 4; 0 of 20) |
| S_DL / S_anat distribution | patch, 2083 | S_DL concentrated near zero; S_anat well spread |
| Gland/patch-size assessment | gland, 1084 | 79% of valid glands estimated larger than the 128 px patch |
| CRAG (partially independent) | image, 211 | correspondence not demonstrated |
| PANDA | case, 10 (pilot) | infeasible under the frozen definition |

---

## 5. Discussion

### 5.1 Main findings

The tested formulation did not demonstrate the intended anatomically supervised orientation-tensor learning objective under the frozen evaluation criteria. Angular correspondence, order-magnitude correlation, and tensor similarity were all far from their requirements, and rotation consistency reached 4 of 9 angles against 7 required. Single-seed ablations, six spatial aggregations, an external dataset, five independent random seeds, and a patient-disjoint data partition did not change this. The pathology analysis showed no association, and PANDA could not be used for the planned target given its annotation semantics. The study does document execution integrity: the rank-2 representation is encoded explicitly and its transformation law holds exactly for the discrete group elements; the anatomical reference is well defined and structurally separated from the image pathway (leakage tests passed, including a poisoned-mask test); the protocol is able to expose disagreement between a learned output and a reference, and it did. None of this is evidence that the learned field corresponds to the reference.

### 5.2 Interpretation

The primary effect sizes are not merely below threshold but close to zero (0.0412, −0.0341, 0.0073), and stayed close to zero across every ablation condition, aggregation scale, alternative seed, and data partition tested. The small p-values accompanying Criteria A and C reflect the large number of overlapping patches, not agreement. Architectural equivariance therefore does not automatically imply that a network will learn an anatomically meaningful orientation field from limited histopathology supervision; this is an interpretation of the present result, not a universal claim about equivariant learning.

The experiments do not establish a single mechanism for the negative result. Plausible contributors include the limited training-cohort size (85 images), the mismatch between patch-level model input and full-gland anatomical targets (Section 4.4: a majority of valid glands exceed the patch's spatial extent), ambiguity or ceiling effects in the mask-derived reference, the near-zero concentration of predicted order magnitude (Section 4.4), representation capacity, and finite-grid rotation/interpolation effects. These remain hypotheses, not established causal explanations: no ablation in this study isolates any one of them as an independent variable, and none is claimed as the definitive cause. Three specific alternative explanations were tested directly and ruled out as sufficient on their own: an unrepresentative single initialization (four additional seeds reproduced the negative result); patient overlap in the canonical split (a patient-disjoint partition, across the same five seeds, reproduced it too); and the rotation-measurement design itself (four alternative metrics gave the same qualitative outcome). None of these tests converted the negative result into a positive one.

### 5.3 Limitations

The GlaS held-out set is the canonical image-level split, not patient-independent (11 of 12 held-out patients also contribute training images); the patient-disjoint sensitivity analysis addressed this directly and does not change the primary conclusion, but is itself a single, non-canonical alternative partition, not a definitive ruling-out of every split-dependent effect. The 2083 (and 2143) evaluated patches overlap and are not independent biological units, so patch-level p-values and intervals are descriptive throughout. The mask-derived reference is a reference, not absolute ground truth, and is not independent of the training objective; it describes gland elongation specifically, not necessarily broader tissue orientation. Training operates on dense per-cell (q₁, q₂) fields, while the frozen criteria evaluate patch-level masked-mean summaries, a summarization choice that was not shown to be optimal. The rotation test is a finite-grid, pixel-domain measurement with interpolation at all but one angle, sampled at a single pixel offset from the true rotation center; a full-field alternative gave the same qualitative outcome under the same thresholds. The patch-level input versus full-gland anatomical supervision mismatch (Section 4.4) means the target may encode morphology extending beyond the model's receptive field. CRAG's independence from GlaS is partial (shared hospital and research group), and its license terms were not independently verified. PANDA's semantic annotation prevented constructing the planned target from only a 10-case pilot. Only 12 distinct patients contribute the 80 GlaS held-out images, limiting the pathology analysis to an inconclusive, non-patient-level test. The parameter-matched non-equivariant ablation cannot isolate equivariance as a single independent variable; the Model 2 permutation null used 100 permutations and one seed; and this study tested one target definition, one architecture family, and one training protocol.

### 5.4 Implications

The study offers a template for evaluating a neural orientation-tensor output against an independently constructed, mask-derived reference under criteria frozen internally in advance, with controls for supervision weight and removal, architecture family, and spatial aggregation, and a full account of measurement limits. The negative outcome should not be over-read: it does not establish that orientation information is absent from histopathology, that equivariant learning is ineffective there, that rank-2 representations are unsuitable, that the anatomical reference is wrong, or that nematic order has no biological relevance in this setting. It shows that, for one reference definition, one architecture family, and the frozen criteria, the required correspondence and rotation consistency were not demonstrated, and, more generally, that a model can satisfy an architectural symmetry constraint by construction while still failing to recover the intended anatomical quantity, motivating explicit anatomical validation rather than reliance on architectural guarantees alone in geometric medical-imaging studies of this kind. Such validation pipelines are, more broadly, one instance of the wider integration of digital and cyber-physical infrastructure into healthcare and diagnostic workflows [20].

---

## 6. Conclusions

This study developed and evaluated an anatomically supervised, rank-2 rotation-equivariant orientation-tensor framework for histopathology, using a network whose Q-tensor output is explicitly typed to the D8 dihedral group and validated against an independently constructed, mask-derived anatomical reference. The predefined anatomical-correspondence and empirical rotation-consistency criteria were not supported on the primary GlaS evaluation, and the negative result reproduced across single-seed ablations, six spatial aggregations, five independent training seeds, a patient-disjoint data partition (also across five seeds), five rotation-consistency metric designs, and an external cross-dataset evaluation on CRAG; no pathology association was demonstrated, and PANDA was incompatible with the target construction under its annotation semantics.

Architectural equivariance (an exact transformation law for the typed representation, verified to floating-point precision for the implemented discrete group) did not by itself yield a network that recovered the intended anatomical orientation field under the tested formulation. Several plausible contributors to this outcome were identified, including the mismatch between patch-level model input and full-gland anatomical targets and a marked concentration of predicted order magnitude near zero, but no single mechanism was established as the definitive cause. These findings indicate that architectural equivariance alone is insufficient evidence that a learned orientation tensor represents the intended anatomical structure, and motivate explicit, independently referenced anatomical validation as a necessary component of geometric medical-imaging studies of this kind.

---

**Funding:** This research received no external funding.

**Data Availability Statement:** The datasets analyzed in this study are publicly available from their respective original repositories or public data platforms. Dataset-specific access conditions, licensing terms, and citation requirements apply. The authors did not generate or collect the underlying datasets.

**Code Availability Statement:** The code developed for this study is available at https://github.com/Islam-DS/nematic-orientation-histopathology.

**Ethics Statement:** This study used publicly available datasets (GlaS, CRAG, and PANDA) and involved no new recruitment or collection of human data by the authors. Dataset-specific ethical approvals, consent procedures, and governance requirements were those established by the respective original data providers. For PANDA, the original publication reports institutional review board or ethics committee approvals and, for the Radboud data used here, a waiver of informed consent for the retrospective use of de-identified specimens [13]. Corresponding ethics documentation for GlaS and CRAG was not available in the sources consulted by the authors.

**AI Use Disclosure:** AI-assisted tools (Claude Code, Anthropic) were used during software development, computational workflow execution and auditing, manuscript drafting, and LaTeX conversion. The authors reviewed and verified the resulting code, analyses, scientific content, numerical results, and manuscript before finalization and remain responsible for the content of the manuscript.

**Author Contributions:** Md. Mahbubul Islam: Conceptualization, methodology, software, data curation, formal analysis, investigation, validation, visualization, writing – original draft, and writing – review and editing. Ainur Yerkos: Supervision, conceptual guidance, methodology review, and writing – review and editing. Nadezhda Kunicina: Supervision, research guidance, and writing – review and editing. All authors have read and agreed to the published version of the manuscript.

**Conflicts of Interest:** The authors declare no conflicts of interest.

**Supplementary Materials:** The following supporting information is provided as a separate document: Synthetic Orientation-Tensor Validation (Phase 1R), a recovery of the project's existing synthetic-benchmark evidence, demonstrating that the framework can be learned under controlled, noise-free geometry; it does not validate the histopathology result reported in this manuscript (see the supplementary document's own interpretation section). Figure S1: the PANDA component-size ratios referenced in Section 4.5 (largest connected same-class region relative to the same-case benign median component size, log scale, for the pilot cases that contain Gleason 3, 4, or 5 and a benign reference; nine bars from six of the ten cases).

---

## References

1. Cohen, T.S.; Welling, M. Group Equivariant Convolutional Networks. In Proceedings of the 33rd International Conference on Machine Learning (ICML 2016); Proceedings of Machine Learning Research, Volume 48; PMLR: 2016; pp. 2990–2999.
2. Weiler, M.; Cesa, G. General E(2)-Equivariant Steerable CNNs. In Advances in Neural Information Processing Systems 32 (NeurIPS 2019); 2019; arXiv:1911.08251.
3. Cesa, G.; Lang, L.; Weiler, M. A Program to Build E(N)-Equivariant Steerable CNNs. In Proceedings of the International Conference on Learning Representations (ICLR 2022); 2022. https://github.com/QUVA-Lab/escnn.
4. Veeling, B.S.; Linmans, J.; Winkens, J.; Cohen, T.; Welling, M. Rotation Equivariant CNNs for Digital Pathology. In Medical Image Computing and Computer Assisted Intervention – MICCAI 2018; Lecture Notes in Computer Science; Springer: 2018; pp. 210–218. https://doi.org/10.1007/978-3-030-00934-2_24.
5. Lafarge, M.W.; Bekkers, E.J.; Pluim, J.P.W.; Duits, R.; Veta, M. Roto-Translation Equivariant Convolutional Networks: Application to Histopathology Image Analysis. Med. Image Anal. 2021, 68, 101849. https://doi.org/10.1016/j.media.2020.101849.
6. de Gennes, P.G.; Prost, J. The Physics of Liquid Crystals, 2nd ed.; Oxford University Press: Oxford, UK, 1993. https://doi.org/10.1093/oso/9780198520245.001.0001.
7. Sirinukunwattana, K.; Pluim, J.P.W.; Chen, H.; Qi, X.; Heng, P.-A.; Guo, Y.B.; Wang, L.Y.; Matuszewski, B.J.; Bruni, E.; Sanchez, U.; Böhm, A.; Ronneberger, O.; Ben Cheikh, B.; Racoceanu, D.; Kainz, P.; Pfeiffer, M.; Urschler, M.; Snead, D.R.J.; Rajpoot, N.M. Gland Segmentation in Colon Histology Images: The GlaS Challenge Contest. Med. Image Anal. 2017, 35, 489–502. https://doi.org/10.1016/j.media.2016.08.008.
8. Graham, S.; Chen, H.; Gamper, J.; Dou, Q.; Heng, P.-A.; Snead, D.; Tsang, Y.W.; Rajpoot, N. MILD-Net: Minimal Information Loss Dilated Network for Gland Instance Segmentation in Colon Histology Images. Med. Image Anal. 2019, 52, 199–211. https://doi.org/10.1016/j.media.2018.12.001.
9. Navarro, J.; Wilkinson, M. On the Equivariant Learning of the Q-tensor Order Parameter. arXiv 2026, arXiv:2605.27679.
10. Alblas, D.; Suk, J.; Brune, C.; Yeung, K.K.; Wolterink, J.M. SIRE: Scale-Invariant, Rotation-Equivariant Estimation of Artery Orientations Using Graph Neural Networks. arXiv 2023, arXiv:2311.05400.
11. Snoussi, H.; Karimi, D. Equivariant Spherical CNNs for Accurate Fiber Orientation Distribution Estimation in Neonatal Diffusion MRI with Reduced Acquisition Time. arXiv 2025, arXiv:2504.01925.
12. Sirinukunwattana, K.; Snead, D.R.J.; Rajpoot, N.M. A Stochastic Polygons Model for Glandular Structures in Colon Histology Images. IEEE Trans. Med. Imaging 2015, 34, 2366–2378. https://doi.org/10.1109/TMI.2015.2433900.
13. Bulten, W.; Kartasalo, K.; Chen, P.-H.C.; Ström, P.; Pinckaers, H.; Nagpal, K.; Cai, Y.; Steiner, D.F.; van Boven, H.; Vink, R.; Hulsbergen-van de Kaa, C.; van der Laak, J.; Amin, M.B.; Evans, A.J.; van der Kwast, T.; Allan, R.; Humphrey, P.A.; Grönberg, H.; Samaratunga, H.; Delahunt, B.; Tsuzuki, T.; Häkkinen, T.; Egevad, L.; Demkin, M.; Dane, S.; Tan, F.; Valkonen, M.; Corrado, G.S.; Peng, L.; Mermel, C.H.; Ruusuvuori, P.; Litjens, G.; Eklund, M.; the PANDA challenge consortium. Artificial Intelligence for Diagnosis and Gleason Grading of Prostate Cancer: The PANDA Challenge. Nat. Med. 2022, 28, 154–163. https://doi.org/10.1038/s41591-021-01620-2.
14. Cohen, J. Statistical Power Analysis for the Behavioral Sciences, 2nd ed.; Lawrence Erlbaum Associates: Hillsdale, NJ, USA, 1988.
15. Bigun, J.; Granlund, G.H. Optimal Orientation Detection of Linear Symmetry. In Proceedings of the 1st International Conference on Computer Vision (ICCV 1987); IEEE Computer Society Press: Washington, DC, USA, 1987; pp. 433–438.
16. Provenzano, P.P.; Eliceiri, K.W.; Campbell, J.M.; Inman, D.R.; White, J.G.; Keely, P.J. Collagen Reorganization at the Tumor-Stromal Interface Facilitates Local Invasion. BMC Med. 2006, 4, 38. https://doi.org/10.1186/1741-7015-4-38.
17. Scholich, A.; Syga, S.; Morales-Navarrete, H.; Segovia-Miranda, F.; Nonaka, H.; Meyer, K.; de Back, W.; Brusch, L.; Kalaidzidis, Y.; Zerial, M.; Jülicher, F.; Friedrich, B.M. Quantification of Nematic Cell Polarity in Three-Dimensional Tissues. PLoS Comput. Biol. 2020, 16, e1008412. https://doi.org/10.1371/journal.pcbi.1008412.
18. Ji, M.-Y.; Yuan, L.; Lu, S.-M.; Gao, M.-T.; Zeng, Z.; Zhan, N.; Ding, Y.-J.; Liu, Z.-R.; Huang, P.-X.; Lu, C.; Dong, W.-G. Glandular Orientation and Shape Determined by Computational Pathology Could Identify Aggressive Tumor for Early Colon Carcinoma: A Triple-Center Study. J. Transl. Med. 2020, 18, 129. https://doi.org/10.1186/s12967-020-02297-w.
19. Lee, G.; Sparks, R.; Ali, S.; Shih, N.N.C.; Feldman, M.D.; Spangler, E.; Rebbeck, T.; Tomaszewski, J.E.; Madabhushi, A. Co-Occurring Gland Angularity in Localized Subgraphs: Predicting Biochemical Recurrence in Intermediate-Risk Prostate Cancer Patients. PLoS ONE 2014, 9, e97954. https://doi.org/10.1371/journal.pone.0097954.
20. Skorobogatjko, A.; Romanovs, A.; Kunicina, N. State of the Art in the Healthcare Cyber-Physical Systems. Inf. Technol. Manag. Sci. 2014, 17. https://doi.org/10.1515/itms-2014-0019.
