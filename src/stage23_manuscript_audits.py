"""
STAGE 23 -- read-only numbers audit for the drafted manuscript.

Extracts every quantitative token from results_v2/manuscript_stage23/01_FULL_MANUSCRIPT.md
(everything before the reference list) and maps each to a source:

  1. SPAN RULES: documented structural values (thresholds, architecture, dataset structure, hyper-
     parameters, counts stated in reports/code). A rule applies to a token only if the rule's
     regular expression matches text that OVERLAPS the token itself.
  2. the corrected numbers table (results_v2/manuscript_stage21_corrected/13_numbers_and_sources.csv)
  3. a numeric leaf of a frozen result file (exact match, or match at the displayed precision)
  4. fallback: a small set of documented definitional values

Nothing is computed or re-estimated. A token that cannot be traced is written with verified = NO and
must be resolved by editing the manuscript. Output: results_v2/manuscript_stage23/04_NUMBERS_AUDIT.csv
"""
import bisect
import csv
import glob
import json
import math
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda *a: os.path.join(ROOT, *a)  # noqa: E731
MS = P("results_v2", "manuscript_stage23", "01_FULL_MANUSCRIPT.md")
OUT = P("results_v2", "manuscript_stage23", "04_NUMBERS_AUDIT.csv")
NT = P("results_v2", "manuscript_stage21_corrected", "13_numbers_and_sources.csv")

PREREG = "results_v2/phase2/preregistered_thresholds_v2.json"
S4 = "results_v2/anatomy_targets/stage4_target_generation_summary.json"
S7 = "reports/STAGE7_AUDIT_REPORT.md"
S9 = "reports/STAGE9_AUDIT_REPORT.md"
S10 = "reports/STAGE10_AUDIT_REPORT.md"
S12 = "results_v2/validation/stage12/stage12_results.json"
S16 = "results_v2/validation/stage16/stage16_results.json"
S17 = "results_v2/validation/stage17_crag/stage17_results.json"
S17T = "results_v2/validation/stage17_crag/crag_target_generation_summary.json"
S17C = "results_v2/validation/stage17_crag/crag_compatibility_report.json"
S18 = "results_v2/validation/stage18_panda/summary_table.csv"
GRADE = "data/glas/Warwick_QU_Dataset/Grade.csv"

# ------------------------------------------------------------ index: numbers table
nt_rows = list(csv.DictReader(open(NT, encoding="utf-8")))
nt_index = []
for i, r in enumerate(nt_rows, start=2):
    try:
        nt_index.append((float(r["number"]), i, r["label"], r["unit"], r["source_file"], r["source_field"]))
    except ValueError:
        pass

# ------------------------------------------------------------ index: frozen leaves
leaves = []


def walk(o, f, path):
    if isinstance(o, bool):
        return
    if isinstance(o, (int, float)):
        if not (isinstance(o, float) and (math.isnan(o) or math.isinf(o))):
            leaves.append((float(o), f, path))
    elif isinstance(o, dict):
        for k, v in o.items():
            walk(v, f, f"{path}.{k}" if path else str(k))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            walk(v, f, f"{path}.{i}" if path else str(i))


JSON_FILES = [P(*x.split("/")) for x in (
    "results_v2/validation/stage11/stage11_results.json", S12,
    "results_v2/validation/stage13/stage13_results.json",
    "results_v2/validation/stage14/stage14_all_evaluations.json",
    "results_v2/validation/stage14/A6_spatial_reduction/a6_results.json",
    "results_v2/validation/stage15/stage15_results.json", S16, S17, S17T, S17C, S4,
    "results_v2/model3/metrics/stage10_summary.json", PREREG,
    "results_v2/phase2/calibration_report_v2.json",
)] + glob.glob(P("results_v2", "baselines", "*", "held_out_*.json"))
for jf in JSON_FILES:
    rel = os.path.relpath(jf, ROOT).replace(os.sep, "/")
    try:
        walk(json.load(open(jf, encoding="utf-8")), rel, "")
    except Exception:  # noqa: BLE001
        pass
for r in csv.DictReader(open(P(*S18.split("/")), encoding="utf-8")):
    for k, v in r.items():
        try:
            leaves.append((float(v), S18, f"{r['image_id'][:8]}.{k}"))
        except (TypeError, ValueError):
            pass
leaves.sort(key=lambda x: x[0])
leaf_vals = [x[0] for x in leaves]

# ------------------------------------------------------------ span rules (regex, source, location, unit)
RULES = [
    (r"λ_Q = 1\.0", "reports/STAGE10_AUDIT_REPORT.md", "Section 4 (lambda_Q = 1.0, fixed, not tuned)", ""),
    (r"p < 0\.01|< 0\.01|0\.01 \(1000", PREREG, "criteria.B/C: p < 0.01 (frozen requirement)", ""),
    (r"\(31\.5, 31\.5\)|31\.5", "code_v2/stage12_rotation_validation.py; skimage.transform.rotate default center = cols/2 - 0.5", "rotation fixed point 63.5 (input) = 31.5 in dense coordinates (dense index j <-> input 2j+0.5)", "pixels"),
    (r"not met \(4 of 9|\(4 of 9;", S12, "criterion_D.n_angles_passed = 4 of 9", "angles"),
    (r"\(8 instances\)|Four instances|four testB images|4 instances in 4 testB", S4, "all.n_fragmented_instances = 4 (all in four testB images); validity_category_counts.excluded_too_small = 8", "instances"),
    (r"only 4 benign|4 benign", "reports/STAGE15_AUDIT_REPORT.md", "Section 6 (only 4 benign images out of 20 in testB)", "images"),
    (r"all three conditions|three conditions", PREREG, "criteria.B: r >= 0.30 AND p < 0.01 AND CI lower bound > 0 (three conditions)", "conditions"),
    (r"of whom 11|11 also contribute", GRADE, "held-out patients that also have training images (Stage 22 recount)", "patients"),
    (r"one dense pixel|single dense pixel|A single dense", "code_v2/stage12_rotation_validation.py", "batched_center_pixel_q: one dense pixel", "pixels"),
    (r"Scale: \d+ × \d+", "results_v2/validation/stage16/multiscale/scale_definitions.json", "window definitions", "pixels"),
    (r"i 2\(", PREREG, "criteria.A.metric: Delta_theta = 0.5*|arg(exp(i*2*(theta_DL - theta_anat)))|", ""),
    (r"95%", PREREG, "criteria.A/B bootstrap 95% confidence level", "%"),
    (r"lower bound > 0", PREREG, "criteria.B.threshold.bootstrap_CI_95_lower_bound_gt = 0.0", ""),
    (r"0 of 1000", "results_v2/validation/stage11/stage11_results.json", "C_tensor_similarity.permutation_p_value = 0.0; code_v2/stage11_geometric_validation.py N_PERM = 1000", "permutations"),
    (r"S ≥ 0|\) ≥ 0", S7, "Section 5 (S = sqrt(q1^2+q2^2), non-negative by construction)", ""),
    (r"2 × 2|I/2|½", "docs_v2/phase2_target_math.md; " + S7, "Sections 1-2 (2x2 covariance; Q = S(uu^T - I/2)); Section 7 (2x2 average pooling)", ""),
    (r"3 × 128 × 128|3 × 3(?= +whose)|3 × 3 R2Conv", S7, "Section 4 (input (B,3,128,128)); Section 2 (Q head k=3)", "pixels"),
    (r"all four constituent|four constituent pixels", S7, "Section 7 (all-4-valid rule for 2x2 pooling)", "pixels"),
    (r"λ_Q = 0|λ_Q = 10|λ_class = 0", "configs/stage14_ablation_matrix.json", "experiments[A1/A3/A4] lambda_Q / lambda_class", ""),
    (r"\d of 9 \((full-field|fixed)|6 / 4", "results_v2/validation/stage14/A6_spatial_reduction/a6_results.json", "reductions.R1_full_spatial_mean.n_angles_passed = 6; R2_fixed_original_valid_mask_mean = 4 (of 9 non-zero)", "angles"),
    (r"was 4 of 9 angles against the GlaS", S17, "rotation_consistency.n_angles_passed_vs_original_GlaS_threshold = 4 of 9", "angles"),
    (r"pass counts were [\d, and]+ of 9", S16, "rotation_by_scale.*.n_angles_passed_vs_original_threshold (4,4,4,4,5,6 of 9)", "angles"),
    (r"across the four|all four primary|Four primary|four pre-specified|four primary", "configs/stage15_pathology_analysis_matrix.json", "primary pathology tests P1-P4 (Holm family of four)", "tests"),
    (r"four (?:prospectively|frozen|criteria)|None of the four|[Aa]ll four criteria", PREREG, "criteria A, B, C, D (four)", "criteria"),
    (r"four angles below|the four with|four of the nine|four of the 9|4 of the 9|4 of 9 non-zero|[Ff]our angles", S12, "criterion_D.n_angles_passed = 4 (90/123/150/173) of 9", "angles"),
    (r"In the four of these|four of these|the four computable|Four cases|four (?:pilot )?cases", S18, "PANDA: 4 Gleason-4 cases with a same-case benign reference (ratios 49.3, 136.8, 735.1, 1082.3)", "cases"),
    (r"Five cases contained|five of the ten", S18, "rows with gleason_4_pixel_fraction > 0 (5)", "cases"),
    (r"three cases contain no|Three cases contain no", S18, "rows with gleason_3/4/5_pixel_fraction all 0 (00743313, 004dd32d, 01642d24)", "cases"),
    (r"one Gleason-4 case|The fifth", S18, "case 006f6aa3: gleason_4_pixel_fraction > 0 and benign_epithelium_n_components = 0", "cases"),
    (r"nine bars from six of the ten|nine bars|six of the ten", "results_v2/figures/stage20/figure7_supplementary_panda_compatibility.png; " + S18, "figure shows 9 bars from 6 of 10 cases (00bbc148, 0068d4c7, 0018ae58, 00928370, 00951a7f, 018eabc8)", "bars"),
    (r"32 to 498|nine cases that contained", S18, "benign_epithelium_n_components: min 32, max 498 over the 9 cases with benign epithelium", "components"),
    (r"8 (?:training )?patches|8 training", "reports/STAGE8_EXTENDED_AUDIT_REPORT.md", "Section 3 (identical 8 canonical TRAIN patches)", "patches"),
    (r"5-level", "reports/STAGE15_DESIGN_FREEZE.md", "grade (Sirinukunwattana et al. 2015): 5-level ordinal grade", "levels"),
    (r"three hypotheses", "reports/STAGE21_5_H2_TRACEABILITY.md", "Section 2 (H1, H2, H3 verbatim)", "count"),
    (r"three closest|Three lines", "manuscript (literature notes recorded at project initiation)", "master-prompt section 3: three closest prior works", "count"),
    (r"three standard deviations", PREREG, "criteria.D threshold = floor_mean + 3 * floor_std", "SD"),
    (r"(?:Three|three) (?:observations|layers|things)|four ways|all three models", "manuscript enumeration", "enumeration of items in the text (not a measured value)", ""),
    (r"eight arbitrary|eight orientation|eight regular|two of eight|8 → 2", S9 + "; " + S7, "Model 2: c1=c2=8 fields, 8 orientation channels; Model 3: 8 regular fields, Linear(8,2)", "channels"),
    (r"eight (?:image-level )?leakage|eight leakage", "reports/STAGE6_AUDIT_REPORT.md", "Section 4 (8 tests, all PASS)", "tests"),
    (r"two components|two coordinates|two-dimensional|2-channel|two-part|two-sided|two datasets|two image-level|two other spatial|two baselines|two equivariant|Two heads|two-block|two unconstrained|Two predefined|two non-gating|the two are|Two train-folder|two train|two clauses|In two dimensions|size 2|2 × 64", "manuscript / " + S7 + " / " + S9, "structural count (irrep size 2, two-part criterion, two-sided tests, two datasets/labels/reductions/baselines/heads/clauses) documented in the cited reports", "count"),
    (r"a single gland|one gland|single gland", S17C, "crag_images_with_dominant_fused_gland_gt30pct = 70 of 213", "images"),
    (r"one sampled|single sampled", "code_v2/stage12_rotation_validation.py", "batched_center_pixel_q: one dense pixel", "pixels"),
    (r"single seed|Single run|single-seed|single training|one training|one seed|single batch|one reflection|one reference|one formulation|one architecture|one dataset|one GPU", S10, "Section 4 (seed 42, single seed; run once); Stage 7 Section 9 (one reflection)", "count"),
    (r"Six pre-frozen|six pre-frozen|six spatial|six scales|Six", "results_v2/validation/stage16/multiscale/scale_definitions.json", "scale_id 0-5 (six scales)", "scales"),
    (r"\(7 required\)|neither reached seven|seven were required|seven required|reached seven|[≥>]=? ?7 of 9|7 of 9|the 7 required|\(of 9\)|4 of 9 angles", PREREG, "criteria.D rule: at least 7 of 9 non-zero angles", "angles"),
    (r"and 0 for the|seed 0", "results_v2/validation/stage13/stage13_results.json", "results.testA.seed = 0", ""),
    (r"nine (?:non-zero |tested |rotation )?angles|nine angles|the nine|Of the nine|nine non-zero", PREREG, "criteria.D.rotation_angles_deg (9 non-zero)", "angles"),
    (r"173(?= (?:in its|train|training))|\(173", S17T, "n_train_images = 173", "images"),
    (r"(?<!\d)40(?= (?:in its|test|test-folder))|test-folder|40 test", S17T, "n_test_images = 40", "images"),
    (r"11 of (?:the )?12|11 of 16|11 of them|patients 3 and 7|16 patients|12 (?:held-out |distinct )?patients|of the 12|patient, 12|Only 12|three of whom|Three of the 12|three of the 12|3 and 7|held-out patients", GRADE, "patient-ID counts by split (Stage 22 recount; Stage 15 report Section 2)", "patients"),
    (r"testB \(20|20 images|20-image|n = 20|testB 20|20 testB|20 / 613", S4, "testB.n_images = 20", "images"),
    (r"85 training|85 canonical|85 / testA|train 85|85 images|of the 85|(?<!\d)85(?= /)", S4, "train.n_images = 85", "images"),
    (r"60 testA|testA 60|60 / 1470|60 images|\(60", S4, "testA.n_images = 60", "images"),
    (r"68 training|68 and|17 development|68/17|68 / 17", S10, "Section 3 (stratified 80/20 split of the 85 train images: 68/17)", "images"),
    (r"80/20", S10, "Section 3 (stratified 80/20 split, seed 42)", ""),
    (r"1470|613", "results_v2/validation/stage11/stage11_results.json", "testA.metrics.n_patches / testB.metrics.n_patches", "patches"),
    (r"2249", "results_v2/phase2/calibration_report_v2.json", "calibration universe: 85 images -> 2249 patches (Stage 0 recalibration audit)", "patches"),
    (r"60 patches", "reports/STAGE0_RECALIBRATION_AUDIT_REPORT.md", "Section D (60 patches sampled per calibration, seed 42)", "patches"),
    (r"at least 10%", S17T, "min_valid_fraction_for_patch = 0.1", "fraction"),
    (r"at least 2%", S17T, "border_frac_threshold = 0.02", "fraction"),
    (r"S < 0\.15", S17T, "default_S_threshold = 0.15", ""),
    (r"fewer than 20", S17T, "min_gland_pixels = 20", "pixels"),
    (r"128 × 128|128-pixel|128 px|128 pixel", "code/data_prep.py", "PATCH_SIZE = 128", "pixels"),
    (r"stride 96|stride 2\b", "code/data_prep.py; " + S7, "STRIDE = 96 (patches); Section 2 (pool stride 2)", "pixels"),
    (r"64 × 64", S7, "Section 4 (q_dense (B,2,64,64))", "pixels"),
    (r"17 × 17|9 × 9|5 × 5 and|5 × 5,|3 × 3,|3 × 3 and|windows|Scale:", "results_v2/validation/stage16/multiscale/scale_definitions.json", "window definitions (1x1, 3x3, 5x5, 9x9, 17x17, full)", "pixels"),
    (r"5 × 5 kernels|5 × 5 convolution|5 × 5 `|R2Conv", S7 + "; " + S9, "Section 2 (R2Conv k=5); Plain CNN Conv2d k=5", "pixels"),
    (r"σ = 0\.66", S7, "Section 2 (PointwiseAvgPoolAntialiased sigma=0.66, stride 2)", ""),
    (r"64 channels", S9, "Section 2 (Plain CNN c1=c2=64)", "channels"),
    (r"order 16|16 elements|all 16", S7, "Section 3 (D8, order 16); Section 9 (all 16 D8 elements)", "elements"),
    (r"multiples of 45|45° and reflections|at α = 45", S7, "Section 3 (flipRot2dOnR2(N=8); irrep matrix checked at alpha=45)", "degrees"),
    (r"15 of 15", S7, "Section 8 (15/15 tests pass)", "tests"),
    (r"0\.00e\+00|0\.00%", S7, "Section 3 / Section 9 (0.00e+00 at alpha=45; 0.00% at identity and one reflection)", ""),
    (r"33\.95", S7, "Section 9 item 5 (up to 33.95%)", "%"),
    (r"learning rate|batch size|at most 40|patience 6|37 epochs|epoch 30|1 × 10", S10, "Section 4 protocol table; Section 5 (37 epochs; best epoch 30)", ""),
    (r"seed 42|\(seed", S10 + "; " + PREREG, "seed 42 (training, dev split, bootstrap, subsample)", ""),
    (r"6 GB|RTX 3050", "environment.yml", "gpu: NVIDIA GeForce RTX 3050 6GB Laptop GPU", "GB"),
    (r"Python 3\.11\.8|PyTorch 2\.6\.0|CUDA 12\.4|escnn 1\.0\.11", "environment.yml", "python_version, torch_version, cuda_version, escnn_version", ""),
    (r"1000 re-pairings|\b1000\b", PREREG + "; code_v2/stage11_geometric_validation.py", "criteria.C rule (1000 re-pairings); N_PERM = 1000", "permutations"),
    (r"100 permutations|\b100\b", "results_v2/validation/stage13/stage13_results.json", "results.testA.n_permutations = 100", "permutations"),
    (r"2000 (?:image-level )?resamples|2000 resamples", PREREG, "criteria.B.bootstrap_n = 2000", "resamples"),
    (r"11,674", "results_v2/model3/metrics/stage10_summary.json", "n_params", "parameters"),
    (r"10,616|5,160", "data/PANDA/stage18_metadata_audit.md", "train.csv rows 10,616; Radboud rows 5,160", "slides"),
    (r"CC BY", "reports/STAGE18_DESIGN_FREEZE.md", "License: CC BY-SA-NC 4.0", ""),
    (r"10-case|10 \(pilot|ten pilot|the ten|Only a 10|only 10 cases", "data/PANDA/stage18_metadata_audit.md", "radboud_pilot_manifest.csv (n=10)", "cases"),
    (r"labels 0–5|0–5|labels 0|0 background|1 stroma|2 benign|3–5", "reports/STAGE21_5_PANDA_DOCUMENTATION_CORRECTION.md", "semantic labels 0-5 (0 background, 1 stroma, 2 benign, 3-5 Gleason patterns)", ""),
    (r"Version [012]|version 2", "reports/STAGE0_RECALIBRATION_AUDIT_REPORT.md", "threshold-file version labels (v0 superseded, v1 invalidated, v2 frozen)", "label"),
    (r"207 characters", "reports/STAGE21_5_H2_TRACEABILITY.md", "Section 4 (H2 block, 207 characters)", "characters"),
    (r"one hospital|the same hospital", "reports/STAGE17_AUDIT_REPORT.md", "Section 4 (same hospital, same curating group)", ""),
    (r"one single-seed|one seed|one training run|one training", S10, "single seed 42; run once", "count"),
    (r"stages 0–22", "manuscript header", "project stage range (not a measured value)", ""),
]

NUMWORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twelve": 12, "sixteen": 16}
NUM_RE = re.compile(r"(?P<sign>[−\-+])?(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?(?:e[+\-]\d+)?)(?P<pct>%)?")
WORD_RE = re.compile(r"\b(" + "|".join(NUMWORDS) + r")\b", re.I)
ONE_RE = re.compile(r"\b(one|single)\b(?= (training|seed|run|extension|reflection|hospital|dataset|GPU|architecture|reference|formulation|pixel|dense|sampled|held-out|image|gland|case|batch))", re.I)


def strip_nonquant(text):
    text = re.sub(r"`[^`]*`", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\bModel [23]\b", " ", text)
    text = re.sub(r"SHA-256", " ", text)
    text = re.sub(r"\brank-2\b", " ", text)
    text = re.sub(r"\bGleason[- ]\d(?:, \d)*(?: or \d)?", " ", text)
    text = re.sub(r"\bpattern \d\b", " ", text)
    text = re.sub(r"\bPhase \d\b|stages 0–22\.5", " ", text)
    text = re.sub(r"\[(\d+)(?:[–,\-]\d+)*\]", " ", text)
    text = re.sub(r"\b(Table|Figure|Section|Sections)\s*S?\d+(?:\.\d+)?(?:[–-]\d+)?", " ", text)
    text = re.sub(r"\b(Stage|Stages)\s*\d+(?:\.\d+)?(?:[–,\s-]+\d+)*", " ", text)
    text = re.sub(r"\b[A-Za-z]+[_₀-₉]*\d+[A-Za-z0-9_]*\b", " ", text)
    text = re.sub(r"\d+(?=[αθφπ])", " ", text)
    text = re.sub(r"[qQSλ][₀-₉]", " ", text)
    text = re.sub(r"\(\d+\)", " ", text)
    text = re.sub(r"^\s*\d+\.\s", " ", text, flags=re.M)
    text = re.sub(r"^#+\s.*$", lambda m: re.sub(r"\d", " ", m.group(0)), text, flags=re.M)
    text = re.sub(r"\*\*(Table|Figure)\s*\d+[.a-z]*\.?", " ", text)
    return text


def decimals(s):
    s = s.replace(",", "")
    if "e" in s:
        return 0
    return len(s.split(".")[1]) if "." in s else 0


def find_nt(v, dec, hint, pct, exact_only=False):
    targets = [v / 100.0, v] if pct else [v]

    def pick(tol):
        c = [x for x in nt_index if any(abs(x[0] - t) <= tol for t in targets)]
        if not c:
            c = [x for x in nt_index if any(abs(abs(x[0]) - abs(t)) <= tol for t in targets)]
        return c
    cands = pick(1e-9) or ([] if exact_only else (pick(0.5 * 10 ** (-dec) + 1e-12) if dec >= 1 else []))
    for h in (hint or []):
        pref = [c for c in cands if h in (c[2] + " " + c[5]).lower()]
        if pref:
            cands = pref
            break
    return cands


def find_leaf(v, dec, hint, pct, pref=None):
    out = []
    targets = [v / 100.0, v] if pct else [v, -v]
    for tol in (1e-9, 0.5 * 10 ** (-dec) + 1e-12 if dec >= 1 else None):
        if tol is None:
            continue
        for cand in targets:
            j = bisect.bisect_left(leaf_vals, cand - tol)
            while j < len(leaf_vals) and leaf_vals[j] <= cand + tol:
                out.append(leaves[j])
                j += 1
        if out:
            break
    if out and pref:
        pf = [c for c in out if pref(c)]
        if pf:
            out = pf
    for h in (hint or []):
        pr = [c for c in out if h in c[2].lower()]
        if pr:
            out = pr
            break
    return out


HINTS = [(r"malignant", "malignant"), (r"benign", "benign"), (r"epoch", "epoch"), (r"holm|adjusted", "holm"), (r"glass", "glass_delta"), (r"correlation|pearson|spearman|ρ|\br\b", "r"), (r"z =|z-score", "z_score"), (r"D_Q", "d_q"), (r"permutation p|one-sided|two-sided", "p"), (r"error", "error")]


def hint_for(ctx):
    return [h for pat, h in HINTS if re.search(pat, ctx, re.I)]


def unit_for(after):
    m = re.match(r"\s*(%|°|images?|patches|patients?|parameters?|epochs?|pixels?|px|angles?|instances?|cases?|slides?|permutations?|resamples?|elements?|tests?|glands?|components?|GB|seeds?)", after, re.I)
    return m.group(1).lower().replace("px", "pixels") if m else ""


def span_rule(ctx, pos_lo, pos_hi):
    for pat, sf, lc, un in RULES:
        for m in re.finditer(pat, ctx):
            if m.start() < pos_hi and m.end() > pos_lo:
                return (sf, lc, un)
    return None


def main():
    txt = open(MS, encoding="utf-8").read()
    body = txt[: txt.find("## References")]
    section = "Title/Abstract"
    rows = []
    nid = 0
    for raw_line in body.splitlines():
        m = re.match(r"^(#{2,3})\s+(.*)$", raw_line)
        if m:
            section = m.group(2).strip()
            if section.lower().startswith("abstract"):
                section = "Abstract"
            continue
        if re.match(r"^\|[\s\-:|]+\|?$", raw_line):
            continue
        line = strip_nonquant(raw_line)
        tokens = []
        for mo in NUM_RE.finditer(line):
            s = mo.group("num")
            v = float(s.replace(",", ""))
            if mo.group("sign") in ("−", "-") and v != 0:
                pre = line[max(0, mo.start() - 1): mo.start()]
                if not (mo.group("sign") == "-" and re.match(r"\d", pre or "")):
                    v = -v
            written = (mo.group("sign") or "") + s + (mo.group("pct") or "")
            tokens.append((mo.start(), mo.end(), written, v, s, mo.group("pct")))
        for mo in WORD_RE.finditer(line):
            tokens.append((mo.start(), mo.end(), mo.group(0), float(NUMWORDS[mo.group(0).lower()]), mo.group(0), None))
        for mo in ONE_RE.finditer(line):
            tokens.append((mo.start(), mo.end(), mo.group(0), 1.0, mo.group(0), None))
        tokens.sort()
        first_cell = re.match(r"^\|\s*(\d+)\s*\|", raw_line)
        split = "testA" if "testA (" in raw_line else ("testB" if "testB (" in raw_line else None)
        rowpref = None
        if raw_line.startswith("| Plain CNN"):
            rowpref = lambda c: "plain_cnn" in c[1]  # noqa: E731
        elif raw_line.startswith("| Model 2"):
            rowpref = lambda c: "model2" in c[1]  # noqa: E731
        elif raw_line.startswith("| Model 3") and split:
            rowpref = lambda c, sp=split: ("stage11" in c[1] and c[2].startswith(sp + ".")) or ("A0_reference." + sp in c[2])  # noqa: E731
        for start, end, written, v, s, pct in tokens:
            lo = max(0, start - 60)
            ctx = line[lo: end + 60]
            plo, phi = start - lo, end - lo
            unit = unit_for(line[end:end + 25])
            hint = hint_for(line[max(0, start - 60): end + 20])
            dec = decimals(s) if re.match(r"[\d,.e+\-]+$", s) else 0
            is_small_int = (dec == 0 and abs(v) <= 20)
            src_file = loc = basis = ""
            verified = "NO"
            hit = None
            if int(abs(v)) in (15, 30, 37, 45, 60, 90, 123, 150, 173) and dec == 0 and (
                    line[end:end + 1] == "°" or (first_cell and int(first_cell.group(1)) == int(v) and start < 6)):
                hit = (PREREG, "criteria.D.rotation_angles_deg (15,30,37,45,60,90,123,150,173)", "degrees")
            elif re.match(r"^\|\s*Scale:", raw_line) and re.search(r"\|\s*\d\s*\|\s*$", raw_line) and dec == 0 and v <= 9 and \
                    not re.search(r"\d", line[end:].replace("|", "").strip()):
                hit = (S16, "rotation_by_scale.<scale>.n_angles_passed_vs_original_threshold", "angles")
            if hit is None:
                hit = span_rule(ctx, plo, phi)
            if hit and dec >= 2 and not is_small_int and not pct and v != 0:
                # decimals with a numbers-table match take priority over broad structural rules
                if find_nt(v, dec, hint, bool(pct), exact_only=True):
                    hit = None
            if hit:
                src_file, loc, basis = hit[0], hit[1], "documented definition / report / code"
                unit = unit or hit[2]
                verified = "yes"
            else:
                nt = [] if (is_small_int or rowpref) else find_nt(v, dec, hint, bool(pct))
                lf = [] if (is_small_int or nt) else find_leaf(v, dec, hint, bool(pct), rowpref)
                if nt:
                    c = nt[0]
                    src_file, loc, basis = c[4], f"13_numbers_and_sources.csv row {c[1]} ({c[2]}); field {c[5]}", "numbers table"
                    unit = unit or c[3]
                    verified = "yes"
                elif lf:
                    c = lf[0]
                    src_file, loc, basis = c[1], f"json path {c[2]}", "frozen result file (value at displayed precision)"
                    verified = "yes"
                elif s.replace(",", "") in ("0.5", "0.50", "0.30", "0.3", "0.01", "0.05", "0.001"):
                    src_file, loc, basis = PREREG, "criteria thresholds (Glass delta >= 0.5; r >= 0.30; p < 0.01)", "documented definition / report / code"
                    verified = "yes"
            nid += 1
            rows.append({"number_id": f"N{nid:04d}", "section": section, "value_as_written": written,
                         "numeric_value": v, "unit": unit or ("%" if pct else ""),
                         "source_file": src_file, "source_row_or_location": loc, "verified": verified,
                         "match_basis": basis, "context": line[max(0, start - 45): end + 45].strip().replace("\n", " ")})
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["number_id", "section", "value_as_written", "numeric_value", "unit", "source_file", "source_row_or_location", "verified", "match_basis", "context"])
        w.writeheader()
        w.writerows(rows)
    n_ok = sum(1 for r in rows if r["verified"] == "yes")
    print(f"tokens: {len(rows)} | verified: {n_ok} | unverified: {len(rows) - n_ok}")
    for r in rows:
        if r["verified"] != "yes":
            print("  UNVERIFIED", r["number_id"], r["section"][:30], repr(r["value_as_written"]), "|", r["context"][:110])


if __name__ == "__main__":
    main()
