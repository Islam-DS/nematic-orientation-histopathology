"""
STAGE 0: real-GlaS (H&E) rotation noise-floor + oracle-vs-anatomy
calibration. This is CALIBRATION ONLY -- no model is trained here. Output
feeds results_v2/phase2/preregistered_thresholds.json, which is frozen
immediately after this script runs and is not touched again.

Per section 18 of the master prompt: the Phase 1R synthetic threshold is
NOT reused directly. This script re-derives a real-image floor using the
same oracle (structure_tensor_estimate) and the same rotation-consistency
methodology, but on actual GlaS patches, from a deterministic calibration
subset drawn ONLY from the TRAIN split (never touching test).

IMPORTANT METHODOLOGICAL NOTE (documented, not a silent assumption):
Phase 1R could regenerate a synthetic object at any target angle directly
(no pixel resampling needed for "rotation"). Real H&E images have no such
analytic generator -- the ONLY way to produce a rotated version of a real
patch is pixel-domain interpolation (skimage.transform.rotate). So the
real-H&E floor computed here unavoidably includes interpolation/resampling
error CONVOLVED with genuine staining/texture noise -- it is not possible
to cleanly separate these for real tissue the way Phase 1R could for
synthetic shapes. This is precisely what section 18 asks the calibration
to characterize (staining variation, texture noise, interpolation,
resampling, all together), not a methodological shortcoming to fix.
"""
import sys
import os
import json

import numpy as np
from skimage.color import rgb2gray
from skimage.transform import rotate as sk_rotate
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
from data_prep import build_image_label_table, extract_patches  # reuse only

sys.path.insert(0, os.path.dirname(__file__))
from anatomy_targets import analyze_glands, build_full_image_field, crop_field_to_patches
from oracle_baselines import structure_tensor_estimate
from nematic_math import rotate_Q_analytic, phi_S_to_q, angular_error_mod_pi

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results_v2", "phase2")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Same split parameters as the frozen Phase 1 pilot (code/data_prep.py::build_patch_dataset)
# -- reused exactly, not redefined, so calibration uses the SAME train/test partition
# already established and never touches test images.
SPLIT_TEST_SIZE = 0.2
SPLIT_SEED = 42

N_CALIBRATION_IMAGES = 20          # deterministic: first N train images by sorted stem
MIN_VALID_FRACTION_FOR_PATCH = 0.10  # patch must have >=10% valid anatomical-target pixels
ROTATION_ANGLES_DEG = [0, 15, 30, 37, 45, 60, 90, 123, 150, 173]  # same set as Phase 1R, for continuity
STRUCTURE_TENSOR_SIGMA = 3.0        # same as used throughout Phase 1R / target diagnostics


def get_deterministic_calibration_image_list():
    """
    TRAIN-only, deterministic. Mirrors code/data_prep.py's exact split call
    (same test_size/seed) without modifying that file, then takes the
    first N_CALIBRATION_IMAGES train images sorted by filename stem.
    """
    table = build_image_label_table()
    train_table, _test_table = train_test_split(
        table, test_size=SPLIT_TEST_SIZE, random_state=SPLIT_SEED, stratify=table["label"]
    )
    train_table = train_table.sort_values("stem").reset_index(drop=True)
    calib = train_table.iloc[:N_CALIBRATION_IMAGES]
    return calib


def build_patch_targets_for_image(image_path):
    """
    Returns list of dicts: {rgb_patch, gray_patch, target_q1, target_S, target_phi,
    valid_fraction} for every patch of this image with sufficient valid
    anatomical-target coverage. Uses the SAME anatomy_targets pipeline and
    the SAME thresholds already frozen in docs_v2/phase2_target_spec.md
    (BORDER_FRAC_THRESHOLD, DEFAULT_S_THRESHOLD) -- not redefined here.
    """
    stem, ext = os.path.splitext(image_path)
    anno_path = stem + "_anno" + ext
    from PIL import Image
    labeled = np.array(Image.open(anno_path))
    img = np.array(Image.open(image_path).convert("RGB"))

    props = analyze_glands(labeled)
    field = build_full_image_field(labeled, props, S_choice="cov")
    field_patches = crop_field_to_patches(field, labeled.shape, patch_size=128, stride=96)
    rgb_patches = extract_patches(image_path, patch_size=128, stride=96)  # same grid, verified in tests_v2

    out = []
    for fp, rgb in zip(field_patches, rgb_patches):
        valid = fp["valid"]
        frac = valid.mean()
        if frac < MIN_VALID_FRACTION_FOR_PATCH:
            continue
        q1_mean = fp["q1"][valid].mean()
        q2_mean = fp["q2"][valid].mean()
        S_target = float(np.hypot(q1_mean, q2_mean))
        phi_target = float(np.mod(0.5 * np.arctan2(q2_mean, q1_mean), np.pi))
        out.append({
            "rgb": rgb, "gray": rgb2gray(rgb).astype(np.float64),
            "target_q1": float(q1_mean), "target_q2": float(q2_mean),
            "target_S": S_target, "target_phi": phi_target,
            "valid_fraction": float(frac),
        })
    return out


def calibration_oracle_vs_anatomy(patches):
    """
    Reference scale A: how well does a FIXED, non-learned, image-only
    method (structure tensor on the raw patch) correspond to the
    anatomical (mask-derived) target on REAL GlaS patches? This provides
    a principled, data-derived floor for the Phase 2 angular/magnitude
    correspondence thresholds -- the physically-typed model, with learned
    capacity, should be judged against what a fixed classical method
    already achieves from image texture alone, not an arbitrary constant.
    """
    ang_errs, S_oracle, S_target, D_Q = [], [], [], []
    rng = np.random.default_rng(0)
    oracle_phi, oracle_q1, oracle_q2 = [], [], []

    for p in patches:
        S_est, phi_est = structure_tensor_estimate(p["gray"], sigma=STRUCTURE_TENSOR_SIGMA)
        q1_o, q2_o = phi_S_to_q(phi_est, S_est)
        oracle_phi.append(phi_est); oracle_q1.append(q1_o); oracle_q2.append(q2_o)

        ang_errs.append(np.degrees(angular_error_mod_pi(phi_est, p["target_phi"])))
        S_oracle.append(S_est)
        S_target.append(p["target_S"])
        D_Q.append(np.hypot(q1_o - p["target_q1"], q2_o - p["target_q2"]))

    oracle_phi = np.array(oracle_phi); oracle_q1 = np.array(oracle_q1); oracle_q2 = np.array(oracle_q2)
    target_phi_arr = np.array([p["target_phi"] for p in patches])
    target_q1_arr = np.array([p["target_q1"] for p in patches])
    target_q2_arr = np.array([p["target_q2"] for p in patches])

    # FULL permutation-null distributions (many re-pairings, not just one) for BOTH
    # mean angular error and mean D_Q -- this is the reusable null-construction
    # PROCEDURE that will later be applied to the MODEL's own test-set predictions
    # (criteria A and C). Computed here on the oracle only to (a) sanity-check the
    # procedure against the closed-form analytic prediction for independent uniform
    # angles [E(Delta_theta) = 45 deg exactly, std = 90/sqrt(12) = 25.98 deg], and
    # (b) report the oracle's own z-score/percentile against this null for context.
    n_perm = 1000
    n = len(patches)
    null_mean_ang = np.empty(n_perm)
    null_mean_DQ = np.empty(n_perm)
    for k in range(n_perm):
        idx = rng.permutation(n)
        d = np.abs(oracle_phi - target_phi_arr[idx]) % np.pi
        d = np.minimum(d, np.pi - d)
        null_mean_ang[k] = np.degrees(d).mean()
        null_mean_DQ[k] = np.hypot(oracle_q1 - target_q1_arr[idx], oracle_q2 - target_q2_arr[idx]).mean()

    # single-draw D_Q_null retained for backward-compat reporting
    idx = rng.permutation(n)
    D_Q_null = list(np.hypot(oracle_q1 - target_q1_arr[idx], oracle_q2 - target_q2_arr[idx]))

    S_oracle, S_target = np.array(S_oracle), np.array(S_target)
    oracle_ang_z = (null_mean_ang.mean() - np.mean(ang_errs)) / null_mean_ang.std()
    oracle_DQ_z = (null_mean_DQ.mean() - np.mean(D_Q)) / null_mean_DQ.std()
    return {
        "n_patches": len(patches),
        "mean_angular_error_deg": float(np.mean(ang_errs)),
        "median_angular_error_deg": float(np.median(ang_errs)),
        "pearson_r_S_oracle_vs_S_target": float(np.corrcoef(S_oracle, S_target)[0, 1]),
        "mean_D_Q": float(np.mean(D_Q)),
        "median_D_Q": float(np.median(D_Q)),
        "mean_D_Q_null_permuted_single_draw": float(np.mean(D_Q_null)),
        "permutation_null_n_permutations": n_perm,
        "permutation_null_mean_angular_error": {
            "null_mean_deg": float(null_mean_ang.mean()), "null_std_deg": float(null_mean_ang.std()),
            "analytic_prediction_mean_deg": 45.0, "analytic_prediction_std_deg": float(90 / np.sqrt(12)),
            "oracle_z_score": float(oracle_ang_z),
        },
        "permutation_null_mean_D_Q": {
            "null_mean": float(null_mean_DQ.mean()), "null_std": float(null_mean_DQ.std()),
            "oracle_z_score": float(oracle_DQ_z),
        },
    }


def calibration_rotation_floor(patches, n_patches=60, seed=SPLIT_SEED):
    """
    Reference scale B: real-H&E rotation-consistency floor. Oracle
    self-consistency under PIXEL-DOMAIN rotation (the only option for real
    images -- see module docstring) of actual GlaS patches.

    SIGN CONVENTION (verified empirically before trusting, see project
    history): for the classical structure-tensor oracle specifically,
    sk_rotate(img, angle=-alpha_deg) produces the image rotation that
    shifts the RECOVERED orientation by +alpha_deg (matching
    rotate_Q_analytic's +alpha convention). This is the OPPOSITE sign from
    what Phase 1R used for the escnn MODEL's own rotation convention --
    the two are governed by different internal conventions (escnn's group
    action vs. skimage's structure_tensor row/col axis handling) and were
    each verified independently rather than assumed to match. Verified
    here via a synthetic object with sigma=3.0: sk_rotate(angle=-alpha)
    recovers phi0+alpha to within ~1 deg for alpha in {20, 45, 70}, while
    sk_rotate(angle=+alpha) does not.
    """
    rng = np.random.default_rng(seed)
    subset = list(rng.choice(patches, size=min(n_patches, len(patches)), replace=False))

    per_angle_errors = {a: [] for a in ROTATION_ANGLES_DEG}
    all_errors = []

    for p in subset:
        gray0 = p["gray"]
        S0, phi0 = structure_tensor_estimate(gray0, sigma=STRUCTURE_TENSOR_SIGMA)
        q0 = np.array(phi_S_to_q(phi0, S0))

        for alpha_deg in ROTATION_ANGLES_DEG:
            if alpha_deg == 0:
                gray_rot = gray0
            else:
                gray_rot = sk_rotate(gray0, angle=-alpha_deg, mode="reflect", order=1)
            S_r, phi_r = structure_tensor_estimate(gray_rot, sigma=STRUCTURE_TENSOR_SIGMA)
            q_r = np.array(phi_S_to_q(phi_r, S_r))

            alpha = np.deg2rad(alpha_deg)
            q_expected = np.array(rotate_Q_analytic(q0[0], q0[1], alpha))
            err = np.linalg.norm(q_r - q_expected) / (np.linalg.norm(q_expected) + 1e-6)
            per_angle_errors[alpha_deg].append(err)
            all_errors.append(err)

    return {
        "n_patches_used": len(subset),
        "rotation_angles_deg": ROTATION_ANGLES_DEG,
        "mean_normalized_equivariance_error": float(np.mean(all_errors)),
        "median_normalized_equivariance_error": float(np.median(all_errors)),
        "per_angle_mean": {str(a): float(np.mean(v)) for a, v in per_angle_errors.items()},
        "per_angle_std": {str(a): float(np.std(v)) for a, v in per_angle_errors.items()},
        "per_angle_n": {str(a): len(v) for a, v in per_angle_errors.items()},
    }


def main():
    print("[phase2_calibration] selecting deterministic TRAIN-only calibration image subset...")
    calib_table = get_deterministic_calibration_image_list()
    print(f"[phase2_calibration] {len(calib_table)} calibration images (train-split only): "
          f"{list(calib_table['stem'])}")

    print("[phase2_calibration] building patch-level anatomical targets + matching RGB patches...")
    all_patches = []
    for _, row in calib_table.iterrows():
        pts = build_patch_targets_for_image(row["path"])
        all_patches.extend(pts)
    print(f"[phase2_calibration] {len(all_patches)} calibration patches with >= "
          f"{MIN_VALID_FRACTION_FOR_PATCH*100:.0f}% valid anatomical-target coverage")

    print("[phase2_calibration] computing oracle-vs-anatomy correspondence (reference scale A)...")
    ref_a = calibration_oracle_vs_anatomy(all_patches)
    print(json.dumps(ref_a, indent=2))

    print("[phase2_calibration] computing real-H&E rotation-consistency floor (reference scale B)...")
    ref_b = calibration_rotation_floor(all_patches)
    print(json.dumps(ref_b, indent=2))

    report = {
        "calibration_image_stems": list(calib_table["stem"]),
        "calibration_subset_selection_rule": (
            f"first {N_CALIBRATION_IMAGES} images (sorted by stem) from the TRAIN split "
            f"produced by the exact same split call as code/data_prep.py::build_patch_dataset "
            f"(test_size={SPLIT_TEST_SIZE}, random_state={SPLIT_SEED}, stratified by label); "
            f"test-split images are never used for calibration"
        ),
        "min_valid_fraction_for_patch": MIN_VALID_FRACTION_FOR_PATCH,
        "structure_tensor_sigma": STRUCTURE_TENSOR_SIGMA,
        "reference_scale_A_oracle_vs_anatomy": ref_a,
        "reference_scale_B_rotation_floor": ref_b,
        "methodological_note": (
            "Reference scale B uses pixel-domain rotation (skimage.transform.rotate) because, "
            "unlike Phase 1R's synthetic objects, real H&E patches have no analytic re-render "
            "path -- this floor necessarily convolves interpolation/resampling error with real "
            "staining/texture noise, which is exactly what section 18 asks it to characterize."
        ),
    }
    with open(os.path.join(RESULTS_DIR, "calibration_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[phase2_calibration] calibration report written to {RESULTS_DIR}/calibration_report.json")

    return report


if __name__ == "__main__":
    main()
