"""
STAGE 0 CORRECTIVE RECALIBRATION (post-Stage-2 split reconciliation).

Regenerates the Stage 0 real-H&E calibration using ONLY the canonical GlaS
`train` split (85 images), per reports/STAGE2_SPLIT_RECONCILIATION.md's
finding that the original calibration was contaminated with 20 canonical
testA images. Produces results_v2/phase2/preregistered_thresholds_v2.json.

Does NOT touch results_v2/phase2/preregistered_thresholds.json or
preregistered_thresholds_v0_superseded.json (both preserved byte-identical
for audit history -- verified in the accompanying audit report).

Reuses the Stage 0 methodology (code_v2/phase2_calibration.py) wherever it
remains scientifically valid -- only the image-selection step changes.
Adds a HARD, PROGRAMMATIC assertion (not just a documentation claim) that
zero non-train-split images enter the calibration universe at any point.
"""
import sys
import os
import json

import numpy as np
import pandas as pd
from skimage.transform import rotate as sk_rotate

sys.path.insert(0, os.path.dirname(__file__))
from phase2_calibration import (
    build_patch_targets_for_image, ROTATION_ANGLES_DEG, STRUCTURE_TENSOR_SIGMA,
    MIN_VALID_FRACTION_FOR_PATCH,
)
from oracle_baselines import structure_tensor_estimate
from nematic_math import rotate_Q_analytic, phi_S_to_q, angular_error_mod_pi

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results_v2", "phase2")
MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "..", "data_manifests", "glas_manifest.csv")
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")

SEED = 42
N_PERM = 1000
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 42


def get_canonical_train_images():
    """
    Reads the Stage-2-verified manifest and returns ONLY images whose
    canonical_split == 'train'. HARD ASSERTION: raises if this set is not
    exactly disjoint from testA/testB, and if it does not have exactly 85
    members (the published GlaS train-set size) -- this is the concrete,
    checkable safeguard against repeating the Stage 0 contamination.
    """
    manifest = pd.read_csv(MANIFEST_PATH)
    train_rows = manifest[manifest["canonical_split"] == "train"].copy()
    non_train_rows = manifest[manifest["canonical_split"] != "train"]

    assert len(train_rows) == 85, f"expected 85 canonical train images, found {len(train_rows)}"
    assert set(train_rows["image_id"]).isdisjoint(set(non_train_rows["image_id"])), \
        "train/non-train image_id sets are not disjoint -- ABORTING, contamination risk"
    assert not any(s.startswith("testA_") or s.startswith("testB_") for s in train_rows["image_id"]), \
        "a testA_/testB_ prefixed image leaked into the train selection -- ABORTING"

    train_rows = train_rows.sort_values("image_id").reset_index(drop=True)
    return train_rows


def q_normalized_error(q_a, q_b, eps=1e-6):
    num = np.linalg.norm(q_a - q_b, axis=-1)
    den = np.linalg.norm(q_b, axis=-1) + eps
    return num / den


def bootstrap_ci_pearson_r(x, y, n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED, alpha=0.05):
    rng = np.random.default_rng(seed)
    n = len(x)
    boot_r = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boot_r[i] = np.corrcoef(x[idx], y[idx])[0, 1]
    lo, hi = np.percentile(boot_r, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def bootstrap_ci_mean(x, n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED, alpha=0.05):
    rng = np.random.default_rng(seed)
    n = len(x)
    boot_mean = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boot_mean[i] = x[idx].mean()
    lo, hi = np.percentile(boot_mean, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def calibration_A_B_C(patches):
    """A: angular correspondence. B: order-magnitude correlation.
    C: tensor-similarity null. All computed together since they share the
    same oracle pass over the patches."""
    oracle_phi, oracle_S = [], []
    target_phi, target_S = [], []
    oracle_q1, oracle_q2, target_q1, target_q2 = [], [], [], []

    for p in patches:
        S_est, phi_est = structure_tensor_estimate(p["gray"], sigma=STRUCTURE_TENSOR_SIGMA)
        q1_o, q2_o = phi_S_to_q(phi_est, S_est)
        oracle_phi.append(phi_est); oracle_S.append(S_est)
        oracle_q1.append(q1_o); oracle_q2.append(q2_o)
        target_phi.append(p["target_phi"]); target_S.append(p["target_S"])
        target_q1.append(p["target_q1"]); target_q2.append(p["target_q2"])

    oracle_phi = np.array(oracle_phi); oracle_S = np.array(oracle_S)
    target_phi = np.array(target_phi); target_S = np.array(target_S)
    oracle_q1 = np.array(oracle_q1); oracle_q2 = np.array(oracle_q2)
    target_q1 = np.array(target_q1); target_q2 = np.array(target_q2)
    n = len(patches)

    ang_err = np.degrees(angular_error_mod_pi(oracle_phi, target_phi))
    D_Q = np.hypot(oracle_q1 - target_q1, oracle_q2 - target_q2)
    r_S = float(np.corrcoef(oracle_S, target_S)[0, 1])
    r_ci_lo, r_ci_hi = bootstrap_ci_pearson_r(oracle_S, target_S)

    # exact individual-level null (full n x n cross-pairing, excluding self-pairs)
    diff = np.abs(oracle_phi[:, None] - target_phi[None, :]) % np.pi
    diff = np.minimum(diff, np.pi - diff)
    mask = ~np.eye(n, dtype=bool)
    null_ang_individual = np.degrees(diff[mask])
    DQ_cross = np.hypot(oracle_q1[:, None] - target_q1[None, :], oracle_q2[:, None] - target_q2[None, :])
    null_DQ_individual = DQ_cross[mask]

    glass_delta_ang = (null_ang_individual.mean() - ang_err.mean()) / null_ang_individual.std()
    glass_delta_DQ = (null_DQ_individual.mean() - D_Q.mean()) / null_DQ_individual.std()

    # permutation-null-of-mean (many re-pairings) for the percentile/p-value check
    rng = np.random.default_rng(SEED)
    null_mean_ang = np.empty(N_PERM)
    null_mean_DQ = np.empty(N_PERM)
    for k in range(N_PERM):
        idx = rng.permutation(n)
        d = np.abs(oracle_phi - target_phi[idx]) % np.pi
        d = np.minimum(d, np.pi - d)
        null_mean_ang[k] = np.degrees(d).mean()
        null_mean_DQ[k] = np.hypot(oracle_q1 - target_q1[idx], oracle_q2 - target_q2[idx]).mean()

    p_perm_ang = float((null_mean_ang <= ang_err.mean()).mean())
    p_perm_DQ = float((null_mean_DQ <= D_Q.mean()).mean())

    ang_ci_lo, ang_ci_hi = bootstrap_ci_mean(ang_err)

    return {
        "A_angular_correspondence": {
            "n_patches": n,
            "oracle_mean_angular_error_deg": float(ang_err.mean()),
            "oracle_median_angular_error_deg": float(np.median(ang_err)),
            "oracle_angular_error_bootstrap_95CI": [ang_ci_lo, ang_ci_hi],
            "individual_null_angular_mean_deg": float(null_ang_individual.mean()),
            "individual_null_angular_std_deg": float(null_ang_individual.std()),
            "oracle_glass_delta_angular": float(glass_delta_ang),
            "permutation_null_of_mean_angular": {"mean": float(null_mean_ang.mean()), "std": float(null_mean_ang.std())},
            "permutation_p_value_angular": p_perm_ang,
        },
        "B_order_magnitude_correlation": {
            "n_patches": n,
            "pearson_r_S_oracle_vs_S_target": r_S,
            "bootstrap_95CI": [r_ci_lo, r_ci_hi],
        },
        "C_tensor_similarity": {
            "n_patches": n,
            "oracle_mean_D_Q": float(D_Q.mean()),
            "oracle_median_D_Q": float(np.median(D_Q)),
            "individual_null_DQ_mean": float(null_DQ_individual.mean()),
            "individual_null_DQ_std": float(null_DQ_individual.std()),
            "oracle_glass_delta_DQ": float(glass_delta_DQ),
            "permutation_null_of_mean_DQ": {"mean": float(null_mean_DQ.mean()), "std": float(null_mean_DQ.std())},
            "permutation_p_value_DQ": p_perm_DQ,
        },
        "n_permutations": N_PERM,
        "seed": SEED,
    }


def calibration_D(patches, n_patches=60, seed=SEED):
    """D: rotation-consistency floor, per-angle mean/std, real H&E pixel-domain
    rotation (sign convention verified in the original Stage 0 work: for the
    structure-tensor oracle, sk_rotate(angle=-alpha) yields the +alpha shift)."""
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
        "seed": seed,
        "rotation_angles_deg": ROTATION_ANGLES_DEG,
        "mean_normalized_equivariance_error": float(np.mean(all_errors)),
        "median_normalized_equivariance_error": float(np.median(all_errors)),
        "per_angle_mean": {str(a): float(np.mean(v)) for a, v in per_angle_errors.items()},
        "per_angle_std": {str(a): float(np.std(v)) for a, v in per_angle_errors.items()},
        "per_angle_n": {str(a): len(v) for a, v in per_angle_errors.items()},
    }


def main():
    print("[recalibration] loading canonical TRAIN-only image list from Stage 2 manifest...")
    train_table = get_canonical_train_images()
    print(f"[recalibration] {len(train_table)} canonical train images (assertion passed: "
          f"exactly 85, disjoint from testA/testB, no testA_/testB_ prefixes)")

    print("[recalibration] building patch-level anatomical targets for ALL 85 train images...")
    all_patches = []
    excluded_images = []
    for _, row in train_table.iterrows():
        img_path = os.path.join(PROJECT_ROOT, row["image_path"])
        pts = build_patch_targets_for_image(img_path)
        if len(pts) == 0:
            excluded_images.append(row["image_id"])
        all_patches.extend(pts)
    print(f"[recalibration] {len(all_patches)} calibration patches with >= "
          f"{MIN_VALID_FRACTION_FOR_PATCH*100:.0f}% valid anatomical-target coverage, "
          f"from {len(train_table) - len(excluded_images)}/{len(train_table)} images "
          f"({len(excluded_images)} images contributed zero qualifying patches: {excluded_images})")

    print("[recalibration] A/B/C calibration (oracle vs anatomy, magnitude correlation, tensor null)...")
    abc = calibration_A_B_C(all_patches)
    print(json.dumps(abc, indent=2))

    print("[recalibration] D calibration (real-H&E rotation-consistency floor)...")
    d = calibration_D(all_patches)
    print(json.dumps(d, indent=2))

    report = {
        "calibration_dataset": "canonical GlaS train split",
        "n_images": int(len(train_table)),
        "image_identifiers": list(train_table["image_id"]),
        "images_excluded_zero_qualifying_patches": excluded_images,
        "n_calibration_patches": len(all_patches),
        "min_valid_fraction_for_patch": MIN_VALID_FRACTION_FOR_PATCH,
        "structure_tensor_sigma": STRUCTURE_TENSOR_SIGMA,
        "A_B_C": abc,
        "D": d,
        "contamination_safeguard": (
            "get_canonical_train_images() asserts exactly 85 images, disjointness from "
            "testA/testB, and absence of testA_/testB_ prefixes before any patch is built. "
            "No testA or testB image path was ever opened by this script."
        ),
    }
    with open(os.path.join(RESULTS_DIR, "calibration_report_v2.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[recalibration] calibration_report_v2.json written")

    return report


if __name__ == "__main__":
    main()
