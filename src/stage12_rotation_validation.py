"""
STAGE 12: rotation consistency validation. Loads the FROZEN Stage 10 Model 3
checkpoint (no retraining) and measures EMPIRICAL, finite-grid, whole-network
rotation consistency under pixel-domain (skimage) rotation at the exact
angle grid and against the exact per-angle local_thresholds already frozen
in results_v2/phase2/preregistered_thresholds_v2.json's D_rotation_consistency
criterion (read-only; nothing there is recomputed, tuned, or flattened here).

This is a DIFFERENT question from Stage 7/8's escnn-native `.transform(g)`
tests (exact algebraic action of the 16 D8 group elements on the internal
representation) -- this stage does NOT claim mathematically exact continuous
equivariance for the complete network; it measures interpolation-and-
architecture-inclusive empirical consistency at specific degree angles, for
direct comparability against the real-H&E floor (which was itself built the
same way, via pixel-domain skimage rotation of real patches).

This stage does NOT alter Stage 11's HOLD status, does not combine with it,
and is not gated on it.

See configs/stage12_rotation_validation_config.json for the full protocol
and the documented resolution of every underspecified detail (population,
scalar-reduction convention, rotation sign convention) -- all resolved by
reusing existing, already-verified project artifacts/precedents, not by
invention.
"""
import sys
import os
import json

import numpy as np
from skimage.transform import rotate as sk_rotate
import torch

sys.path.insert(0, os.path.dirname(__file__))
from model3_typed_equivariant import Model3TypedEquivariant
from nematic_math import rotate_Q_analytic, q_to_phi_S, angular_error_mod_pi

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "results_v2", "model3", "checkpoints", "checkpoint_final.pt")
THRESHOLDS_PATH = os.path.join(PROJECT_ROOT, "results_v2", "phase2", "preregistered_thresholds_v2.json")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "stage12_rotation_validation_config.json")
CACHE_DIR = os.path.join(PROJECT_ROOT, "results_v2", "baselines", "_patch_cache")
OUT_DIR = os.path.join(PROJECT_ROOT, "results_v2", "validation", "stage12")
os.makedirs(OUT_DIR, exist_ok=True)

ROTATION_ANGLES_DEG = [0, 15, 30, 37, 45, 60, 90, 123, 150, 173]
NONZERO_ANGLES_DEG = [15, 30, 37, 45, 60, 90, 123, 150, 173]
BATCH_SIZE = 64


def load_frozen_thresholds():
    with open(THRESHOLDS_PATH) as f:
        d = json.load(f)
    crit = d["criteria"]["D_rotation_consistency"]
    assert crit["rotation_angles_deg"] == ROTATION_ANGLES_DEG, (
        "frozen angle grid does not match expected -- STOP, do not proceed with a mismatched grid"
    )
    local_thresholds = {int(k): float(v) for k, v in crit["threshold_primary"]["local_thresholds"].items()}
    assert set(local_thresholds.keys()) == set(NONZERO_ANGLES_DEG)
    return local_thresholds, crit


def load_model(device):
    model = Model3TypedEquivariant().to(device)
    model.eval()  # escnn filter-buffer lifecycle: eval() BEFORE load_state_dict (Stage 8 precedent)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    assert model.q_out_type.size == 2
    assert model.group.order() == 16
    return model


def load_pooled_test_patches():
    a = np.load(os.path.join(CACHE_DIR, "testA.npz"))
    b = np.load(os.path.join(CACHE_DIR, "testB.npz"))
    rgb = np.concatenate([a["rgb"], b["rgb"]], axis=0)  # (N, 128, 128, 3) uint8
    image_id = np.concatenate([a["image_id"], b["image_id"]], axis=0)
    return rgb, image_id


@torch.no_grad()
def batched_center_pixel_q(model, rgb_float_batch, device):
    """
    rgb_float_batch: (N, H, W, 3) float32 in [0,1].
    Returns (N, 2) numpy array = q_dense at the spatial center pixel, for
    each image in the batch -- same center-pixel reduction convention as
    oracle_baselines.py::structure_tensor_estimate (see config doc).
    """
    n = rgb_float_batch.shape[0]
    out = np.empty((n, 2), dtype=np.float32)
    for i in range(0, n, BATCH_SIZE):
        chunk = rgb_float_batch[i:i + BATCH_SIZE]
        x = torch.from_numpy(chunk).permute(0, 3, 1, 2).to(device)
        _, q_dense = model(x)
        cy, cx = q_dense.shape[-2] // 2, q_dense.shape[-1] // 2
        out[i:i + BATCH_SIZE] = q_dense[:, :, cy, cx].cpu().numpy()
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[stage12] device = {device}")

    local_thresholds, crit_raw = load_frozen_thresholds()
    print(f"[stage12] loaded FROZEN D_rotation_consistency criterion from {THRESHOLDS_PATH} (read-only)")
    print(f"[stage12] frozen angle grid: {ROTATION_ANGLES_DEG}")
    print(f"[stage12] frozen local_thresholds (9 non-zero angles): {local_thresholds}")

    model = load_model(device)
    print(f"[stage12] loaded frozen Stage 10 checkpoint: {CHECKPOINT_PATH}")
    print(f"[stage12] verified: q_out_type.size={model.q_out_type.size}, group.order()={model.group.order()}")

    rgb, image_id = load_pooled_test_patches()
    n_patches = rgb.shape[0]
    print(f"[stage12] pooled testA+testB: {n_patches} patches (reused from Stage 11's population, "
          f"no exclusions, fragmented testB images included)")

    rgb_float = rgb.astype(np.float32) / 255.0  # (N, 128, 128, 3)

    print("[stage12] computing q0 (angle=0 reference) via center-pixel reduction...")
    q0 = batched_center_pixel_q(model, rgb_float, device)  # (N, 2)
    S0 = np.sqrt(q0[:, 0] ** 2 + q0[:, 1] ** 2)
    phi0 = 0.5 * np.arctan2(q0[:, 1], q0[:, 0])
    phi0 = np.mod(phi0, np.pi)

    per_angle_records = []
    per_angle_errors_for_criterion = {}
    nan_or_inf_detected = False

    for alpha_deg in ROTATION_ANGLES_DEG:
        print(f"[stage12] angle = {alpha_deg} deg ...")
        if alpha_deg == 0:
            q_rot = q0.copy()
        else:
            rot_imgs = np.empty_like(rgb_float)
            for i in range(n_patches):
                # MODEL's own verified sign convention: angle=+alpha_deg
                # (code_v2/run_synthetic_benchmark.py precedent) -- NOT the
                # oracle's angle=-alpha_deg convention. See config doc.
                rot_imgs[i] = sk_rotate(rgb_float[i], angle=alpha_deg, mode="reflect", order=1).astype(np.float32)
            q_rot = batched_center_pixel_q(model, rot_imgs, device)

        finite = np.isfinite(q_rot).all()
        if not finite:
            nan_or_inf_detected = True

        alpha_rad = np.deg2rad(alpha_deg)
        q_expected = np.stack(rotate_Q_analytic(q0[:, 0], q0[:, 1], alpha_rad), axis=-1)  # (N, 2)

        num = np.linalg.norm(q_rot - q_expected, axis=-1)
        den = np.linalg.norm(q_expected, axis=-1) + 1e-8
        err_per_patch = num / den
        model_error_mean = float(np.mean(err_per_patch))
        model_error_median = float(np.median(err_per_patch))

        S_rot = np.sqrt(q_rot[:, 0] ** 2 + q_rot[:, 1] ** 2)
        S_abs_diff_mean = float(np.mean(np.abs(S_rot - S0)))

        phi_rot = np.mod(0.5 * np.arctan2(q_rot[:, 1], q_rot[:, 0]), np.pi)
        phi_expected = np.mod(phi0 + alpha_rad, np.pi)
        phi_deviation_deg_mean = float(np.degrees(np.mean(angular_error_mod_pi(phi_rot, phi_expected))))

        rec = {
            "angle_deg": alpha_deg,
            "is_identity_sanity_check_only": alpha_deg == 0,
            "n_patches": int(n_patches),
            "model_error_mean": model_error_mean,
            "model_error_median": model_error_median,
            "model_error_std": float(np.std(err_per_patch)),
            "S_abs_diff_mean": S_abs_diff_mean,
            "phi_deviation_from_expected_shift_deg_mean": phi_deviation_deg_mean,
            "finite": bool(finite),
        }
        if alpha_deg != 0:
            thresh = local_thresholds[alpha_deg]
            passes = bool(model_error_mean < thresh)
            rec["frozen_local_threshold"] = thresh
            rec["PASSES_criterion_D_this_angle"] = passes
            per_angle_errors_for_criterion[alpha_deg] = passes
        per_angle_records.append(rec)
        print(f"    model_error_mean={model_error_mean:.4f}"
              + (f"  threshold={local_thresholds[alpha_deg]:.4f}  PASS={per_angle_errors_for_criterion.get(alpha_deg)}"
                 if alpha_deg != 0 else "  (identity sanity check, not scored)"))

    n_passed = sum(1 for v in per_angle_errors_for_criterion.values() if v)
    n_total_nonzero = len(NONZERO_ANGLES_DEG)
    criterion_D_pass = bool(n_passed >= 7)

    overall_mean_error_nonzero = float(np.mean([r["model_error_mean"] for r in per_angle_records if not r["is_identity_sanity_check_only"]]))

    results = {
        "stage": "STAGE 12 - Rotation Consistency Validation",
        "device": device,
        "n_patches_pooled_testA_testB": int(n_patches),
        "rotation_angles_deg": ROTATION_ANGLES_DEG,
        "nonzero_angles_deg": NONZERO_ANGLES_DEG,
        "per_angle": per_angle_records,
        "criterion_D": {
            "rule": "model_error(angle) < frozen local_threshold(angle) for >= 7 of 9 non-zero angles",
            "n_angles_passed": n_passed,
            "n_angles_total_nonzero": n_total_nonzero,
            "PASSES_criterion_D": criterion_D_pass,
        },
        "overall_mean_error_nonzero_angles_context_only_nongating": overall_mean_error_nonzero,
        "nan_or_inf_detected": nan_or_inf_detected,
        "stage_11_relationship": "Stage 12 does NOT alter, rescue, or combine with Stage 11's HOLD status. Stage 11 remains HOLD regardless of this stage's Criterion D outcome.",
    }

    with open(os.path.join(OUT_DIR, "stage12_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    import csv
    with open(os.path.join(OUT_DIR, "per_angle_metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["angle_deg", "is_identity_sanity_check_only", "n_patches", "model_error_mean",
                    "model_error_median", "model_error_std", "S_abs_diff_mean",
                    "phi_deviation_from_expected_shift_deg_mean", "finite",
                    "frozen_local_threshold", "PASSES_criterion_D_this_angle"])
        for r in per_angle_records:
            w.writerow([r["angle_deg"], r["is_identity_sanity_check_only"], r["n_patches"],
                        r["model_error_mean"], r["model_error_median"], r["model_error_std"],
                        r["S_abs_diff_mean"], r["phi_deviation_from_expected_shift_deg_mean"], r["finite"],
                        r.get("frozen_local_threshold", ""), r.get("PASSES_criterion_D_this_angle", "")])

    with open(os.path.join(OUT_DIR, "threshold_comparison.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["angle_deg", "model_error_mean", "frozen_local_threshold", "margin_threshold_minus_error", "PASS"])
        for r in per_angle_records:
            if r["is_identity_sanity_check_only"]:
                continue
            margin = r["frozen_local_threshold"] - r["model_error_mean"]
            w.writerow([r["angle_deg"], r["model_error_mean"], r["frozen_local_threshold"], margin, r["PASSES_criterion_D_this_angle"]])

    import shutil
    shutil.copy(CONFIG_PATH, os.path.join(OUT_DIR, "stage12_config_copy.json"))

    print("\n" + "=" * 60)
    print("STAGE 12 CRITERION D SUMMARY")
    print("=" * 60)
    print(json.dumps(results["criterion_D"], indent=2))
    print(f"nan_or_inf_detected: {nan_or_inf_detected}")
    print("Stage 11 remains HOLD (unaffected by this stage).")
    print("=" * 60)
    return results


if __name__ == "__main__":
    main()
