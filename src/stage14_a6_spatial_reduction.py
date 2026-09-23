"""
STAGE 14 / A6: spatial-reduction diagnostic. Uses the EXACT SAME frozen
Model 3 checkpoint, rotation angle grid, sign convention, and frozen
per-angle thresholds as code_v2/stage12_rotation_validation.py -- the only
thing that changes is the scalar (q1,q2) reduction from the dense output,
per the two reductions pre-specified in configs/stage14_ablation_matrix.json
(A6): R1 (full spatial mean) and R2 (fixed original-patch valid-mask mean).

No retraining. Diagnostic only -- does NOT replace or override Stage 12's
Criterion D verdict (which remains HOLD, using center-pixel, as frozen).
"""
import sys
import os
import json
import csv

import numpy as np
from skimage.transform import rotate as sk_rotate
import torch

sys.path.insert(0, os.path.dirname(__file__))
from model3_typed_equivariant import Model3TypedEquivariant
from model3_loss import downsample_target_for_loss
from nematic_math import rotate_Q_analytic

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "results_v2", "model3", "checkpoints", "checkpoint_final.pt")
THRESHOLDS_PATH = os.path.join(PROJECT_ROOT, "results_v2", "phase2", "preregistered_thresholds_v2.json")
CACHE_DIR = os.path.join(PROJECT_ROOT, "results_v2", "baselines", "_patch_cache")
STAGE12_RESULTS = os.path.join(PROJECT_ROOT, "results_v2", "validation", "stage12", "stage12_results.json")
OUT_DIR = os.path.join(PROJECT_ROOT, "results_v2", "validation", "stage14", "A6_spatial_reduction")
os.makedirs(OUT_DIR, exist_ok=True)

ROTATION_ANGLES_DEG = [0, 15, 30, 37, 45, 60, 90, 123, 150, 173]
NONZERO_ANGLES_DEG = [15, 30, 37, 45, 60, 90, 123, 150, 173]
BATCH_SIZE = 64


def load_frozen_thresholds():
    with open(THRESHOLDS_PATH) as f:
        d = json.load(f)
    crit = d["criteria"]["D_rotation_consistency"]
    local_thresholds = {int(k): float(v) for k, v in crit["threshold_primary"]["local_thresholds"].items()}
    return local_thresholds


def load_model(device):
    model = Model3TypedEquivariant().to(device)
    model.eval()
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    return model


def load_pooled_test_patches():
    a = np.load(os.path.join(CACHE_DIR, "testA.npz"))
    b = np.load(os.path.join(CACHE_DIR, "testB.npz"))
    rgb = np.concatenate([a["rgb"], b["rgb"]], axis=0)
    q1 = np.concatenate([a["q1"], b["q1"]], axis=0)
    q2 = np.concatenate([a["q2"], b["q2"]], axis=0)
    valid = np.concatenate([a["valid"], b["valid"]], axis=0)
    return rgb, q1, q2, valid


@torch.no_grad()
def batched_q_dense(model, rgb_float_batch, device):
    """Returns the full dense (N, 2, H2, W2) q field (as numpy), batched."""
    n = rgb_float_batch.shape[0]
    outs = []
    for i in range(0, n, BATCH_SIZE):
        chunk = rgb_float_batch[i:i + BATCH_SIZE]
        x = torch.from_numpy(chunk).permute(0, 3, 1, 2).to(device)
        _, q_dense = model(x)
        outs.append(q_dense.cpu().numpy())
    return np.concatenate(outs, axis=0)


def reduce_full_mean(q_dense, fixed_mask_ds=None):
    """R1: unweighted mean over ALL spatial locations."""
    return q_dense.mean(axis=(2, 3))  # (N, 2)


def reduce_fixed_mask_mean(q_dense, fixed_mask_ds):
    """R2: mean over the fixed (per-patch) original-orientation valid mask, resolution-matched.
    fixed_mask_ds: (N, H2, W2) bool. Patches with zero valid pixels fall back to the full mean
    (documented, not silently zero-filled)."""
    N = q_dense.shape[0]
    out = np.empty((N, 2), dtype=np.float32)
    for i in range(N):
        m = fixed_mask_ds[i]
        if m.any():
            out[i, 0] = q_dense[i, 0][m].mean()
            out[i, 1] = q_dense[i, 1][m].mean()
        else:
            out[i, 0] = q_dense[i, 0].mean()
            out[i, 1] = q_dense[i, 1].mean()
    return out


def run_reduction(name, reduce_fn, model, rgb_float, fixed_mask_ds, local_thresholds, device):
    print(f"[stage14-A6] running reduction {name} ...")
    n_patches = rgb_float.shape[0]

    q0_dense = batched_q_dense(model, rgb_float, device)
    q0 = reduce_fn(q0_dense, fixed_mask_ds)
    S0 = np.sqrt(q0[:, 0] ** 2 + q0[:, 1] ** 2)
    phi0 = np.mod(0.5 * np.arctan2(q0[:, 1], q0[:, 0]), np.pi)

    per_angle_records = []
    n_passed = 0
    for alpha_deg in ROTATION_ANGLES_DEG:
        if alpha_deg == 0:
            q_rot = q0.copy()
        else:
            rot_imgs = np.empty_like(rgb_float)
            for i in range(n_patches):
                rot_imgs[i] = sk_rotate(rgb_float[i], angle=alpha_deg, mode="reflect", order=1).astype(np.float32)
            q_rot_dense = batched_q_dense(model, rot_imgs, device)
            q_rot = reduce_fn(q_rot_dense, fixed_mask_ds)

        alpha_rad = np.deg2rad(alpha_deg)
        q_expected = np.stack(rotate_Q_analytic(q0[:, 0], q0[:, 1], alpha_rad), axis=-1)
        num = np.linalg.norm(q_rot - q_expected, axis=-1)
        den = np.linalg.norm(q_expected, axis=-1) + 1e-8
        err = num / den
        model_error_mean = float(np.mean(err))

        rec = {"angle_deg": alpha_deg, "reduction": name, "model_error_mean": model_error_mean,
               "model_error_median": float(np.median(err)), "n_patches": int(n_patches)}
        if alpha_deg != 0:
            thresh = local_thresholds[alpha_deg]
            passes = bool(model_error_mean < thresh)
            rec["frozen_local_threshold"] = thresh
            rec["PASSES_criterion_D_this_angle_DIAGNOSTIC_ONLY"] = passes
            n_passed += int(passes)
        per_angle_records.append(rec)
        print(f"    [{name}] angle={alpha_deg}: model_error_mean={model_error_mean:.4f}"
              + (f"  threshold={local_thresholds[alpha_deg]:.4f}  PASS={rec.get('PASSES_criterion_D_this_angle_DIAGNOSTIC_ONLY')}" if alpha_deg != 0 else ""))

    return per_angle_records, n_passed


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[stage14-A6] device = {device}")

    local_thresholds = load_frozen_thresholds()
    model = load_model(device)
    print(f"[stage14-A6] loaded frozen Stage 10/A0 checkpoint (no retraining): {CHECKPOINT_PATH}")

    rgb, q1_native, q2_native, valid_native = load_pooled_test_patches()
    n_patches = rgb.shape[0]
    rgb_float = rgb.astype(np.float32) / 255.0
    print(f"[stage14-A6] pooled testA+testB: {n_patches} patches (same population as Stage 12)")

    q1_t = torch.from_numpy(q1_native)
    q2_t = torch.from_numpy(q2_native)
    valid_t = torch.from_numpy(valid_native)
    _, _, valid_ds_t = downsample_target_for_loss(q1_t, q2_t, valid_t, factor=2)
    fixed_mask_ds = valid_ds_t.numpy()  # (N, H2, W2) bool, from the ORIGINAL (angle=0) patch
    n_with_valid = int((fixed_mask_ds.any(axis=(1, 2))).sum())
    print(f"[stage14-A6] R2 fixed valid-mask available for {n_with_valid}/{n_patches} patches "
          f"(others fall back to full-mean for R2, documented)")

    all_results = {}
    for name, fn in [("R1_full_spatial_mean", reduce_full_mean), ("R2_fixed_original_valid_mask_mean", reduce_fixed_mask_mean)]:
        records, n_passed = run_reduction(name, fn, model, rgb_float, fixed_mask_ds, local_thresholds, device)
        all_results[name] = {"per_angle": records, "n_angles_passed_diagnostic_only": n_passed, "n_angles_total_nonzero": len(NONZERO_ANGLES_DEG)}

    stage12_center_pixel = None
    if os.path.exists(STAGE12_RESULTS):
        with open(STAGE12_RESULTS) as f:
            s12 = json.load(f)
        stage12_center_pixel = {"n_angles_passed": s12["criterion_D"]["n_angles_passed"],
                                 "per_angle_model_error_mean": {str(r["angle_deg"]): r["model_error_mean"] for r in s12["per_angle"] if not r["is_identity_sanity_check_only"]}}

    final = {
        "stage": "STAGE 14 / A6 - spatial-reduction diagnostic",
        "note": "DIAGNOSTIC ONLY. Does not replace or override Stage 12's Criterion D verdict (center-pixel, frozen, HOLD).",
        "checkpoint": CHECKPOINT_PATH,
        "n_patches": n_patches,
        "reductions": all_results,
        "stage12_center_pixel_comparison": stage12_center_pixel,
    }
    with open(os.path.join(OUT_DIR, "a6_results.json"), "w") as f:
        json.dump(final, f, indent=2)

    with open(os.path.join(OUT_DIR, "a6_per_angle_metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["reduction", "angle_deg", "model_error_mean", "model_error_median", "frozen_local_threshold", "PASSES_diagnostic_only"])
        for name, res in all_results.items():
            for r in res["per_angle"]:
                w.writerow([name, r["angle_deg"], r["model_error_mean"], r["model_error_median"],
                            r.get("frozen_local_threshold", ""), r.get("PASSES_criterion_D_this_angle_DIAGNOSTIC_ONLY", "")])

    print("\n" + "=" * 60)
    print("STAGE 14 / A6 SUMMARY (diagnostic only, does not alter Stage 12)")
    print("=" * 60)
    for name, res in all_results.items():
        print(f"{name}: {res['n_angles_passed_diagnostic_only']}/{res['n_angles_total_nonzero']} angles pass frozen threshold (context only)")
    if stage12_center_pixel:
        print(f"Stage 12 center-pixel (frozen, official): {stage12_center_pixel['n_angles_passed']}/9")
    print("=" * 60)
    return final


if __name__ == "__main__":
    main()
