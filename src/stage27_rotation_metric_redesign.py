"""
STAGE 27 TASK 3: rotation metric redesign. The NETWORK IS NOT CHANGED (the frozen Stage 10 Model 3
checkpoint is loaded read-only, same as Stage 12). Only the EVALUATION of rotation consistency is
redesigned, to separate four possible explanations for Criterion D's failure: network behavior,
pixel-domain interpolation, pixel-grid/fixed-point geometry, and the normalization/evaluation design
itself (small-S instability). Five metrics are computed on the SAME pooled testA+testB population
(2083 patches) and the SAME nine rotation angles as the frozen Criterion D:

  M1 center-pixel (historical)   -- exactly Stage 12's own method, reproduced here for direct comparison
  M2 exact-lattice-valid         -- restricts to angles where pixel-domain rotation is exact (0, 90 deg;
                                     the other 7 angles require interpolation and cannot be made "exact" by
                                     construction -- reported as not-applicable for those angles, not
                                     silently interpolated and relabeled "exact")
  M3 local-neighborhood (5x5)    -- mean over a 5x5 neighborhood of the dense output around the rotation's
                                     true fixed point, instead of a single sampled pixel
  M4 full-field                  -- compares the network's ENTIRE 64x64 dense output field (spatially
                                     realigned) against the analytically rotated field, not one point
  M5 stable-normalization        -- same full-field comparison as M4, but reports BOTH the normalized error
                                     (Stage 12's definition) AND an absolute (non-normalized) error, plus a
                                     floor-normalized error using max(||q_expected||, S_FLOOR) to avoid
                                     blow-up when S approx 0

Writes only under results_v2/stage27_robustness/rotation_metrics/. Never touches results_v2/model3/ or
results_v2/validation/ (frozen Stage 10-12 outputs).
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from scipy.ndimage import shift as nd_shift
from skimage.transform import rotate as sk_rotate

sys.path.insert(0, os.path.dirname(__file__))
from model3_typed_equivariant import Model3TypedEquivariant
from nematic_math import rotate_Q_analytic

ROOT = os.path.join(os.path.dirname(__file__), "..")
CACHE_DIR = os.path.join(ROOT, "results_v2", "baselines", "_patch_cache")
THRESHOLDS_PATH = os.path.join(ROOT, "results_v2", "phase2", "preregistered_thresholds_v2.json")
OUT_DIR = os.path.join(ROOT, "results_v2", "stage27_robustness", "rotation_metrics")
os.makedirs(OUT_DIR, exist_ok=True)

ROTATION_ANGLES_DEG = [0, 15, 30, 37, 45, 60, 90, 123, 150, 173]
NONZERO_ANGLES_DEG = [15, 30, 37, 45, 60, 90, 123, 150, 173]
EXACT_LATTICE_ANGLES = {0, 90}  # only multiples of 90 deg are exact on a square pixel grid (no interpolation)
BATCH_SIZE = 64
S_FLOOR = 0.05  # fixed floor, same order of magnitude as the mean predicted S reported in the manuscript (~0.03)


def load_model(ckpt_path, device):
    model = Model3TypedEquivariant().to(device)
    model.eval()
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model


def load_pooled_test_patches(cache_dir):
    a = np.load(os.path.join(cache_dir, "testA.npz"))
    b = np.load(os.path.join(cache_dir, "testB.npz"))
    rgb = np.concatenate([a["rgb"], b["rgb"]], axis=0)
    return rgb


@torch.no_grad()
def dense_output(model, rgb_float_batch, device, batch_size=BATCH_SIZE):
    """Returns the full (N, 2, 64, 64) dense Q field for a batch of 128x128x3 float images."""
    n = rgb_float_batch.shape[0]
    outs = []
    for i in range(0, n, batch_size):
        chunk = rgb_float_batch[i:i + batch_size]
        x = torch.from_numpy(chunk).permute(0, 3, 1, 2).to(device)
        _, q_dense = model(x)
        outs.append(q_dense.cpu().numpy())
    return np.concatenate(outs, axis=0)  # (N, 2, 64, 64)


def rotate_field_bilinear(field, angle_deg):
    """Rotates a (2, H, W) dense field about its OWN center by angle_deg using bilinear interpolation
    (scipy.ndimage), applied independently to each of the 2 channels. This "unrotates" a dense output
    computed on a rotated input, to spatially realign it with the unrotated field's grid, for the
    full-field comparison (M4/M5). Uses the same +alpha convention as the rest of the project
    (skimage.transform.rotate with a positive angle argument corresponds to this same sense of rotation)."""
    from scipy.ndimage import rotate as nd_rotate
    out = np.stack([nd_rotate(field[c], angle=angle_deg, reshape=False, order=1, mode="reflect") for c in range(2)])
    return out


def crop_center(field, k):
    H, W = field.shape[-2:]
    cy, cx = H // 2, W // 2
    half = k // 2
    return field[..., cy - half:cy + half + 1, cx - half:cx + half + 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.path.join(ROOT, "results_v2", "model3", "checkpoints", "checkpoint_final.pt"))
    ap.add_argument("--cache_dir", default=CACHE_DIR)
    ap.add_argument("--tag", default="seed42_frozen")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(THRESHOLDS_PATH) as f:
        local_thresholds = {int(k): float(v) for k, v in json.load(f)["criteria"]["D_rotation_consistency"]["threshold_primary"]["local_thresholds"].items()}

    model = load_model(args.checkpoint, device)
    rgb = load_pooled_test_patches(args.cache_dir)
    rgb_float = rgb.astype(np.float32) / 255.0
    n_patches = rgb_float.shape[0]
    print(f"[stage27-task3] {args.tag}: {n_patches} pooled testA+testB patches, checkpoint={args.checkpoint}")

    # dense field at angle 0 (reference)
    field0 = dense_output(model, rgb_float, device)  # (N, 2, 64, 64)
    q0_center = field0[:, :, 32, 32]  # (N, 2) -- same center-pixel convention as Stage 12
    S0_center = np.hypot(q0_center[:, 0], q0_center[:, 1])
    phi0_center = np.mod(0.5 * np.arctan2(q0_center[:, 1], q0_center[:, 0]), np.pi)
    q0_5x5 = crop_center(field0, 5).mean(axis=(-2, -1))  # (N, 2) M3 reference

    rows = []
    for alpha_deg in ROTATION_ANGLES_DEG:
        print(f"[stage27-task3] angle={alpha_deg} deg")
        if alpha_deg == 0:
            field_rot = field0
        else:
            rot_imgs = np.empty_like(rgb_float)
            for i in range(n_patches):
                rot_imgs[i] = sk_rotate(rgb_float[i], angle=alpha_deg, mode="reflect", order=1).astype(np.float32)
            field_rot = dense_output(model, rot_imgs, device)  # (N, 2, 64, 64)

        alpha_rad = np.deg2rad(alpha_deg)

        # ---- M1: center pixel (historical, = Stage 12) ----
        q_rot_center = field_rot[:, :, 32, 32]
        q_exp_center = np.stack(rotate_Q_analytic(q0_center[:, 0], q0_center[:, 1], alpha_rad), axis=-1)
        err_m1 = np.linalg.norm(q_rot_center - q_exp_center, axis=-1) / (np.linalg.norm(q_exp_center, axis=-1) + 1e-8)

        # ---- M2: exact-lattice-valid (only meaningful at 0, 90 deg) ----
        if alpha_deg in EXACT_LATTICE_ANGLES:
            k90 = int(round(alpha_deg / 90))
            rot_imgs_exact = np.rot90(rgb_float, k=k90, axes=(1, 2)).copy()
            field_rot_exact = dense_output(model, rot_imgs_exact.astype(np.float32), device)
            q_rot_exact_center = field_rot_exact[:, :, 32, 32]
            q_exp_exact = np.stack(rotate_Q_analytic(q0_center[:, 0], q0_center[:, 1], alpha_rad), axis=-1)
            err_m2 = np.linalg.norm(q_rot_exact_center - q_exp_exact, axis=-1) / (np.linalg.norm(q_exp_exact, axis=-1) + 1e-8)
            m2_mean = float(err_m2.mean())
        else:
            m2_mean = None  # not applicable: no exact (interpolation-free) pixel-lattice rotation exists at this angle

        # ---- M3: local 5x5 neighborhood mean ----
        q_rot_5x5 = crop_center(field_rot, 5).mean(axis=(-2, -1))
        q_exp_5x5 = np.stack(rotate_Q_analytic(q0_5x5[:, 0], q0_5x5[:, 1], alpha_rad), axis=-1)
        err_m3 = np.linalg.norm(q_rot_5x5 - q_exp_5x5, axis=-1) / (np.linalg.norm(q_exp_5x5, axis=-1) + 1e-8)

        # ---- M4 / M5: full-field agreement ----
        # Realign field_rot to the unrotated grid by rotating it back by -alpha, then compare pointwise
        # against the analytic rotation of field0 at every spatial location (not just the center).
        if alpha_deg == 0:
            field_rot_realigned = field_rot
        else:
            field_rot_realigned = np.stack([rotate_field_bilinear(field_rot[i], -alpha_deg) for i in range(n_patches)])
        q1_0, q2_0 = field0[:, 0], field0[:, 1]  # (N, 64, 64) each
        q1_exp, q2_exp = rotate_Q_analytic(q1_0, q2_0, alpha_rad)
        diff = np.stack([field_rot_realigned[:, 0] - q1_exp, field_rot_realigned[:, 1] - q2_exp], axis=1)  # (N,2,64,64)
        num_field = np.linalg.norm(diff, axis=1)  # (N, 64, 64)
        den_field = np.hypot(q1_exp, q2_exp)
        err_m4 = num_field / (den_field + 1e-8)  # Stage-12-style normalization, but full-field
        err_m4_abs = num_field  # absolute, non-normalized
        den_floor = np.maximum(den_field, S_FLOOR)
        err_m5_floor = num_field / den_floor  # stable normalization

        rec = {
            "angle_deg": alpha_deg,
            "M1_center_pixel_mean": float(err_m1.mean()),
            "M1_center_pixel_median": float(np.median(err_m1)),
            "M2_exact_lattice_mean": m2_mean,
            "M2_applicable": alpha_deg in EXACT_LATTICE_ANGLES,
            "M3_local_5x5_mean": float(err_m3.mean()),
            "M3_local_5x5_median": float(np.median(err_m3)),
            "M4_full_field_normalized_mean": float(err_m4.mean()),
            "M4_full_field_normalized_median": float(np.median(err_m4)),
            "M4b_full_field_absolute_mean": float(err_m4_abs.mean()),
            "M5_full_field_floor_normalized_mean": float(err_m5_floor.mean()),
            "M5_full_field_floor_normalized_median": float(np.median(err_m5_floor)),
            "mean_predicted_S_at_center": float(S0_center.mean()) if alpha_deg == 0 else None,
        }
        if alpha_deg != 0:
            thresh = local_thresholds[alpha_deg]
            rec["frozen_local_threshold_M1_convention"] = thresh
            rec["M1_PASSES"] = bool(err_m1.mean() < thresh)
            rec["M3_PASSES_same_threshold"] = bool(err_m3.mean() < thresh)
            rec["M4_PASSES_same_threshold"] = bool(err_m4.mean() < thresh)
        rows.append(rec)
        print(f"    M1(center)={rec['M1_center_pixel_mean']:.4f}  M3(5x5)={rec['M3_local_5x5_mean']:.4f}  "
              f"M4(full-field norm)={rec['M4_full_field_normalized_mean']:.4f}  "
              f"M4b(full-field abs)={rec['M4b_full_field_absolute_mean']:.4f}  "
              f"M5(floor-norm)={rec['M5_full_field_floor_normalized_mean']:.4f}"
              + (f"  M2(exact-lattice)={m2_mean:.4f}" if m2_mean is not None else "  M2=n/a (no exact rotation at this angle)"))

    n_m1_pass = sum(1 for r in rows if r.get("M1_PASSES"))
    n_m3_pass = sum(1 for r in rows if r.get("M3_PASSES_same_threshold"))
    n_m4_pass = sum(1 for r in rows if r.get("M4_PASSES_same_threshold"))

    out = {
        "tag": args.tag, "checkpoint": args.checkpoint, "n_patches": int(n_patches),
        "S_FLOOR": S_FLOOR, "per_angle": rows,
        "pass_counts_vs_frozen_M1_thresholds_informational_only": {
            "M1_center_pixel_n_passed_of_9": n_m1_pass,
            "M3_local_5x5_n_passed_of_9": n_m3_pass,
            "M4_full_field_n_passed_of_9": n_m4_pass,
            "note": "The frozen local_thresholds were calibrated for the CENTER-PIXEL convention (M1) only. "
                    "Applying them to M3/M4 (different summarization) is diagnostic context, not a re-application "
                    "of Criterion D under a different, uncalibrated rule -- no new gating criterion is created here.",
        },
    }
    with open(os.path.join(OUT_DIR, f"rotation_metrics_{args.tag}.json"), "w") as f:
        json.dump(out, f, indent=2)

    csv_path = os.path.join(OUT_DIR, "STAGE27_ROTATION_ANALYSIS.csv")
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["tag", "angle_deg", "M1_center_pixel_mean", "M2_exact_lattice_mean", "M2_applicable",
                        "M3_local_5x5_mean", "M4_full_field_normalized_mean", "M4b_full_field_absolute_mean",
                        "M5_full_field_floor_normalized_mean", "frozen_local_threshold_M1_convention",
                        "M1_PASSES", "M3_PASSES_same_threshold", "M4_PASSES_same_threshold"])
        for r in rows:
            w.writerow([args.tag, r["angle_deg"], r["M1_center_pixel_mean"], r["M2_exact_lattice_mean"], r["M2_applicable"],
                        r["M3_local_5x5_mean"], r["M4_full_field_normalized_mean"], r["M4b_full_field_absolute_mean"],
                        r["M5_full_field_floor_normalized_mean"], r.get("frozen_local_threshold_M1_convention", ""),
                        r.get("M1_PASSES", ""), r.get("M3_PASSES_same_threshold", ""), r.get("M4_PASSES_same_threshold", "")])
    print(f"[stage27-task3] wrote {OUT_DIR}/rotation_metrics_{args.tag}.json and appended to {csv_path}")
    print(json.dumps(out["pass_counts_vs_frozen_M1_thresholds_informational_only"], indent=2))
    return out


if __name__ == "__main__":
    main()
