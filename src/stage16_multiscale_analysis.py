"""
STAGE 16: multi-scale orientation-tensor analysis. Executes EXACTLY the
pre-frozen, checksummed scale set in
configs/stage16_multiscale_analysis_matrix.json against the frozen Stage
10 Model 3 checkpoint (no retraining). Reuses Stage 11's anatomical
correspondence metrics and Stage 12's rotation-consistency protocol
unmodified; only the spatial reduction applied to the existing dense
(B,2,H2,W2) prediction is varied across 6 frozen scales.

Diagnostic only. Does not alter Stage 11 (HOLD) or Stage 12 (HOLD)'s
official verdicts; does not select or promote a "best" scale.
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
from nematic_math import rotate_Q_analytic, angular_error_mod_pi
from stage9_train_baselines import to_tensors
from stage11_geometric_validation import get_predictions, compute_full_metrics

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "results_v2", "model3", "checkpoints", "checkpoint_final.pt")
THRESHOLDS_PATH = os.path.join(PROJECT_ROOT, "results_v2", "phase2", "preregistered_thresholds_v2.json")
CACHE_DIR = os.path.join(PROJECT_ROOT, "results_v2", "baselines", "_patch_cache")
OUT_DIR = os.path.join(PROJECT_ROOT, "results_v2", "validation", "stage16")

# Internal arrays are keyed by WINDOW SIZE directly: 1 (center pixel), 3, 5, 9, 17, "full"
# (six entries), matching the frozen matrix's scale_id 0..5 respectively.
WINDOWS = [1, 3, 5, 9, 17, "full"]
SCALE_ID_OF_WINDOW = {1: 0, 3: 1, 5: 2, 9: 3, 17: 4, "full": 5}
SCALE_NAMES = {1: "center_pixel", 3: "3x3_mean", 5: "5x5_mean", 9: "9x9_mean", 17: "17x17_mean", "full": "full_field"}

ROTATION_ANGLES_DEG = [0, 15, 30, 37, 45, 60, 90, 123, 150, 173]
NONZERO_ANGLES_DEG = [15, 30, 37, 45, 60, 90, 123, 150, 173]
BATCH_SIZE = 64
H2 = 64  # dense field spatial resolution for 128x128 input (block1's one stride-2 pool)


def load_frozen_thresholds():
    with open(THRESHOLDS_PATH) as f:
        d = json.load(f)
    crit = d["criteria"]["D_rotation_consistency"]
    return {int(k): float(v) for k, v in crit["threshold_primary"]["local_thresholds"].items()}


def load_model(device):
    model = Model3TypedEquivariant().to(device)
    model.eval()
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    return model


def window_slice(window):
    if window == "full":
        return slice(0, H2)
    half = window // 2
    start = H2 // 2 - half
    return slice(start, start + window)


def windowed_qmean_unmasked(q1, q2, window):
    """q1, q2: (N, H2, W2) numpy arrays. Returns (N,) S and phi from an
    unweighted vector-space mean of (q1,q2) over the window (or the entire
    field if window == 'full', unmasked -- matches Stage 14/A6's R1)."""
    sl = window_slice(window)
    q1w = q1[:, sl, sl].mean(axis=(1, 2))
    q2w = q2[:, sl, sl].mean(axis=(1, 2))
    S = np.hypot(q1w, q2w)
    phi = np.mod(0.5 * np.arctan2(q2w, q1w), np.pi)
    return S, phi, q1w, q2w


def windowed_qmean_masked_full_valid(q1, q2, valid):
    """Scale 5's ANATOMY definition: masked mean over ALL anatomically-valid
    pixels in the patch (== Stage 11's per_patch_scalar_qs reduction, unchanged)."""
    N = q1.shape[0]
    S = np.zeros(N); phi = np.zeros(N)
    for i in range(N):
        v = valid[i]
        if v.any():
            m1, m2 = q1[i][v].mean(), q2[i][v].mean()
        else:
            m1, m2 = 0.0, 0.0
        S[i] = np.hypot(m1, m2)
        phi[i] = np.mod(0.5 * np.arctan2(m2, m1), np.pi)
    return S, phi


# ==================================================================== ANATOMY ANALYSIS

def run_anatomy_analysis(model, device):
    print("\n" + "=" * 60 + "\n[stage16] ANATOMY CORRESPONDENCE PER SCALE\n" + "=" * 60)
    per_split_arrays = {}
    for split_name in ["testA", "testB"]:
        d = np.load(os.path.join(CACHE_DIR, f"{split_name}.npz"), allow_pickle=True)
        arrs = {k: d[k] for k in d.files}
        assert all(str(i).startswith(split_name + "_") for i in arrs["image_id"])
        q1_pred, q2_pred, q1_t, q2_t, valid = get_predictions(model, arrs, device)
        q1p, q2p = q1_pred.numpy(), q2_pred.numpy()
        q1t, q2t = q1_t.numpy(), q2_t.numpy()
        v = valid.numpy()
        has_valid = v.any(axis=(1, 2))

        # FIXED target reduction (Stage 11's own definition), same for every scale
        S_t, phi_t = windowed_qmean_masked_full_valid(q1t, q2t, v)

        per_split_arrays[split_name] = {"q1p": q1p, "q2p": q2p, "S_t": S_t, "phi_t": phi_t, "valid": v,
                                          "has_valid": has_valid, "image_id": arrs["image_id"], "n_images": len(set(arrs["image_id"]))}
        print(f"[stage16] {split_name}: {per_split_arrays[split_name]['n_images']} images, "
              f"{len(has_valid)} patches, {int(has_valid.sum())} with valid target")

    results_by_scale = {}
    for window in WINDOWS:
        scale_id = SCALE_ID_OF_WINDOW[window]
        print(f"[stage16] anatomy scale_id={scale_id} ({SCALE_NAMES[window]}, window={window}) ...")
        per_pop = {}
        pooled = {"S_pred": [], "phi_pred": [], "S_t": [], "phi_t": []}
        for split_name in ["testA", "testB"]:
            d = per_split_arrays[split_name]
            hv = d["has_valid"]
            if window == "full":
                # ANATOMY scale-5 definition: masked mean over anatomically-valid pixels
                # (same mask used for the fixed target reduction) -- reproduces Stage 11 exactly.
                S_pred, phi_pred = windowed_qmean_masked_full_valid(d["q1p"], d["q2p"], d["valid"])
            else:
                S_pred, phi_pred, _, _ = windowed_qmean_unmasked(d["q1p"], d["q2p"], window)

            S_pred_v, phi_pred_v = S_pred[hv], phi_pred[hv]
            S_t_v, phi_t_v = d["S_t"][hv], d["phi_t"][hv]
            n_valid = int(hv.sum())
            metrics = compute_full_metrics(S_pred_v, phi_pred_v, S_t_v, phi_t_v, seed=scale_id * 10 + (0 if split_name == "testA" else 1))
            per_pop[split_name] = {"n_images": d["n_images"], "n_patches_total": len(hv), "n_patches_valid": n_valid, "metrics": metrics}
            pooled["S_pred"].append(S_pred_v); pooled["phi_pred"].append(phi_pred_v)
            pooled["S_t"].append(S_t_v); pooled["phi_t"].append(phi_t_v)

        pooled_S_pred = np.concatenate(pooled["S_pred"]); pooled_phi_pred = np.concatenate(pooled["phi_pred"])
        pooled_S_t = np.concatenate(pooled["S_t"]); pooled_phi_t = np.concatenate(pooled["phi_t"])
        pooled_metrics = compute_full_metrics(pooled_S_pred, pooled_phi_pred, pooled_S_t, pooled_phi_t, seed=scale_id * 10 + 5)
        per_pop["pooled"] = {"n_images": 80, "n_patches_total": len(pooled_S_pred), "n_patches_valid": len(pooled_S_pred), "metrics": pooled_metrics}

        results_by_scale[str(window)] = per_pop
        print(f"    pooled: ang_mean={pooled_metrics['A_angular_correspondence']['mean_deg']:.2f} "
              f"glass_A={pooled_metrics['A_angular_correspondence']['glass_delta']:.4f} "
              f"pearson_r={pooled_metrics['B_order_magnitude_correlation']['pearson_r']:.4f} "
              f"D_Q={pooled_metrics['C_tensor_similarity']['mean_D_Q']:.4f}")

    return results_by_scale


# ==================================================================== ROTATION ANALYSIS

@torch.no_grad()
def batched_q_dense(model, rgb_float_batch, device):
    n = rgb_float_batch.shape[0]
    outs = []
    for i in range(0, n, BATCH_SIZE):
        chunk = rgb_float_batch[i:i + BATCH_SIZE]
        x = torch.from_numpy(chunk).permute(0, 3, 1, 2).to(device)
        _, q_dense = model(x)
        outs.append(q_dense.cpu().numpy())
    return np.concatenate(outs, axis=0)


def run_rotation_analysis(model, device, local_thresholds):
    print("\n" + "=" * 60 + "\n[stage16] ROTATION CONSISTENCY PER SCALE\n" + "=" * 60)
    a = np.load(os.path.join(CACHE_DIR, "testA.npz"))
    b = np.load(os.path.join(CACHE_DIR, "testB.npz"))
    rgb = np.concatenate([a["rgb"], b["rgb"]], axis=0)
    image_id = np.concatenate([a["image_id"], b["image_id"]], axis=0)
    n_patches = rgb.shape[0]
    rgb_float = rgb.astype(np.float32) / 255.0
    is_testA = np.array([str(i).startswith("testA_") for i in image_id])
    print(f"[stage16] pooled testA+testB: {n_patches} patches (testA={is_testA.sum()}, testB={(~is_testA).sum()})")

    # q0 (angle=0) per window
    q0_dense = batched_q_dense(model, rgb_float, device)
    q0_by_window = {}
    for window in WINDOWS:
        if window == "full":
            S0, phi0, q1_0, q2_0 = windowed_qmean_unmasked(q0_dense[:, 0], q0_dense[:, 1], "full")
        else:
            S0, phi0, q1_0, q2_0 = windowed_qmean_unmasked(q0_dense[:, 0], q0_dense[:, 1], window)
        q0_by_window[window] = (q1_0, q2_0)

    results_by_scale = {str(w): {"per_angle": []} for w in WINDOWS}

    for alpha_deg in ROTATION_ANGLES_DEG:
        print(f"[stage16] angle={alpha_deg} ...")
        if alpha_deg == 0:
            q_rot_dense = q0_dense
        else:
            rot_imgs = np.empty_like(rgb_float)
            for i in range(n_patches):
                rot_imgs[i] = sk_rotate(rgb_float[i], angle=alpha_deg, mode="reflect", order=1).astype(np.float32)
            q_rot_dense = batched_q_dense(model, rot_imgs, device)

        alpha_rad = np.deg2rad(alpha_deg)
        for window in WINDOWS:
            q1_0, q2_0 = q0_by_window[window]
            q1_r, q2_r = windowed_qmean_unmasked(q_rot_dense[:, 0], q_rot_dense[:, 1], window)[2:4]

            q_expected = np.stack(rotate_Q_analytic(q1_0, q2_0, alpha_rad), axis=-1)
            q_rot_vec = np.stack([q1_r, q2_r], axis=-1)
            num = np.linalg.norm(q_rot_vec - q_expected, axis=-1)
            den = np.linalg.norm(q_expected, axis=-1) + 1e-8
            err = num / den

            S0 = np.hypot(q1_0, q2_0); S_r = np.hypot(q1_r, q2_r)
            S_abs_diff = np.abs(S_r - S0)
            phi0 = np.mod(0.5 * np.arctan2(q2_0, q1_0), np.pi)
            phi_r = np.mod(0.5 * np.arctan2(q2_r, q1_r), np.pi)
            phi_expected = np.mod(phi0 + alpha_rad, np.pi)
            phi_dev = np.degrees(angular_error_mod_pi(phi_r, phi_expected))

            rec = {
                "angle_deg": alpha_deg, "n_patches": int(n_patches),
                "pooled_model_error_mean": float(err.mean()), "pooled_model_error_median": float(np.median(err)),
                "testA_model_error_mean": float(err[is_testA].mean()), "testB_model_error_mean": float(err[~is_testA].mean()),
                "S_abs_diff_mean": float(S_abs_diff.mean()),
                "phi_deviation_deg_mean": float(phi_dev.mean()),
            }
            if alpha_deg != 0:
                thresh = local_thresholds[alpha_deg]
                rec["frozen_local_threshold"] = thresh
                rec["PASSES_original_stage12_threshold"] = bool(rec["pooled_model_error_mean"] < thresh)
            results_by_scale[str(window)]["per_angle"].append(rec)

    for window in WINDOWS:
        recs = [r for r in results_by_scale[str(window)]["per_angle"] if r["angle_deg"] != 0]
        n_passed = sum(1 for r in recs if r["PASSES_original_stage12_threshold"])
        results_by_scale[str(window)]["n_angles_passed_vs_original_threshold"] = n_passed
        results_by_scale[str(window)]["n_angles_total_nonzero"] = len(NONZERO_ANGLES_DEG)
        print(f"    window={window}: {n_passed}/9 angles pass ORIGINAL Stage 12 threshold (descriptive comparison only)")

    return results_by_scale


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[stage16] device = {device}")

    local_thresholds = load_frozen_thresholds()
    model = load_model(device)
    print(f"[stage16] loaded frozen Stage 10 checkpoint (no retraining): {CHECKPOINT_PATH}")

    anatomy_results = run_anatomy_analysis(model, device)
    rotation_results = run_rotation_analysis(model, device, local_thresholds)

    final = {
        "stage": "STAGE 16 - Multi-Scale Orientation-Tensor Analysis",
        "scales": WINDOWS, "scale_names": {str(w): SCALE_NAMES[w] for w in WINDOWS},
        "anatomy_by_scale": anatomy_results,
        "rotation_by_scale": rotation_results,
        "stage_11_relationship": "Stage 16 does NOT alter Stage 11's HOLD status.",
        "stage_12_relationship": "Stage 16 does NOT alter Stage 12's HOLD status.",
    }
    with open(os.path.join(OUT_DIR, "stage16_results.json"), "w") as f:
        json.dump(final, f, indent=2, default=str)

    print("\n[stage16] DONE. Results written to stage16_results.json")
    return final


if __name__ == "__main__":
    main()
