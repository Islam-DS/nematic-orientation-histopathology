"""
STAGE 17: CRAG external validation -- primary (image-level) anatomical
correspondence, secondary (patch-level, Stage-11-identical) diagnostic,
and rotation consistency (Stage-12-identical protocol), exactly per the
frozen configs/stage17_crag_validation_matrix.json. Frozen Stage 10 Model
3 checkpoint only; no retraining anywhere in this script.
"""
import sys
import os
import json

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
CRAG_DIR = os.path.join(PROJECT_ROOT, "results_v2", "validation", "stage17_crag")
OUT_DIR = CRAG_DIR

MAX_PATCH_SUBSAMPLE = 5000
SUBSAMPLE_SEED = 42
ROTATION_ANGLES_DEG = [0, 15, 30, 37, 45, 60, 90, 123, 150, 173]
NONZERO_ANGLES_DEG = [15, 30, 37, 45, 60, 90, 123, 150, 173]
BATCH_SIZE = 64


def load_crag_split(split_name):
    d = np.load(os.path.join(CRAG_DIR, "targets", f"crag_{split_name}_patches.npz"), allow_pickle=True)
    arrs = {k: d[k] for k in d.files}
    arrs["label"] = np.zeros(len(arrs["image_id"]), dtype=np.int64)  # placeholder, unused (no pathology labels exist)
    return arrs


def load_frozen_thresholds():
    with open(THRESHOLDS_PATH) as f:
        d = json.load(f)
    return {int(k): float(v) for k, v in d["criteria"]["D_rotation_consistency"]["threshold_primary"]["local_thresholds"].items()}


def load_model(device):
    model = Model3TypedEquivariant().to(device)
    model.eval()
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    return model


def masked_patch_qmean(q1, q2, valid):
    """Per-patch masked mean of (q1,q2) over valid pixels -- identical
    convention to per_patch_scalar_qs, but returns (q1,q2) not (S,phi)."""
    N = q1.shape[0]
    out_q1 = np.zeros(N); out_q2 = np.zeros(N)
    for i in range(N):
        v = valid[i]
        if v.any():
            out_q1[i] = q1[i][v].mean()
            out_q2[i] = q2[i][v].mean()
    return out_q1, out_q2


def image_level_aggregate(q1_patch, q2_patch, image_ids, has_valid):
    """Groups patch-level (q1,q2) vectors by image_id and takes the plain
    mean within each image (unweighted across that image's valid patches),
    per the frozen matrix's image-level aggregation rule."""
    df_image_ids = np.asarray(image_ids)
    unique_images = sorted(set(df_image_ids[has_valid]))
    S_img, phi_img, n_patches_img = [], [], []
    for iid in unique_images:
        m = (df_image_ids == iid) & has_valid
        q1m, q2m = q1_patch[m].mean(), q2_patch[m].mean()
        S_img.append(np.hypot(q1m, q2m))
        phi_img.append(np.mod(0.5 * np.arctan2(q2m, q1m), np.pi))
        n_patches_img.append(int(m.sum()))
    return np.array(S_img), np.array(phi_img), unique_images, np.array(n_patches_img)


def run_anatomy_image_level(model, device):
    print("\n" + "=" * 60 + "\n[stage17-crag] IMAGE-LEVEL PRIMARY ANATOMY ANALYSIS\n" + "=" * 60)
    all_data = {}
    for split_name in ["train", "test"]:
        arrs = load_crag_split(split_name)
        q1_pred, q2_pred, q1_t, q2_t, valid = get_predictions(model, arrs, device)
        q1p, q2p = q1_pred.numpy(), q2_pred.numpy()
        q1t, q2t = q1_t.numpy(), q2_t.numpy()
        v = valid.numpy()
        has_valid = v.any(axis=(1, 2))

        q1p_patch, q2p_patch = masked_patch_qmean(q1p, q2p, v)
        q1t_patch, q2t_patch = masked_patch_qmean(q1t, q2t, v)

        S_pred_img, phi_pred_img, image_ids, n_patches_img = image_level_aggregate(q1p_patch, q2p_patch, arrs["image_id"], has_valid)
        S_t_img, phi_t_img, _, _ = image_level_aggregate(q1t_patch, q2t_patch, arrs["image_id"], has_valid)

        all_data[split_name] = {"S_pred_img": S_pred_img, "phi_pred_img": phi_pred_img,
                                  "S_t_img": S_t_img, "phi_t_img": phi_t_img,
                                  "image_ids": image_ids, "n_patches_per_image": n_patches_img,
                                  "n_images": len(image_ids), "n_patches_total": len(has_valid), "n_patches_valid": int(has_valid.sum())}
        print(f"[stage17-crag] {split_name}: {len(image_ids)} images, {len(has_valid)} patches ({int(has_valid.sum())} valid)")

    results = {}
    for split_name in ["train", "test"]:
        d = all_data[split_name]
        m = compute_full_metrics(d["S_pred_img"], d["phi_pred_img"], d["S_t_img"], d["phi_t_img"],
                                  seed=0 if split_name == "train" else 1)
        results[split_name] = {"n_images": d["n_images"], "n_patches_total": d["n_patches_total"],
                                 "n_patches_valid": d["n_patches_valid"], "metrics": m}

    pooled_S_pred = np.concatenate([all_data["train"]["S_pred_img"], all_data["test"]["S_pred_img"]])
    pooled_phi_pred = np.concatenate([all_data["train"]["phi_pred_img"], all_data["test"]["phi_pred_img"]])
    pooled_S_t = np.concatenate([all_data["train"]["S_t_img"], all_data["test"]["S_t_img"]])
    pooled_phi_t = np.concatenate([all_data["train"]["phi_t_img"], all_data["test"]["phi_t_img"]])
    pooled_m = compute_full_metrics(pooled_S_pred, pooled_phi_pred, pooled_S_t, pooled_phi_t, seed=3)
    results["pooled"] = {"n_images": len(pooled_S_pred),
                          "n_patches_total": all_data["train"]["n_patches_total"] + all_data["test"]["n_patches_total"],
                          "n_patches_valid": all_data["train"]["n_patches_valid"] + all_data["test"]["n_patches_valid"],
                          "metrics": pooled_m}

    for split_name in ["train", "test", "pooled"]:
        m = results[split_name]["metrics"]
        print(f"[stage17-crag] {split_name} (image-level): n_images={results[split_name]['n_images']} "
              f"ang_mean={m['A_angular_correspondence']['mean_deg']:.2f} glass_A={m['A_angular_correspondence']['glass_delta']:.4f} "
              f"pearson_r={m['B_order_magnitude_correlation']['pearson_r']:.4f} D_Q={m['C_tensor_similarity']['mean_D_Q']:.4f}")

    return results, all_data


def run_anatomy_patch_level_diagnostic(model, device):
    print("\n" + "=" * 60 + "\n[stage17-crag] PATCH-LEVEL SECONDARY DIAGNOSTIC (Stage 11-identical metric)\n" + "=" * 60)
    pooled_S_pred, pooled_phi_pred, pooled_S_t, pooled_phi_t = [], [], [], []
    per_split_idx = {}
    offset = 0
    for split_name in ["train", "test"]:
        arrs = load_crag_split(split_name)
        q1_pred, q2_pred, q1_t, q2_t, valid = get_predictions(model, arrs, device)
        q1p, q2p = q1_pred.numpy(), q2_pred.numpy()
        q1t, q2t = q1_t.numpy(), q2_t.numpy()
        v = valid.numpy()
        has_valid = v.any(axis=(1, 2))
        q1p_patch, q2p_patch = masked_patch_qmean(q1p, q2p, v)
        q1t_patch, q2t_patch = masked_patch_qmean(q1t, q2t, v)
        S_p = np.hypot(q1p_patch, q2p_patch); phi_p = np.mod(0.5 * np.arctan2(q2p_patch, q1p_patch), np.pi)
        S_t = np.hypot(q1t_patch, q2t_patch); phi_t = np.mod(0.5 * np.arctan2(q2t_patch, q1t_patch), np.pi)
        pooled_S_pred.append(S_p[has_valid]); pooled_phi_pred.append(phi_p[has_valid])
        pooled_S_t.append(S_t[has_valid]); pooled_phi_t.append(phi_t[has_valid])
        n = int(has_valid.sum())
        per_split_idx[split_name] = (offset, offset + n)
        offset += n

    pooled_S_pred = np.concatenate(pooled_S_pred); pooled_phi_pred = np.concatenate(pooled_phi_pred)
    pooled_S_t = np.concatenate(pooled_S_t); pooled_phi_t = np.concatenate(pooled_phi_t)
    n_total = len(pooled_S_pred)
    print(f"[stage17-crag] total valid patches (pooled train+test): {n_total}")

    rng = np.random.default_rng(SUBSAMPLE_SEED)
    if n_total > MAX_PATCH_SUBSAMPLE:
        sub_idx = rng.choice(n_total, size=MAX_PATCH_SUBSAMPLE, replace=False)
        sub_idx.sort()
    else:
        sub_idx = np.arange(n_total)
    print(f"[stage17-crag] patch-level diagnostic uses {len(sub_idx)} patches "
          f"(deterministic seed={SUBSAMPLE_SEED} subsample, cap={MAX_PATCH_SUBSAMPLE})")

    m = compute_full_metrics(pooled_S_pred[sub_idx], pooled_phi_pred[sub_idx], pooled_S_t[sub_idx], pooled_phi_t[sub_idx], seed=5)
    print(f"[stage17-crag] patch-level pooled subsample: ang_mean={m['A_angular_correspondence']['mean_deg']:.2f} "
          f"glass_A={m['A_angular_correspondence']['glass_delta']:.4f} pearson_r={m['B_order_magnitude_correlation']['pearson_r']:.4f} "
          f"D_Q={m['C_tensor_similarity']['mean_D_Q']:.4f}")

    return {"n_patches_total_pooled": n_total, "n_patches_subsampled": len(sub_idx),
            "subsample_seed": SUBSAMPLE_SEED, "subsample_cap": MAX_PATCH_SUBSAMPLE, "metrics": m}, sub_idx, per_split_idx


@torch.no_grad()
def batched_q_dense_from_rgb(model, rgb_float_batch, device):
    n = rgb_float_batch.shape[0]
    outs = []
    for i in range(0, n, BATCH_SIZE):
        chunk = rgb_float_batch[i:i + BATCH_SIZE]
        x = torch.from_numpy(chunk).permute(0, 3, 1, 2).to(device)
        _, q_dense = model(x)
        outs.append(q_dense.cpu().numpy())
    return np.concatenate(outs, axis=0)


def run_rotation_consistency(model, device, local_thresholds, sub_idx, per_split_idx):
    print("\n" + "=" * 60 + "\n[stage17-crag] ROTATION CONSISTENCY (Stage 12-identical protocol, center-pixel)\n" + "=" * 60)
    rgb_all = []
    for split_name in ["train", "test"]:
        arrs = load_crag_split(split_name)
        rgb_all.append(arrs["rgb"])
    rgb_all = np.concatenate(rgb_all, axis=0)
    rgb_sub = rgb_all[sub_idx].astype(np.float32) / 255.0
    n_patches = rgb_sub.shape[0]
    print(f"[stage17-crag] rotation analysis on the same {n_patches}-patch subsample")

    q0_dense = batched_q_dense_from_rgb(model, rgb_sub, device)
    cy, cx = q0_dense.shape[-2] // 2, q0_dense.shape[-1] // 2
    q0 = q0_dense[:, :, cy, cx]

    per_angle = []
    for alpha_deg in ROTATION_ANGLES_DEG:
        if alpha_deg == 0:
            q_rot = q0.copy()
        else:
            rot_imgs = np.empty_like(rgb_sub)
            for i in range(n_patches):
                rot_imgs[i] = sk_rotate(rgb_sub[i], angle=alpha_deg, mode="reflect", order=1).astype(np.float32)
            q_rot_dense = batched_q_dense_from_rgb(model, rot_imgs, device)
            q_rot = q_rot_dense[:, :, cy, cx]

        alpha_rad = np.deg2rad(alpha_deg)
        q_expected = np.stack(rotate_Q_analytic(q0[:, 0], q0[:, 1], alpha_rad), axis=-1)
        num = np.linalg.norm(q_rot - q_expected, axis=-1)
        den = np.linalg.norm(q_expected, axis=-1) + 1e-8
        err = num / den
        S0 = np.hypot(q0[:, 0], q0[:, 1]); S_r = np.hypot(q_rot[:, 0], q_rot[:, 1])
        phi0 = np.mod(0.5 * np.arctan2(q0[:, 1], q0[:, 0]), np.pi)
        phi_r = np.mod(0.5 * np.arctan2(q_rot[:, 1], q_rot[:, 0]), np.pi)
        phi_expected = np.mod(phi0 + alpha_rad, np.pi)

        rec = {"angle_deg": alpha_deg, "n_patches": int(n_patches), "model_error_mean": float(err.mean()),
               "model_error_median": float(np.median(err)), "S_abs_diff_mean": float(np.abs(S_r - S0).mean()),
               "phi_deviation_deg_mean": float(np.degrees(angular_error_mod_pi(phi_r, phi_expected)).mean())}
        if alpha_deg != 0:
            thresh = local_thresholds[alpha_deg]
            rec["frozen_local_threshold_from_GlaS_Stage12"] = thresh
            rec["PASSES_ORIGINAL_STAGE12_THRESHOLD_descriptive_only"] = bool(rec["model_error_mean"] < thresh)
        per_angle.append(rec)
        print(f"    angle={alpha_deg}: model_error_mean={rec['model_error_mean']:.4f}"
              + (f"  threshold={local_thresholds[alpha_deg]:.4f} PASS={rec['PASSES_ORIGINAL_STAGE12_THRESHOLD_descriptive_only']}" if alpha_deg else ""))

    n_passed = sum(1 for r in per_angle if r["angle_deg"] != 0 and r["PASSES_ORIGINAL_STAGE12_THRESHOLD_descriptive_only"])
    print(f"[stage17-crag] CRAG rotation consistency: {n_passed}/9 angles pass the ORIGINAL GlaS Stage 12 threshold (descriptive only)")
    return {"n_patches": n_patches, "per_angle": per_angle, "n_angles_passed_vs_original_GlaS_threshold": n_passed, "n_angles_total_nonzero": 9}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[stage17-crag] device = {device}")

    local_thresholds = load_frozen_thresholds()
    model = load_model(device)
    print(f"[stage17-crag] loaded frozen Stage 10 checkpoint (no retraining): {CHECKPOINT_PATH}")

    image_level_results, _ = run_anatomy_image_level(model, device)
    patch_level_results, sub_idx, per_split_idx = run_anatomy_patch_level_diagnostic(model, device)
    rotation_results = run_rotation_consistency(model, device, local_thresholds, sub_idx, per_split_idx)

    final = {
        "stage": "STAGE 17 - CRAG External Cross-Dataset Validation",
        "device": device,
        "image_level_primary": image_level_results,
        "patch_level_secondary_diagnostic": patch_level_results,
        "rotation_consistency": rotation_results,
        "stage_11_relationship": "Stage 17 does NOT alter Stage 11's HOLD status (GlaS-specific).",
        "stage_12_relationship": "Stage 17 does NOT alter Stage 12's HOLD status (GlaS-specific).",
    }
    with open(os.path.join(OUT_DIR, "stage17_results.json"), "w") as f:
        json.dump(final, f, indent=2, default=str)

    print("\n[stage17-crag] DONE.")
    return final


if __name__ == "__main__":
    main()
