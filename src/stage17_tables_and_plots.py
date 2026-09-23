"""
STAGE 17: assembles required tables (1-4), the master results CSV, the
compatibility report, and figures 1-5 (+6 rotation curve) from already-
computed Stage 17 results. No new statistics computed here; no GlaS/CRAG
pooling anywhere.
"""
import os
import json
import csv

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
CRAG_DIR = os.path.join(PROJECT_ROOT, "results_v2", "validation", "stage17_crag")
STAGE11_PATH = os.path.join(PROJECT_ROOT, "reports", "STAGE11_AUDIT_REPORT.md")
STAGE12_RESULTS = os.path.join(PROJECT_ROOT, "results_v2", "validation", "stage12", "stage12_results.json")

with open(os.path.join(CRAG_DIR, "stage17_results.json")) as f:
    R = json.load(f)
with open(os.path.join(CRAG_DIR, "crag_target_generation_summary.json")) as f:
    TGEN = json.load(f)

gland_df = pd.read_csv(os.path.join(CRAG_DIR, "manifests", "crag_gland_manifest.csv"))
image_df = pd.read_csv(os.path.join(CRAG_DIR, "manifests", "crag_dataset_manifest.csv"))
glas_gland_df = pd.read_csv(os.path.join(PROJECT_ROOT, "results_v2", "anatomy_targets", "stage4_gland_manifest.csv"))

# ============================================================ TABLE 1: dataset audit
table1 = {
    "dataset": "CRAG", "n_images": TGEN["n_images"], "n_train_images": TGEN["n_train_images"],
    "n_test_images": TGEN["n_test_images"], "image_dims": "~1500-1514 x 1516 (8 distinct sizes)",
    "n_masks": TGEN["n_images"], "n_valid_masks_paired": TGEN["n_images"],
    "n_gland_instances_total": TGEN["n_gland_instances_total"],
    "n_valid_gland_instances": TGEN["validity_category_counts"].get("valid", 0),
    "n_invalid_gland_instances": TGEN["n_gland_instances_total"] - TGEN["validity_category_counts"].get("valid", 0),
    "n_fragmented_instances": TGEN["n_fragmented_instances"],
    "pathology_labels_available": False, "patient_metadata_available": False,
    "n_images_with_gland_covering_gt30pct_area": int((gland_df.groupby("image_id")["area_fraction_of_image"].max() > 0.3).sum()),
}
with open(os.path.join(CRAG_DIR, "tables", "table1_dataset_audit.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(table1.keys())); w.writeheader(); w.writerow(table1)

# ============================================================ TABLE 2: target generation per image
table2_rows = []
for _, row in image_df.iterrows():
    reasons = gland_df[(gland_df.image_id == row.image_id) & (gland_df.validity_category != "valid")]["validity_category"].value_counts().to_dict()
    table2_rows.append({"image_id": row.image_id, "crag_split": row.crag_split, "gland_count": row.n_gland_instances,
                          "valid_count": row.n_valid, "invalid_count": row.n_excluded, "reasons": json.dumps(reasons)})
pd.DataFrame(table2_rows).to_csv(os.path.join(CRAG_DIR, "tables", "table2_target_generation.csv"), index=False)

# ============================================================ TABLE 3: external anatomical validation
table3_rows = []
for level, key in [("image-level (primary)", "image_level_primary"), ("patch-level (secondary diagnostic)", "patch_level_secondary_diagnostic")]:
    if key == "image_level_primary":
        for pop in ["train", "test", "pooled"]:
            m = R[key][pop]["metrics"]
            n = R[key][pop]["n_images"]
            table3_rows.append({"metric": "angular correspondence (Glass's delta)", "unit": f"image ({pop})", "n": n,
                                  "estimate": m["A_angular_correspondence"]["glass_delta"], "confidence_interval": "NA (Glass's delta has no closed-form CI here; bootstrap mean CI on raw error reported instead)",
                                  "p_value": m["A_angular_correspondence"]["permutation_p_value"]})
            table3_rows.append({"metric": "order-magnitude correlation (Pearson r)", "unit": f"image ({pop})", "n": n,
                                  "estimate": m["B_order_magnitude_correlation"]["pearson_r"],
                                  "confidence_interval": json.dumps(m["B_order_magnitude_correlation"]["bootstrap_95CI"]),
                                  "p_value": m["B_order_magnitude_correlation"]["pearson_p"]})
            table3_rows.append({"metric": "tensor similarity (mean D_Q)", "unit": f"image ({pop})", "n": n,
                                  "estimate": m["C_tensor_similarity"]["mean_D_Q"], "confidence_interval": "NA",
                                  "p_value": m["C_tensor_similarity"]["permutation_p_value"]})
    else:
        m = R[key]["metrics"]; n = R[key]["n_patches_subsampled"]
        table3_rows.append({"metric": "angular correspondence (Glass's delta)", "unit": "patch (pooled subsample)", "n": n,
                              "estimate": m["A_angular_correspondence"]["glass_delta"], "confidence_interval": "NA", "p_value": m["A_angular_correspondence"]["permutation_p_value"]})
        table3_rows.append({"metric": "order-magnitude correlation (Pearson r)", "unit": "patch (pooled subsample)", "n": n,
                              "estimate": m["B_order_magnitude_correlation"]["pearson_r"], "confidence_interval": json.dumps(m["B_order_magnitude_correlation"]["bootstrap_95CI"]), "p_value": m["B_order_magnitude_correlation"]["pearson_p"]})
        table3_rows.append({"metric": "tensor similarity (mean D_Q)", "unit": "patch (pooled subsample)", "n": n,
                              "estimate": m["C_tensor_similarity"]["mean_D_Q"], "confidence_interval": "NA", "p_value": m["C_tensor_similarity"]["permutation_p_value"]})
pd.DataFrame(table3_rows).to_csv(os.path.join(CRAG_DIR, "tables", "table3_external_validation.csv"), index=False)

# ============================================================ TABLE 4: GlaS vs CRAG (no pooling, no ranking)
glas_pooled_ang = 43.38; glas_pooled_r = -0.0341; glas_pooled_dq = 0.5092  # Stage 11's frozen pooled numbers, cited read-only
table4_rows = [
    {"dataset": "GlaS", "unit": "patch (Stage 11 pooled testA+testB)", "images": 80, "glands": "NA (patch-level target)", "patches": 2083,
     "angular_correspondence_mean_deg": glas_pooled_ang, "order_correlation_pearson_r": glas_pooled_r, "tensor_similarity_mean_D_Q": glas_pooled_dq},
    {"dataset": "CRAG", "unit": "image (Stage 17 pooled, primary)", "images": R["image_level_primary"]["pooled"]["n_images"], "glands": int(TGEN["validity_category_counts"].get("valid", 0)), "patches": R["image_level_primary"]["pooled"]["n_patches_valid"],
     "angular_correspondence_mean_deg": R["image_level_primary"]["pooled"]["metrics"]["A_angular_correspondence"]["mean_deg"],
     "order_correlation_pearson_r": R["image_level_primary"]["pooled"]["metrics"]["B_order_magnitude_correlation"]["pearson_r"],
     "tensor_similarity_mean_D_Q": R["image_level_primary"]["pooled"]["metrics"]["C_tensor_similarity"]["mean_D_Q"]},
    {"dataset": "CRAG", "unit": "patch (Stage 17 pooled subsample, secondary)", "images": "NA", "glands": "NA", "patches": R["patch_level_secondary_diagnostic"]["n_patches_subsampled"],
     "angular_correspondence_mean_deg": R["patch_level_secondary_diagnostic"]["metrics"]["A_angular_correspondence"]["mean_deg"],
     "order_correlation_pearson_r": R["patch_level_secondary_diagnostic"]["metrics"]["B_order_magnitude_correlation"]["pearson_r"],
     "tensor_similarity_mean_D_Q": R["patch_level_secondary_diagnostic"]["metrics"]["C_tensor_similarity"]["mean_D_Q"]},
]
pd.DataFrame(table4_rows).to_csv(os.path.join(CRAG_DIR, "tables", "table4_glas_vs_crag.csv"), index=False)

# ============================================================ compatibility report + master results
compat = {
    "n_glas_images": 165, "n_crag_images": 213,
    "glas_gland_area_px": {"mean": float(glas_gland_df.n_pixels.mean()), "median": float(glas_gland_df.n_pixels.median()), "max": float(glas_gland_df.n_pixels.max())},
    "crag_gland_area_px": {"mean": float(gland_df.n_pixels.mean()), "median": float(gland_df.n_pixels.median()), "max": float(gland_df.n_pixels.max())},
    "glas_glands_per_image": {"mean": float(glas_gland_df.groupby("image_id").size().mean())},
    "crag_glands_per_image": {"mean": float(image_df.n_gland_instances.mean())},
    "crag_images_with_dominant_fused_gland_gt30pct": int((gland_df.groupby("image_id")["area_fraction_of_image"].max() > 0.3).sum()),
    "crag_images_with_dominant_fused_gland_gt50pct": int((gland_df.groupby("image_id")["area_fraction_of_image"].max() > 0.5).sum()),
    "glas_images_with_dominant_gland_gt30pct": "not computed for GlaS (not part of Stage 4's original audit; area-fraction-of-image was not a tracked quantity there) -- reported for CRAG only, as a CRAG-specific compatibility characterization, not a retroactive GlaS re-analysis",
    "images_excluded_from_image_level_analysis_zero_valid_patches": ["train_156", "train_43"],
    "images_excluded_reason": "train_156's sole gland instance (92.9% of image area) and both of train_43's instances were excluded under the UNCHANGED Stage 4 validity rules (near_isotropic and border_truncated respectively) -- a direct, non-invented consequence of applying GlaS's existing rules to CRAG's large-fused-instance annotation convention.",
    "independence_summary": "Same source hospital (UHCW Coventry) and curating group (Warwick TIA) as GlaS; zero exact file-content duplicates (exhaustive 165x213 check); sampled (non-exhaustive) perceptual check found no near-duplicates.",
}
with open(os.path.join(CRAG_DIR, "crag_compatibility_report.json"), "w") as f:
    json.dump(compat, f, indent=2, default=str)

master_rows = table4_rows
with open(os.path.join(CRAG_DIR, "stage17_master_results.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(master_rows[0].keys())); w.writeheader(); [w.writerow(r) for r in master_rows]

print("Tables + compatibility report + master CSV written.")

# ============================================================ FIGURES

# FIGURE 1: dataset examples (image + annotation overlay-style)
crag_root = os.path.join(PROJECT_ROOT, "data", "CRAG", "OpenDataLab___CRAG", "raw", "CRAG", "extracted", "CRAG")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
examples = [("train", "train_1"), ("train", "train_156"), ("valid", "test_1")]
for col, (folder, stem) in enumerate(examples):
    img = np.array(Image.open(os.path.join(crag_root, folder, "Images", f"{stem}.png")))
    ann = np.array(Image.open(os.path.join(crag_root, folder, "Annotation", f"{stem}.png")))
    axes[0, col].imshow(img); axes[0, col].set_title(f"{stem} (image)"); axes[0, col].axis("off")
    axes[1, col].imshow(ann, cmap="nipy_spectral"); axes[1, col].set_title(f"{stem} (instance mask, {len(np.unique(ann))-1} glands)"); axes[1, col].axis("off")
fig.suptitle("Figure 1: CRAG dataset examples (train_156 shown deliberately -- the single-fused-instance compatibility case, not a favorable example)")
fig.tight_layout()
fig.savefig(os.path.join(CRAG_DIR, "figures", "figure1_dataset_examples.png"), dpi=120)
plt.close(fig)

# FIGURE 2: CRAG gland morphology distributions (vs GlaS, side by side, not pooled)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].hist(np.log10(glas_gland_df.n_pixels.clip(lower=1)), bins=30, alpha=0.6, label="GlaS", density=True)
axes[0].hist(np.log10(gland_df.n_pixels.clip(lower=1)), bins=30, alpha=0.6, label="CRAG", density=True)
axes[0].set_xlabel("log10(gland area, pixels)"); axes[0].set_title("Gland area distribution (density, not pooled)"); axes[0].legend()
axes[1].hist(glas_gland_df.groupby("image_id").size(), bins=20, alpha=0.6, label="GlaS", density=True)
axes[1].hist(image_df.n_gland_instances, bins=20, alpha=0.6, label="CRAG", density=True)
axes[1].set_xlabel("gland instances per image"); axes[1].set_title("Glands-per-image distribution"); axes[1].legend()
fig.suptitle("Figure 2: CRAG gland morphology vs GlaS (descriptive comparison, datasets never pooled)")
fig.tight_layout()
fig.savefig(os.path.join(CRAG_DIR, "figures", "figure2_morphology_distributions.png"), dpi=130)
plt.close(fig)

# FIGURE 3 + 4: predicted vs anatomical Q, and angular correspondence distribution (image-level, pooled)
import torch
import sys
sys.path.insert(0, os.path.dirname(__file__))
from model3_typed_equivariant import Model3TypedEquivariant
from stage17_crag_evaluate import load_crag_split, CHECKPOINT_PATH, masked_patch_qmean, image_level_aggregate
from stage11_geometric_validation import get_predictions
from nematic_math import angular_error_mod_pi

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Model3TypedEquivariant().to(device); model.eval()
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))

all_S_pred, all_phi_pred, all_S_t, all_phi_t = [], [], [], []
for split_name in ["train", "test"]:
    arrs = load_crag_split(split_name)
    q1_pred, q2_pred, q1_t, q2_t, valid = get_predictions(model, arrs, device)
    q1p, q2p = q1_pred.numpy(), q2_pred.numpy(); q1t, q2t = q1_t.numpy(), q2_t.numpy(); v = valid.numpy()
    has_valid = v.any(axis=(1, 2))
    q1p_patch, q2p_patch = masked_patch_qmean(q1p, q2p, v)
    q1t_patch, q2t_patch = masked_patch_qmean(q1t, q2t, v)
    S_pred_img, phi_pred_img, _, _ = image_level_aggregate(q1p_patch, q2p_patch, arrs["image_id"], has_valid)
    S_t_img, phi_t_img, _, _ = image_level_aggregate(q1t_patch, q2t_patch, arrs["image_id"], has_valid)
    all_S_pred.append(S_pred_img); all_phi_pred.append(phi_pred_img); all_S_t.append(S_t_img); all_phi_t.append(phi_t_img)
S_pred = np.concatenate(all_S_pred); phi_pred = np.concatenate(all_phi_pred); S_t = np.concatenate(all_S_t); phi_t = np.concatenate(all_phi_t)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
axes[0].scatter(S_t, S_pred, alpha=0.5, s=15)
axes[0].set_xlabel("S_anat (image-level)"); axes[0].set_ylabel("S_DL (image-level)")
axes[0].set_title(f"Figure 3: predicted vs anatomical S (n={len(S_pred)} images, pooled train+test)\nPearson r={R['image_level_primary']['pooled']['metrics']['B_order_magnitude_correlation']['pearson_r']:.3f}")
ang_err = np.degrees(angular_error_mod_pi(phi_pred, phi_t))
axes[1].hist(ang_err, bins=30, color="tab:orange")
axes[1].set_xlabel("angular error (deg)"); axes[1].set_ylabel("count (images)")
axes[1].set_title(f"Figure 4: angular correspondence distribution (image-level, n={len(ang_err)})\nmean={ang_err.mean():.1f} deg")
fig.tight_layout()
fig.savefig(os.path.join(CRAG_DIR, "figures", "figure3_4_predicted_vs_anatomical_and_angular_error.png"), dpi=130)
plt.close(fig)

# FIGURE 5: GlaS vs CRAG correspondence metrics, side by side (no ranking, no pooling)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
labels = ["GlaS\n(patch, Stage 11)", "CRAG\n(image, primary)", "CRAG\n(patch, secondary)"]
ang_vals = [glas_pooled_ang, R["image_level_primary"]["pooled"]["metrics"]["A_angular_correspondence"]["mean_deg"], R["patch_level_secondary_diagnostic"]["metrics"]["A_angular_correspondence"]["mean_deg"]]
r_vals = [glas_pooled_r, R["image_level_primary"]["pooled"]["metrics"]["B_order_magnitude_correlation"]["pearson_r"], R["patch_level_secondary_diagnostic"]["metrics"]["B_order_magnitude_correlation"]["pearson_r"]]
dq_vals = [glas_pooled_dq, R["image_level_primary"]["pooled"]["metrics"]["C_tensor_similarity"]["mean_D_Q"], R["patch_level_secondary_diagnostic"]["metrics"]["C_tensor_similarity"]["mean_D_Q"]]
for ax, vals, title in [(axes[0], ang_vals, "mean angular error (deg)"), (axes[1], r_vals, "order-magnitude Pearson r"), (axes[2], dq_vals, "tensor similarity mean D_Q")]:
    ax.bar(labels, vals, color=["tab:blue", "tab:orange", "tab:orange"])
    ax.set_title(title); ax.axhline(0, color="black", linewidth=0.5)
    if title.startswith("order"):
        ax.axhline(0.30, color="crimson", linestyle="--", linewidth=1, label="GlaS threshold (context only)")
        ax.legend(fontsize=7)
fig.suptitle("Figure 5: GlaS vs CRAG correspondence metrics -- descriptive comparison, datasets never pooled, no ranking implied")
fig.tight_layout()
fig.savefig(os.path.join(CRAG_DIR, "figures", "figure5_glas_vs_crag_comparison.png"), dpi=130)
plt.close(fig)

# FIGURE 6: rotation-consistency curve
angles = [r["angle_deg"] for r in R["rotation_consistency"]["per_angle"] if r["angle_deg"] != 0]
crag_err = [r["model_error_mean"] for r in R["rotation_consistency"]["per_angle"] if r["angle_deg"] != 0]
thresh = [r["frozen_local_threshold_from_GlaS_Stage12"] for r in R["rotation_consistency"]["per_angle"] if r["angle_deg"] != 0]
glas_err = None
if os.path.exists(STAGE12_RESULTS):
    with open(STAGE12_RESULTS) as f:
        s12 = json.load(f)
    glas_err = [r["model_error_mean"] for r in s12["per_angle"] if not r["is_identity_sanity_check_only"]]

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(angles, crag_err, marker="o", label="CRAG (Stage 17)")
if glas_err:
    ax.plot(angles, glas_err, marker="s", label="GlaS (Stage 12, frozen)")
ax.plot(angles, thresh, marker="x", linestyle="--", color="crimson", label="frozen GlaS threshold (context only)")
ax.set_xlabel("rotation angle (deg)"); ax.set_ylabel("model_error (normalized equivariance error)")
ax.set_title("Figure 6: CRAG rotation-consistency curve vs GlaS (Stage 12), same frozen checkpoint\n(descriptive comparison; no new CRAG-specific threshold)")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(CRAG_DIR, "figures", "figure6_rotation_consistency_curve.png"), dpi=130)
plt.close(fig)

print("[stage17] figures written")
