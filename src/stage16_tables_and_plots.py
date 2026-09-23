"""
STAGE 16: assembles the required master results table + Tables 1-3 (CSV)
and Figures 1-4 from results_v2/validation/stage16/stage16_results.json
(already produced by stage16_multiscale_analysis.py). No new statistics
are computed here -- tabulation and visualization only. No scale is
ranked or labeled "best"/"correct".
"""
import os
import json
import csv

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR = os.path.join(PROJECT_ROOT, "results_v2", "validation", "stage16")

with open(os.path.join(OUT_DIR, "stage16_results.json")) as f:
    R = json.load(f)

WINDOWS = R["scales"]  # [1,3,5,9,17,"full"]
WINDOW_LABELS = {"1": "center", "3": "3x3", "5": "5x5", "9": "9x9", "17": "17x17", "full": "full-field"}


# ---------------------------------------------------------------- TABLE 1: anatomy summary
rows1 = []
for w in WINDOWS:
    ws = str(w)
    pooled = R["anatomy_by_scale"][ws]["pooled"]
    m = pooled["metrics"]
    rows1.append({
        "scale": WINDOW_LABELS[ws], "window": ws,
        "n_images": pooled["n_images"], "n_patches": pooled["n_patches_total"],
        "n_patches_valid": pooled["n_patches_valid"],
        "angular_correspondence_mean_deg": m["A_angular_correspondence"]["mean_deg"],
        "angular_correspondence_glass_delta": m["A_angular_correspondence"]["glass_delta"],
        "order_correlation_pearson_r": m["B_order_magnitude_correlation"]["pearson_r"],
        "order_correlation_p": m["B_order_magnitude_correlation"]["pearson_p"],
        "tensor_similarity_mean_D_Q": m["C_tensor_similarity"]["mean_D_Q"],
        "tensor_similarity_glass_delta": m["C_tensor_similarity"]["glass_delta"],
    })
with open(os.path.join(OUT_DIR, "tables", "table1_anatomy_by_scale.csv"), "w", newline="") as f:
    w_ = csv.DictWriter(f, fieldnames=list(rows1[0].keys())); w_.writeheader(); [w_.writerow(r) for r in rows1]

# ---------------------------------------------------------------- TABLE 2: rotation per angle x scale
rows2 = []
for w in WINDOWS:
    ws = str(w)
    for rec in R["rotation_by_scale"][ws]["per_angle"]:
        if rec["angle_deg"] == 0:
            continue
        rows2.append({
            "scale": WINDOW_LABELS[ws], "window": ws, "angle_deg": rec["angle_deg"],
            "rotation_error_pooled_mean": rec["pooled_model_error_mean"],
            "S_invariance_error_mean": rec["S_abs_diff_mean"],
            "angular_transformation_error_deg_mean": rec["phi_deviation_deg_mean"],
            "original_stage12_threshold": rec["frozen_local_threshold"],
            "passes_original_threshold": rec["PASSES_original_stage12_threshold"],
        })
with open(os.path.join(OUT_DIR, "tables", "table2_rotation_by_scale_angle.csv"), "w", newline="") as f:
    w_ = csv.DictWriter(f, fieldnames=list(rows2[0].keys())); w_.writeheader(); [w_.writerow(r) for r in rows2]

# ---------------------------------------------------------------- TABLE 3: scale summary
rows3 = []
for w in WINDOWS:
    ws = str(w)
    m = R["anatomy_by_scale"][ws]["pooled"]["metrics"]
    rot = R["rotation_by_scale"][ws]
    ang_errs = [rec["pooled_model_error_mean"] for rec in rot["per_angle"] if rec["angle_deg"] != 0]
    rows3.append({
        "scale": WINDOW_LABELS[ws], "window": ws,
        "mean_angular_error_deg": m["A_angular_correspondence"]["mean_deg"],
        "median_angular_error_deg": m["A_angular_correspondence"]["median_deg"],
        "order_correlation_pearson_r": m["B_order_magnitude_correlation"]["pearson_r"],
        "tensor_similarity_mean_D_Q": m["C_tensor_similarity"]["mean_D_Q"],
        "rotation_angles_passed_of_9": rot["n_angles_passed_vs_original_threshold"],
        "mean_rotation_error_across_nonzero_angles": float(np.mean(ang_errs)),
    })
with open(os.path.join(OUT_DIR, "tables", "table3_scale_summary.csv"), "w", newline="") as f:
    w_ = csv.DictWriter(f, fieldnames=list(rows3[0].keys())); w_.writeheader(); [w_.writerow(r) for r in rows3]

# ---------------------------------------------------------------- master results csv (project-standard format)
master_rows = []
for w in WINDOWS:
    ws = str(w)
    m = R["anatomy_by_scale"][ws]["pooled"]["metrics"]
    rot = R["rotation_by_scale"][ws]
    master_rows.append({
        "scale_id": {"1": 0, "3": 1, "5": 2, "9": 3, "17": 4, "full": 5}[ws],
        "scale_name": WINDOW_LABELS[ws], "window": ws,
        "n_images_pooled": 80, "n_patches_pooled": R["anatomy_by_scale"][ws]["pooled"]["n_patches_total"],
        "angular_correspondence_mean_deg": m["A_angular_correspondence"]["mean_deg"],
        "angular_correspondence_glass_delta": m["A_angular_correspondence"]["glass_delta"],
        "PASSES_original_criterion_A_threshold_context_only": m["A_angular_correspondence"]["PASSES_PRIMARY_glass_delta_ge_0.5"],
        "order_magnitude_pearson_r": m["B_order_magnitude_correlation"]["pearson_r"],
        "order_magnitude_95CI": json.dumps(m["B_order_magnitude_correlation"]["bootstrap_95CI"]),
        "PASSES_original_criterion_B_context_only": m["B_order_magnitude_correlation"]["PASSES_PRIMARY_ALL_THREE"],
        "tensor_similarity_mean_D_Q": m["C_tensor_similarity"]["mean_D_Q"],
        "PASSES_original_criterion_C_context_only": m["C_tensor_similarity"]["PASSES_PRIMARY_BOTH"],
        "rotation_angles_passed_of_9_vs_original_stage12_threshold": rot["n_angles_passed_vs_original_threshold"],
    })
with open(os.path.join(OUT_DIR, "stage16_master_results.csv"), "w", newline="") as f:
    w_ = csv.DictWriter(f, fieldnames=list(master_rows[0].keys())); w_.writeheader(); [w_.writerow(r) for r in master_rows]

print("Tables written.")
for r in master_rows:
    print(r)


# ================================================================== FIGURES

x_labels = [WINDOW_LABELS[str(w)] for w in WINDOWS]
x_pos = np.arange(len(WINDOWS))

# FIGURE 1: anatomical correspondence vs scale
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
glass_a = [R["anatomy_by_scale"][str(w)]["pooled"]["metrics"]["A_angular_correspondence"]["glass_delta"] for w in WINDOWS]
pearson_r = [R["anatomy_by_scale"][str(w)]["pooled"]["metrics"]["B_order_magnitude_correlation"]["pearson_r"] for w in WINDOWS]
dq = [R["anatomy_by_scale"][str(w)]["pooled"]["metrics"]["C_tensor_similarity"]["mean_D_Q"] for w in WINDOWS]

axes[0].plot(x_pos, glass_a, marker="o", color="tab:blue")
axes[0].axhline(0.5, color="crimson", linestyle="--", label="frozen Criterion A threshold (0.5)")
axes[0].set_title("Criterion A: Glass's delta (angular)"); axes[0].legend()
axes[1].plot(x_pos, pearson_r, marker="o", color="tab:green")
axes[1].axhline(0.30, color="crimson", linestyle="--", label="frozen Criterion B threshold (r>=0.30)")
axes[1].set_title("Criterion B: order-magnitude Pearson r"); axes[1].legend()
axes[2].plot(x_pos, dq, marker="o", color="tab:purple")
axes[2].set_title("Criterion C: mean tensor distance D_Q (lower = more similar)")
for ax in axes:
    ax.set_xticks(x_pos); ax.set_xticklabels(x_labels, rotation=20)
    ax.set_xlabel("spatial aggregation scale (pooled testA+testB, diagnostic only)")
    ax.axhline(0, color="black", linewidth=0.5)
fig.suptitle("Figure 1: Anatomical correspondence vs spatial scale (diagnostic; does not alter Stage 11's HOLD verdict)")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "figures", "figure1_anatomy_vs_scale.png"), dpi=130)
plt.close(fig)

# FIGURE 2: rotation consistency vs scale
fig, ax = plt.subplots(figsize=(8, 5.5))
n_passed = [R["rotation_by_scale"][str(w)]["n_angles_passed_vs_original_threshold"] for w in WINDOWS]
ax.plot(x_pos, n_passed, marker="o", color="tab:orange")
ax.axhline(7, color="crimson", linestyle="--", label="original Stage 12 pass rule (>=7/9)")
ax.set_ylim(0, 9.5)
ax.set_xticks(x_pos); ax.set_xticklabels(x_labels, rotation=20)
ax.set_ylabel("angles passing the ORIGINAL frozen Stage 12 threshold (of 9)")
ax.set_xlabel("spatial aggregation scale (pooled testA+testB, descriptive comparison only)")
ax.set_title("Figure 2: Rotation consistency vs spatial scale\n(diagnostic; does not alter Stage 12's HOLD verdict or create a new Criterion D decision)")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "figures", "figure2_rotation_vs_scale.png"), dpi=130)
plt.close(fig)

# FIGURE 3: S invariance and angular transformation error vs scale (averaged over 9 angles)
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
s_inv = []
phi_dev = []
for w in WINDOWS:
    recs = [r for r in R["rotation_by_scale"][str(w)]["per_angle"] if r["angle_deg"] != 0]
    s_inv.append(np.mean([r["S_abs_diff_mean"] for r in recs]))
    phi_dev.append(np.mean([r["phi_deviation_deg_mean"] for r in recs]))
axes[0].plot(x_pos, s_inv, marker="o", color="tab:cyan")
axes[0].set_title("S invariance error (mean |S_rot - S_0|, averaged over 9 angles)")
axes[1].plot(x_pos, phi_dev, marker="o", color="tab:pink")
axes[1].set_title("angular transformation error (deg, averaged over 9 angles)")
for ax in axes:
    ax.set_xticks(x_pos); ax.set_xticklabels(x_labels, rotation=20)
    ax.set_xlabel("spatial aggregation scale")
fig.suptitle("Figure 3: Rotation error decomposition vs spatial scale (magnitude vs orientation component)")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "figures", "figure3_error_decomposition_vs_scale.png"), dpi=130)
plt.close(fig)

# FIGURE 4: representative Q-field visualization across scales, for a handful of example patches
import sys
sys.path.insert(0, os.path.dirname(__file__))
from model3_typed_equivariant import Model3TypedEquivariant

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Model3TypedEquivariant().to(device); model.eval()
model.load_state_dict(torch.load(os.path.join(PROJECT_ROOT, "results_v2", "model3", "checkpoints", "checkpoint_final.pt"), map_location=device))

d = np.load(os.path.join(PROJECT_ROOT, "results_v2", "baselines", "_patch_cache", "testA.npz"), allow_pickle=True)
example_idx = [0, 200, 400]  # deterministic, pre-existing indices -- not selected to look favorable
with torch.no_grad():
    x = torch.from_numpy(d["rgb"][example_idx].astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(device)
    _, q_dense = model(x)
q_dense = q_dense.cpu().numpy()

fig, axes = plt.subplots(len(example_idx), len(WINDOWS) + 1, figsize=(3 * (len(WINDOWS) + 1), 3 * len(example_idx)))
for row, idx in enumerate(example_idx):
    axes[row, 0].imshow(d["rgb"][idx])
    axes[row, 0].set_title(f"patch {d['image_id'][idx]}[{idx}]\n(input image)")
    axes[row, 0].axis("off")
    for col, w in enumerate(WINDOWS):
        ax = axes[row, col + 1]
        S_field = np.hypot(q_dense[row, 0], q_dense[row, 1])
        im = ax.imshow(S_field, cmap="viridis", vmin=0, vmax=S_field.max() + 1e-8)
        if w != "full":
            half = w // 2
            c = 32
            rect = plt.Rectangle((c - half - 0.5, c - half - 0.5), w, w, edgecolor="red", facecolor="none", linewidth=1.5)
            ax.add_patch(rect)
        ax.set_title(WINDOW_LABELS[str(w)]); ax.axis("off")
fig.suptitle("Figure 4: dense S=|Q| field (viridis) with the aggregation window overlaid (red) at each frozen scale\n(3 representative testA patches, deterministic indices; not selected for favorable appearance)")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "figures", "figure4_qfield_visualization_by_scale.png"), dpi=130)
plt.close(fig)

print("[stage16] figures written")
