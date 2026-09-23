"""
STAGE 14: diagnostic plots only (per instruction section 19). No ranking,
no "best model" labeling, no truncated/misleading axes, no color emphasis.
"""
import os
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
STAGE14_DIR = os.path.join(PROJECT_ROOT, "results_v2", "validation", "stage14")
STAGE10_HISTORY = os.path.join(PROJECT_ROOT, "results_v2", "model3", "training", "training_history.json")

TRAIN_EXPS = [
    ("A0", STAGE10_HISTORY, "tab:blue"),
    ("A1", os.path.join(STAGE14_DIR, "A1_no_q", "training_history.json"), "tab:orange"),
    ("A3", os.path.join(STAGE14_DIR, "A3_q_dominant", "training_history.json"), "tab:green"),
    ("A4", os.path.join(STAGE14_DIR, "A4_q_only", "training_history.json"), "tab:red"),
    ("A5", os.path.join(STAGE14_DIR, "A5_plain_cnn_control", "training_history.json"), "tab:purple"),
]


def load_history(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def plot_loss_curves():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for exp_id, path, color in TRAIN_EXPS:
        h = load_history(path)
        if h is None:
            continue
        epochs = [r["epoch"] for r in h]
        axes[0].plot(epochs, [r["train_loss"] for r in h], label=exp_id, color=color)
        axes[1].plot(epochs, [r["train_L_Q"] for r in h], label=exp_id, color=color)
        axes[2].plot(epochs, [r["train_L_class"] for r in h], label=exp_id, color=color)
    axes[0].set_title("total training loss (each experiment's own weighted objective)")
    axes[1].set_title("L_Q (unweighted) by ablation")
    axes[2].set_title("L_class (unweighted) by ablation")
    for ax in axes:
        ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend(fontsize=8)
    fig.suptitle("Stage 14: training loss curves by ablation (diagnostic; no ranking implied)")
    fig.tight_layout()
    fig.savefig(os.path.join(STAGE14_DIR, "stage14_loss_curves.png"), dpi=130)
    plt.close(fig)


def plot_geometric_comparison():
    rows = json.load(open(os.path.join(STAGE14_DIR, "stage14_master_results.json")))
    exp_ids, ang_A, ang_B, glass_A, glass_B, r_pooled = [], [], [], [], [], []
    for r in rows:
        if r["testA_angular_error_deg"] in (None, "NA"):
            continue
        exp_ids.append(r["experiment_id"])
        ang_A.append(r["testA_angular_error_deg"]); ang_B.append(r["testB_angular_error_deg"])
        glass_A.append(r["testA_Glass_delta"]); glass_B.append(r["testB_Glass_delta"])
        r_pooled.append(r["pooled_r"])

    if not exp_ids:
        print("[stage14-plots] no evaluated experiments with geometric metrics yet -- skipping comparison plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x = range(len(exp_ids))
    axes[0].scatter(x, ang_A, label="testA", marker="o")
    axes[0].scatter(x, ang_B, label="testB", marker="s")
    axes[0].set_title("mean angular correspondence error (deg) -- lower is not automatically 'better' without CI/effect-size context")
    axes[0].set_ylabel("degrees"); axes[0].legend()

    axes[1].scatter(x, glass_A, label="testA", marker="o")
    axes[1].scatter(x, glass_B, label="testB", marker="s")
    axes[1].axhline(0.5, color="gray", linestyle="--", linewidth=1, label="frozen Criterion A threshold (0.5)")
    axes[1].set_title("Glass's delta (angular)")
    axes[1].legend()

    axes[2].scatter(x, r_pooled, marker="D")
    axes[2].axhline(0.30, color="gray", linestyle="--", linewidth=1, label="frozen Criterion B threshold (r>=0.30)")
    axes[2].set_title("pooled testA+testB order-magnitude Pearson r")
    axes[2].legend()

    for ax in axes:
        ax.set_xticks(list(x)); ax.set_xticklabels(exp_ids)
        ax.axhline(0, color="black", linewidth=0.5)

    fig.suptitle("Stage 14: anatomical-correspondence diagnostics by ablation (context only, not new criterion verdicts)")
    fig.tight_layout()
    fig.savefig(os.path.join(STAGE14_DIR, "stage14_geometric_comparison.png"), dpi=130)
    plt.close(fig)


def plot_a6_diagnostic():
    path = os.path.join(STAGE14_DIR, "A6_spatial_reduction", "a6_results.json")
    if not os.path.exists(path):
        return
    a6 = json.load(open(path))
    stage12_path = os.path.join(PROJECT_ROOT, "results_v2", "validation", "stage12", "stage12_results.json")
    s12 = json.load(open(stage12_path)) if os.path.exists(stage12_path) else None

    fig, ax = plt.subplots(figsize=(9, 5.5))
    angles = [15, 30, 37, 45, 60, 90, 123, 150, 173]
    for name, marker in [("R1_full_spatial_mean", "o"), ("R2_fixed_original_valid_mask_mean", "s")]:
        errs = [r["model_error_mean"] for r in a6["reductions"][name]["per_angle"] if r["angle_deg"] != 0]
        ax.plot(angles, errs, marker=marker, label=name)
    if s12:
        errs = [r["model_error_mean"] for r in s12["per_angle"] if not r["is_identity_sanity_check_only"]]
        ax.plot(angles, errs, marker="^", label="Stage 12 center-pixel (frozen, official)", color="black")
    thresholds = [a6["reductions"]["R1_full_spatial_mean"]["per_angle"][i]["frozen_local_threshold"]
                  for i in range(len(a6["reductions"]["R1_full_spatial_mean"]["per_angle"])) if a6["reductions"]["R1_full_spatial_mean"]["per_angle"][i]["angle_deg"] != 0]
    ax.plot(angles, thresholds, marker="x", linestyle="--", color="crimson", label="frozen local threshold (Criterion D)")
    ax.set_xlabel("rotation angle (deg)"); ax.set_ylabel("model_error (normalized equivariance error)")
    ax.set_title("Stage 14 / A6: spatial-reduction diagnostic vs Stage 12 center-pixel (same frozen checkpoint)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(STAGE14_DIR, "A6_spatial_reduction", "a6_diagnostic_plot.png"), dpi=130)
    plt.close(fig)


def main():
    plot_loss_curves()
    plot_geometric_comparison()
    plot_a6_diagnostic()
    print("[stage14-plots] done")


if __name__ == "__main__":
    main()
