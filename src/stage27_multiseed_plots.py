"""
STAGE 27 TASK 1: seed-variance plots and training-loss trajectories for the five-seed robustness
study. Reads only results_v2/stage27_robustness/multi_seed/seed_*/ (evaluation_results.json,
training_summary.json, training/training_history.json or the frozen seed-42 equivalent) and
STAGE27_MULTI_SEED_SUMMARY.json. Writes only into results_v2/stage27_robustness/multi_seed/plots/.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
MS_DIR = os.path.join(ROOT, "results_v2", "stage27_robustness", "multi_seed")
OUT_DIR = os.path.join(MS_DIR, "plots")
os.makedirs(OUT_DIR, exist_ok=True)
SEEDS = [11, 22, 33, 42, 55]
COLORS = {11: "#1f77b4", 22: "#ff7f0e", 33: "#2ca02c", 42: "#000000", 55: "#9467bd"}
NEUTRAL_GRAY = "#6b6b6b"
FAIL_RED = "#b23a3a"


def load_seed(seed):
    d = os.path.join(MS_DIR, f"seed_{seed}")
    ev = json.load(open(os.path.join(d, "evaluation_results.json")))
    hist_path = os.path.join(d, "training", "training_history.json")
    hist = json.load(open(hist_path)) if os.path.isfile(hist_path) else None
    return ev, hist


def main():
    summary = json.load(open(os.path.join(MS_DIR, "STAGE27_MULTI_SEED_SUMMARY.json")))

    # ---- Figure: seed variance across Criteria A-D + AUROC ----
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.4))
    panels = [("A_glass_delta", "Criterion A: Glass's $\\Delta$", 0.5, "$\\Delta \\geq 0.5$"),
              ("B_pearson_r", "Criterion B: Pearson r", 0.30, "$r \\geq 0.30$"),
              ("C_glass_delta", "Criterion C: Glass's $\\Delta$", 0.5, "$\\Delta \\geq 0.5$"),
              ("D_n_angles_passed", "Criterion D: angles passed", 7, "$\\geq 7$ of 9"),
              ("eval_auroc", "Classification AUROC", None, None)]
    for ax, (field, title, thresh, label) in zip(axes, panels):
        s = summary[field]
        vals = [s["values_by_seed"][str(sd)] for sd in SEEDS]
        ax.bar(range(5), vals, color=[COLORS[sd] for sd in SEEDS])
        ax.axhline(s["mean"], color="black", linewidth=1.0, linestyle="-", label=f"mean={s['mean']:.3f}")
        if thresh is not None:
            ax.axhline(thresh, color=FAIL_RED, linestyle="--", linewidth=1.0, label=label)
        ax.set_xticks(range(5)); ax.set_xticklabels([str(sd) for sd in SEEDS], fontsize=8)
        ax.set_xlabel("seed", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=6, loc="best")
    fig.suptitle("Multi-seed robustness (5 seeds): Criteria A-D and classification AUROC, pooled testA+testB", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(OUT_DIR, "seed_variance.png"), dpi=200)
    plt.close(fig)
    print("[plots] wrote seed_variance.png")

    # ---- Figure: training-loss trajectories ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for seed in SEEDS:
        _, hist = load_seed(seed)
        if hist is None:
            continue
        epochs = [r["epoch"] for r in hist]
        train_loss = [r["train_loss"] for r in hist]
        dev_loss = [r["dev_loss"] for r in hist]
        axes[0].plot(epochs, train_loss, color=COLORS[seed], label=f"seed {seed}", linewidth=1.2)
        axes[1].plot(epochs, dev_loss, color=COLORS[seed], label=f"seed {seed}", linewidth=1.2)
    axes[0].set_title("Training loss"); axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[1].set_title("Dev loss"); axes[1].set_xlabel("epoch"); axes[1].set_ylabel("loss")
    axes[0].legend(fontsize=7); axes[1].legend(fontsize=7)
    fig.suptitle("Training-loss trajectories, five seeds (identical architecture/hyperparameters)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(OUT_DIR, "training_trajectories.png"), dpi=200)
    plt.close(fig)
    print("[plots] wrote training_trajectories.png")


if __name__ == "__main__":
    main()
