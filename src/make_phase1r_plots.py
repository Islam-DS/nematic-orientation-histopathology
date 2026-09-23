"""Plots for Phase 1R results."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.join(os.path.dirname(__file__), "..", "results_v2", "synthetic_replication")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results_v2", "plots")


def main():
    with open(os.path.join(BASE, "phase1r_results.json")) as f:
        results = json.load(f)
    with open(os.path.join(BASE, "training_history.json")) as f:
        history = json.load(f)

    # 1. angular MAE across methods
    methods = ["B_naive_gradient_baseline", "C_structure_tensor_baseline", "D_p8m_model", "E_nonequivariant_cnn_control"]
    labels = ["naive\ngradient", "structure\ntensor", "p8m\n(equivariant)", "plain CNN\n(control)"]
    maes = [results["test_set_results"][m]["mean_angular_error_deg"] for m in methods]

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["tab:gray", "tab:blue", "tab:green", "tab:orange"]
    ax.bar(labels, maes, color=colors)
    ax.axhline(10, color="k", linestyle="--", linewidth=1, label="threshold = 10 deg")
    ax.set_ylabel("mean angular error (deg, mod pi)")
    ax.set_title("Phase 1R: held-out test set angular accuracy")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "phase1r_angular_mae_by_method.png"), dpi=150)
    plt.close(fig)

    # 2. rotation-consistency error: p8m vs CNN vs calibration floor
    rc = results["rotation_consistency"]
    fig, ax = plt.subplots(figsize=(6, 4))
    names = ["E_render\n(calibration floor)", "p8m\nE_model", "CNN control\nE_model"]
    vals = [rc["E_render_rotation_floor_calibration"], rc["p8m_E_model"], rc["cnn_control_E_model"]]
    ax.bar(names, vals, color=["tab:gray", "tab:green", "tab:orange"])
    ax.set_ylabel("normalized rotation-consistency error")
    ax.set_title("Phase 1R: equivariance error vs rendering-noise floor")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "phase1r_equivariance_vs_floor.png"), dpi=150)
    plt.close(fig)

    # 3. training curves (train vs val, both models) -- shows early stopping, not fit-to-zero
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title in [(axes[0], "p8m", "p8m"), (axes[1], "cnn_control", "CNN control")]:
        h = history[key]
        epochs = [r["epoch"] for r in h]
        tr = [r["train_loss"] for r in h]
        va = [r["val_loss"] for r in h]
        ax.plot(epochs, tr, label="train loss")
        ax.plot(epochs, va, label="val loss")
        ax.set_xlabel("epoch")
        ax.set_ylabel("MSE loss (q1,q2)")
        ax.set_title(f"{title} training curve (early-stopped)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "phase1r_training_curves.png"), dpi=150)
    plt.close(fig)

    print(f"[make_phase1r_plots] saved 3 plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
