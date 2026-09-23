"""Plots for the Phase 1 synthetic benchmark results."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SYN_DIR = os.path.join(os.path.dirname(__file__), "..", "results_v2", "synthetic")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results_v2", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def main():
    with open(os.path.join(SYN_DIR, "synthetic_results.json")) as f:
        results = json.load(f)
    data = np.load(os.path.join(SYN_DIR, "eval_arrays.npz"))

    test_phi_deg = np.degrees(data["test_phi"])
    p8m_pred_deg = np.degrees(data["p8m_phi_pred"])
    p4m_pred_deg = np.degrees(data["p4m_phi_pred"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, pred_deg, title, err in [
        (axes[0], p8m_pred_deg, "p8m (proper 2D irrep)", results["p8m"]["mean_angular_error_deg"]),
        (axes[1], p4m_pred_deg, "p4m (degenerate 1D irrep)", results["p4m_sanity_comparison"]["mean_angular_error_deg"]),
    ]:
        ax.scatter(test_phi_deg, pred_deg, s=10, alpha=0.4)
        ax.plot([0, 180], [0, 180], 'r--', linewidth=1, label="ideal")
        ax.set_xlabel("true phi (deg)")
        ax.set_ylabel("predicted phi (deg)")
        ax.set_title(f"{title}\nmean angular error = {err:.2f} deg")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "synthetic_phi_pred_vs_true.png"), dpi=150)
    plt.close(fig)

    # equivariance error per angle
    equiv = results["p8m"]["equivariance_error_per_angle_deg"]
    angles = sorted(equiv.keys(), key=lambda a: int(a))
    values = [equiv[a] for a in angles]
    is_lattice = [int(a) % 45 == 0 for a in angles]

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["tab:green" if L else "tab:red" for L in is_lattice]
    ax.bar(angles, values, color=colors)
    ax.axhline(0.05, color="k", linestyle="--", linewidth=1, label="threshold = 0.05")
    ax.set_xlabel("rotation angle (deg)")
    ax.set_ylabel("mean normalized equivariance error")
    ax.set_title("p8m equivariance error by angle\n(green = exact p8m lattice angle, red = off-lattice)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "synthetic_equivariance_error_by_angle.png"), dpi=150)
    plt.close(fig)

    print(f"[make_synthetic_plots] saved 2 plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
