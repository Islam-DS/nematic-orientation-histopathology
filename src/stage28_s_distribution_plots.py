"""
STAGE 28 TASK B: neutral diagnostic plots for S_DL / S_anat (histogram/density, scatter). Reads only
results_v2/stage28/s_distribution/s_values_pooled.npz (written by stage28_s_distribution_diagnostics.py).
Writes only into results_v2/stage28/s_distribution/plots/. No claim of success or failure is encoded in
the plotting code; axis ranges and binning are shared between S_DL and S_anat so the comparison is fair.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
IN_PATH = os.path.join(ROOT, "results_v2", "stage28", "s_distribution", "s_values_pooled.npz")
OUT_DIR = os.path.join(ROOT, "results_v2", "stage28", "s_distribution", "plots")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    d = np.load(IN_PATH)
    S_DL, S_anat, label = d["S_DL"], d["S_anat"], d["label"]
    bins = np.linspace(0, 1, 51)

    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.6))
    axes[0].hist(S_DL, bins=bins, color="#6b6b6b", edgecolor="none")
    axes[0].set_title(f"S_DL (n={len(S_DL)})\nmean={S_DL.mean():.4f}, median={np.median(S_DL):.4f}")
    axes[0].set_xlabel("S_DL"); axes[0].set_ylabel("patch count"); axes[0].set_xlim(0, 1)
    axes[1].hist(S_anat, bins=bins, color="#2f6690", edgecolor="none")
    axes[1].set_title(f"S_anat (n={len(S_anat)})\nmean={S_anat.mean():.4f}, median={np.median(S_anat):.4f}")
    axes[1].set_xlabel("S_anat"); axes[1].set_ylabel("patch count"); axes[1].set_xlim(0, 1)
    fig.suptitle("Patch-level order-magnitude distributions, pooled held-out GlaS (2083 patches).\n"
                 "Same x-axis range and bin width for both panels; diagnostic only, no criterion is evaluated here.",
                 fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(os.path.join(OUT_DIR, "s_histograms.png"), dpi=650)
    plt.close(fig)
    print("[plots] wrote s_histograms.png")

    # overlay (log-scale y, since S_DL is heavily concentrated near zero) -- scale interpretation kept
    # explicit via the shared x-axis and the legend; not used to imply a pass/fail reading
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(S_DL, bins=bins, alpha=0.55, color="#6b6b6b", label=f"S_DL (median {np.median(S_DL):.3f})")
    ax.hist(S_anat, bins=bins, alpha=0.55, color="#2f6690", label=f"S_anat (median {np.median(S_anat):.3f})")
    ax.set_yscale("log")
    ax.set_xlabel("order magnitude S"); ax.set_ylabel("patch count (log scale)")
    ax.set_title("S_DL vs S_anat, overlaid (log-scale count axis)\nDiagnostic only -- log scale chosen only to make both distributions visible together")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "s_histograms_overlay_log.png"), dpi=200)
    plt.close(fig)
    print("[plots] wrote s_histograms_overlay_log.png")

    # scatter S_DL vs S_anat
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(S_anat, S_DL, s=4, alpha=0.25, color="#6b6b6b", linewidths=0)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("S_anat"); ax.set_ylabel("S_DL")
    r = np.corrcoef(S_DL, S_anat)[0, 1]
    ax.set_title(f"S_DL vs S_anat, pooled held-out GlaS (n={len(S_DL)})\nPearson r = {r:.4f} (Criterion B's own reported value)")
    ax.plot([0, 1], [0, 1], color="black", linewidth=0.6, linestyle="--", label="y = x (reference)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "s_scatter.png"), dpi=200)
    plt.close(fig)
    print("[plots] wrote s_scatter.png, r =", r)


if __name__ == "__main__":
    main()
