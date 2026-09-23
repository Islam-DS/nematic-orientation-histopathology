"""STAGE 5: dataset-level distribution summaries, split by canonical split. Characterization only."""
import os
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AT_DIR = os.path.join(os.path.dirname(__file__), "..", "results_v2", "anatomy_targets")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "results_v2", "figures", "stage5")
os.makedirs(FIG_DIR, exist_ok=True)


def main():
    df = pd.read_csv(os.path.join(AT_DIR, "stage4_gland_manifest.csv"))
    img = pd.read_csv(os.path.join(AT_DIR, "stage4_per_image_summary.csv"))
    valid = df[df.validity_category == "valid"].copy()
    valid["q1"] = valid.S_cov * np.cos(2 * valid.phi_cov)
    valid["q2"] = valid.S_cov * np.sin(2 * valid.phi_cov)

    splits = ["train", "testA", "testB"]
    colors = {"train": "tab:blue", "testA": "tab:orange", "testB": "tab:green"}

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))

    ax = axes[0, 0]
    for s in splits:
        ax.hist(np.log10(df[df.canonical_split == s].n_pixels), bins=30, alpha=0.5, label=s, color=colors[s], density=True)
    ax.set_title("gland area (log10 n_pixels), all instances"); ax.set_xlabel("log10(pixels)"); ax.legend()

    ax = axes[0, 1]
    for s in splits:
        ax.hist(img[img.canonical_split == s].n_gland_instances, bins=range(0, 34, 2), alpha=0.5, label=s, color=colors[s], density=True)
    ax.set_title("glands per image"); ax.set_xlabel("n_gland_instances"); ax.legend()

    ax = axes[0, 2]
    for s in splits:
        ax.hist(valid[valid.canonical_split == s].S_cov, bins=30, alpha=0.5, label=s, color=colors[s], density=True, range=(0, 1))
    ax.set_title("S_cov (valid instances only)"); ax.set_xlabel("S_cov"); ax.legend()

    ax = axes[0, 3]
    for s in splits:
        ax.hist(np.degrees(valid[valid.canonical_split == s].phi_cov), bins=30, alpha=0.5, label=s, color=colors[s], density=True, range=(0, 180))
    ax.set_title("phi_cov (deg, valid instances only)"); ax.set_xlabel("phi_cov (deg)"); ax.legend()

    ax = axes[1, 0]
    for s in splits:
        ax.hist(valid[valid.canonical_split == s].q1, bins=30, alpha=0.5, label=s, color=colors[s], density=True, range=(-1, 1))
    ax.set_title("q1 (valid instances only)"); ax.set_xlabel("q1"); ax.legend()

    ax = axes[1, 1]
    for s in splits:
        ax.hist(valid[valid.canonical_split == s].q2, bins=30, alpha=0.5, label=s, color=colors[s], density=True, range=(-1, 1))
    ax.set_title("q2 (valid instances only)"); ax.set_xlabel("q2"); ax.legend()

    ax = axes[1, 2]
    cat_counts = df.groupby(["canonical_split", "validity_category"]).size().unstack(fill_value=0)
    cat_counts = cat_counts.reindex(splits)
    cat_counts.plot(kind="bar", stacked=True, ax=ax, legend=True)
    ax.set_title("validity category counts by split"); ax.set_xlabel(""); plt.setp(ax.get_xticklabels(), rotation=0)

    ax = axes[1, 3]
    ax.scatter(valid.S_cov, valid.S_contour, s=4, alpha=0.2)
    ax.plot([0, 1], [0, 1], "r--", linewidth=1)
    ax.set_xlabel("S_cov"); ax.set_ylabel("S_contour")
    r = np.corrcoef(valid.S_cov, valid.S_contour.fillna(valid.S_cov))[0, 1]
    ax.set_title(f"S_cov vs S_contour (valid), r={r:.3f}")

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "stage5_dataset_distributions.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[stage5_distributions] figure saved: {path}")

    # numeric summary table
    summary = {}
    for s in splits:
        sub = df[df.canonical_split == s]
        vsub = valid[valid.canonical_split == s]
        isub = img[img.canonical_split == s]
        summary[s] = {
            "n_images": int(len(isub)),
            "n_gland_instances": int(len(sub)),
            "glands_per_image": {"mean": float(isub.n_gland_instances.mean()), "median": float(isub.n_gland_instances.median()),
                                   "min": int(isub.n_gland_instances.min()), "max": int(isub.n_gland_instances.max())},
            "gland_area_pixels": {"mean": float(sub.n_pixels.mean()), "median": float(sub.n_pixels.median()),
                                    "min": int(sub.n_pixels.min()), "max": int(sub.n_pixels.max())},
            "validity_category_counts": sub.validity_category.value_counts().to_dict(),
            "S_cov_valid": {"mean": float(vsub.S_cov.mean()), "median": float(vsub.S_cov.median()), "std": float(vsub.S_cov.std())} if len(vsub) else None,
            "phi_cov_valid_deg": {"mean": float(np.degrees(vsub.phi_cov).mean()), "std": float(np.degrees(vsub.phi_cov).std())} if len(vsub) else None,
            "q1_valid": {"mean": float(vsub.q1.mean()), "std": float(vsub.q1.std())} if len(vsub) else None,
            "q2_valid": {"mean": float(vsub.q2.mean()), "std": float(vsub.q2.std())} if len(vsub) else None,
        }

    summary_path = os.path.join(FIG_DIR, "stage5_dataset_distributions_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[stage5_distributions] summary saved: {summary_path}")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
