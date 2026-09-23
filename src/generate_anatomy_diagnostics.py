"""
Diagnostics only -- no model training. Generates:
  1. Full-dataset gland statistics (border truncation, size distribution,
     connected-component fragmentation) -- Addendum 4 mask-quality check.
  2. S_cov vs S_contour agreement/disagreement analysis -- Addendum 3 (J).
  3. Per-image visual diagnostic panels for a sample of GlaS images.
  4. A numeric summary JSON tying it all together, used to justify the
     final S_threshold / border_frac_threshold choices in
     docs_v2/phase2_target_spec.md.
"""
import sys
import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from skimage.measure import label as cc_label

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
from data_prep import build_image_label_table  # reuse only

sys.path.insert(0, os.path.dirname(__file__))
from anatomy_targets import (
    analyze_glands, build_full_image_field, BORDER_FRAC_THRESHOLD, DEFAULT_S_THRESHOLD,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results_v2", "anatomy_targets")
os.makedirs(OUT_DIR, exist_ok=True)


def anno_path_for(image_path):
    stem, ext = os.path.splitext(image_path)
    return stem + "_anno" + ext


def load_labeled_mask(image_path):
    p = anno_path_for(image_path)
    return np.array(Image.open(p))


# ---------------------------------------------------------------------
# 1 & 2: full-dataset statistics (mask quality + S measure agreement)
# ---------------------------------------------------------------------
def full_dataset_analysis():
    table = build_image_label_table()
    all_props = []
    fragment_counts = []

    for _, row in table.iterrows():
        labeled = load_labeled_mask(row["path"])
        props = analyze_glands(labeled)
        for p in props:
            p["image"] = os.path.basename(row["path"])
        all_props.extend(props)

        ids = np.unique(labeled)
        ids = ids[ids > 0]
        for gid in ids:
            cc = cc_label(labeled == gid)
            fragment_counts.append(int(cc.max()))

    n_total = len(all_props)
    n_too_small = sum(p["too_small"] for p in all_props)
    n_border = sum(p["border_truncated"] for p in all_props)
    n_border_any_touch = sum(p["border_pixels"] > 0 for p in all_props)
    n_fragmented = sum(1 for f in fragment_counts if f > 1)

    valid_cov = [p for p in all_props if not p["too_small"] and p.get("S_cov") is not None]
    S_cov_vals = np.array([p["S_cov"] for p in valid_cov])
    S_contour_vals = np.array([p["S_contour"] for p in valid_cov if p.get("S_contour") is not None])
    both = [(p["S_cov"], p["S_contour"], p["phi_cov"], p["phi_contour"])
            for p in valid_cov if p.get("S_contour") is not None and p.get("phi_contour") is not None]
    S_cov_b = np.array([b[0] for b in both])
    S_con_b = np.array([b[1] for b in both])
    phi_cov_b = np.array([b[2] for b in both])
    phi_con_b = np.array([b[3] for b in both])
    phi_diff_deg = np.degrees(np.minimum(np.abs(phi_cov_b - phi_con_b) % np.pi,
                                          np.pi - (np.abs(phi_cov_b - phi_con_b) % np.pi)))

    n_low_aniso_cov = int((S_cov_vals < DEFAULT_S_THRESHOLD).sum())

    summary = {
        "n_total_gland_instances": n_total,
        "n_too_small_excluded": n_too_small,
        "n_border_touching_any": n_border_any_touch,
        "n_border_touching_any_pct": round(100 * n_border_any_touch / n_total, 1),
        "n_border_truncated_by_chosen_threshold": n_border,
        "n_border_truncated_pct": round(100 * n_border / n_total, 1),
        "border_frac_threshold_used": BORDER_FRAC_THRESHOLD,
        "n_fragmented_multi_component_labels": n_fragmented,
        "n_fragmented_pct": round(100 * n_fragmented / n_total, 2),
        "S_cov_distribution": {
            "mean": float(S_cov_vals.mean()), "median": float(np.median(S_cov_vals)),
            "p10": float(np.percentile(S_cov_vals, 10)), "p25": float(np.percentile(S_cov_vals, 25)),
            "p50": float(np.percentile(S_cov_vals, 50)), "p75": float(np.percentile(S_cov_vals, 75)),
        },
        "S_threshold_used": DEFAULT_S_THRESHOLD,
        "n_below_S_threshold": n_low_aniso_cov,
        "n_below_S_threshold_pct": round(100 * n_low_aniso_cov / len(S_cov_vals), 1),
        "S_cov_vs_S_contour": {
            "n_compared": len(both),
            "pearson_r": float(np.corrcoef(S_cov_b, S_con_b)[0, 1]),
            "mean_abs_diff": float(np.mean(np.abs(S_cov_b - S_con_b))),
        },
        "phi_cov_vs_phi_contour": {
            "mean_angular_diff_deg": float(np.mean(phi_diff_deg)),
            "median_angular_diff_deg": float(np.median(phi_diff_deg)),
            "frac_agree_within_10deg": float(np.mean(phi_diff_deg < 10)),
        },
        "final_valid_gland_count_estimate": int(sum(
            1 for p in all_props if not p["too_small"] and not p["border_truncated"]
            and p.get("S_cov") is not None and p["S_cov"] >= DEFAULT_S_THRESHOLD
        )),
    }

    with open(os.path.join(OUT_DIR, "target_generation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # scatter plot: S_cov vs S_contour, phi agreement
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(S_cov_b, S_con_b, s=4, alpha=0.25)
    axes[0].plot([0, 1], [0, 1], 'r--', linewidth=1)
    axes[0].set_xlabel("S_cov (pixel-covariance)")
    axes[0].set_ylabel("S_contour (boundary ellipse fit)")
    axes[0].set_title(f"r = {summary['S_cov_vs_S_contour']['pearson_r']:.3f}")

    axes[1].hist(phi_diff_deg, bins=40)
    axes[1].set_xlabel("|phi_cov - phi_contour| (deg, mod pi)")
    axes[1].set_ylabel("count")
    axes[1].set_title(f"median diff = {summary['phi_cov_vs_phi_contour']['median_angular_diff_deg']:.1f} deg")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "S_measure_agreement.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(S_cov_vals, bins=40)
    ax.axvline(DEFAULT_S_THRESHOLD, color="r", linestyle="--", label=f"threshold = {DEFAULT_S_THRESHOLD}")
    ax.set_xlabel("S_cov")
    ax.set_ylabel("gland count")
    ax.set_title("Distribution of covariance-based anisotropy across all GlaS glands")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "S_cov_distribution.png"), dpi=150)
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    return summary, table


# ---------------------------------------------------------------------
# 3: per-image visual diagnostic panels
# ---------------------------------------------------------------------
def visualize_image(image_path, out_path):
    img = np.array(Image.open(image_path).convert("RGB"))
    labeled = load_labeled_mask(image_path)
    props = analyze_glands(labeled)
    field = build_full_image_field(labeled, props, S_choice="cov")

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    axes[0, 0].imshow(img)
    axes[0, 0].set_title("original image")

    axes[0, 1].imshow(labeled, cmap="tab20")
    axes[0, 1].set_title(f"gland instance masks (n={len(props)})")

    axes[0, 2].imshow(img)
    for p in props:
        if p["too_small"] or p["phi_cov"] is None:
            continue
        mask = labeled == p["gland_id"]
        ys, xs = np.nonzero(mask)
        cy, cx = ys.mean(), xs.mean()
        L = 0.6 * np.sqrt(mask.sum())
        dx, dy = L * np.cos(p["phi_cov"]), L * np.sin(p["phi_cov"])
        color = "lime" if (not p["border_truncated"] and p["S_cov"] >= DEFAULT_S_THRESHOLD) else "red"
        axes[0, 2].plot([cx - dx, cx + dx], [cy - dy, cy + dy], color=color, linewidth=2)
    axes[0, 2].set_title("principal axes (green=valid, red=excluded)")

    conf = np.zeros(labeled.shape)
    for p in props:
        if p["too_small"] or p["phi_cov"] is None:
            continue
        mask = labeled == p["gland_id"]
        conf[mask] = p["S_cov"]
    im = axes[0, 3].imshow(conf, cmap="viridis", vmin=0, vmax=1)
    axes[0, 3].set_title("S_cov (anisotropy/confidence) map")
    plt.colorbar(im, ax=axes[0, 3], fraction=0.046)

    im = axes[1, 0].imshow(field["q1"], cmap="RdBu", vmin=-1, vmax=1)
    axes[1, 0].set_title("q1 = S*cos(2phi) map")
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

    im = axes[1, 1].imshow(field["q2"], cmap="RdBu", vmin=-1, vmax=1)
    axes[1, 1].set_title("q2 = S*sin(2phi) map")
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046)

    im = axes[1, 2].imshow(field["S"], cmap="viridis", vmin=0, vmax=1)
    axes[1, 2].set_title("final S magnitude (after exclusion)")
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046)

    hue = field["phi"] / np.pi
    sat = field["valid"].astype(np.float32)
    val = field["valid"].astype(np.float32)
    hsv = np.stack([hue, sat, val], axis=-1)
    from matplotlib.colors import hsv_to_rgb
    rgb = hsv_to_rgb(hsv)
    axes[1, 3].imshow(rgb)
    axes[1, 3].set_title("phi director field (hue=angle, black=invalid)")

    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(os.path.basename(image_path))
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    print("[diagnostics] running full-dataset analysis (mask quality + S-measure agreement)...")
    summary, table = full_dataset_analysis()

    print("[diagnostics] generating per-image visual panels...")
    sample = table.sample(n=15, random_state=0)
    diag_dir = os.path.join(OUT_DIR, "panels")
    os.makedirs(diag_dir, exist_ok=True)
    for _, row in sample.iterrows():
        out_path = os.path.join(diag_dir, os.path.splitext(os.path.basename(row["path"]))[0] + "_diagnostic.png")
        visualize_image(row["path"], out_path)
        print(f"    saved {out_path}")

    print(f"[diagnostics] done. Outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
