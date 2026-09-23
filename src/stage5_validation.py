"""
STAGE 5: visualization + sanity-check of the FROZEN Stage 4 anatomical
targets. VALIDATION/INSPECTION ONLY -- loads Stage 4's persisted outputs
(results_v2/anatomy_targets/stage4_*.csv, stage4_fields/*.npz) as the
reference; does not recompute or alter any target, rule, or threshold.

Sample selection is by PRE-STATED, DETERMINISTIC rules (documented in each
selector function's docstring), applied to the Stage 4 manifest before any
figure is inspected -- not chosen after looking at candidate images.
"""
import sys
import os
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from nematic_math import phi_S_to_q, rotate_Q_analytic

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
AT_DIR = os.path.join(PROJECT_ROOT, "results_v2", "anatomy_targets")
FIELDS_DIR = os.path.join(AT_DIR, "stage4_fields")
FIG_DIR = os.path.join(PROJECT_ROOT, "results_v2", "figures", "stage5")
os.makedirs(FIG_DIR, exist_ok=True)

FRAGMENTED_TESTB_IDS = ["testB_3", "testB_11", "testB_18", "testB_19"]


def load_stage4():
    gland_df = pd.read_csv(os.path.join(AT_DIR, "stage4_gland_manifest.csv"))
    image_df = pd.read_csv(os.path.join(AT_DIR, "stage4_per_image_summary.csv"))
    return gland_df, image_df


def load_field(image_id):
    d = np.load(os.path.join(FIELDS_DIR, f"{image_id}.npz"))
    return {k: d[k] for k in d.files}


def load_mask(image_id, split):
    return np.array(Image.open(os.path.join(PROJECT_ROOT, "data", "glas", "Warwick_QU_Dataset", f"{image_id}_anno.bmp")))


def load_image(image_id):
    return np.array(Image.open(os.path.join(PROJECT_ROOT, "data", "glas", "Warwick_QU_Dataset", f"{image_id}.bmp")).convert("RGB"))


# ---------------------------------------------------------------------
# DETERMINISTIC, PRE-STATED SAMPLE SELECTION (documented before any figure is viewed)
# ---------------------------------------------------------------------
def select_examples(gland_df, image_df):
    picks = {}

    valid = gland_df[gland_df.validity_category == "valid"]
    near_iso = gland_df[gland_df.validity_category == "excluded_near_isotropic"]

    # A: typical elongated gland = valid instance with S_cov closest to the
    # 75th percentile of the valid S_cov distribution (a clearly-elongated,
    # non-extreme-outlier case), ties broken by (image_id, gland_id) order.
    q75 = valid.S_cov.quantile(0.75)
    a_row = valid.loc[[(valid.S_cov - q75).abs().idxmin()]]
    picks["A_typical_elongated"] = a_row.iloc[0].to_dict()
    picks["A_typical_elongated"]["_rule"] = "valid instance with S_cov closest to the 75th percentile of all valid S_cov"

    # B: approximately isotropic/round gland = excluded_near_isotropic instance
    # with S_cov closest to the MEDIAN of that category (representative, not
    # the most extreme near-zero case).
    med_iso = near_iso.S_cov.median()
    b_row = near_iso.loc[[(near_iso.S_cov - med_iso).abs().idxmin()]]
    picks["B_isotropic_round"] = b_row.iloc[0].to_dict()
    picks["B_isotropic_round"]["_rule"] = "excluded_near_isotropic instance with S_cov closest to the median of that category"

    # C: small valid gland = valid instance with the SMALLEST n_pixels (deterministic min).
    c_row = valid.loc[[valid.n_pixels.idxmin()]]
    picks["C_small_valid"] = c_row.iloc[0].to_dict()
    picks["C_small_valid"]["_rule"] = "valid instance with the minimum n_pixels among all valid instances"

    # D: multiple glands within one image = the image (any split) with the
    # MAXIMUM n_gland_instances overall (deterministic max).
    d_row = image_df.loc[[image_df.n_gland_instances.idxmax()]]
    picks["D_multi_gland_image"] = d_row.iloc[0].to_dict()
    picks["D_multi_gland_image"]["_rule"] = "image with the maximum n_gland_instances across the whole dataset"

    # E: low-anisotropy VALID gland = valid instance with the SMALLEST S_cov
    # (closest to the 0.15 threshold from below-excluded side, but still valid).
    e_row = valid.loc[[valid.S_cov.idxmin()]]
    picks["E_low_anisotropy_valid"] = e_row.iloc[0].to_dict()
    picks["E_low_anisotropy_valid"]["_rule"] = "valid instance with the minimum S_cov among all valid instances"

    # F/G/H: typical (median n_gland_instances) image per split; H excludes
    # the 4 fragmented testB images (handled separately as case I).
    for split, key in [("train", "F_typical_train"), ("testA", "G_typical_testA")]:
        sub = image_df[image_df.canonical_split == split].sort_values("image_id").reset_index(drop=True)
        med = sub.n_gland_instances.median()
        row = sub.loc[[(sub.n_gland_instances - med).abs().idxmin()]]
        picks[key] = row.iloc[0].to_dict()
        picks[key]["_rule"] = f"{split} image with n_gland_instances closest to the {split} median ({med})"

    subB = image_df[(image_df.canonical_split == "testB") & (~image_df.image_id.isin(FRAGMENTED_TESTB_IDS))].sort_values("image_id").reset_index(drop=True)
    medB = subB.n_gland_instances.median()
    rowH = subB.loc[[(subB.n_gland_instances - medB).abs().idxmin()]]
    picks["H_typical_testB"] = rowH.iloc[0].to_dict()
    picks["H_typical_testB"]["_rule"] = f"testB image (excluding the 4 flagged fragmented images) with n_gland_instances closest to the testB median ({medB})"

    # I: all 4 fragmented testB images (all included, per instruction's "preferably more than one")
    picks["I_fragmented_testB"] = {"image_ids": FRAGMENTED_TESTB_IDS,
                                    "_rule": "all 4 images previously flagged fragmented in Stage 2/3/4"}

    return picks


# ---------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------
def phi_field_rgb(field):
    hue = field["phi"] / np.pi
    sat = field["valid"].astype(np.float32)
    val = field["valid"].astype(np.float32)
    return hsv_to_rgb(np.stack([hue, sat, val], axis=-1))


def draw_axis(ax, cy, cx, phi, npix, color, label=None):
    L = 0.6 * np.sqrt(npix)
    dx, dy = L * np.cos(phi), L * np.sin(phi)
    ax.plot([cx - dx, cx + dx], [cy - dy, cy + dy], color=color, linewidth=2.5, label=label)
    ax.scatter([cx], [cy], color=color, s=30, marker="o", zorder=5)


def figure_full_image(image_id, split, gland_df, highlight_gland_id=None, title_suffix="", out_name=None):
    img = load_image(image_id)
    mask = load_mask(image_id, split)
    field = load_field(image_id)
    rows = gland_df[gland_df.image_id == image_id]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes[0, 0].imshow(img); axes[0, 0].set_title(f"{image_id} ({split}) -- H&E (context only)")
    axes[0, 1].imshow(mask, cmap="tab20"); axes[0, 1].set_title(f"gland instance mask (n={len(rows)})")

    axes[0, 2].imshow(img)
    for _, r in rows.iterrows():
        gmask = mask == r["gland_id"]
        ys, xs = np.nonzero(gmask)
        cy, cx = ys.mean(), xs.mean()
        if r["validity_category"] == "valid":
            color = "yellow" if (highlight_gland_id is not None and r["gland_id"] == highlight_gland_id) else "lime"
        else:
            color = "red"
        if pd.notna(r["phi_cov"]):
            draw_axis(axes[0, 2], cy, cx, r["phi_cov"], r["n_pixels"], color)
    axes[0, 2].set_title("principal axes (lime=valid, red=excluded, yellow=highlighted)")

    im = axes[0, 3].imshow(field["S"], cmap="viridis", vmin=0, vmax=1)
    axes[0, 3].set_title("S (anisotropy) field"); plt.colorbar(im, ax=axes[0, 3], fraction=0.046)

    im = axes[1, 0].imshow(field["q1"], cmap="RdBu", vmin=-1, vmax=1)
    axes[1, 0].set_title("q1 = S*cos(2phi)"); plt.colorbar(im, ax=axes[1, 0], fraction=0.046)
    im = axes[1, 1].imshow(field["q2"], cmap="RdBu", vmin=-1, vmax=1)
    axes[1, 1].set_title("q2 = S*sin(2phi)"); plt.colorbar(im, ax=axes[1, 1], fraction=0.046)
    axes[1, 2].imshow(phi_field_rgb(field)); axes[1, 2].set_title("phi director field (hue=angle, black=invalid)")
    axes[1, 3].imshow(field["valid"], cmap="gray"); axes[1, 3].set_title(f"valid mask (frac={field['valid'].mean():.2f})")

    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(f"{image_id} {title_suffix}")
    fig.tight_layout()
    out_name = out_name or f"{image_id}_full.png"
    path = os.path.join(FIG_DIR, out_name)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def figure_gland_closeup(image_id, split, gland_id, gland_df, case_label, out_name, margin=40):
    img = load_image(image_id)
    mask = load_mask(image_id, split)
    row = gland_df[(gland_df.image_id == image_id) & (gland_df.gland_id == gland_id)].iloc[0]

    gmask = mask == gland_id
    ys, xs = np.nonzero(gmask)
    y0, y1 = max(0, ys.min() - margin), min(mask.shape[0], ys.max() + margin)
    x0, x1 = max(0, xs.min() - margin), min(mask.shape[1], xs.max() + margin)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img[y0:y1, x0:x1]); axes[0].set_title(f"{image_id} gland {gland_id} -- H&E crop (context only)")

    axes[1].imshow(gmask[y0:y1, x0:x1], cmap="gray")
    cy, cx = ys.mean() - y0, xs.mean() - x0
    if pd.notna(row["phi_cov"]):
        draw_axis(axes[1], cy, cx, row["phi_cov"], row["n_pixels"], "red")
    axes[1].set_title(f"mask crop + principal axis\nphi={np.degrees(row['phi_cov']) if pd.notna(row['phi_cov']) else float('nan'):.1f} deg  S={row['S_cov']:.3f}"
                       if pd.notna(row["phi_cov"]) else "mask crop (phi undefined -- too_small)")

    q1, q2 = (phi_S_to_q(row["phi_cov"], row["S_cov"]) if pd.notna(row["phi_cov"]) else (float("nan"), float("nan")))
    axes[2].axis("off")
    txt = (f"gland_id: {gland_id}\nn_pixels: {row['n_pixels']}\n"
           f"validity_category: {row['validity_category']}\n"
           f"S_cov: {row['S_cov']}\nphi_cov (deg): {np.degrees(row['phi_cov']) if pd.notna(row['phi_cov']) else 'NA'}\n"
           f"q1: {q1:.4f}\nq2: {q2:.4f}\n"
           f"S_contour: {row['S_contour']}\nphi_contour (deg): {np.degrees(row['phi_contour']) if pd.notna(row['phi_contour']) else 'NA'}\n"
           f"border_truncated: {row['border_truncated']}\nnear_isotropic: {row['near_isotropic']}\n"
           f"fragmented: {row['fragmented']}  n_components: {row['n_connected_components']}")
    axes[2].text(0.05, 0.95, txt, va="top", fontsize=11, family="monospace")

    for ax in axes[:2]:
        ax.axis("off")
    fig.suptitle(f"Case {case_label}: {image_id} gland {gland_id}")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, out_name)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main():
    gland_df, image_df = load_stage4()
    picks = select_examples(gland_df, image_df)

    # log the selection deterministically, before generating/inspecting any figure
    picks_log = {k: {kk: (vv if not isinstance(vv, (np.integer, np.floating)) else float(vv))
                      for kk, vv in v.items()} for k, v in picks.items()}
    with open(os.path.join(FIG_DIR, "stage5_sample_selection.json"), "w") as f:
        json.dump(picks_log, f, indent=2, default=str)
    print("[stage5] sample selection (deterministic, pre-stated rules):")
    print(json.dumps(picks_log, indent=2, default=str))

    generated_figures = []

    # A, B, C, E: gland close-ups
    for key, label in [("A_typical_elongated", "A"), ("B_isotropic_round", "B"),
                        ("C_small_valid", "C"), ("E_low_anisotropy_valid", "E")]:
        p = picks[key]
        path = figure_gland_closeup(p["image_id"], p["canonical_split"], int(p["gland_id"]), gland_df,
                                     label, f"case_{label}_{p['image_id']}_gland{int(p['gland_id'])}.png")
        generated_figures.append(path)

    # D, F, G, H: full-image panels
    for key, label in [("D_multi_gland_image", "D"), ("F_typical_train", "F"),
                        ("G_typical_testA", "G"), ("H_typical_testB", "H")]:
        p = picks[key]
        path = figure_full_image(p["image_id"], p["canonical_split"], gland_df,
                                  title_suffix=f"(case {label})", out_name=f"case_{label}_{p['image_id']}_full.png")
        generated_figures.append(path)

    # I: all 4 fragmented testB images, full-image panels
    for image_id in FRAGMENTED_TESTB_IDS:
        path = figure_full_image(image_id, "testB", gland_df, title_suffix="(case I, fragmented)",
                                  out_name=f"case_I_{image_id}_full.png")
        generated_figures.append(path)

    print(f"[stage5] {len(generated_figures)} figures generated in {FIG_DIR}")
    for p in generated_figures:
        print("   ", p)

    return picks, generated_figures


if __name__ == "__main__":
    main()
