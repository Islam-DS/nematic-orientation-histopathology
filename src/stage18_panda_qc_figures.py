"""
STAGE 18: QC figures for the PANDA Radboud compatibility test. Selected
without reference to prediction quality (Model 3 was never run on PANDA)
-- selection is purely by ISUP grade to show the compatibility finding
across the grade spectrum.
"""
import os

import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from skimage.measure import label as cc_label

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
PANDA_DIR = os.path.join(PROJECT_ROOT, "data", "PANDA")
OUT_DIR = os.path.join(PROJECT_ROOT, "results_v2", "validation", "stage18_panda", "qc_figures")
os.makedirs(OUT_DIR, exist_ok=True)

CLASS_COLORS = ["white", "tan", "green", "gold", "orange", "red"]  # 0..5
CMAP = ListedColormap(CLASS_COLORS)


def downsample_read(path, level):
    return tifffile.imread(path, key=level)


def make_case_figure(image_id, isup, gleason):
    img_path = os.path.join(PANDA_DIR, "train_images", f"{image_id}.tiff")
    mask_path = os.path.join(PANDA_DIR, "train_label_masks", f"{image_id}_mask.tiff")

    img_low = downsample_read(img_path, 1)  # level 1 (downsampled ~4x) for display
    mask_low = downsample_read(mask_path, 1)[..., 0]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    axes[0].imshow(img_low)
    axes[0].set_title(f"{image_id[:8]} (H&E)\nISUP={isup}, Gleason={gleason}")
    axes[0].axis("off")

    axes[1].imshow(mask_low, cmap=CMAP, vmin=0, vmax=5)
    axes[1].set_title("semantic mask\n(0=bg,1=stroma,2=benign,3/4/5=Gleason 3/4/5)")
    axes[1].axis("off")

    # connected components of benign (2) and highest-grade class present, colored distinctly
    present_classes = [c for c in [4, 5, 3] if (mask_low == c).any()]
    target_cls = present_classes[0] if present_classes else 2
    binary = (mask_low == target_cls)
    cc = cc_label(binary, connectivity=2)
    rng = np.random.default_rng(0)
    colors = rng.random((cc.max() + 1, 3))
    colors[0] = [1, 1, 1]
    cc_rgb = colors[cc]
    axes[2].imshow(cc_rgb)
    axes[2].set_title(f"connected components of class {target_cls}\n({cc.max()} components, random colors per component)")
    axes[2].axis("off")

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, f"qc_{image_id[:8]}_isup{isup}.png")
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"[qc] wrote {out_path}")


def main():
    pilot = pd.read_csv(os.path.join(PANDA_DIR, "radboud_pilot_manifest.csv"))
    # Selected purely by ISUP grade coverage (low/mid/high), not by any prediction or appearance criterion
    selection = pilot[pilot.isup_grade.isin([0, 2, 4, 5])].drop_duplicates("isup_grade").sort_values("isup_grade")
    for _, row in selection.iterrows():
        make_case_figure(row["image_id"], row["isup_grade"], row["gleason_score"])


if __name__ == "__main__":
    main()
