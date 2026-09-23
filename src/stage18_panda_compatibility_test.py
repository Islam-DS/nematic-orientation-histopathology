"""
STAGE 18: PANDA Radboud annotation compatibility test. Reads the 10-case
pilot's masks (semantic Gleason-pattern labels: 0=background, 1=stroma,
2=benign epithelium, 3/4/5=Gleason pattern 3/4/5) and tests, via
connected-component analysis, whether each class's regions correspond to
individually-separable, gland-scale objects (as GlaS/CRAG's instance
masks do) or large continuous/merged tissue sheets. No gland boundary is
invented; this only measures what the existing semantic labels already
contain.
"""
import os
import json

import numpy as np
import pandas as pd
import tifffile
from skimage.measure import label as cc_label, regionprops

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
PANDA_DIR = os.path.join(PROJECT_ROOT, "data", "PANDA")
OUT_DIR = os.path.join(PROJECT_ROOT, "results_v2", "validation", "stage18_panda")
os.makedirs(os.path.join(OUT_DIR, "qc_figures"), exist_ok=True)

CLASS_NAMES = {0: "background", 1: "stroma", 2: "benign_epithelium", 3: "gleason_3", 4: "gleason_4", 5: "gleason_5"}
GLAND_LIKE_CLASSES = [2, 3, 4, 5]  # epithelial/cancerous classes -- the closest analogue to "gland" tissue
MIN_COMPONENT_PIXELS = 20  # same MIN_GLAND_PIXELS threshold as Stage 4, for a like-for-like size filter


def analyze_mask(mask_path, image_id):
    label_channel = tifffile.imread(mask_path, key=0)[..., 0]
    H, W = label_channel.shape
    total_px = H * W

    class_stats = {}
    for cls, name in CLASS_NAMES.items():
        px_count = int((label_channel == cls).sum())
        class_stats[name] = {"pixel_count": px_count, "fraction_of_image": px_count / total_px}

    component_stats = {}
    for cls in GLAND_LIKE_CLASSES:
        binary = (label_channel == cls)
        if not binary.any():
            component_stats[CLASS_NAMES[cls]] = {"n_components_total": 0, "n_components_ge_min_pixels": 0}
            continue
        cc = cc_label(binary, connectivity=2)
        props = regionprops(cc)
        areas = [p.area for p in props]
        areas_filtered = [a for a in areas if a >= MIN_COMPONENT_PIXELS]
        largest = max(areas) if areas else 0
        component_stats[CLASS_NAMES[cls]] = {
            "n_components_total": len(areas),
            "n_components_ge_min_pixels": len(areas_filtered),
            "area_px_min": int(min(areas)) if areas else None,
            "area_px_median": float(np.median(areas)) if areas else None,
            "area_px_mean": float(np.mean(areas)) if areas else None,
            "area_px_max": int(largest) if areas else None,
            "largest_component_fraction_of_class_area": float(largest / sum(areas)) if areas else None,
            "largest_component_fraction_of_image_area": float(largest / total_px) if areas else None,
        }

    return {"image_id": image_id, "mask_shape": [int(H), int(W)], "total_pixels": total_px,
            "class_pixel_stats": class_stats, "component_stats": component_stats}


def main():
    pilot = pd.read_csv(os.path.join(PANDA_DIR, "radboud_pilot_manifest.csv"))
    results = []
    for _, row in pilot.iterrows():
        iid = row["image_id"]
        mask_path = os.path.join(PANDA_DIR, "train_label_masks", f"{iid}_mask.tiff")
        print(f"[stage18-compat] analyzing {iid} (ISUP={row['isup_grade']}, Gleason={row['gleason_score']}) ...")
        r = analyze_mask(mask_path, iid)
        r["isup_grade"] = int(row["isup_grade"]); r["gleason_score"] = row["gleason_score"]
        results.append(r)
        cs = r["component_stats"]["benign_epithelium"]
        print(f"    benign_epithelium: n_components={cs['n_components_total']}, "
              f"median_area={cs.get('area_px_median')}, largest={cs.get('area_px_max')}, "
              f"largest_frac_of_image={cs.get('largest_component_fraction_of_image_area')}")

    with open(os.path.join(OUT_DIR, "compatibility_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n[stage18-compat] DONE.")
    return results


if __name__ == "__main__":
    main()
