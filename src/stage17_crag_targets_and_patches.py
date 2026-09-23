"""
STAGE 17: CRAG anatomical target generation + patch extraction. Reuses the
EXACT, unmodified Stage 4 mathematical pipeline (anatomy_targets.py) and
the EXACT, unmodified patch framework (code/data_prep.py::extract_patches,
anatomy_targets.py::crop_field_to_patches, phase2_calibration.py's
MIN_VALID_FRACTION_FOR_PATCH=0.10) -- nothing here is CRAG-specific except
the file paths. No pixel of the H&E image is ever used to determine
orientation (mask-only), exactly mirroring Stage 4's own data-independence
rule.
"""
import sys
import os
import glob
import json
import time

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
from data_prep import extract_patches  # reuse only, unmodified

sys.path.insert(0, os.path.dirname(__file__))
from anatomy_targets import analyze_glands, build_full_image_field, crop_field_to_patches, \
    DEFAULT_S_THRESHOLD, BORDER_FRAC_THRESHOLD, MIN_GLAND_PIXELS
from phase2_calibration import MIN_VALID_FRACTION_FOR_PATCH  # reuse only, unmodified (0.10)

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
CRAG_ROOT = os.path.join(PROJECT_ROOT, "data", "CRAG", "OpenDataLab___CRAG", "raw", "CRAG", "extracted", "CRAG")
OUT_DIR = os.path.join(PROJECT_ROOT, "results_v2", "validation", "stage17_crag")
TARGETS_DIR = os.path.join(OUT_DIR, "targets", "dense_fields")
os.makedirs(TARGETS_DIR, exist_ok=True)

PATCH_SIZE = 128
STRIDE = 96

SPLIT_FOLDERS = [("train", "train"), ("test", "valid")]  # (our_split_name, CRAG's own folder name)


def classify_instance(entry):
    """Identical to stage4_generate_targets.py::classify_instance -- reused unchanged."""
    if entry["too_small"]:
        return "excluded_too_small"
    if entry["border_truncated"]:
        return "excluded_border_truncated"
    if entry["numerically_unstable"]:
        return "excluded_numerically_unstable"
    if entry["near_isotropic"]:
        return "excluded_near_isotropic"
    return "valid"


def main():
    t0 = time.time()
    gland_rows = []
    per_image_summary = []
    patch_cache = {"train": {"rgb": [], "q1": [], "q2": [], "valid": [], "image_id": []},
                   "test": {"rgb": [], "q1": [], "q2": [], "valid": [], "image_id": []}}

    for our_split, folder in SPLIT_FOLDERS:
        img_files = sorted(glob.glob(os.path.join(CRAG_ROOT, folder, "Images", "*.png")))
        print(f"[stage17-targets] {our_split} ({folder}/): {len(img_files)} images")
        for imgf in img_files:
            stem = os.path.splitext(os.path.basename(imgf))[0]
            annf = os.path.join(CRAG_ROOT, folder, "Annotation", stem + ".png")
            labeled_mask = np.array(Image.open(annf))
            H, W = labeled_mask.shape

            props = analyze_glands(labeled_mask, min_pixels=MIN_GLAND_PIXELS, border_frac_threshold=BORDER_FRAC_THRESHOLD)
            field = build_full_image_field(labeled_mask, props, S_choice="cov",
                                            S_threshold=DEFAULT_S_THRESHOLD, exclude_border_truncated=True)

            np.savez_compressed(os.path.join(TARGETS_DIR, f"{stem}.npz"),
                                 q1=field["q1"].astype(np.float32), q2=field["q2"].astype(np.float32),
                                 S=field["S"].astype(np.float32), phi=field["phi"].astype(np.float32),
                                 valid=field["valid"].astype(bool))

            n_valid = n_excluded = n_fragmented = 0
            for entry in props:
                category = classify_instance(entry)
                if category == "valid":
                    n_valid += 1
                else:
                    n_excluded += 1
                if entry["fragmented"]:
                    n_fragmented += 1
                gland_rows.append({
                    "image_id": stem, "crag_split": our_split, "gland_id": entry["gland_id"],
                    "n_pixels": entry["n_pixels"], "border_frac": entry["border_frac"],
                    "border_truncated": entry["border_truncated"], "too_small": entry["too_small"],
                    "n_connected_components": entry["n_connected_components"], "fragmented": entry["fragmented"],
                    "phi_cov": entry["phi_cov"], "S_cov": entry["S_cov"],
                    "lambda1": entry.get("lambda1"), "lambda2": entry.get("lambda2"),
                    "phi_contour": entry["phi_contour"], "S_contour": entry["S_contour"],
                    "near_isotropic": entry["near_isotropic"], "numerically_unstable": entry["numerically_unstable"],
                    "validity_category": category, "area_fraction_of_image": entry["n_pixels"] / (H * W),
                })

            per_image_summary.append({
                "image_id": stem, "crag_split": our_split, "width": int(W), "height": int(H),
                "n_gland_instances": len(props), "n_valid": n_valid, "n_excluded": n_excluded,
                "n_fragmented_instances": n_fragmented, "valid_pixel_fraction": float(field["valid"].mean()),
            })

            # ---- patch extraction (reuses the exact established framework) ----
            field_patches = crop_field_to_patches(field, labeled_mask.shape, patch_size=PATCH_SIZE, stride=STRIDE)
            rgb_patches = extract_patches(imgf, patch_size=PATCH_SIZE, stride=STRIDE)
            assert len(field_patches) == len(rgb_patches), f"patch grid mismatch for {stem}"
            for fp, rgb in zip(field_patches, rgb_patches):
                if fp["valid"].mean() < MIN_VALID_FRACTION_FOR_PATCH:
                    continue
                patch_cache[our_split]["rgb"].append(rgb.astype(np.uint8))
                patch_cache[our_split]["q1"].append(fp["q1"].astype(np.float32))
                patch_cache[our_split]["q2"].append(fp["q2"].astype(np.float32))
                patch_cache[our_split]["valid"].append(fp["valid"].astype(bool))
                patch_cache[our_split]["image_id"].append(stem)

        print(f"[stage17-targets] {our_split}: {sum(1 for r in gland_rows if r['crag_split']==our_split)} gland rows so far, "
              f"{len(patch_cache[our_split]['rgb'])} valid patches so far, elapsed={time.time()-t0:.1f}s")

    gland_df = pd.DataFrame(gland_rows)
    gland_manifest_path = os.path.join(OUT_DIR, "manifests", "crag_gland_manifest.csv")
    gland_df.to_csv(gland_manifest_path, index=False)

    image_df = pd.DataFrame(per_image_summary)
    image_summary_path = os.path.join(OUT_DIR, "manifests", "crag_dataset_manifest.csv")
    image_df.to_csv(image_summary_path, index=False)

    for split_name in ["train", "test"]:
        pc = patch_cache[split_name]
        out_path = os.path.join(OUT_DIR, "targets", f"crag_{split_name}_patches.npz")
        np.savez_compressed(out_path,
                             rgb=np.stack(pc["rgb"]), q1=np.stack(pc["q1"]), q2=np.stack(pc["q2"]),
                             valid=np.stack(pc["valid"]), image_id=np.array(pc["image_id"]))
        print(f"[stage17-targets] wrote {out_path}: {len(pc['rgb'])} patches")

    summary = {
        "n_images": len(image_df), "n_train_images": int((image_df.crag_split == "train").sum()),
        "n_test_images": int((image_df.crag_split == "test").sum()),
        "n_gland_instances_total": len(gland_df),
        "validity_category_counts": gland_df["validity_category"].value_counts().to_dict(),
        "n_fragmented_instances": int(gland_df["fragmented"].sum()),
        "n_patches_valid": {s: len(patch_cache[s]["rgb"]) for s in ["train", "test"]},
        "min_gland_pixels": MIN_GLAND_PIXELS, "border_frac_threshold": BORDER_FRAC_THRESHOLD,
        "default_S_threshold": DEFAULT_S_THRESHOLD, "min_valid_fraction_for_patch": MIN_VALID_FRACTION_FOR_PATCH,
        "patch_size": PATCH_SIZE, "stride": STRIDE,
        "duration_sec": time.time() - t0,
    }
    with open(os.path.join(OUT_DIR, "crag_target_generation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    main()
