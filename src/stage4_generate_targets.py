"""
STAGE 4: anatomical target generation.

Constructs the independent anatomical orientation-tensor target for every
gland instance in all 165 GlaS images (train + testA + testB), using ONLY
the segmentation mask (never the H&E image) as input to the orientation
math. Deterministic: no randomness anywhere in this script.

CRITICAL DATA-INDEPENDENCE RULE (verified by construction, not just
asserted): this script's only per-image inputs are the labeled mask array
(via anatomy_targets.analyze_glands / build_full_image_field, both of
which take a `labeled_mask` argument and nothing else pixel-related) and
the image's (width, height) shape (needed only to size the output arrays
and check border-touching, not to inspect pixel content). No RGB image
array is ever loaded or passed into any orientation-determining function
in this script -- confirmed by code inspection: search this file for any
call that opens an image_path (there is none).

Native geometry is preserved throughout: all computation happens on each
mask at its own native resolution (no resizing), consistent with the 14
non-standard-dimension images identified in Stage 2/3. No coordinate
transformation is applied (none is needed, since nothing is resized).
"""
import sys
import os
import json

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from anatomy_targets import analyze_glands, build_full_image_field, DEFAULT_S_THRESHOLD, BORDER_FRAC_THRESHOLD, MIN_GLAND_PIXELS

MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "..", "data_manifests", "glas_manifest_stage3_finalized.csv")
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results_v2", "anatomy_targets")
FIELDS_DIR = os.path.join(OUT_DIR, "stage4_fields")
os.makedirs(FIELDS_DIR, exist_ok=True)

FRAGMENTED_TESTB_IDS = {"testB_3", "testB_11", "testB_18", "testB_19"}  # from Stage 2/3, for targeted reporting only


def classify_instance(entry):
    """Single validity category per instance, from the flags already computed
    in analyze_glands. 'fragmented' is informational and does not by itself
    change the category, per Stage 4 instruction ("do not create special
    rules merely because they are in testB")."""
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
    manifest = pd.read_csv(MANIFEST_PATH)
    assert manifest["stage3_include"].all(), "unexpected: some images excluded by Stage 3 -- Stage 4 expects the full 165"
    print(f"[stage4] processing {len(manifest)} images from the Stage 3 finalized manifest "
          f"(train={  (manifest.canonical_split=='train').sum()}, "
          f"testA={(manifest.canonical_split=='testA').sum()}, "
          f"testB={(manifest.canonical_split=='testB').sum()})")

    gland_rows = []
    per_image_summary = []

    for _, row in manifest.iterrows():
        mask_path = os.path.join(PROJECT_ROOT, row["mask_path"])
        # MASK ONLY -- no image_path is ever opened in this script.
        labeled_mask = np.array(Image.open(mask_path))
        H, W = labeled_mask.shape

        props = analyze_glands(labeled_mask, min_pixels=MIN_GLAND_PIXELS,
                                 border_frac_threshold=BORDER_FRAC_THRESHOLD)
        field = build_full_image_field(labeled_mask, props, S_choice="cov",
                                        S_threshold=DEFAULT_S_THRESHOLD, exclude_border_truncated=True)

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
                "image_id": row["image_id"], "canonical_split": row["canonical_split"],
                "gland_id": entry["gland_id"], "n_pixels": entry["n_pixels"],
                "border_frac": entry["border_frac"], "border_truncated": entry["border_truncated"],
                "too_small": entry["too_small"],
                "n_connected_components": entry["n_connected_components"], "fragmented": entry["fragmented"],
                "phi_cov": entry["phi_cov"], "S_cov": entry["S_cov"],
                "lambda1": entry.get("lambda1"), "lambda2": entry.get("lambda2"),
                "phi_contour": entry["phi_contour"], "S_contour": entry["S_contour"],
                "near_isotropic": entry["near_isotropic"], "numerically_unstable": entry["numerically_unstable"],
                "validity_category": category,
            })

        np.savez_compressed(
            os.path.join(FIELDS_DIR, f"{row['image_id']}.npz"),
            q1=field["q1"].astype(np.float32), q2=field["q2"].astype(np.float32),
            S=field["S"].astype(np.float32), phi=field["phi"].astype(np.float32),
            valid=field["valid"].astype(bool),
        )

        per_image_summary.append({
            "image_id": row["image_id"], "canonical_split": row["canonical_split"],
            "width": int(W), "height": int(H),
            "n_gland_instances": len(props), "n_valid": n_valid, "n_excluded": n_excluded,
            "n_fragmented_instances": n_fragmented,
            "valid_pixel_fraction": float(field["valid"].mean()),
        })

    gland_df = pd.DataFrame(gland_rows)
    gland_manifest_path = os.path.join(OUT_DIR, "stage4_gland_manifest.csv")
    gland_df.to_csv(gland_manifest_path, index=False)
    print(f"[stage4] gland-level manifest written: {gland_manifest_path} ({len(gland_df)} rows)")

    image_df = pd.DataFrame(per_image_summary)
    image_summary_path = os.path.join(OUT_DIR, "stage4_per_image_summary.csv")
    image_df.to_csv(image_summary_path, index=False)
    print(f"[stage4] per-image summary written: {image_summary_path}")

    # ---- aggregate statistics ----
    def split_stats(split_name):
        sub = gland_df if split_name == "ALL" else gland_df[gland_df["canonical_split"] == split_name]
        img_sub = image_df if split_name == "ALL" else image_df[image_df["canonical_split"] == split_name]
        valid_sub = sub[sub["validity_category"] == "valid"]
        return {
            "n_images": int(len(img_sub)),
            "n_gland_instances_total": int(len(sub)),
            "n_glands_per_image": {"min": int(img_sub["n_gland_instances"].min()),
                                     "max": int(img_sub["n_gland_instances"].max()),
                                     "mean": float(img_sub["n_gland_instances"].mean())},
            "validity_category_counts": sub["validity_category"].value_counts().to_dict(),
            "n_fragmented_instances": int(sub["fragmented"].sum()),
            "n_numerically_unstable": int(sub["numerically_unstable"].sum()),
            "S_cov_distribution_valid_only": {
                "mean": float(valid_sub["S_cov"].mean()), "median": float(valid_sub["S_cov"].median()),
                "std": float(valid_sub["S_cov"].std()), "min": float(valid_sub["S_cov"].min()),
                "max": float(valid_sub["S_cov"].max()),
            } if len(valid_sub) > 0 else None,
            "phi_cov_distribution_valid_only_deg": {
                "mean": float(np.degrees(valid_sub["phi_cov"]).mean()),
                "std": float(np.degrees(valid_sub["phi_cov"]).std()),
            } if len(valid_sub) > 0 else None,
            "lambda1_stats": {"mean": float(sub["lambda1"].dropna().mean()), "max": float(sub["lambda1"].dropna().max())},
            "lambda2_stats": {"mean": float(sub["lambda2"].dropna().mean()), "min": float(sub["lambda2"].dropna().min())},
        }

    # targeted reporting for the 4 previously-identified fragmented testB images
    fragmented_testb_detail = []
    for img_id in sorted(FRAGMENTED_TESTB_IDS):
        img_rows = gland_df[gland_df["image_id"] == img_id]
        fragmented_testb_detail.append({
            "image_id": img_id,
            "n_instances": len(img_rows),
            "instances": img_rows[["gland_id", "n_pixels", "n_connected_components", "fragmented",
                                    "validity_category", "S_cov", "phi_cov"]].to_dict("records"),
        })

    summary = {
        "coordinate_convention": "x = column, y = row (standard math, ccw-positive), matching docs_v2/phase2_target_math.md section 1; NOT the row/col-swapped skimage.feature.structure_tensor convention (irrelevant here -- no structure tensor is used in target generation)",
        "angle_convention": "phi = atan2(v1_y, v1_x) mod pi, where v1 is the covariance matrix's dominant eigenvector; head-tail symmetry (phi === phi+pi) falls out of the eigenvector sign ambiguity for free (verified in tests_v2/test_anatomy_targets.py #9)",
        "Q_construction_convention": "Q = [[q1,q2],[q2,-q1]], (q1,q2) = S*(cos 2phi, sin 2phi); equals 2x the literal S*(uu^T - 0.5I) form specified in the master prompt -- pure normalization choice, documented in docs_v2/phase2_target_math.md section 2",
        "S_definition_used": "S_cov = (lambda1-lambda2)/(lambda1+lambda2+eps), the covariance-eigenvalue anisotropy selected in docs_v2/phase2_target_spec.md; S_contour retained and recorded alongside for every instance, not used for the primary target",
        "thresholds_reused_unchanged": {"MIN_GLAND_PIXELS": MIN_GLAND_PIXELS, "BORDER_FRAC_THRESHOLD": BORDER_FRAC_THRESHOLD, "DEFAULT_S_THRESHOLD": DEFAULT_S_THRESHOLD},
        "native_geometry_preserved": True,
        "no_resizing_performed": True,
        "no_H&E_image_pixels_used_for_target_generation": True,
        "all": split_stats("ALL"),
        "train": split_stats("train"),
        "testA": split_stats("testA"),
        "testB": split_stats("testB"),
        "fragmented_testB_images_detail": fragmented_testb_detail,
        "files_generated": {
            "gland_level_manifest": "results_v2/anatomy_targets/stage4_gland_manifest.csv",
            "per_image_summary": "results_v2/anatomy_targets/stage4_per_image_summary.csv",
            "dense_fields_per_image": "results_v2/anatomy_targets/stage4_fields/<image_id>.npz (q1,q2,S,phi,valid arrays at native resolution)",
        },
    }

    summary_path = os.path.join(OUT_DIR, "stage4_target_generation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[stage4] summary written: {summary_path}")
    print(json.dumps(summary["all"], indent=2, default=str))

    return summary


if __name__ == "__main__":
    main()
