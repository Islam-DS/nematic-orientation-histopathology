"""
STAGE 28 TASK D: gland-size vs. patch-size descriptive assessment. Uses ONLY the existing, frozen
Stage 4 gland manifest (results_v2/anatomy_targets/stage4_gland_manifest.csv, read-only) -- no new
mask reading, no change to the frozen target-construction pipeline. Two simple, defensible descriptive
measures, both derivable from columns already in that manifest:

  (1) gland area relative to patch area: n_pixels / (128*128), for every VALID gland instance (the
      population Q_anat's patch-level summaries are actually built from).
  (2) an approximate major-axis extent, from the pixel-coordinate covariance eigenvalue lambda1 already
      computed in Stage 4 (for a uniformly-filled elongated blob, the major-axis full extent is
      approximately 4*sqrt(lambda1); this is a standard second-moment size estimate, not a new
      measurement of the mask), compared with the 128 px patch edge length.

Writes only into results_v2/stage28/gland_patch_size/. Does not alter the frozen manifest or targets.
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
MANIFEST = os.path.join(ROOT, "results_v2", "anatomy_targets", "stage4_gland_manifest.csv")
OUT_DIR = os.path.join(ROOT, "results_v2", "stage28", "gland_patch_size")
os.makedirs(OUT_DIR, exist_ok=True)

PATCH_SIZE = 128
PATCH_AREA = PATCH_SIZE * PATCH_SIZE


def describe(x):
    x = np.asarray(x, dtype=float)
    return {"n": int(len(x)), "mean": float(x.mean()), "median": float(np.median(x)),
            "sd": float(x.std(ddof=1)), "min": float(x.min()), "max": float(x.max()),
            "p90": float(np.percentile(x, 90)), "p95": float(np.percentile(x, 95)), "p99": float(np.percentile(x, 99))}


def main():
    g = pd.read_csv(MANIFEST)
    valid = g[g.validity_category == "valid"].copy()
    print(f"[stage28-taskD] {len(valid)} valid gland instances (of {len(g)} total)")

    valid["area_fraction_of_patch"] = valid["n_pixels"] / PATCH_AREA
    # approximate major-axis full extent for a uniformly-filled elongated blob: 4*sqrt(lambda1)
    # (second-moment size estimate; lambda1 is the larger pixel-coordinate covariance eigenvalue,
    # already computed and frozen in Stage 4 -- not recomputed here beyond this one derived column)
    valid["major_axis_extent_px_est"] = 4.0 * np.sqrt(valid["lambda1"].clip(lower=0))

    thresholds_area = [0.05, 0.10, 0.25, 0.50, 1.00]
    frac_area = {f"frac_area_gt_{t}": float((valid["area_fraction_of_patch"] > t).mean()) for t in thresholds_area}

    thresholds_extent = [96, 128, 160, 192]  # px; 128 = patch edge, 96 = patch stride
    frac_extent = {f"frac_major_axis_gt_{t}px": float((valid["major_axis_extent_px_est"] > t).mean()) for t in thresholds_extent}

    summary = {
        "patch_size_px": PATCH_SIZE, "patch_area_px": PATCH_AREA, "patch_stride_px": 96,
        "n_valid_glands": int(len(valid)),
        "area_fraction_of_patch": describe(valid["area_fraction_of_patch"]),
        "major_axis_extent_px_est": describe(valid["major_axis_extent_px_est"]),
        "fraction_of_glands_exceeding_area_threshold": frac_area,
        "fraction_of_glands_with_major_axis_exceeding_length": frac_extent,
        "methodology_note": (
            "area_fraction_of_patch = n_pixels / 16384 (patch area), for every valid gland instance "
            "(the population underlying the patch-level Q_anat summaries used by Criteria A-C). "
            "major_axis_extent_px_est = 4*sqrt(lambda1), the standard second-moment full-extent estimate "
            "for a uniformly-filled elongated 2D blob, using the ALREADY-COMPUTED, frozen lambda1 column "
            "(the larger eigenvalue of the gland's pixel-coordinate covariance matrix, Section 3.3). "
            "Neither measure re-reads a mask or changes any frozen artifact. Both are descriptive proxies, "
            "not an exact truncation count: a gland whose major-axis extent exceeds 128 px is NOT "
            "guaranteed to be truncated in every patch that contains part of it (patches can be positioned "
            "so the gland's long axis runs diagonally, or the patch may capture only part of the gland "
            "deliberately, which is exactly the scenario this diagnostic is meant to characterize -- the "
            "network at most sees a 128x128 window, while Q_anat for that gland was computed from the "
            "COMPLETE unpatched mask)."
        ),
    }
    with open(os.path.join(OUT_DIR, "gland_patch_size_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    valid[["image_id", "canonical_split", "gland_id", "n_pixels", "area_fraction_of_patch", "major_axis_extent_px_est"]].to_csv(
        os.path.join(OUT_DIR, "valid_gland_patch_size_table.csv"), index=False)

    print(json.dumps({k: v for k, v in summary.items() if k not in ("methodology_note",)}, indent=2))
    return summary


if __name__ == "__main__":
    main()
