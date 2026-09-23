"""
STAGE 27 TASK 2: build the patch cache for the patient-disjoint sensitivity split (constructed by
stage27_patient_split_analysis.py -> results_v2/stage27_robustness/patient_sensitivity/
glas_manifest_patient_disjoint.csv). Reuses stage27_build_patch_cache.py's frozen-construction utility
(verified byte-identical to the canonical Stage 9/10 cache before use, see that script's own sanity
check). Splits the "train" group (80 images) into train_inner/dev (stratified 80/20, seed 42, same
rule as the canonical split); the "held_out" group (85 images) becomes one pooled evaluation cache.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from stage27_build_patch_cache import build_cache, save_cache, stratified_inner_dev_split

ROOT = os.path.join(os.path.dirname(__file__), "..")
PS_DIR = os.path.join(ROOT, "results_v2", "stage27_robustness", "patient_sensitivity")
OUT_DIR = os.path.join(PS_DIR, "_patch_cache")


def main():
    manifest = pd.read_csv(os.path.join(PS_DIR, "glas_manifest_patient_disjoint.csv"))
    train_ids = sorted(manifest.loc[manifest.patient_disjoint_group == "train", "image_id"])
    held_ids = sorted(manifest.loc[manifest.patient_disjoint_group == "held_out", "image_id"])
    print(f"[stage27-task2] patient-disjoint groups: train={len(train_ids)} images, held_out={len(held_ids)} images")

    inner_ids, dev_ids = stratified_inner_dev_split(train_ids, manifest, test_size=0.2, seed=42)
    print(f"[stage27-task2] train split into inner={len(inner_ids)}, dev={len(dev_ids)} (stratified by grade_label, seed=42)")

    for name, ids in (("train_inner", inner_ids), ("dev", dev_ids), ("held_out", held_ids)):
        cache = build_cache(sorted(ids), manifest)
        save_cache(cache, os.path.join(OUT_DIR, f"{name}.npz"))
        print(f"[stage27-task2] {name}: {len(ids)} images -> {len(cache['label'])} patches "
              f"({int((cache['label']==0).sum())} benign, {int((cache['label']==1).sum())} malignant)")

    summary = {
        "train_images": len(train_ids), "held_out_images": len(held_ids),
        "train_inner_images": len(inner_ids), "dev_images": len(dev_ids),
        "note": "This is a DIFFERENT patient-disjoint partition of the same 165 GlaS images, not a "
                "recreation of the canonical 85/60/20 benchmark. See "
                "results_v2/stage27_robustness/patient_sensitivity/patient_split_feasibility.json "
                "for why the canonical split cannot be reproduced patient-disjointly.",
    }
    import json
    with open(os.path.join(PS_DIR, "patch_cache_build_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
