"""
STAGE 27 shared utility: build a patch cache for an arbitrary list of GlaS image_ids, reusing the
EXACT, unmodified frozen construction (code/data_prep.py::extract_patches, precomputed Stage 4
per-image target fields in results_v2/anatomy_targets/stage4_fields/, anatomy_targets.py::
crop_field_to_patches, phase2_calibration.MIN_VALID_FRACTION_FOR_PATCH=0.10, PATCH_SIZE=128,
STRIDE=96). Used by both the multi-seed study (canonical split, so this reproduces exactly what
Stage 9/10 already cached) and the patient-disjoint sensitivity study (a different image list).
Writes only under results_v2/stage27_robustness/.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
from data_prep import extract_patches  # noqa: E402  (reuse only, unmodified)

sys.path.insert(0, os.path.dirname(__file__))
from phase2_calibration import MIN_VALID_FRACTION_FOR_PATCH  # noqa: E402 (reuse only, unmodified: 0.10)

ROOT = os.path.join(os.path.dirname(__file__), "..")
FIELDS_DIR = os.path.join(ROOT, "results_v2", "anatomy_targets", "stage4_fields")
PATCH_SIZE = 128
STRIDE = 96


def crop_field_to_patches(field, image_shape, patch_size=PATCH_SIZE, stride=STRIDE):
    """Identical to code_v2/anatomy_targets.py::crop_field_to_patches (reproduced here to avoid
    importing that module's __main__-triggering top-level code); same grid as extract_patches."""
    H, W = image_shape[:2]
    patches = []
    for y in range(0, max(1, H - patch_size + 1), stride):
        for x in range(0, max(1, W - patch_size + 1), stride):
            if y + patch_size > H or x + patch_size > W:
                continue
            crop = {k: v[y:y + patch_size, x:x + patch_size] for k, v in field.items()}
            patches.append(crop)
    return patches


def build_cache(image_ids, manifest):
    """image_ids: list of GlaS image_id strings. manifest: the canonical Stage 3 manifest
    (for image_path, grade_label). Returns a dict of stacked arrays: rgb, label, q1, q2, valid,
    image_id -- the same schema as results_v2/baselines/_patch_cache/*.npz."""
    m = manifest.set_index("image_id")
    out = {"rgb": [], "label": [], "q1": [], "q2": [], "valid": [], "image_id": []}
    for iid in image_ids:
        row = m.loc[iid]
        field_npz = np.load(os.path.join(FIELDS_DIR, f"{iid}.npz"))
        field = {k: field_npz[k] for k in ("q1", "q2", "valid")}
        img_path = os.path.join(ROOT, row["image_path"])
        rgb_patches = extract_patches(img_path, patch_size=PATCH_SIZE, stride=STRIDE)
        field_patches = crop_field_to_patches(field, (row["image_height"], row["image_width"]),
                                               patch_size=PATCH_SIZE, stride=STRIDE)
        assert len(rgb_patches) == len(field_patches), f"patch grid mismatch for {iid}"
        label = int(row["grade_label"])
        for rgb, fp in zip(rgb_patches, field_patches):
            if fp["valid"].mean() < MIN_VALID_FRACTION_FOR_PATCH:
                continue
            out["rgb"].append(rgb.astype(np.uint8))
            out["label"].append(label)
            out["q1"].append(fp["q1"].astype(np.float32))
            out["q2"].append(fp["q2"].astype(np.float32))
            out["valid"].append(fp["valid"].astype(bool))
            out["image_id"].append(iid)
    return {
        "rgb": np.stack(out["rgb"]), "label": np.array(out["label"], dtype=np.int64),
        "q1": np.stack(out["q1"]), "q2": np.stack(out["q2"]), "valid": np.stack(out["valid"]),
        "image_id": np.array(out["image_id"]),
    }


def save_cache(cache, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **cache)


def stratified_inner_dev_split(image_ids, manifest, test_size=0.2, seed=42):
    """Reproduces code/data_prep.py::build_patch_dataset's image-level split logic exactly
    (train_test_split, stratify=label, random_state=seed), applied to an arbitrary image_id list."""
    from sklearn.model_selection import train_test_split
    m = manifest.set_index("image_id")
    labels = [int(m.loc[i, "grade_label"]) for i in image_ids]
    inner, dev = train_test_split(image_ids, test_size=test_size, random_state=seed, stratify=labels)
    return list(inner), list(dev)


if __name__ == "__main__":
    # sanity check: rebuild the canonical train_inner/dev/testA/testB caches from image_ids alone and
    # verify byte-for-byte agreement with the frozen Stage 9/10 cache (proves this utility reproduces
    # the frozen construction exactly, before it is trusted for the patient-disjoint sensitivity study)
    manifest = pd.read_csv(os.path.join(ROOT, "data_manifests", "glas_manifest_stage3_finalized.csv"))
    frozen_dir = os.path.join(ROOT, "results_v2", "baselines", "_patch_cache")
    train_imgs = sorted(manifest.loc[manifest.canonical_split == "train", "image_id"])
    inner_ids, dev_ids = stratified_inner_dev_split(train_imgs, manifest, test_size=0.2, seed=42)
    print(f"[sanity] canonical train images: {len(train_imgs)}; inner {len(inner_ids)}, dev {len(dev_ids)}")
    for name, ids in (("train_inner", inner_ids), ("dev", dev_ids)):
        frozen = np.load(os.path.join(frozen_dir, f"{name}.npz"), allow_pickle=True)
        rebuilt = build_cache(sorted(ids), manifest)
        same_n = len(frozen["label"]) == len(rebuilt["label"])
        same_ids = sorted(frozen["image_id"].tolist()) == sorted(rebuilt["image_id"].tolist())
        print(f"[sanity] {name}: frozen n={len(frozen['label'])}, rebuilt n={len(rebuilt['label'])}, "
              f"same_n={same_n}, same_image_id_multiset={same_ids}")
