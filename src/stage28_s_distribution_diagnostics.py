"""
STAGE 28 TASK B: S_DL / S_anat distribution diagnostics. Uses the EXISTING frozen Stage 10 Model 3
checkpoint (read-only; no retraining) and the EXISTING frozen patch caches
(results_v2/baselines/_patch_cache/testA.npz, testB.npz). Computes the same patch-level (S, phi)
summary already defined and used by Criteria A-C (Section 3.9's patch-level masked-mean rule,
reproduced from code_v2/stage11_geometric_validation.py's per_patch_scalar_qs, unmodified) -- this
script does not redefine or recompute the summarization rule, only reports its distribution.

Writes only under results_v2/stage28/s_distribution/. Never touches results_v2/model3/ or
results_v2/validation/.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from model3_typed_equivariant import Model3TypedEquivariant
from stage11_geometric_validation import get_predictions, per_patch_scalar_qs

ROOT = os.path.join(os.path.dirname(__file__), "..")
CACHE_DIR = os.path.join(ROOT, "results_v2", "baselines", "_patch_cache")
CKPT = os.path.join(ROOT, "results_v2", "model3", "checkpoints", "checkpoint_final.pt")
OUT_DIR = os.path.join(ROOT, "results_v2", "stage28", "s_distribution")
os.makedirs(OUT_DIR, exist_ok=True)

PERCENTILES = [1, 5, 25, 50, 75, 95, 99]
THRESHOLDS = [0.001, 0.005, 0.01, 0.02, 0.05]


def load_cache(name):
    d = np.load(os.path.join(CACHE_DIR, f"{name}.npz"), allow_pickle=True)
    return {k: d[k] for k in d.files}


def describe(x, name):
    x = np.asarray(x, dtype=float)
    n = len(x)
    out = {
        "n": int(n),
        "mean": float(np.mean(x)), "sd": float(np.std(x, ddof=1)) if n > 1 else 0.0,
        "median": float(np.median(x)),
        "iqr": float(np.percentile(x, 75) - np.percentile(x, 25)),
        "min": float(np.min(x)), "max": float(np.max(x)),
    }
    for p in PERCENTILES:
        out[f"p{p}"] = float(np.percentile(x, p))
    for t in THRESHOLDS:
        out[f"frac_below_{t}"] = float(np.mean(x < t))
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model3TypedEquivariant(n_classes=2, c1=8, c2=8).to(device)
    model.eval()
    model.load_state_dict(torch.load(CKPT, map_location=device))
    print(f"[stage28-taskB] frozen Stage 10 checkpoint loaded, device={device}")

    testA = load_cache("testA")
    testB = load_cache("testB")

    per_split = {}
    for name, arrs in (("testA", testA), ("testB", testB)):
        q1_pred, q2_pred, q1_t, q2_t, valid = get_predictions(model, arrs, device)
        S_pred, phi_pred, S_t, phi_t, has_valid = per_patch_scalar_qs(q1_pred, q2_pred, q1_t, q2_t, valid)
        per_split[name] = {"S_pred": S_pred, "S_t": S_t, "has_valid": has_valid, "label": arrs["label"]}
        print(f"[stage28-taskB] {name}: {len(S_pred)} patches, {int(has_valid.sum())} with a valid reference")

    pooled_S_pred = np.concatenate([per_split[s]["S_pred"][per_split[s]["has_valid"]] for s in ("testA", "testB")])
    pooled_S_t = np.concatenate([per_split[s]["S_t"][per_split[s]["has_valid"]] for s in ("testA", "testB")])
    pooled_label = np.concatenate([per_split[s]["label"][per_split[s]["has_valid"]] for s in ("testA", "testB")])

    summary = {
        "checkpoint": CKPT, "device": device,
        "note": "S_DL and S_anat are the patch-level masked-mean summaries defined in Section 3.9 "
                "(reused unchanged from code_v2/stage11_geometric_validation.py::per_patch_scalar_qs); "
                "only patches with at least one valid reference cell are included (has_valid=True), "
                "matching the population Criteria A-C are evaluated on.",
        "S_DL": {
            "pooled_testA_testB": describe(pooled_S_pred, "S_DL_pooled"),
            "testA": describe(per_split["testA"]["S_pred"][per_split["testA"]["has_valid"]], "S_DL_testA"),
            "testB": describe(per_split["testB"]["S_pred"][per_split["testB"]["has_valid"]], "S_DL_testB"),
            "benign_pooled": describe(pooled_S_pred[pooled_label == 0], "S_DL_benign"),
            "malignant_pooled": describe(pooled_S_pred[pooled_label == 1], "S_DL_malignant"),
        },
        "S_anat": {
            "pooled_testA_testB": describe(pooled_S_t, "S_anat_pooled"),
            "testA": describe(per_split["testA"]["S_t"][per_split["testA"]["has_valid"]], "S_anat_testA"),
            "testB": describe(per_split["testB"]["S_t"][per_split["testB"]["has_valid"]], "S_anat_testB"),
            "benign_pooled": describe(pooled_S_t[pooled_label == 0], "S_anat_benign"),
            "malignant_pooled": describe(pooled_S_t[pooled_label == 1], "S_anat_malignant"),
        },
    }
    with open(os.path.join(OUT_DIR, "s_distribution_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # save raw arrays for the plots (pooled, with labels) -- small, patch-level scalars only, no images
    np.savez_compressed(os.path.join(OUT_DIR, "s_values_pooled.npz"),
                         S_DL=pooled_S_pred, S_anat=pooled_S_t, label=pooled_label)

    print(json.dumps(summary["S_DL"]["pooled_testA_testB"], indent=2))
    print(json.dumps(summary["S_anat"]["pooled_testA_testB"], indent=2))
    return summary


if __name__ == "__main__":
    main()
