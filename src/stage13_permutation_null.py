"""
STAGE 13: dedicated, reproducible Model 2 channel-to-angle permutation-null
analysis. Reuses the EXACT statistic, permutation mechanism, permutation
count (100), and seed (0) already established in
code_v2/stage9_train_baselines.py::model2_permutation_null -- this script
does not redefine the statistic or the null procedure, it re-runs it
against the frozen Stage 9 Model 2 checkpoint as a standalone, fully
reproducible, per-permutation-logged analysis, separately for testA and
testB.

SCOPE: Model 2 (code/model.py::P4MOrientationNet) ONLY. Model 3 is never
loaded, evaluated, or modified in this script.

See configs/stage13_permutation_null_config.json for the documented
resolution of the one underspecified detail (the empirical p-value, which
Stage 9's function computed a z-score for but never a p-value).
"""
import sys
import os
import json
import csv

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
from model import P4MOrientationNet

sys.path.insert(0, os.path.dirname(__file__))
from model2_posthoc_q import posthoc_q_dense, permute_channel_angle_mapping
from stage9_train_baselines import load_split, to_tensors

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "results_v2", "baselines", "model2", "checkpoint_final.pt")
THRESHOLDS_PATH = os.path.join(PROJECT_ROOT, "results_v2", "phase2", "preregistered_thresholds_v2.json")
CACHE_DIR = os.path.join(PROJECT_ROOT, "results_v2", "baselines", "_patch_cache")
OUT_DIR = os.path.join(PROJECT_ROOT, "results_v2", "validation", "stage13")
os.makedirs(OUT_DIR, exist_ok=True)

N_PERMUTATIONS = 100  # frozen, identical to code_v2/stage9_train_baselines.py::N_PERMUTATIONS
SEED = 0  # frozen, identical to model2_permutation_null's default seed


def load_model2(device):
    model = P4MOrientationNet(n_classes=2, c1=8, c2=8).to(device)
    model.eval()
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    return model


@torch.no_grad()
def get_all_features(model, arrs, device, batch_size=64):
    n = len(arrs["label"])
    all_feats = []
    for i in range(0, n, batch_size):
        idx = np.arange(i, min(i + batch_size, n))
        x, _, _, _, _ = to_tensors(arrs, device, idx=idx)
        _, feats = model(x, return_features=True)
        all_feats.append(feats.cpu())
    return torch.cat(all_feats)  # (N, c2, 8, H2, W2)


def run_permutation_null(model, arrs, device, split_name, n_perm=N_PERMUTATIONS, seed=SEED):
    thetas = model.orientation_angles()
    feats = get_all_features(model, arrs, device)

    obs_q = posthoc_q_dense(feats, thetas)
    obs_S = torch.hypot(obs_q[:, 0], obs_q[:, 1]).mean().item()

    rng = np.random.default_rng(seed)
    null_S = np.empty(n_perm)
    perm_records = []
    for k in range(n_perm):
        perm_idx = rng.permutation(8)
        perm_feats = permute_channel_angle_mapping(feats, perm_idx)
        q_perm = posthoc_q_dense(perm_feats, thetas)
        null_S[k] = torch.hypot(q_perm[:, 0], q_perm[:, 1]).mean().item()
        perm_records.append({"permutation_index": k, "perm_indices": ",".join(map(str, perm_idx.tolist())),
                              "null_S": float(null_S[k])})

    null_mean = float(null_S.mean())
    null_std = float(null_S.std())
    z = (obs_S - null_mean) / (null_std + 1e-8)
    p_one_sided_le = float((null_S <= obs_S).mean())  # documented resolution, see config
    p_two_sided = float((np.abs(null_S - null_mean) >= abs(obs_S - null_mean)).mean())

    result = {
        "split": split_name,
        "n_patches": int(len(arrs["label"])),
        "n_images": int(len(set(arrs["image_id"]))),
        "observed_statistic_mean_S": obs_S,
        "null_mean_S": null_mean,
        "null_std_S": null_std,
        "z_score": float(z),
        "empirical_p_value_one_sided_null_le_observed": p_one_sided_le,
        "empirical_p_value_two_sided_secondary_diagnostic": p_two_sided,
        "n_permutations": n_perm,
        "seed": seed,
    }
    return result, perm_records, null_S


def write_permutation_csv(path, perm_records):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["permutation_index", "perm_indices", "null_S"])
        w.writeheader()
        for r in perm_records:
            w.writerow(r)


def make_diagnostic_plot(split_name, obs_S, null_S, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(null_S, bins=20, color="#7a9cc6", edgecolor="black", alpha=0.85, label="permutation null (n=100)")
    ax.axvline(obs_S, color="crimson", linewidth=2, label=f"observed statistic ({obs_S:.4f})")
    ax.axvline(null_S.mean(), color="black", linestyle="--", linewidth=1.5, label=f"null mean ({null_S.mean():.4f})")
    z = (obs_S - null_S.mean()) / (null_S.std() + 1e-8)
    ax.set_title(f"Stage 13: Model 2 channel-permutation null -- {split_name} (z={z:.3f})")
    ax.set_xlabel("mean S (order magnitude) over all patches/pixels")
    ax.set_ylabel("count (of 100 permutations)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[stage13] device = {device}")
    print(f"[stage13] Model 2 checkpoint: {CHECKPOINT_PATH}")

    model = load_model2(device)
    print("[stage13] loaded frozen Stage 9 Model 2 checkpoint (no retraining, eval() mode)")

    thetas = model.orientation_angles()
    print(f"[stage13] model.orientation_angles() (radians): {thetas}")

    all_results = {}
    for split_name in ["testA", "testB"]:
        print(f"[stage13] loading held-out split: {split_name}")
        arrs = load_split(split_name)
        assert not np.any([not (iid.startswith("testA_") or iid.startswith("testB_")) for iid in arrs["image_id"]]), \
            "held-out set contains a non-test image_id -- contamination check failed"

        result, perm_records, null_S = run_permutation_null(model, arrs, device, split_name)
        print(f"[stage13] {split_name}: {json.dumps(result, indent=2)}")

        write_permutation_csv(os.path.join(OUT_DIR, f"{split_name}_permutation_results.csv"), perm_records)
        make_diagnostic_plot(split_name, result["observed_statistic_mean_S"], null_S,
                              os.path.join(OUT_DIR, f"{split_name}_null_diagnostic.png"))
        all_results[split_name] = result

    stage9_reference = {
        "testA_z": -1.7105639019182926,
        "testB_z": -1.7044436835891512,
        "source": "results_v2/baselines/model2/held_out_{testA,testB}_results.json (Stage 9 embedded call)",
    }
    replication_check = {
        "testA_stage13_z": all_results["testA"]["z_score"],
        "testA_stage9_z": stage9_reference["testA_z"],
        "testA_abs_diff": abs(all_results["testA"]["z_score"] - stage9_reference["testA_z"]),
        "testB_stage13_z": all_results["testB"]["z_score"],
        "testB_stage9_z": stage9_reference["testB_z"],
        "testB_abs_diff": abs(all_results["testB"]["z_score"] - stage9_reference["testB_z"]),
    }

    with open(os.path.join(OUT_DIR, "null_distribution_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "n_patches", "n_images", "observed_statistic_mean_S", "null_mean_S", "null_std_S",
                    "z_score", "p_one_sided_null_le_observed", "p_two_sided", "n_permutations", "seed"])
        for split_name in ["testA", "testB"]:
            r = all_results[split_name]
            w.writerow([r["split"], r["n_patches"], r["n_images"], r["observed_statistic_mean_S"],
                        r["null_mean_S"], r["null_std_S"], r["z_score"],
                        r["empirical_p_value_one_sided_null_le_observed"],
                        r["empirical_p_value_two_sided_secondary_diagnostic"], r["n_permutations"], r["seed"]])

    final = {
        "stage": "STAGE 13 - Model 2 Permutation-Null Analysis",
        "device": device,
        "checkpoint": CHECKPOINT_PATH,
        "n_permutations": N_PERMUTATIONS,
        "seed": SEED,
        "results": all_results,
        "stage9_replication_check": {**stage9_reference, **replication_check},
        "stage_11_relationship": "Stage 13 does NOT alter Stage 11's HOLD status.",
        "stage_12_relationship": "Stage 13 does NOT alter Stage 12's HOLD status.",
    }
    with open(os.path.join(OUT_DIR, "stage13_results.json"), "w") as f:
        json.dump(final, f, indent=2)

    import shutil
    shutil.copy(os.path.join(PROJECT_ROOT, "configs", "stage13_permutation_null_config.json"),
                os.path.join(OUT_DIR, "stage13_config_copy.json"))

    print("\n" + "=" * 60)
    print("STAGE 13 SUMMARY")
    print("=" * 60)
    print(json.dumps({"results": all_results, "stage9_replication_check": final["stage9_replication_check"]}, indent=2))
    print("Stage 11 remains HOLD. Stage 12 remains HOLD. (unaffected by this stage)")
    print("=" * 60)
    return final


if __name__ == "__main__":
    main()
