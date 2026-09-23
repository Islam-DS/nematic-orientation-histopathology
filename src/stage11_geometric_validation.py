"""
STAGE 11: primary geometric validation. Loads the FROZEN Stage 10 Model 3
checkpoint, evaluates on testA and testB (held out until now), computes
the preregistered geometric metrics exactly as defined in
results_v2/phase2/preregistered_thresholds_v2.json (read-only), and
applies the frozen decision rule for criteria A/B/C (D is reserved for
Stage 12). No threshold, target, or split is modified. No cherry-picking:
every testB image (including the 4 previously-flagged fragmented ones) is
included in the primary result; a diagnostic-only breakdown with/without
them is additionally reported for transparency, never substituted as the
primary number.

Unit of analysis: PATCH-level (S, phi) via masked mean over each patch's
valid, resolution-matched pixels -- the same convention used throughout
the calibration work underlying the frozen thresholds (Stage 0
recalibration) and Stage 9's baseline evaluation.
"""
import sys
import os
import json

import numpy as np
from scipy import stats
import torch

sys.path.insert(0, os.path.dirname(__file__))
from model3_typed_equivariant import Model3TypedEquivariant
from model3_loss import downsample_target_for_loss
from nematic_math import angular_error_mod_pi
from stage9_train_baselines import to_tensors, per_patch_scalar_qs

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "results_v2", "model3", "checkpoints", "checkpoint_final.pt")
THRESHOLDS_PATH = os.path.join(PROJECT_ROOT, "results_v2", "phase2", "preregistered_thresholds_v2.json")
OUT_DIR = os.path.join(PROJECT_ROOT, "results_v2", "validation", "stage11")
os.makedirs(OUT_DIR, exist_ok=True)

FRAGMENTED_TESTB_IDS = {"testB_3", "testB_11", "testB_18", "testB_19"}
N_PERM = 1000
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 42


@torch.no_grad()
def get_predictions(model, arrs, device, batch_size=64):
    n = len(arrs["label"])
    all_q1_pred, all_q2_pred, all_q1_t, all_q2_t, all_valid = [], [], [], [], []
    for i in range(0, n, batch_size):
        idx = np.arange(i, min(i + batch_size, n))
        x, labels, q1, q2, valid = to_tensors(arrs, device, idx=idx)
        logits, q_dense = model(x)
        factor = q1.shape[-1] // q_dense.shape[-1]
        q1_ds, q2_ds, valid_ds = downsample_target_for_loss(q1, q2, valid, factor=factor)
        all_q1_pred.append(q_dense[:, 0].cpu()); all_q2_pred.append(q_dense[:, 1].cpu())
        all_q1_t.append(q1_ds.cpu()); all_q2_t.append(q2_ds.cpu()); all_valid.append(valid_ds.cpu())
    return (torch.cat(all_q1_pred), torch.cat(all_q2_pred), torch.cat(all_q1_t), torch.cat(all_q2_t), torch.cat(all_valid))


def bootstrap_ci_pearson_r(x, y, n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED, alpha=0.05):
    rng = np.random.default_rng(seed)
    n = len(x)
    boot_r = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boot_r[i] = np.corrcoef(x[idx], y[idx])[0, 1]
    lo, hi = np.percentile(boot_r, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def bootstrap_ci_mean(x, n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED, alpha=0.05):
    rng = np.random.default_rng(seed)
    n = len(x)
    boot_mean = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boot_mean[i] = x[idx].mean()
    lo, hi = np.percentile(boot_mean, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def compute_full_metrics(S_pred, phi_pred, S_t, phi_t, seed=0):
    """Full metric set matching preregistered_thresholds_v2.json's A/B/C definitions exactly."""
    n = len(S_pred)
    rng = np.random.default_rng(seed)

    ang_err = np.degrees(angular_error_mod_pi(phi_pred, phi_t))
    ang_ci_lo, ang_ci_hi = bootstrap_ci_mean(ang_err)

    diff = np.abs(phi_pred[:, None] - phi_t[None, :]) % np.pi
    diff = np.minimum(diff, np.pi - diff)
    mask = ~np.eye(n, dtype=bool)
    null_ang_individual = np.degrees(diff[mask])
    glass_delta_ang = (null_ang_individual.mean() - ang_err.mean()) / (null_ang_individual.std() + 1e-8)

    null_mean_ang = np.empty(N_PERM)
    for k in range(N_PERM):
        idx = rng.permutation(n)
        d = np.abs(phi_pred - phi_t[idx]) % np.pi
        d = np.minimum(d, np.pi - d)
        null_mean_ang[k] = np.degrees(d).mean()
    p_perm_ang = float((null_mean_ang <= ang_err.mean()).mean())

    r_S, p_S = stats.pearsonr(S_pred, S_t)
    rho_S, p_rho = stats.spearmanr(S_pred, S_t)
    r_ci_lo, r_ci_hi = bootstrap_ci_pearson_r(S_pred, S_t)

    q1p, q2p = S_pred * np.cos(2 * phi_pred), S_pred * np.sin(2 * phi_pred)
    q1t, q2t = S_t * np.cos(2 * phi_t), S_t * np.sin(2 * phi_t)
    D_Q = np.hypot(q1p - q1t, q2p - q2t)
    DQ_cross = np.hypot(q1p[:, None] - q1t[None, :], q2p[:, None] - q2t[None, :])
    null_DQ_individual = DQ_cross[mask]
    glass_delta_DQ = (null_DQ_individual.mean() - D_Q.mean()) / (null_DQ_individual.std() + 1e-8)

    null_mean_DQ = np.empty(N_PERM)
    for k in range(N_PERM):
        idx = rng.permutation(n)
        null_mean_DQ[k] = np.hypot(q1p - q1t[idx], q2p - q2t[idx]).mean()
    p_perm_DQ = float((null_mean_DQ <= D_Q.mean()).mean())
    z_DQ = (null_mean_DQ.mean() - D_Q.mean()) / (null_mean_DQ.std() + 1e-8)

    return {
        "n_patches": int(n),
        "A_angular_correspondence": {
            "mean_deg": float(ang_err.mean()), "median_deg": float(np.median(ang_err)),
            "std_deg": float(ang_err.std()), "IQR_deg": float(np.percentile(ang_err, 75) - np.percentile(ang_err, 25)),
            "bootstrap_95CI_mean": [ang_ci_lo, ang_ci_hi],
            "glass_delta": float(glass_delta_ang),
            "permutation_p_value": p_perm_ang,
            "individual_null_mean_deg": float(null_ang_individual.mean()),
            "individual_null_std_deg": float(null_ang_individual.std()),
            "PASSES_PRIMARY_glass_delta_ge_0.5": bool(glass_delta_ang >= 0.5),
            "PASSES_SECONDARY_perm_p_lt_0.01": bool(p_perm_ang < 0.01),
            "PASSES_SECONDARY_abs_backstop_lt_40deg": bool(ang_err.mean() < 40.0),
        },
        "B_order_magnitude_correlation": {
            "pearson_r": float(r_S), "pearson_p": float(p_S),
            "spearman_rho": float(rho_S), "spearman_p": float(p_rho),
            "bootstrap_95CI": [r_ci_lo, r_ci_hi],
            "PASSES_r_ge_0.30": bool(r_S >= 0.30),
            "PASSES_p_lt_0.01": bool(p_S < 0.01),
            "PASSES_CI_lower_gt_0": bool(r_ci_lo > 0.0),
            "PASSES_PRIMARY_ALL_THREE": bool(r_S >= 0.30 and p_S < 0.01 and r_ci_lo > 0.0),
        },
        "C_tensor_similarity": {
            "mean_D_Q": float(D_Q.mean()), "median_D_Q": float(np.median(D_Q)),
            "individual_null_mean": float(null_DQ_individual.mean()), "individual_null_std": float(null_DQ_individual.std()),
            "glass_delta": float(glass_delta_DQ),
            "permutation_p_value": p_perm_DQ,
            "null_of_mean_mean": float(null_mean_DQ.mean()), "null_of_mean_std": float(null_mean_DQ.std()),
            "z_score_context_only_nongating": float(z_DQ),
            "PASSES_PRIMARY_perm_p_lt_0.01": bool(p_perm_DQ < 0.01),
            "PASSES_PRIMARY_glass_delta_ge_0.5": bool(glass_delta_DQ >= 0.5),
            "PASSES_PRIMARY_BOTH": bool(p_perm_DQ < 0.01 and glass_delta_DQ >= 0.5),
        },
    }



def load_split_testAB(split_name):
    d = np.load(os.path.join(PROJECT_ROOT, "results_v2", "baselines", "_patch_cache", f"{split_name}.npz"), allow_pickle=True)
    return {k: d[k] for k in d.files}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[stage11] device = {device}")

    with open(THRESHOLDS_PATH) as f:
        thresholds = json.load(f)
    print(f"[stage11] loaded FROZEN thresholds from {THRESHOLDS_PATH} (read-only)")

    model = Model3TypedEquivariant(n_classes=2, c1=8, c2=8).to(device)
    model.eval()
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    print(f"[stage11] loaded frozen Stage 10 checkpoint: {CHECKPOINT_PATH}")
    print(f"[stage11] verified: q_out_type.size={model.q_out_type.size}, group.order()={model.group.order()}")

    results = {}
    per_split_arrays = {}
    for split_name in ["testA", "testB"]:
        arrs = load_split_testAB(split_name)
        assert all(str(i).startswith(split_name + "_") for i in arrs["image_id"]), \
            f"provenance check failed for {split_name}"
        n_images = len(set(arrs["image_id"]))
        print(f"[stage11] {split_name}: {n_images} images, {len(arrs['label'])} patches (provenance verified)")

        q1_pred, q2_pred, q1_t, q2_t, valid = get_predictions(model, arrs, device)
        S_pred, phi_pred, S_t, phi_t, has_valid = per_patch_scalar_qs(q1_pred, q2_pred, q1_t, q2_t, valid)
        per_split_arrays[split_name] = {
            "S_pred": S_pred, "phi_pred": phi_pred, "S_t": S_t, "phi_t": phi_t,
            "has_valid": has_valid, "image_id": arrs["image_id"],
        }

        valid_S_pred, valid_phi_pred = S_pred[has_valid], phi_pred[has_valid]
        valid_S_t, valid_phi_t = S_t[has_valid], phi_t[has_valid]
        n_valid = int(has_valid.sum())
        n_total = len(has_valid)

        metrics = compute_full_metrics(valid_S_pred, valid_phi_pred, valid_S_t, valid_phi_t, seed=0 if split_name == "testA" else 1)
        results[split_name] = {
            "n_images": n_images, "n_patches_total": n_total, "n_patches_with_valid_target": n_valid,
            "n_patches_invalid_excluded_from_metrics": n_total - n_valid,
            "metrics": metrics,
        }
        print(f"[stage11] {split_name}: n_valid={n_valid}/{n_total}")
        print(json.dumps(metrics, indent=2))

    # ---- diagnostic-only: fragmented testB subgroup breakdown (NOT the primary result) ----
    print("[stage11] diagnostic-only: fragmented testB image breakdown (not primary, not excluded from above)...")
    testB_data = per_split_arrays["testB"]
    is_fragmented = np.array([iid in FRAGMENTED_TESTB_IDS for iid in testB_data["image_id"]])
    hv = testB_data["has_valid"]

    frag_mask = is_fragmented & hv
    nonfrag_mask = (~is_fragmented) & hv
    diagnostic = {}
    for name, m in [("fragmented_testB_only", frag_mask), ("testB_excluding_fragmented", nonfrag_mask)]:
        n = int(m.sum())
        if n >= 5:
            diagnostic[name] = {"n_patches": n, "metrics": compute_full_metrics(
                testB_data["S_pred"][m], testB_data["phi_pred"][m], testB_data["S_t"][m], testB_data["phi_t"][m], seed=2)}
        else:
            diagnostic[name] = {"n_patches": n, "note": "too few valid patches for full metrics"}
    print(json.dumps(diagnostic, indent=2))

    # ---- pooled testA+testB, exactly as the frozen decision_rule specifies ----
    print("[stage11] pooled testA+testB (80 images) -- per the frozen decision_rule's specified evaluation set...")
    pooled_S_pred = np.concatenate([per_split_arrays[s]["S_pred"][per_split_arrays[s]["has_valid"]] for s in ["testA", "testB"]])
    pooled_phi_pred = np.concatenate([per_split_arrays[s]["phi_pred"][per_split_arrays[s]["has_valid"]] for s in ["testA", "testB"]])
    pooled_S_t = np.concatenate([per_split_arrays[s]["S_t"][per_split_arrays[s]["has_valid"]] for s in ["testA", "testB"]])
    pooled_phi_t = np.concatenate([per_split_arrays[s]["phi_t"][per_split_arrays[s]["has_valid"]] for s in ["testA", "testB"]])
    pooled_metrics = compute_full_metrics(pooled_S_pred, pooled_phi_pred, pooled_S_t, pooled_phi_t, seed=3)
    print(json.dumps(pooled_metrics, indent=2))

    crit = thresholds["criteria"]
    pass_A = pooled_metrics["A_angular_correspondence"]["PASSES_PRIMARY_glass_delta_ge_0.5"]
    pass_B = pooled_metrics["B_order_magnitude_correlation"]["PASSES_PRIMARY_ALL_THREE"]
    pass_C = pooled_metrics["C_tensor_similarity"]["PASSES_PRIMARY_BOTH"]

    final = {
        "checkpoint": CHECKPOINT_PATH,
        "thresholds_file": THRESHOLDS_PATH,
        "testA": results["testA"],
        "testB": results["testB"],
        "diagnostic_fragmented_testB_breakdown": diagnostic,
        "pooled_testA_testB_80_images": {
            "n_patches_valid": len(pooled_S_pred),
            "metrics": pooled_metrics,
        },
        "preregistered_criteria_evaluation": {
            "A_angular_correspondence_PASS": pass_A,
            "B_order_magnitude_correspondence_PASS": pass_B,
            "C_tensor_similarity_PASS": pass_C,
            "D_rotation_consistency": "RESERVED FOR STAGE 12 -- not evaluated this stage",
            "note": "Full GO per the frozen decision_rule requires A, B, C, AND D to all pass; D is not evaluated in Stage 11, so no full GO/NO-GO determination is made here.",
        },
    }
    with open(os.path.join(OUT_DIR, "stage11_results.json"), "w") as f:
        json.dump(final, f, indent=2)

    print("\n" + "=" * 60)
    print("STAGE 11 PREREGISTERED CRITERIA (A/B/C, pooled testA+testB) -- D reserved for Stage 12")
    print("=" * 60)
    print(f"  A (angular, Glass's delta >= 0.5): {pass_A}")
    print(f"  B (order magnitude, r>=0.30 AND p<0.01 AND CI>0): {pass_B}")
    print(f"  C (tensor similarity, perm p<0.01 AND Glass's delta>=0.5): {pass_C}")
    print("=" * 60)

    return final


if __name__ == "__main__":
    main()
