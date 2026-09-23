"""
STAGE 14: shared held-out evaluation for A0/A1/A3/A4/A5. Reuses the EXACT
metric definitions from code_v2/stage11_geometric_validation.py
(compute_full_metrics: angular correspondence w/ Glass's delta, order-
magnitude Pearson/Spearman r w/ bootstrap CI, tensor similarity D_Q w/
permutation p) and code_v2/stage9_train_baselines.py (per_patch_scalar_qs
masked-mean reduction), applied identically to every ablation checkpoint,
plus classification metrics (accuracy, AUROC, F1, precision, recall) via
sklearn, not previously computed elsewhere in this project.

Per configs/stage14_ablation_matrix.json: results here are DIAGNOSTIC /
EXPLANATORY CONTEXT ONLY relative to the frozen preregistered thresholds.
No ablation's pass/fail constitutes a new official Criterion A/B/C
verdict. Stage 11's HOLD status is not affected by anything in this script.
"""
import sys
import os
import json

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score

sys.path.insert(0, os.path.dirname(__file__))
from model3_typed_equivariant import Model3TypedEquivariant
from baseline_cnn_dualhead import PlainCNNDualHead
from model3_loss import downsample_target_for_loss
from stage9_train_baselines import to_tensors, per_patch_scalar_qs
from stage11_geometric_validation import compute_full_metrics

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
CACHE_DIR = os.path.join(PROJECT_ROOT, "results_v2", "baselines", "_patch_cache")
STAGE14_DIR = os.path.join(PROJECT_ROOT, "results_v2", "validation", "stage14")

EXPERIMENTS = {
    "A0_reference":          dict(model="model3", checkpoint=os.path.join(PROJECT_ROOT, "results_v2", "model3", "checkpoints", "checkpoint_final.pt")),
    "A1_no_q":               dict(model="model3", checkpoint=os.path.join(STAGE14_DIR, "A1_no_q", "checkpoint_final.pt")),
    "A3_q_dominant":         dict(model="model3", checkpoint=os.path.join(STAGE14_DIR, "A3_q_dominant", "checkpoint_final.pt")),
    "A4_q_only":             dict(model="model3", checkpoint=os.path.join(STAGE14_DIR, "A4_q_only", "checkpoint_final.pt")),
    "A5_plain_cnn_control":  dict(model="plain_cnn_matched", checkpoint=os.path.join(STAGE14_DIR, "A5_plain_cnn_control", "checkpoint_final.pt")),
}


def build_model(name):
    if name == "model3":
        return Model3TypedEquivariant(n_classes=2, c1=8, c2=8)
    elif name == "plain_cnn_matched":
        return PlainCNNDualHead(in_channels=3, c1=38, c2=9, n_classes=2)
    raise ValueError(name)


def load_split_testAB(split_name):
    d = np.load(os.path.join(CACHE_DIR, f"{split_name}.npz"), allow_pickle=True)
    return {k: d[k] for k in d.files}


@torch.no_grad()
def get_predictions(model, arrs, device, batch_size=64):
    n = len(arrs["label"])
    all_logits, all_q1_pred, all_q2_pred, all_q1_t, all_q2_t, all_valid = [], [], [], [], [], []
    for i in range(0, n, batch_size):
        idx = np.arange(i, min(i + batch_size, n))
        x, labels, q1, q2, valid = to_tensors(arrs, device, idx=idx)
        logits, q_dense = model(x)
        factor = q1.shape[-1] // q_dense.shape[-1]
        q1_ds, q2_ds, valid_ds = downsample_target_for_loss(q1, q2, valid, factor=factor)
        all_logits.append(logits.cpu()); all_q1_pred.append(q_dense[:, 0].cpu()); all_q2_pred.append(q_dense[:, 1].cpu())
        all_q1_t.append(q1_ds.cpu()); all_q2_t.append(q2_ds.cpu()); all_valid.append(valid_ds.cpu())
    return (torch.cat(all_logits), torch.cat(all_q1_pred), torch.cat(all_q2_pred),
            torch.cat(all_q1_t), torch.cat(all_q2_t), torch.cat(all_valid))


def classification_metrics(logits, labels):
    probs = torch.softmax(logits, dim=1)[:, 1].numpy()
    preds = logits.argmax(1).numpy()
    labels = np.asarray(labels)
    out = {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
    }
    try:
        out["auroc"] = float(roc_auc_score(labels, probs))
    except ValueError as e:
        out["auroc"] = None
        out["auroc_note"] = f"undefined ({e})"
    return out


def evaluate_experiment(exp_id, device):
    cfg = EXPERIMENTS[exp_id]
    print(f"\n{'='*60}\n[stage14-eval] {exp_id}\n{'='*60}")
    assert os.path.exists(cfg["checkpoint"]), f"checkpoint not found for {exp_id}: {cfg['checkpoint']}"

    model = build_model(cfg["model"]).to(device)
    model.eval()
    model.load_state_dict(torch.load(cfg["checkpoint"], map_location=device))
    print(f"[stage14-eval] {exp_id}: loaded checkpoint {cfg['checkpoint']}")

    result = {"experiment_id": exp_id, "model": cfg["model"], "checkpoint": cfg["checkpoint"]}
    per_split_arrays = {}

    for split_name in ["testA", "testB"]:
        arrs = load_split_testAB(split_name)
        assert all(str(i).startswith(split_name + "_") for i in arrs["image_id"]), \
            f"provenance check failed for {split_name}"
        n_images = len(set(arrs["image_id"]))

        logits, q1_pred, q2_pred, q1_t, q2_t, valid = get_predictions(model, arrs, device)
        S_pred, phi_pred, S_t, phi_t, has_valid = per_patch_scalar_qs(q1_pred, q2_pred, q1_t, q2_t, valid)
        per_split_arrays[split_name] = {"S_pred": S_pred, "phi_pred": phi_pred, "S_t": S_t, "phi_t": phi_t,
                                          "has_valid": has_valid, "image_id": arrs["image_id"]}

        clf = classification_metrics(logits, arrs["label"])
        n_valid = int(has_valid.sum())
        geo = compute_full_metrics(S_pred[has_valid], phi_pred[has_valid], S_t[has_valid], phi_t[has_valid],
                                    seed=0 if split_name == "testA" else 1)

        result[split_name] = {
            "n_images": n_images, "n_patches_total": len(has_valid), "n_patches_with_valid_target": n_valid,
            "classification_metrics": clf,
            "geometric_metrics": geo,
        }
        print(f"[stage14-eval] {exp_id} {split_name}: acc={clf['accuracy']:.4f} auroc={clf.get('auroc')} "
              f"ang_mean_deg={geo['A_angular_correspondence']['mean_deg']:.2f} "
              f"pearson_r={geo['B_order_magnitude_correlation']['pearson_r']:.4f}")

    pooled_S_pred = np.concatenate([per_split_arrays[s]["S_pred"][per_split_arrays[s]["has_valid"]] for s in ["testA", "testB"]])
    pooled_phi_pred = np.concatenate([per_split_arrays[s]["phi_pred"][per_split_arrays[s]["has_valid"]] for s in ["testA", "testB"]])
    pooled_S_t = np.concatenate([per_split_arrays[s]["S_t"][per_split_arrays[s]["has_valid"]] for s in ["testA", "testB"]])
    pooled_phi_t = np.concatenate([per_split_arrays[s]["phi_t"][per_split_arrays[s]["has_valid"]] for s in ["testA", "testB"]])
    pooled_metrics = compute_full_metrics(pooled_S_pred, pooled_phi_pred, pooled_S_t, pooled_phi_t, seed=3)
    result["pooled_testA_testB"] = {"n_patches": int(len(pooled_S_pred)), "geometric_metrics": pooled_metrics}
    print(f"[stage14-eval] {exp_id} pooled: ang_mean_deg={pooled_metrics['A_angular_correspondence']['mean_deg']:.2f} "
          f"glass_delta_A={pooled_metrics['A_angular_correspondence']['glass_delta']:.4f} "
          f"pearson_r={pooled_metrics['B_order_magnitude_correlation']['pearson_r']:.4f}")

    out_dir = os.path.join(STAGE14_DIR, exp_id)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "held_out_evaluation.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[stage14-eval] device = {device}")
    all_results = {}
    for exp_id in EXPERIMENTS:
        all_results[exp_id] = evaluate_experiment(exp_id, device)
    with open(os.path.join(STAGE14_DIR, "stage14_all_evaluations.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print("\n[stage14-eval] ALL EXPERIMENTS EVALUATED.")
    return all_results


if __name__ == "__main__":
    main()
