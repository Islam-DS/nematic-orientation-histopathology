"""
STAGE 27 TASK 1 (and shared by TASK 2): train and evaluate one Model 3 run, parameterized by seed
and by the train/eval patch caches. Reuses, UNMODIFIED: Model3TypedEquivariant, model3_loss
(downsample_target_for_loss, masked_q_loss), the training loop and hyperparameters of
code_v2/stage10_train_model3.py (SEED-dependent only where Stage 10 itself was seed-dependent:
torch.manual_seed(seed) and the per-epoch permutation rng), the Stage 11 Criteria A-C computation
(compute_full_metrics), and the Stage 12 Criterion D computation (center-pixel rotation test against
the FROZEN thresholds in results_v2/phase2/preregistered_thresholds_v2.json, read-only, never
recalibrated). Adds AUROC/accuracy for the classification head (not computed by Stage 10/11).

Never writes into results_v2/model3/ or results_v2/validation/ (the frozen Stage 10-12 outputs).
Every output goes under the --out_dir given on the command line (results_v2/stage27_robustness/...).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from model3_typed_equivariant import Model3TypedEquivariant
from model3_loss import downsample_target_for_loss, masked_q_loss
from nematic_math import rotate_Q_analytic, angular_error_mod_pi
from stage11_geometric_validation import compute_full_metrics, get_predictions

ROOT = os.path.join(os.path.dirname(__file__), "..")
FROZEN_CACHE = os.path.join(ROOT, "results_v2", "baselines", "_patch_cache")
THRESHOLDS_PATH = os.path.join(ROOT, "results_v2", "phase2", "preregistered_thresholds_v2.json")

LR = 1e-3
LAMBDA_Q = 1.0
BATCH_SIZE = 32
MAX_EPOCHS = 40
PATIENCE = 6
ROTATION_ANGLES_DEG = [0, 15, 30, 37, 45, 60, 90, 123, 150, 173]
NONZERO_ANGLES_DEG = [15, 30, 37, 45, 60, 90, 123, 150, 173]


def to_tensors(arrs, device, idx=None):
    if idx is not None:
        arrs = {k: v[idx] for k, v in arrs.items()}
    x = torch.from_numpy(arrs["rgb"].astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(device)
    labels = torch.from_numpy(arrs["label"]).long().to(device)
    q1 = torch.from_numpy(arrs["q1"]).to(device)
    q2 = torch.from_numpy(arrs["q2"]).to(device)
    valid = torch.from_numpy(arrs["valid"]).to(device)
    return x, labels, q1, q2, valid


def per_patch_scalar_qs(q1_pred, q2_pred, q1_t, q2_t, valid):
    def reduce(q1, q2, v):
        out_S, out_phi = [], []
        for i in range(q1.shape[0]):
            if v[i].any():
                m1, m2 = q1[i][v[i]].mean().item(), q2[i][v[i]].mean().item()
            else:
                m1, m2 = 0.0, 0.0
            out_S.append(float(np.hypot(m1, m2)))
            out_phi.append(float(np.mod(0.5 * np.arctan2(m2, m1), np.pi)))
        return np.array(out_S), np.array(out_phi)

    S_pred, phi_pred = reduce(q1_pred.numpy(), q2_pred.numpy(), valid.numpy())
    S_t, phi_t = reduce(q1_t.numpy(), q2_t.numpy(), valid.numpy())
    has_any_valid = valid.numpy().any(axis=(1, 2))
    return S_pred, phi_pred, S_t, phi_t, has_any_valid


def total_loss(logits, labels, q_dense, q1_t, q2_t, valid_t, lambda_Q=LAMBDA_Q):
    L_class = F.cross_entropy(logits, labels)
    factor = q1_t.shape[-1] // q_dense.shape[-1]
    q1_ds, q2_ds, valid_ds = downsample_target_for_loss(q1_t, q2_t, valid_t, factor=factor)
    L_Q = masked_q_loss(q_dense, q1_ds, q2_ds, valid_ds)
    return L_class + lambda_Q * L_Q, L_class, L_Q


def grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return total ** 0.5


def train_one(seed, train_arrs, dev_arrs, out_dir, device):
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    train_dir = os.path.join(out_dir, "training")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)

    torch.manual_seed(seed)
    model = Model3TypedEquivariant(n_classes=2, c1=8, c2=8).to(device)
    model.eval()
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "checkpoint_initial.pt"))
    n_params = sum(p.numel() for p in model.parameters())

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n_train_patches = len(train_arrs["label"])
    n_dev_patches = len(dev_arrs["label"])

    history = []
    best_dev_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    nan_or_inf_detected = False
    t0 = time.time()

    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = np.random.default_rng(seed + epoch).permutation(n_train_patches)
        total_train_loss, total_class_loss, total_q_loss, n_batches = 0.0, 0.0, 0.0, 0
        epoch_grad_norms = []

        for i in range(0, n_train_patches, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            x, labels, q1, q2, valid = to_tensors(train_arrs, device, idx=idx)
            opt.zero_grad()
            logits, q_dense = model(x)
            loss, L_class, L_Q = total_loss(logits, labels, q_dense, q1, q2, valid)
            if not torch.isfinite(loss):
                nan_or_inf_detected = True
                break
            loss.backward()
            gn = grad_norm(model)
            if not np.isfinite(gn):
                nan_or_inf_detected = True
                break
            epoch_grad_norms.append(gn)
            opt.step()
            total_train_loss += loss.item(); total_class_loss += L_class.item(); total_q_loss += L_Q.item()
            n_batches += 1

        if nan_or_inf_detected:
            break

        model.eval()
        with torch.no_grad():
            dev_loss_sum, dev_Lc_sum, dev_LQ_sum, dev_n = 0.0, 0.0, 0.0, 0
            for j in range(0, n_dev_patches, BATCH_SIZE):
                idx = np.arange(j, min(j + BATCH_SIZE, n_dev_patches))
                dx, dlabels, dq1, dq2, dvalid = to_tensors(dev_arrs, device, idx=idx)
                dlogits, dq_dense = model(dx)
                dloss, dLc, dLQ = total_loss(dlogits, dlabels, dq_dense, dq1, dq2, dvalid)
                bs = len(idx)
                dev_loss_sum += dloss.item() * bs; dev_Lc_sum += dLc.item() * bs; dev_LQ_sum += dLQ.item() * bs
                dev_n += bs
            dev_loss = dev_loss_sum / dev_n
            dev_L_class = dev_Lc_sum / dev_n
            dev_L_Q = dev_LQ_sum / dev_n

        gpu_mem_mb = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else None
        row = {"epoch": epoch, "train_loss": total_train_loss / n_batches, "train_L_class": total_class_loss / n_batches,
               "train_L_Q": total_q_loss / n_batches, "dev_loss": dev_loss, "dev_L_class": dev_L_class,
               "dev_L_Q": dev_L_Q, "mean_grad_norm": float(np.mean(epoch_grad_norms)),
               "max_grad_norm": float(np.max(epoch_grad_norms)), "lr": LR, "gpu_mem_mb": gpu_mem_mb}
        history.append(row)

        improved = dev_loss < best_dev_loss - 1e-5
        if improved:
            best_dev_loss = dev_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        print(f"    [seed {seed}] epoch {epoch:3d}  train_loss={row['train_loss']:.4f}  dev_loss={row['dev_loss']:.4f}"
              f"{' *' if improved else ''}")
        if epochs_no_improve >= PATIENCE:
            print(f"    [seed {seed}] early stopping at epoch {epoch}")
            break

    train_duration_sec = time.time() - t0
    with open(os.path.join(train_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "checkpoint_final.pt"))

    summary = {
        "seed": seed, "n_params": n_params, "device": device, "n_epochs_run": len(history),
        "L_initial_total": history[0]["train_loss"] if history else None,
        "L_final_total": history[-1]["train_loss"] if history else None,
        "best_dev_loss": best_dev_loss, "nan_or_inf_detected": nan_or_inf_detected,
        "train_duration_sec": train_duration_sec,
    }
    with open(os.path.join(out_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return model, summary


@torch.no_grad()
def classification_metrics(model, arrs, device, batch_size=64):
    n = len(arrs["label"])
    all_logits, all_labels = [], []
    for i in range(0, n, batch_size):
        idx = np.arange(i, min(i + batch_size, n))
        x, labels, _, _, _ = to_tensors(arrs, device, idx=idx)
        logits, _ = model(x)
        all_logits.append(logits.cpu()); all_labels.append(labels.cpu())
    logits = torch.cat(all_logits); labels = torch.cat(all_labels)
    probs = torch.softmax(logits, dim=1)[:, 1].numpy()
    preds = logits.argmax(dim=1).numpy()
    y = labels.numpy()
    acc = float((preds == y).mean())
    try:
        auroc = float(roc_auc_score(y, probs)) if len(set(y.tolist())) == 2 else None
    except Exception:
        auroc = None
    return {"n_patches": int(n), "accuracy": acc, "auroc": auroc}


@torch.no_grad()
def batched_center_pixel_q(model, rgb_float_batch, device, batch_size=64):
    n = rgb_float_batch.shape[0]
    out = np.empty((n, 2), dtype=np.float32)
    for i in range(0, n, batch_size):
        chunk = rgb_float_batch[i:i + batch_size]
        x = torch.from_numpy(chunk).permute(0, 3, 1, 2).to(device)
        _, q_dense = model(x)
        cy, cx = q_dense.shape[-2] // 2, q_dense.shape[-1] // 2
        out[i:i + batch_size] = q_dense[:, :, cy, cx].cpu().numpy()
    return out


def evaluate_criteria_ABC(model, eval_splits, device):
    """eval_splits: dict of split_name -> arrs (e.g. {'testA':..,'testB':..} or {'held_out':..}).
    Returns per-split results plus the pooled result across all given splits, exactly as Stage 11."""
    results = {}
    per_split_arrays = {}
    for split_name, arrs in eval_splits.items():
        q1_pred, q2_pred, q1_t, q2_t, valid = get_predictions(model, arrs, device)
        S_pred, phi_pred, S_t, phi_t, has_valid = per_patch_scalar_qs(q1_pred, q2_pred, q1_t, q2_t, valid)
        per_split_arrays[split_name] = {"S_pred": S_pred, "phi_pred": phi_pred, "S_t": S_t, "phi_t": phi_t, "has_valid": has_valid}
        n_valid = int(has_valid.sum())
        metrics = compute_full_metrics(S_pred[has_valid], phi_pred[has_valid], S_t[has_valid], phi_t[has_valid], seed=0)
        results[split_name] = {"n_images": len(set(arrs["image_id"].tolist())), "n_patches_total": len(has_valid),
                                "n_patches_with_valid_target": n_valid, "metrics": metrics}

    pooled_S_pred = np.concatenate([per_split_arrays[s]["S_pred"][per_split_arrays[s]["has_valid"]] for s in eval_splits])
    pooled_phi_pred = np.concatenate([per_split_arrays[s]["phi_pred"][per_split_arrays[s]["has_valid"]] for s in eval_splits])
    pooled_S_t = np.concatenate([per_split_arrays[s]["S_t"][per_split_arrays[s]["has_valid"]] for s in eval_splits])
    pooled_phi_t = np.concatenate([per_split_arrays[s]["phi_t"][per_split_arrays[s]["has_valid"]] for s in eval_splits])
    pooled_metrics = compute_full_metrics(pooled_S_pred, pooled_phi_pred, pooled_S_t, pooled_phi_t, seed=3)
    results["pooled"] = {"n_patches_valid": len(pooled_S_pred), "metrics": pooled_metrics}
    return results


def evaluate_criterion_D(model, rgb, device, local_thresholds):
    rgb_float = rgb.astype(np.float32) / 255.0
    n_patches = rgb_float.shape[0]
    from skimage.transform import rotate as sk_rotate
    q0 = batched_center_pixel_q(model, rgb_float, device)
    phi0 = np.mod(0.5 * np.arctan2(q0[:, 1], q0[:, 0]), np.pi)
    per_angle = []
    for alpha_deg in ROTATION_ANGLES_DEG:
        if alpha_deg == 0:
            q_rot = q0.copy()
        else:
            rot_imgs = np.empty_like(rgb_float)
            for i in range(n_patches):
                rot_imgs[i] = sk_rotate(rgb_float[i], angle=alpha_deg, mode="reflect", order=1).astype(np.float32)
            q_rot = batched_center_pixel_q(model, rot_imgs, device)
        alpha_rad = np.deg2rad(alpha_deg)
        q_expected = np.stack(rotate_Q_analytic(q0[:, 0], q0[:, 1], alpha_rad), axis=-1)
        num = np.linalg.norm(q_rot - q_expected, axis=-1)
        den = np.linalg.norm(q_expected, axis=-1) + 1e-8
        err = num / den
        rec = {"angle_deg": alpha_deg, "model_error_mean": float(err.mean())}
        if alpha_deg != 0:
            thresh = local_thresholds[alpha_deg]
            rec["frozen_local_threshold"] = thresh
            rec["PASSES_criterion_D_this_angle"] = bool(err.mean() < thresh)
        per_angle.append(rec)
    n_passed = sum(1 for r in per_angle if r.get("PASSES_criterion_D_this_angle"))
    return {"per_angle": per_angle, "n_angles_passed": n_passed, "n_angles_total_nonzero": len(NONZERO_ANGLES_DEG),
            "PASSES_criterion_D": bool(n_passed >= 7)}


def load_cache(path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--train_cache", default=None, help="dir with train_inner.npz/dev.npz; default = frozen canonical cache")
    ap.add_argument("--eval_cache", default=None, help="dir with testA.npz/testB.npz; default = frozen canonical cache")
    ap.add_argument("--eval_single_split", default=None, help="if set, a single .npz file used as one pooled eval split (e.g. patient-disjoint held-out)")
    ap.add_argument("--skip_training", action="store_true", help="reuse an existing checkpoint_final.pt in out_dir/checkpoints instead of retraining")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_cache_dir = args.train_cache or FROZEN_CACHE
    eval_cache_dir = args.eval_cache or FROZEN_CACHE
    os.makedirs(args.out_dir, exist_ok=True)

    with open(THRESHOLDS_PATH) as f:
        local_thresholds = {int(k): float(v) for k, v in json.load(f)["criteria"]["D_rotation_consistency"]["threshold_primary"]["local_thresholds"].items()}

    ckpt_path = os.path.join(args.out_dir, "checkpoints", "checkpoint_final.pt")
    if args.skip_training and os.path.isfile(ckpt_path):
        model = Model3TypedEquivariant(n_classes=2, c1=8, c2=8).to(device)
        model.eval()
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"[stage27] reused existing checkpoint at {ckpt_path}")
    else:
        train_arrs = load_cache(os.path.join(train_cache_dir, "train_inner.npz"))
        dev_arrs = load_cache(os.path.join(train_cache_dir, "dev.npz"))
        print(f"[stage27] seed={args.seed}: train_inner={len(train_arrs['label'])}, dev={len(dev_arrs['label'])}")
        model, train_summary = train_one(args.seed, train_arrs, dev_arrs, args.out_dir, device)
        print(f"[stage27] seed={args.seed} training done: {train_summary}")

    # ---------------------------------------------------------------- evaluation
    if args.eval_single_split:
        eval_arrs = load_cache(args.eval_single_split)
        eval_splits = {"held_out": eval_arrs}
        rgb_for_rotation = eval_arrs["rgb"]
        train_class_arrs = None
    else:
        testA = load_cache(os.path.join(eval_cache_dir, "testA.npz"))
        testB = load_cache(os.path.join(eval_cache_dir, "testB.npz"))
        eval_splits = {"testA": testA, "testB": testB}
        rgb_for_rotation = np.concatenate([testA["rgb"], testB["rgb"]], axis=0)

    abc = evaluate_criteria_ABC(model, eval_splits, device)
    d = evaluate_criterion_D(model, rgb_for_rotation, device, local_thresholds)
    cls_eval = classification_metrics(model, {k: np.concatenate([eval_splits[s][k] for s in eval_splits]) for k in eval_splits[list(eval_splits)[0]]}, device)

    if not args.skip_training:
        cls_train = classification_metrics(model, train_arrs, device)
        cls_dev = classification_metrics(model, dev_arrs, device)
    else:
        cls_train = cls_dev = None

    out = {
        "seed": args.seed,
        "train_cache_dir": train_cache_dir, "eval_cache_dir": eval_cache_dir,
        "eval_single_split": args.eval_single_split,
        "criteria_A_B_C": abc,
        "criterion_D": d,
        "classification": {"eval_pooled": cls_eval, "train_inner": cls_train, "dev": cls_dev},
    }
    with open(os.path.join(args.out_dir, "evaluation_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[stage27] seed={args.seed} evaluation done -> {args.out_dir}/evaluation_results.json")
    print(json.dumps({"pooled_A_glass_delta": abc["pooled"]["metrics"]["A_angular_correspondence"]["glass_delta"],
                       "pooled_B_r": abc["pooled"]["metrics"]["B_order_magnitude_correlation"]["pearson_r"],
                       "pooled_C_glass_delta": abc["pooled"]["metrics"]["C_tensor_similarity"]["glass_delta"],
                       "criterion_D_n_passed": d["n_angles_passed"], "eval_auroc": cls_eval["auroc"],
                       "eval_accuracy": cls_eval["accuracy"]}, indent=2))
    return out


if __name__ == "__main__":
    main()
