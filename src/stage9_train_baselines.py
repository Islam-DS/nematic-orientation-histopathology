"""
STAGE 9: baseline training (Plain CNN, Model 2) + held-out evaluation on
testA/testB (reported separately), geometric metrics, and Model 2's
channel-permutation null. Model 3 is explicitly NOT trained here.

TEST-SET PROTECTION: testA/testB patch arrays are loaded ONLY inside
evaluate_on_held_out_set(), which is called ONLY after training (early
stopping, checkpoint selection) is fully complete for both models. This
is enforced by the calling structure in main() (train first end-to-end,
capture the final checkpoints, THEN load testA/testB) and by explicit
assertions logged below.
"""
import sys
import os
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
from model import P4MOrientationNet

sys.path.insert(0, os.path.dirname(__file__))
from baseline_cnn_dualhead import PlainCNNDualHead
from model2_posthoc_q import posthoc_q_dense, permute_channel_angle_mapping, FILTER_INDEX
from model3_loss import downsample_target_for_loss, masked_q_loss
from nematic_math import angular_error_mod_pi

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
CACHE_DIR = os.path.join(PROJECT_ROOT, "results_v2", "baselines", "_patch_cache")
BASELINES_DIR = os.path.join(PROJECT_ROOT, "results_v2", "baselines")
THRESHOLDS_PATH = os.path.join(PROJECT_ROOT, "results_v2", "phase2", "preregistered_thresholds_v2.json")

SEED = 42
LR = 1e-3
LAMBDA_Q = 1.0
BATCH_SIZE = 32
MAX_EPOCHS = 40
PATIENCE = 6
N_PERMUTATIONS = 100  # Model 2 channel-permutation null, same order of magnitude as the original pilot


def load_split(name):
    d = np.load(os.path.join(CACHE_DIR, f"{name}.npz"), allow_pickle=True)
    return {k: d[k] for k in d.files}


def to_tensors(arrs, device, idx=None):
    if idx is not None:
        arrs = {k: v[idx] for k, v in arrs.items()}
    x = torch.from_numpy(arrs["rgb"].astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(device)
    labels = torch.from_numpy(arrs["label"]).long().to(device)
    q1 = torch.from_numpy(arrs["q1"]).to(device)
    q2 = torch.from_numpy(arrs["q2"]).to(device)
    valid = torch.from_numpy(arrs["valid"]).to(device)
    return x, labels, q1, q2, valid


def total_loss(logits, labels, q_dense, q1_t, q2_t, valid_t, lambda_Q=LAMBDA_Q):
    L_class = F.cross_entropy(logits, labels)
    factor = q1_t.shape[-1] // q_dense.shape[-1]
    q1_ds, q2_ds, valid_ds = downsample_target_for_loss(q1_t, q2_t, valid_t, factor=factor)
    L_Q = masked_q_loss(q_dense, q1_ds, q2_ds, valid_ds)
    total = L_class + lambda_Q * L_Q
    return total, L_class, L_Q


def train_model(model_name, model, train_arrs, dev_arrs, device):
    print(f"[stage9] training {model_name} (seed={SEED}, lr={LR}, batch={BATCH_SIZE}, max_epochs={MAX_EPOCHS})")
    torch.manual_seed(SEED)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    n_train = len(train_arrs["label"])
    dev_x, dev_labels, dev_q1, dev_q2, dev_valid = to_tensors(dev_arrs, device)

    best_dev_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = []

    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = np.random.default_rng(SEED + epoch).permutation(n_train)
        total_train_loss, total_class_loss, total_q_loss, n_batches = 0.0, 0.0, 0.0, 0

        for i in range(0, n_train, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            x, labels, q1, q2, valid = to_tensors(train_arrs, device, idx=idx)
            opt.zero_grad()
            if model_name == "model2":
                logits, feats = model(x, return_features=True)
                q_dense = posthoc_q_dense(feats, model.orientation_angles())
            else:
                logits, q_dense = model(x)
            loss, L_class, L_Q = total_loss(logits, labels, q_dense, q1, q2, valid)
            loss.backward()
            opt.step()
            total_train_loss += loss.item(); total_class_loss += L_class.item(); total_q_loss += L_Q.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            if model_name == "model2":
                dev_logits, dev_feats = model(dev_x, return_features=True)
                dev_q_dense = posthoc_q_dense(dev_feats, model.orientation_angles())
            else:
                dev_logits, dev_q_dense = model(dev_x)
            dev_loss, dev_L_class, dev_L_Q = total_loss(dev_logits, dev_labels, dev_q_dense, dev_q1, dev_q2, dev_valid)

        row = {"epoch": epoch, "train_loss": total_train_loss / n_batches,
               "train_L_class": total_class_loss / n_batches, "train_L_Q": total_q_loss / n_batches,
               "dev_loss": dev_loss.item(), "dev_L_class": dev_L_class.item(), "dev_L_Q": dev_L_Q.item()}
        history.append(row)

        improved = dev_loss.item() < best_dev_loss - 1e-5
        if improved:
            best_dev_loss = dev_loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        flag = " *" if improved else ""
        print(f"    epoch {epoch:3d}  train_loss={row['train_loss']:.4f}  dev_loss={row['dev_loss']:.4f}{flag}")

        if epochs_no_improve >= PATIENCE:
            print(f"    early stopping at epoch {epoch} (no dev improvement for {PATIENCE} epochs)")
            break

    model.eval()
    model.load_state_dict(best_state)
    print(f"[stage9] {model_name}: best dev_loss={best_dev_loss:.4f} after {len(history)} epochs")
    return model, history, best_dev_loss


@torch.no_grad()
def get_predictions(model_name, model, arrs, device, batch_size=64):
    """TEST-SET PROTECTION: arrs must already be loaded by the caller; this
    function only runs inference, never selects/tunes anything from the result."""
    n = len(arrs["label"])
    all_logits, all_q1_pred, all_q2_pred, all_q1_t, all_q2_t, all_valid = [], [], [], [], [], []
    for i in range(0, n, batch_size):
        idx = np.arange(i, min(i + batch_size, n))
        x, labels, q1, q2, valid = to_tensors(arrs, device, idx=idx)
        if model_name == "model2":
            logits, feats = model(x, return_features=True)
            q_dense = posthoc_q_dense(feats, model.orientation_angles())
        else:
            logits, q_dense = model(x)
        factor = q1.shape[-1] // q_dense.shape[-1]
        q1_ds, q2_ds, valid_ds = downsample_target_for_loss(q1, q2, valid, factor=factor)
        all_logits.append(logits.cpu()); all_q1_pred.append(q_dense[:, 0].cpu()); all_q2_pred.append(q_dense[:, 1].cpu())
        all_q1_t.append(q1_ds.cpu()); all_q2_t.append(q2_ds.cpu()); all_valid.append(valid_ds.cpu())
    return (torch.cat(all_logits), torch.cat(all_q1_pred), torch.cat(all_q2_pred),
            torch.cat(all_q1_t), torch.cat(all_q2_t), torch.cat(all_valid))


def per_patch_scalar_qs(q1_pred, q2_pred, q1_t, q2_t, valid):
    """Reduce each patch's dense field to one (S,phi) via the masked mean, matching
    the same patch-summarization rule established in Stage 0's calibration."""
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


def geometric_metrics(S_pred, phi_pred, S_t, phi_t, valid_mask, rng_seed=0):
    S_pred, phi_pred, S_t, phi_t = S_pred[valid_mask], phi_pred[valid_mask], S_t[valid_mask], phi_t[valid_mask]
    n = len(S_pred)
    if n < 5:
        return {"n_patches": n, "note": "too few valid patches for geometric metrics"}

    ang_err = np.degrees(angular_error_mod_pi(phi_pred, phi_t))
    r_S = float(np.corrcoef(S_pred, S_t)[0, 1]) if np.std(S_t) > 1e-8 and np.std(S_pred) > 1e-8 else float("nan")

    rng = np.random.default_rng(rng_seed)
    diff = np.abs(phi_pred[:, None] - phi_t[None, :]) % np.pi
    diff = np.minimum(diff, np.pi - diff)
    mask = ~np.eye(n, dtype=bool)
    null_ang_individual = np.degrees(diff[mask])
    glass_delta_ang = (null_ang_individual.mean() - ang_err.mean()) / (null_ang_individual.std() + 1e-8)

    q1p, q2p = S_pred * np.cos(2 * phi_pred), S_pred * np.sin(2 * phi_pred)
    q1t, q2t = S_t * np.cos(2 * phi_t), S_t * np.sin(2 * phi_t)
    D_Q = np.hypot(q1p - q1t, q2p - q2t)
    DQ_cross = np.hypot(q1p[:, None] - q1t[None, :], q2p[:, None] - q2t[None, :])
    null_DQ_individual = DQ_cross[mask]
    glass_delta_DQ = (null_DQ_individual.mean() - D_Q.mean()) / (null_DQ_individual.std() + 1e-8)

    n_perm = 1000
    null_mean_DQ = np.empty(n_perm)
    for k in range(n_perm):
        idx = rng.permutation(n)
        null_mean_DQ[k] = np.hypot(q1p - q1t[idx], q2p - q2t[idx]).mean()
    p_perm_DQ = float((null_mean_DQ <= D_Q.mean()).mean())

    return {
        "n_patches": n,
        "A_angular_correspondence": {"mean_deg": float(ang_err.mean()), "median_deg": float(np.median(ang_err)), "glass_delta": float(glass_delta_ang)},
        "B_order_magnitude_correlation": {"pearson_r": r_S},
        "C_tensor_similarity": {"mean_D_Q": float(D_Q.mean()), "glass_delta": float(glass_delta_DQ), "permutation_p_value": p_perm_DQ},
    }


def model2_permutation_null(model, arrs, device, n_perm=N_PERMUTATIONS, seed=0):
    """Channel-to-angle permutation null, Model 2 ONLY -- reuses the exact
    methodology of code/experiments.py::channel_permutation_null (original pilot)."""
    rng = np.random.default_rng(seed)
    thetas = model.orientation_angles()

    n = len(arrs["label"])
    all_feats = []
    with torch.no_grad():
        for i in range(0, n, 64):
            idx = np.arange(i, min(i + 64, n))
            x, _, _, _, _ = to_tensors(arrs, device, idx=idx)
            _, feats = model(x, return_features=True)
            all_feats.append(feats.cpu())
    feats = torch.cat(all_feats)  # (N, c2, 8, H2, W2)

    obs_q = posthoc_q_dense(feats, thetas)
    obs_S = torch.hypot(obs_q[:, 0], obs_q[:, 1]).mean().item()

    null_S = []
    for _ in range(n_perm):
        perm_idx = rng.permutation(8)
        perm_feats = permute_channel_angle_mapping(feats, perm_idx)
        q_perm = posthoc_q_dense(perm_feats, thetas)
        null_S.append(torch.hypot(q_perm[:, 0], q_perm[:, 1]).mean().item())
    null_S = np.array(null_S)
    z = (obs_S - null_S.mean()) / (null_S.std() + 1e-8)
    return {"observed_mean_S": obs_S, "null_mean_S": float(null_S.mean()), "null_std_S": float(null_S.std()), "z_score": float(z), "n_permutations": n_perm}


def evaluate_on_held_out_set(model_name, model, split_name, device):
    print(f"[stage9] loading held-out set: {split_name} (evaluation only, never used for training/selection)")
    assert split_name in ("testA", "testB")
    arrs = load_split(split_name)
    assert not np.any([not (iid.startswith("testA_") or iid.startswith("testB_")) for iid in arrs["image_id"]]), \
        "held-out set contains a non-test image_id -- contamination check failed"

    logits, q1_pred, q2_pred, q1_t, q2_t, valid = get_predictions(model_name, model, arrs, device)
    preds = logits.argmax(1).numpy()
    labels = arrs["label"]
    accuracy = float((preds == labels).mean())

    S_pred, phi_pred, S_t, phi_t, has_valid = per_patch_scalar_qs(q1_pred, q2_pred, q1_t, q2_t, valid)
    geo = geometric_metrics(S_pred, phi_pred, S_t, phi_t, has_valid)

    result = {"split": split_name, "n_patches": int(len(labels)), "n_images": int(len(set(arrs["image_id"]))),
               "classification_accuracy": accuracy, "geometric_metrics": geo}

    if model_name == "model2":
        result["permutation_null"] = model2_permutation_null(model, arrs, device)

    return result


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[stage9] device = {device}")

    train_arrs = load_split("train_inner")
    dev_arrs = load_split("dev")
    print(f"[stage9] train_inner={len(train_arrs['label'])} patches, dev={len(dev_arrs['label'])} patches "
          f"(TEST-SET PROTECTION: no testA/testB file loaded yet)")

    results = {}

    for model_name, model_ctor, out_subdir in [
        ("plain_cnn", lambda: PlainCNNDualHead(in_channels=3, c1=64, c2=64, n_classes=2), "plain_cnn"),
        ("model2", lambda: P4MOrientationNet(n_classes=2, c1=8, c2=8), "model2"),
    ]:
        print(f"\n{'='*60}\nTRAINING {model_name}\n{'='*60}")
        model, history, best_dev_loss = train_model(model_name, model_ctor(), train_arrs, dev_arrs, device)

        out_dir = os.path.join(BASELINES_DIR, out_subdir)
        os.makedirs(out_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(out_dir, "checkpoint_final.pt"))
        with open(os.path.join(out_dir, "training_history.json"), "w") as f:
            json.dump(history, f, indent=2)

        print(f"[stage9] {model_name}: TRAINING COMPLETE. Now proceeding to held-out evaluation (testA, testB separately).")
        eval_testA = evaluate_on_held_out_set(model_name, model, "testA", device)
        eval_testB = evaluate_on_held_out_set(model_name, model, "testB", device)

        with open(os.path.join(out_dir, "held_out_testA_results.json"), "w") as f:
            json.dump(eval_testA, f, indent=2)
        with open(os.path.join(out_dir, "held_out_testB_results.json"), "w") as f:
            json.dump(eval_testB, f, indent=2)

        results[model_name] = {"best_dev_loss": best_dev_loss, "n_epochs": len(history),
                                 "testA": eval_testA, "testB": eval_testB}
        print(f"[stage9] {model_name} testA: acc={eval_testA['classification_accuracy']:.4f}")
        print(f"[stage9] {model_name} testB: acc={eval_testB['classification_accuracy']:.4f}")

    with open(os.path.join(BASELINES_DIR, "stage9_all_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n[stage9] ALL RESULTS:")
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
