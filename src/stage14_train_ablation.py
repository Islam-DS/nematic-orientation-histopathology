"""
STAGE 14: shared ablation training loop for A1, A3, A4, A5. Reuses Stage
10's training loop structure EXACTLY (same SEED, LR, BATCH_SIZE,
MAX_EPOCHS, PATIENCE, optimizer, dev-batching, NaN/Inf checks, checkpoint
selection rule) -- the only things that vary between experiments are the
model constructor and the (lambda_class, lambda_Q) loss weights, per the
frozen configs/stage14_ablation_matrix.json. A0 is NOT trained by this
script (it reuses the exact frozen Stage 10 checkpoint, see the matrix).

TEST-SET PROTECTION: identical to Stage 10 -- only train_inner/dev are
loaded here; testA/testB are never opened in this script.
"""
import sys
import os
import json
import time
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from model3_typed_equivariant import Model3TypedEquivariant
from baseline_cnn_dualhead import PlainCNNDualHead
from model3_loss import downsample_target_for_loss, masked_q_loss

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
CACHE_DIR = os.path.join(PROJECT_ROOT, "results_v2", "baselines", "_patch_cache")
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "data_manifests", "glas_manifest_stage3_finalized.csv")
STAGE14_DIR = os.path.join(PROJECT_ROOT, "results_v2", "validation", "stage14")

SEED = 42
LR = 1e-3
BATCH_SIZE = 32
MAX_EPOCHS = 40
PATIENCE = 6

EXPERIMENTS = {
    "A1_no_q":               dict(model="model3", lambda_class=1.0, lambda_Q=0.0),
    "A3_q_dominant":         dict(model="model3", lambda_class=1.0, lambda_Q=10.0),
    "A4_q_only":             dict(model="model3", lambda_class=0.0, lambda_Q=1.0),
    "A5_plain_cnn_control":  dict(model="plain_cnn_matched", lambda_class=1.0, lambda_Q=1.0),
}


def assert_split_protection():
    manifest = pd.read_csv(MANIFEST_PATH)
    n_train = int((manifest.canonical_split == "train").sum())
    n_testA = int((manifest.canonical_split == "testA").sum())
    n_testB = int((manifest.canonical_split == "testB").sum())
    assert n_train == 85 and n_testA == 60 and n_testB == 20
    print(f"[stage14] split protection assertion PASSED: train={n_train}, testA={n_testA}, testB={n_testB}")


def load_split(name):
    assert name in ("train_inner", "dev"), f"Stage 14 training may only load train_inner/dev, got '{name}'"
    d = np.load(os.path.join(CACHE_DIR, f"{name}.npz"), allow_pickle=True)
    arrs = {k: d[k] for k in d.files}
    assert not any(str(i).startswith(("testA_", "testB_")) for i in arrs["image_id"]), \
        f"contamination check failed: a testA/testB image_id was found in cached split '{name}'"
    return arrs


def to_tensors(arrs, device, idx=None):
    if idx is not None:
        arrs = {k: v[idx] for k, v in arrs.items()}
    x = torch.from_numpy(arrs["rgb"].astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(device)
    labels = torch.from_numpy(arrs["label"]).long().to(device)
    q1 = torch.from_numpy(arrs["q1"]).to(device)
    q2 = torch.from_numpy(arrs["q2"]).to(device)
    valid = torch.from_numpy(arrs["valid"]).to(device)
    return x, labels, q1, q2, valid


def total_loss(logits, labels, q_dense, q1_t, q2_t, valid_t, lambda_class, lambda_Q):
    L_class = F.cross_entropy(logits, labels)
    factor = q1_t.shape[-1] // q_dense.shape[-1]
    q1_ds, q2_ds, valid_ds = downsample_target_for_loss(q1_t, q2_t, valid_t, factor=factor)
    L_Q = masked_q_loss(q_dense, q1_ds, q2_ds, valid_ds)
    total = lambda_class * L_class + lambda_Q * L_Q
    return total, L_class, L_Q


def grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return total ** 0.5


def build_model(name):
    if name == "model3":
        return Model3TypedEquivariant(n_classes=2, c1=8, c2=8)
    elif name == "plain_cnn_matched":
        return PlainCNNDualHead(in_channels=3, c1=38, c2=9, n_classes=2)
    raise ValueError(name)


def run_experiment(exp_id):
    cfg = EXPERIMENTS[exp_id]
    out_dir = os.path.join(STAGE14_DIR, exp_id)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*60}\n[stage14] EXPERIMENT {exp_id}: {cfg}\n{'='*60}")

    assert_split_protection()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[stage14] device = {device}")

    train_arrs = load_split("train_inner")
    dev_arrs = load_split("dev")
    print(f"[stage14] train_inner={len(train_arrs['label'])} patches, dev={len(dev_arrs['label'])} patches "
          f"(TEST-SET PROTECTION: testA/testB never referenced in this script)")

    torch.manual_seed(SEED)
    model = build_model(cfg["model"]).to(device)
    model.eval()
    torch.save(model.state_dict(), os.path.join(out_dir, "checkpoint_initial.pt"))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[stage14] {exp_id}: model={cfg['model']}, {n_params} parameters, "
          f"lambda_class={cfg['lambda_class']}, lambda_Q={cfg['lambda_Q']}")

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
        perm = np.random.default_rng(SEED + epoch).permutation(n_train_patches)
        total_train_loss, total_class_loss, total_q_loss, n_batches = 0.0, 0.0, 0.0, 0
        epoch_grad_norms = []

        for i in range(0, n_train_patches, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            x, labels, q1, q2, valid = to_tensors(train_arrs, device, idx=idx)
            opt.zero_grad()
            logits, q_dense = model(x)
            loss, L_class, L_Q = total_loss(logits, labels, q_dense, q1, q2, valid,
                                             cfg["lambda_class"], cfg["lambda_Q"])

            if not torch.isfinite(loss):
                nan_or_inf_detected = True
                print(f"[stage14] {exp_id}: NaN/Inf loss at epoch {epoch}, batch starting {i} -- STOPPING")
                break

            loss.backward()
            gn = grad_norm(model)
            if not np.isfinite(gn):
                nan_or_inf_detected = True
                print(f"[stage14] {exp_id}: non-finite gradient norm at epoch {epoch}, batch starting {i} -- STOPPING")
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
                assert torch.isfinite(dq_dense).all(), "non-finite Q output on dev set"
                dloss, dLc, dLQ = total_loss(dlogits, dlabels, dq_dense, dq1, dq2, dvalid,
                                              cfg["lambda_class"], cfg["lambda_Q"])
                bs = len(idx)
                dev_loss_sum += dloss.item() * bs; dev_Lc_sum += dLc.item() * bs; dev_LQ_sum += dLQ.item() * bs
                dev_n += bs
            dev_loss = dev_loss_sum / dev_n
            dev_L_class = dev_Lc_sum / dev_n
            dev_L_Q = dev_LQ_sum / dev_n

        gpu_mem_mb = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else None

        row = {
            "epoch": epoch, "train_loss": total_train_loss / n_batches,
            "train_L_class": total_class_loss / n_batches, "train_L_Q": total_q_loss / n_batches,
            "dev_loss": dev_loss, "dev_L_class": dev_L_class, "dev_L_Q": dev_L_Q,
            "mean_grad_norm": float(np.mean(epoch_grad_norms)), "max_grad_norm": float(np.max(epoch_grad_norms)),
            "lr": LR, "gpu_mem_mb": gpu_mem_mb,
        }
        history.append(row)

        improved = dev_loss < best_dev_loss - 1e-5
        if improved:
            best_dev_loss = dev_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        flag = " *" if improved else ""
        print(f"    [{exp_id}] epoch {epoch:3d}  train_loss={row['train_loss']:.4f} (class={row['train_L_class']:.4f} "
              f"Q={row['train_L_Q']:.4f})  dev_loss={row['dev_loss']:.4f}{flag}  grad_norm={row['mean_grad_norm']:.4f}")

        if epochs_no_improve >= PATIENCE:
            print(f"    [{exp_id}] early stopping at epoch {epoch} (no dev improvement for {PATIENCE} epochs)")
            break

    train_duration_sec = time.time() - t0

    with open(os.path.join(out_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    torch.save(model.state_dict(), os.path.join(out_dir, "checkpoint_final.pt"))

    repro_x, _, _, _, _ = to_tensors(dev_arrs, device, idx=np.arange(8))
    model2 = build_model(cfg["model"]).to(device)
    model2.eval()
    model2.load_state_dict(torch.load(os.path.join(out_dir, "checkpoint_final.pt"), map_location=device))
    with torch.no_grad():
        logits_a, q_a = model(repro_x)
        logits_b, q_b = model2(repro_x)
    checkpoint_reproducible = torch.equal(logits_a, logits_b) and torch.equal(q_a, q_b)

    summary = {
        "experiment_id": exp_id,
        "config": cfg,
        "n_params": n_params,
        "seed": SEED, "device": device, "dtype": "float32",
        "n_epochs_run": len(history),
        "best_epoch": int(np.argmin([r["dev_loss"] for r in history])) if history else None,
        "L_initial_total": history[0]["train_loss"] if history else None,
        "L_final_total": history[-1]["train_loss"] if history else None,
        "best_dev_loss": best_dev_loss,
        "nan_or_inf_detected": nan_or_inf_detected,
        "checkpoint_reproducible": checkpoint_reproducible,
        "train_duration_sec": train_duration_sec,
        "peak_gpu_mem_mb": max((r["gpu_mem_mb"] for r in history if r["gpu_mem_mb"]), default=None),
    }
    with open(os.path.join(out_dir, "stage14_train_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[stage14] {exp_id} SUMMARY: {json.dumps(summary, indent=2)}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True, choices=list(EXPERIMENTS.keys()))
    args = parser.parse_args()
    run_experiment(args.exp)


if __name__ == "__main__":
    main()
