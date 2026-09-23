"""
STAGE 10: final Model 3 training. Uses the SAME frozen protocol applied to
the baselines in Stage 9 (configs/stage9_baseline_training_config.json,
reused as the authoritative Stage 10 configuration -- see
configs/stage10_model3_training_config.json for the documented reasoning).
Model architecture and loss are imported unmodified from Stage 7.

TEST-SET PROTECTION: this script never imports or references testA/testB
in any form. It reuses ONLY the already-cached train_inner/dev patch
arrays built in Stage 9 (results_v2/baselines/_patch_cache/), which were
themselves built exclusively from the canonical 85-image TRAIN split
(verified there, reverified here by an explicit count assertion).

No testA/testB evaluation is performed in this stage (deferred to Stage 11).
"""
import sys
import os
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from model3_typed_equivariant import Model3TypedEquivariant
from model3_loss import downsample_target_for_loss, masked_q_loss

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
CACHE_DIR = os.path.join(PROJECT_ROOT, "results_v2", "baselines", "_patch_cache")
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "data_manifests", "glas_manifest_stage3_finalized.csv")
OUT_ROOT = os.path.join(PROJECT_ROOT, "results_v2", "model3")

SEED = 42
LR = 1e-3
LAMBDA_Q = 1.0
BATCH_SIZE = 32
MAX_EPOCHS = 40
PATIENCE = 6


def assert_split_protection():
    """Final Stage 10 test-set-protection audit, per instruction section 11."""
    manifest = pd.read_csv(MANIFEST_PATH)
    n_train = int((manifest.canonical_split == "train").sum())
    n_testA = int((manifest.canonical_split == "testA").sum())
    n_testB = int((manifest.canonical_split == "testB").sum())
    assert n_train == 85, f"expected 85 canonical train images, found {n_train}"
    assert n_testA == 60, f"expected 60 canonical testA images, found {n_testA}"
    assert n_testB == 20, f"expected 20 canonical testB images, found {n_testB}"
    print(f"[stage10] split protection assertion PASSED: train={n_train}, testA={n_testA}, testB={n_testB}")
    return n_train, n_testA, n_testB


def load_split(name):
    assert name in ("train_inner", "dev"), f"Stage 10 may only load train_inner/dev, got '{name}'"
    d = np.load(os.path.join(CACHE_DIR, f"{name}.npz"), allow_pickle=True)
    arrs = {k: d[k] for k in d.files}
    # reverify every image_id in this cached split is a canonical TRAIN image (not testA/testB)
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


def total_loss(logits, labels, q_dense, q1_t, q2_t, valid_t, lambda_Q=LAMBDA_Q):
    L_class = F.cross_entropy(logits, labels)
    factor = q1_t.shape[-1] // q_dense.shape[-1]
    q1_ds, q2_ds, valid_ds = downsample_target_for_loss(q1_t, q2_t, valid_t, factor=factor)
    L_Q = masked_q_loss(q_dense, q1_ds, q2_ds, valid_ds)
    total = L_class + lambda_Q * L_Q
    return total, L_class, L_Q


def grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return total ** 0.5


def main():
    print("[stage10] === FINAL MODEL 3 TRAINING ===")
    n_train, n_testA, n_testB = assert_split_protection()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[stage10] device = {device}")

    train_arrs = load_split("train_inner")
    dev_arrs = load_split("dev")
    print(f"[stage10] train_inner={len(train_arrs['label'])} patches, dev={len(dev_arrs['label'])} patches "
          f"(reused from Stage 9's cache -- TEST-SET PROTECTION: testA/testB never referenced in this script)")

    torch.manual_seed(SEED)
    model = Model3TypedEquivariant(n_classes=2, c1=8, c2=8).to(device)
    model.eval()
    torch.save(model.state_dict(), os.path.join(OUT_ROOT, "checkpoints", "checkpoint_initial.pt"))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[stage10] model constructed: {n_params} parameters, D8.irrep(1,2) output type = {model.q_out_type}")

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
            loss, L_class, L_Q = total_loss(logits, labels, q_dense, q1, q2, valid)

            if not torch.isfinite(loss):
                nan_or_inf_detected = True
                print(f"[stage10] NaN/Inf loss at epoch {epoch}, batch starting {i} -- STOPPING")
                break

            loss.backward()
            gn = grad_norm(model)
            if not np.isfinite(gn):
                nan_or_inf_detected = True
                print(f"[stage10] non-finite gradient norm at epoch {epoch}, batch starting {i} -- STOPPING")
                break
            epoch_grad_norms.append(gn)
            opt.step()

            total_train_loss += loss.item(); total_class_loss += L_class.item(); total_q_loss += L_Q.item()
            n_batches += 1

        if nan_or_inf_detected:
            break

        # LEVEL 1 FIX (documented in reports/STAGE10_AUDIT_REPORT.md): the dev set
        # (433 patches) must be evaluated in mini-batches like training, not as one
        # unbatched forward pass -- an initial unbatched attempt caused a CUDA OOM
        # (Model3's Q-head forward pass over 433 patches at once exceeds the 6GB
        # budget). This does not change the loss definition, only how it is computed
        # in chunks and aggregated (patch-count-weighted mean, exactly equivalent to
        # the unbatched computation mathematically).
        model.eval()
        with torch.no_grad():
            dev_loss_sum, dev_Lc_sum, dev_LQ_sum, dev_n = 0.0, 0.0, 0.0, 0
            for j in range(0, n_dev_patches, BATCH_SIZE):
                idx = np.arange(j, min(j + BATCH_SIZE, n_dev_patches))
                dx, dlabels, dq1, dq2, dvalid = to_tensors(dev_arrs, device, idx=idx)
                dlogits, dq_dense = model(dx)
                assert torch.isfinite(dq_dense).all(), "non-finite Q output on dev set"
                dloss, dLc, dLQ = total_loss(dlogits, dlabels, dq_dense, dq1, dq2, dvalid)
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
        print(f"    epoch {epoch:3d}  train_loss={row['train_loss']:.4f} (class={row['train_L_class']:.4f} "
              f"Q={row['train_L_Q']:.4f})  dev_loss={row['dev_loss']:.4f}{flag}  grad_norm={row['mean_grad_norm']:.4f}")

        if epochs_no_improve >= PATIENCE:
            print(f"    early stopping at epoch {epoch} (no dev improvement for {PATIENCE} epochs)")
            break

    train_duration_sec = time.time() - t0

    with open(os.path.join(OUT_ROOT, "training", "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    torch.save(model.state_dict(), os.path.join(OUT_ROOT, "checkpoints", "checkpoint_final.pt"))

    # reproducibility check: reload final checkpoint into a fresh instance (small 8-patch slice, no OOM risk)
    repro_x, _, _, _, _ = to_tensors(dev_arrs, device, idx=np.arange(8))
    model2 = Model3TypedEquivariant(n_classes=2, c1=8, c2=8).to(device)
    model2.eval()
    model2.load_state_dict(torch.load(os.path.join(OUT_ROOT, "checkpoints", "checkpoint_final.pt"), map_location=device))
    with torch.no_grad():
        logits_a, q_a = model(repro_x)
        logits_b, q_b = model2(repro_x)
    checkpoint_reproducible = torch.equal(logits_a, logits_b) and torch.equal(q_a, q_b)
    correct_type = model2.q_out_type.size == 2 and model2.group.order() == 16

    summary = {
        "stage": "STAGE 10 - final Model 3 training",
        "n_params": n_params,
        "seed": SEED, "device": device, "dtype": "float32",
        "n_epochs_run": len(history),
        "L_initial_total": history[0]["train_loss"] if history else None,
        "L_final_total": history[-1]["train_loss"] if history else None,
        "best_dev_loss": best_dev_loss,
        "nan_or_inf_detected": nan_or_inf_detected,
        "checkpoint_reproducible": checkpoint_reproducible,
        "output_type_correct_after_reload": correct_type,
        "train_duration_sec": train_duration_sec,
        "peak_gpu_mem_mb": max((r["gpu_mem_mb"] for r in history if r["gpu_mem_mb"]), default=None),
        "split_protection": {"train": n_train, "testA": n_testA, "testB": n_testB,
                              "testA_testB_used_this_stage": False},
    }
    with open(os.path.join(OUT_ROOT, "metrics", "stage10_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("STAGE 10 TRAINING SUMMARY")
    print("=" * 60)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
