"""
STAGE 8: 8-sample / 500-epoch tiny overfit test. ENGINEERING VALIDATION
ONLY -- confirms the implementation can optimize, not that Model 3
generalizes. testA/testB are never opened anywhere in this file.

Sample selection rule (deterministic, documented, not cherry-picked):
canonical TRAIN images (data_manifests/glas_manifest_stage3_finalized.csv,
canonical_split == 'train'), sorted by image_id. For each of the first 8
such images, scan that image's own patch grid (code/data_prep.py's exact
128x128/stride-96 grid, in raster order) and take the FIRST patch with
>= 10% valid anatomical-target coverage (same threshold already
established in the Stage 0/0-recalibration work, reused for consistency,
not re-chosen here). This guarantees each of the 8 samples has some
learnable Q signal without hand-picking a "good-looking" patch.

Anatomical targets are loaded from the FROZEN Stage 4 persisted fields
(results_v2/anatomy_targets/stage4_fields/*.npz) -- not regenerated.
"""
import sys
import os
import json
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
from data_prep import extract_patches, build_image_label_table

sys.path.insert(0, os.path.dirname(__file__))
from model3_typed_equivariant import Model3TypedEquivariant
from model3_loss import model3_total_loss, downsample_target_for_loss
from nematic_math import rotate_Q_analytic

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "data_manifests", "glas_manifest_stage3_finalized.csv")
FIELDS_DIR = os.path.join(PROJECT_ROOT, "results_v2", "anatomy_targets", "stage4_fields")
OUT_DIR = os.path.join(PROJECT_ROOT, "results_v2", "phase2", "stage8_tiny_overfit")
os.makedirs(OUT_DIR, exist_ok=True)

N_SAMPLES = 8
MAX_EPOCHS = 500
MIN_VALID_FRACTION = 0.10  # same threshold as Stage 0/0-recalibration, reused for consistency
PATCH_SIZE = 128
STRIDE = 96
SEED = 42
LR = 1e-3  # predefined engineering configuration; NOT tuned/searched


def crop_field_patch(field, y, x, size=PATCH_SIZE):
    return {k: v[y:y + size, x:x + size] for k, v in field.items()}


def select_8_samples():
    """Deterministic, documented -- see module docstring. TRAIN split only."""
    manifest = pd.read_csv(MANIFEST_PATH)
    train_images = manifest[manifest.canonical_split == "train"].sort_values("image_id").reset_index(drop=True)
    assert not train_images["image_id"].str.startswith(("testA_", "testB_")).any(), \
        "contamination check failed -- a non-train image leaked into the selection pool"

    table = build_image_label_table()
    label_lookup = dict(zip(table["stem"], table["label"]))

    samples = []
    for _, row in train_images.iterrows():
        if len(samples) >= N_SAMPLES:
            break
        image_id = row["image_id"]
        field = np.load(os.path.join(FIELDS_DIR, f"{image_id}.npz"))
        field = {k: field[k] for k in field.files}
        H, W = field["valid"].shape

        rgb_patches = extract_patches(os.path.join(PROJECT_ROOT, row["image_path"]), patch_size=PATCH_SIZE, stride=STRIDE)
        idx = 0
        found = None
        for y in range(0, max(1, H - PATCH_SIZE + 1), STRIDE):
            for x in range(0, max(1, W - PATCH_SIZE + 1), STRIDE):
                if y + PATCH_SIZE > H or x + PATCH_SIZE > W:
                    continue
                patch_field = crop_field_patch(field, y, x)
                if patch_field["valid"].mean() >= MIN_VALID_FRACTION:
                    found = {
                        "image_id": image_id, "patch_offset": (int(y), int(x)), "patch_grid_index": idx,
                        "rgb": rgb_patches[idx], "field": patch_field,
                        "label": int(label_lookup[image_id]),
                    }
                    break
                idx += 1
            if found is not None:
                break
        assert found is not None, f"no qualifying patch found for {image_id} -- unexpected, investigate"
        samples.append(found)

    assert len(samples) == N_SAMPLES
    return samples


def build_tensors(samples, device):
    x = torch.stack([torch.from_numpy(s["rgb"].astype(np.float32) / 255.0).permute(2, 0, 1) for s in samples]).to(device)
    labels = torch.tensor([s["label"] for s in samples], dtype=torch.long, device=device)
    q1_t = torch.stack([torch.from_numpy(s["field"]["q1"]) for s in samples]).to(device)
    q2_t = torch.stack([torch.from_numpy(s["field"]["q2"]) for s in samples]).to(device)
    valid_t = torch.stack([torch.from_numpy(s["field"]["valid"]) for s in samples]).to(device)
    return x, labels, q1_t, q2_t, valid_t


def grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return total ** 0.5


def param_norm(model):
    total = 0.0
    for p in model.parameters():
        total += p.detach().pow(2).sum().item()
    return total ** 0.5


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[stage8] device = {device}")

    print("[stage8] selecting 8 deterministic samples from canonical TRAIN split...")
    samples = select_8_samples()
    sample_manifest = [{"image_id": s["image_id"], "patch_offset": s["patch_offset"],
                         "patch_grid_index": s["patch_grid_index"], "label": s["label"],
                         "valid_fraction": float(s["field"]["valid"].mean())} for s in samples]
    print(json.dumps(sample_manifest, indent=2))
    with open(os.path.join(OUT_DIR, "selected_samples.json"), "w") as f:
        json.dump(sample_manifest, f, indent=2)

    x, labels, q1_t, q2_t, valid_t = build_tensors(samples, device)
    print(f"[stage8] x={tuple(x.shape)} labels={labels.tolist()} "
          f"valid_fraction per sample={[round(v['valid_fraction'],3) for v in sample_manifest]}")

    torch.manual_seed(SEED)
    model = Model3TypedEquivariant(n_classes=2, c1=8, c2=8).to(device)
    model.eval()  # materialize escnn's lazy filter buffers before saving, so this checkpoint reloads cleanly too
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "checkpoint_initial.pt"))

    opt = torch.optim.Adam(model.parameters(), lr=LR)

    history = []
    L_initial = None
    nan_or_inf_detected = False

    for epoch in range(MAX_EPOCHS):
        model.train()
        opt.zero_grad()
        logits, q_dense = model(x)
        loss, comps = model3_total_loss(logits, labels, q_dense, q1_t, q2_t, valid_t)

        if not torch.isfinite(loss):
            nan_or_inf_detected = True
            print(f"[stage8] NaN/Inf loss detected at epoch {epoch} -- STOPPING")
            break

        loss.backward()

        gnorm = grad_norm(model)
        pnorm = param_norm(model)
        if not np.isfinite(gnorm):
            nan_or_inf_detected = True
            print(f"[stage8] non-finite gradient norm at epoch {epoch} -- STOPPING")
            break

        opt.step()

        with torch.no_grad():
            q1_pred, q2_pred = q_dense[:, 0], q_dense[:, 1]
            S_pred = torch.hypot(q1_pred, q2_pred)
            theta_pred = 0.5 * torch.atan2(q2_pred, q1_pred)

        if epoch == 0:
            L_initial = comps["L_total"]

        row = {
            "epoch": epoch, "L_total": comps["L_total"], "L_class": comps["L_class"], "L_Q": comps["L_Q"],
            "grad_norm": gnorm, "param_norm": pnorm, "lr": LR,
            "q_pred_mean": q_dense.mean().item(), "q_pred_std": q_dense.std().item(),
            "q_pred_absmax": q_dense.abs().max().item(),
            "S_pred_mean": S_pred.mean().item(), "S_pred_std": S_pred.std().item(),
            "theta_pred_std": theta_pred.std().item(),
        }
        history.append(row)
        if epoch % 25 == 0 or epoch == MAX_EPOCHS - 1:
            print(f"    epoch {epoch:4d}  L_total={row['L_total']:.5f}  L_class={row['L_class']:.5f}  "
                  f"L_Q={row['L_Q']:.5f}  grad_norm={gnorm:.4f}  S_pred_mean={row['S_pred_mean']:.4f}")

    L_final = history[-1]["L_total"] if history else float("nan")
    ratio = L_final / L_initial if L_initial and np.isfinite(L_initial) else float("nan")
    print(f"[stage8] L_initial={L_initial:.5f}  L_final={L_final:.5f}  ratio={ratio:.5f}")

    # escnn's R2Conv discards its cached `filter` buffer in train() mode and only
    # materializes it in eval() mode (documented escnn behavior -- "we recommend
    # converting the model to eval mode before storing or loading the state dict").
    # The training loop leaves the model in train() mode; switch to eval() before saving.
    model.eval()
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "checkpoint_final.pt"))

    with open(os.path.join(OUT_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # target statistics (fixed, don't change across epochs)
    target_stats = {
        "q1_target_mean_valid": float(q1_t[valid_t].mean().item()) if valid_t.any() else None,
        "q2_target_mean_valid": float(q2_t[valid_t].mean().item()) if valid_t.any() else None,
        "S_target_mean_valid": float(torch.hypot(q1_t, q2_t)[valid_t].mean().item()) if valid_t.any() else None,
        "n_valid_pixels_total": int(valid_t.sum().item()),
        "n_pixels_total": int(valid_t.numel()),
    }
    print("[stage8] target stats:", target_stats)

    # ---- checkpoint reproducibility ----
    model.eval()
    with torch.no_grad():
        logits_before, q_before = model(x)

    # NOTE (Level 1 implementation detail, fixed): escnn's R2Conv registers its expanded
    # `filter` as a lazily-materialized buffer, only populated on the first train()/eval()
    # call. A freshly constructed model's state_dict() is missing that key until eval()/
    # train() is called once, so we must call eval() BEFORE load_state_dict, not after.
    model2 = Model3TypedEquivariant(n_classes=2, c1=8, c2=8).to(device)
    model2.eval()
    model2.load_state_dict(torch.load(os.path.join(OUT_DIR, "checkpoint_final.pt"), map_location=device))
    with torch.no_grad():
        logits_after, q_after = model2(x)
    checkpoint_reproducible = torch.equal(logits_before, logits_after) and torch.equal(q_before, q_after)
    print(f"[stage8] checkpoint reproducibility: {checkpoint_reproducible}")

    # ---- rotation sanity check (NOT gated against preregistered thresholds) ----
    print("[stage8] post-training rotation sanity check (D8 elements, informational only)...")
    import escnn.nn as enn
    model.eval()
    x0 = x[:1]
    xg = enn.GeometricTensor(x0, model.input_type)

    def run(t):
        f1 = model.pool1(model.block1(t))
        shared = model.block2(f1)
        return model.q_head(shared)

    rotation_check = []
    with torch.no_grad():
        for g in model.gspace.testing_elements:
            q_rot = run(xg.transform(g))
            q_then = run(xg).transform(g)
            finite = torch.isfinite(q_rot.tensor).all().item() and torch.isfinite(q_then.tensor).all().item()
            correct_type = q_rot.type == model.q_out_type
            err = (q_rot.tensor - q_then.tensor).abs().mean().item()
            rotation_check.append({"element": str(g), "finite": finite, "correct_fieldtype": correct_type, "mean_abs_diff": err})
    print(json.dumps(rotation_check, indent=2))
    all_finite_and_typed = all(r["finite"] and r["correct_fieldtype"] for r in rotation_check)

    summary = {
        "seed": SEED, "device": device, "lr": LR, "max_epochs": MAX_EPOCHS,
        "n_epochs_run": len(history),
        "L_initial": L_initial, "L_final": L_final, "ratio": ratio,
        "criterion_L_final_over_L_initial_le_0.10": bool(ratio <= 0.10) if np.isfinite(ratio) else False,
        "nan_or_inf_detected": nan_or_inf_detected,
        "checkpoint_reproducible": checkpoint_reproducible,
        "rotation_check_all_finite_and_correctly_typed": all_finite_and_typed,
        "target_stats": target_stats,
        "final_grad_norm": history[-1]["grad_norm"] if history else None,
        "final_param_norm": history[-1]["param_norm"] if history else None,
    }
    with open(os.path.join(OUT_DIR, "stage8_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(OUT_DIR, "rotation_check.json"), "w") as f:
        json.dump(rotation_check, f, indent=2)

    print("\n" + "=" * 60)
    print("STAGE 8 ENGINEERING VALIDATION SUMMARY")
    print("=" * 60)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
