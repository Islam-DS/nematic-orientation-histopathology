"""
STAGE 8 EXTENDED: single controlled extended engineering run, authorized
after the original 500-epoch Stage 8 run did not meet the primary
criterion (ratio 0.139 > 0.10).

ONLY CHANGE from the original run: MAX_EPOCHS 500 -> 1000. Every other
setting is IMPORTED, not re-typed, from code_v2/stage8_tiny_overfit.py to
guarantee byte-for-byte identical sample selection, preprocessing, model
construction, loss, optimizer, and seed. This is not a hyperparameter
search -- one fixed configuration, run once, with a longer epoch budget.

The original run's script, outputs, and audit report are NOT modified --
this writes to an entirely separate output directory.
"""
import sys
import os
import json

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from stage8_tiny_overfit import (
    select_8_samples, build_tensors, grad_norm, param_norm,
    SEED, LR, MIN_VALID_FRACTION, PATCH_SIZE, STRIDE, N_SAMPLES,
)
from model3_typed_equivariant import Model3TypedEquivariant
from model3_loss import model3_total_loss

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR = os.path.join(PROJECT_ROOT, "results_v2", "phase2", "stage8_extended_tiny_overfit")
os.makedirs(OUT_DIR, exist_ok=True)

MAX_EPOCHS = 1000  # ONLY CHANGE from the original run (was 500)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[stage8_extended] device = {device}")
    print(f"[stage8_extended] ONLY CHANGE from the original Stage 8 run: MAX_EPOCHS 500 -> {MAX_EPOCHS}")
    print(f"[stage8_extended] reusing unchanged: SEED={SEED}, LR={LR}, "
          f"MIN_VALID_FRACTION={MIN_VALID_FRACTION}, PATCH_SIZE={PATCH_SIZE}, STRIDE={STRIDE}, N_SAMPLES={N_SAMPLES}")

    print("[stage8_extended] selecting samples (identical function/rule as the original run)...")
    samples = select_8_samples()
    sample_manifest = [{"image_id": s["image_id"], "patch_offset": s["patch_offset"],
                         "patch_grid_index": s["patch_grid_index"], "label": s["label"],
                         "valid_fraction": float(s["field"]["valid"].mean())} for s in samples]
    with open(os.path.join(OUT_DIR, "selected_samples.json"), "w") as f:
        json.dump(sample_manifest, f, indent=2)
    print(json.dumps(sample_manifest, indent=2))

    x, labels, q1_t, q2_t, valid_t = build_tensors(samples, device)
    print(f"[stage8_extended] x={tuple(x.shape)} labels={labels.tolist()}")

    torch.manual_seed(SEED)
    model = Model3TypedEquivariant(n_classes=2, c1=8, c2=8).to(device)
    model.eval()
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
            print(f"[stage8_extended] NaN/Inf loss detected at epoch {epoch} -- STOPPING")
            break

        loss.backward()
        gnorm = grad_norm(model)
        pnorm = param_norm(model)
        if not np.isfinite(gnorm):
            nan_or_inf_detected = True
            print(f"[stage8_extended] non-finite gradient norm at epoch {epoch} -- STOPPING")
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
        if epoch % 50 == 0 or epoch == MAX_EPOCHS - 1:
            print(f"    epoch {epoch:4d}  L_total={row['L_total']:.5f}  L_class={row['L_class']:.5f}  "
                  f"L_Q={row['L_Q']:.5f}  grad_norm={gnorm:.4f}  S_pred_mean={row['S_pred_mean']:.4f}")

    L_final = history[-1]["L_total"] if history else float("nan")
    ratio = L_final / L_initial if L_initial and np.isfinite(L_initial) else float("nan")
    print(f"[stage8_extended] L_initial={L_initial:.5f}  L_final={L_final:.5f}  ratio={ratio:.5f}")

    model.eval()  # escnn filter-buffer lifecycle -- same handling as the original run
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "checkpoint_final.pt"))

    with open(os.path.join(OUT_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    target_stats = {
        "q1_target_mean_valid": float(q1_t[valid_t].mean().item()) if valid_t.any() else None,
        "q2_target_mean_valid": float(q2_t[valid_t].mean().item()) if valid_t.any() else None,
        "S_target_mean_valid": float(torch.hypot(q1_t, q2_t)[valid_t].mean().item()) if valid_t.any() else None,
        "n_valid_pixels_total": int(valid_t.sum().item()),
        "n_pixels_total": int(valid_t.numel()),
    }

    # checkpoint reproducibility
    with torch.no_grad():
        logits_before, q_before = model(x)
    model2 = Model3TypedEquivariant(n_classes=2, c1=8, c2=8).to(device)
    model2.eval()
    model2.load_state_dict(torch.load(os.path.join(OUT_DIR, "checkpoint_final.pt"), map_location=device))
    with torch.no_grad():
        logits_after, q_after = model2(x)
    checkpoint_reproducible = torch.equal(logits_before, logits_after) and torch.equal(q_before, q_after)
    print(f"[stage8_extended] checkpoint reproducibility: {checkpoint_reproducible}")

    # rotation check -- informational only, not gated, not used to tune anything
    import escnn.nn as enn
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
    with open(os.path.join(OUT_DIR, "rotation_check.json"), "w") as f:
        json.dump(rotation_check, f, indent=2)
    all_finite_and_typed = all(r["finite"] and r["correct_fieldtype"] for r in rotation_check)

    verdict = "PASS" if ratio <= 0.10 else "NOT MET"
    summary = {
        "run": "STAGE 8 EXTENDED (1000 epochs) -- authorized single controlled extended run",
        "only_change_from_original": "max_epochs 500 -> 1000; all else identical",
        "seed": SEED, "device": device, "lr": LR, "max_epochs": MAX_EPOCHS,
        "n_epochs_run": len(history),
        "L_initial": L_initial, "L_final": L_final, "ratio": ratio,
        "STAGE_8_ENGINEERING_CRITERION": verdict,
        "nan_or_inf_detected": nan_or_inf_detected,
        "checkpoint_reproducible": checkpoint_reproducible,
        "rotation_check_all_finite_and_correctly_typed": all_finite_and_typed,
        "target_stats": target_stats,
        "final_grad_norm": history[-1]["grad_norm"] if history else None,
        "final_param_norm": history[-1]["param_norm"] if history else None,
    }
    with open(os.path.join(OUT_DIR, "stage8_extended_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"STAGE 8 EXTENDED ENGINEERING CRITERION = {verdict}")
    print("=" * 60)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
