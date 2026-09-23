"""
Phase 1R: Corrected Synthetic Benchmark Replication.

Execution order (enforces "thresholds frozen before results are seen"):
  1. Render a CALIBRATION set (separate seed from train/val/test).
  2. Measure E_render (rasterization floor, oracle vs ground truth) and
     the rotation-pipeline floor E_render_rotation (oracle self-consistency
     under rotation, via clean re-rendering vs the OLD pixel-rotate method)
     -- entirely from the calibration set, no model involved yet.
  3. WRITE results_v2/synthetic_replication/preregistered_thresholds.json
     using the calibration numbers to set the equivariance margin. This
     file is not edited again after this point.
  4. Generate train/val/test sets (distinct seeds/streams).
  5. Train p8m nematic model and the non-equivariant CNN control, with
     early stopping on validation loss (not to zero training loss).
  6. Evaluate all of {ground truth (trivial), naive-gradient baseline,
     structure-tensor baseline, p8m, CNN control} on the held-out test set.
  7. Rotation-consistency test (clean re-rendering) for p8m and CNN control.
  8. Apply the FROZEN decision rule from step 3. Save everything.
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from nematic_math import q_to_phi_S, rotate_Q_analytic, angular_error_mod_pi
from nematic_model import NematicNet
from baseline_cnn import PlainCNN
from oracle_baselines import structure_tensor_estimate, naive_gradient_estimate
from synthetic_shapes_v2 import render_object, sample_object_params, pixel_domain_rotate

BASE = os.path.join(os.path.dirname(__file__), "..", "results_v2", "synthetic_replication")
CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "results_v2", "checkpoints")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results_v2", "plots")
os.makedirs(BASE, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

SIZE = 128
SEED_CALIB = 2001
SEED_TRAIN = 42
SEED_VAL = 142
SEED_TEST = 242
SEED_EQUIV_PROBE = 342
SEED_MODEL_INIT = 7


def q_normalized_error(q_a, q_b, eps=1e-6):
    num = np.linalg.norm(q_a - q_b, axis=-1)
    den = np.linalg.norm(q_b, axis=-1) + eps
    return num / den


# ---------------------------------------------------------------------
# Step 1-2: calibration -- rendering floor and rotation-pipeline floor
# ---------------------------------------------------------------------
def run_calibration(n_samples=120, seed=SEED_CALIB):
    rng = np.random.default_rng(seed)
    angular_errs = []
    render_equiv_errs = []       # oracle self-consistency, CLEAN re-render rotation
    pixelrotate_equiv_errs = []  # oracle self-consistency, OLD pixel-domain rotation

    for _ in range(n_samples):
        phi0 = rng.uniform(0, np.pi)
        S0 = rng.uniform(0.3, 0.9)
        params = sample_object_params(rng, size=SIZE)
        img0, (q1_0, q2_0) = render_object(phi0, S0, size=SIZE, rng=rng, **params)

        # (a) absolute rasterization floor: oracle vs analytic ground truth
        S_est, phi_est = structure_tensor_estimate(img0, sigma=3.0)
        angular_errs.append(np.degrees(angular_error_mod_pi(phi_est, phi0)))
        q0_oracle = np.array([S_est * np.cos(2 * phi_est), S_est * np.sin(2 * phi_est)])

        alpha_deg = rng.uniform(5, 175)
        alpha = np.deg2rad(alpha_deg)
        q_expected = np.array(rotate_Q_analytic(q0_oracle[0], q0_oracle[1], alpha))

        # (b) clean re-render rotation: regenerate at phi0+alpha, same object params
        img_clean, _ = render_object(phi0 + alpha, S0, size=SIZE, rng=rng, **params)
        S_c, phi_c = structure_tensor_estimate(img_clean, sigma=3.0)
        q_clean_oracle = np.array([S_c * np.cos(2 * phi_c), S_c * np.sin(2 * phi_c)])
        render_equiv_errs.append(q_normalized_error(q_clean_oracle, q_expected))

        # (c) OLD pixel-domain rotate, for comparison only
        img_px = pixel_domain_rotate(img0, alpha_deg)
        S_p, phi_p = structure_tensor_estimate(img_px, sigma=3.0)
        q_px_oracle = np.array([S_p * np.cos(2 * phi_p), S_p * np.sin(2 * phi_p)])
        pixelrotate_equiv_errs.append(q_normalized_error(q_px_oracle, q_expected))

    return {
        "n_calibration_samples": n_samples,
        "E_render_absolute_deg": float(np.mean(angular_errs)),
        "E_render_rotation_clean": float(np.mean(render_equiv_errs)),
        "E_render_rotation_pixel_domain_OLD_METHOD": float(np.mean(pixelrotate_equiv_errs)),
    }


# ---------------------------------------------------------------------
# Dataset generation (train/val/test, distinct seeds)
# ---------------------------------------------------------------------
def generate_split(n_samples, seed, size=SIZE):
    rng = np.random.default_rng(seed)
    imgs, qs, phis, Ss = [], [], [], []
    for _ in range(n_samples):
        phi = rng.uniform(0, np.pi)
        S = rng.uniform(0.2, 0.95)
        params = sample_object_params(rng, size=size)
        img, (q1, q2) = render_object(phi, S, size=size, rng=rng, **params)
        imgs.append(img[None])
        qs.append([q1, q2])
        phis.append(phi)
        Ss.append(S)
    return (np.stack(imgs).astype(np.float32), np.array(qs, dtype=np.float32),
            np.array(phis, dtype=np.float32), np.array(Ss, dtype=np.float32))


# ---------------------------------------------------------------------
# Training with early stopping on validation loss
# ---------------------------------------------------------------------
def train_with_early_stopping(model, train_imgs, train_q, val_imgs, val_q, device,
                               max_epochs=60, patience=6, batch_size=64, lr=2e-3, seed=SEED_MODEL_INIT):
    torch.manual_seed(seed)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    x_tr = torch.from_numpy(train_imgs).to(device)
    y_tr = torch.from_numpy(train_q).to(device)
    x_val = torch.from_numpy(val_imgs).to(device)
    y_val = torch.from_numpy(val_q).to(device)
    n = x_tr.shape[0]

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = []
    t0 = time.time()

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            pred = model(x_tr[idx])
            loss = loss_fn(pred, y_tr[idx])
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        train_loss = total_loss / n

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(x_val), y_val).item()
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})

        improved = val_loss < best_val - 1e-5
        if improved:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 5 == 0 or epoch == 0 or improved:
            flag = " *" if improved else ""
            print(f"      epoch {epoch+1:3d}  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}{flag}")

        if epochs_no_improve >= patience:
            print(f"      early stopping at epoch {epoch+1} (no val improvement for {patience} epochs)")
            break

    model.load_state_dict(best_state)
    model.eval()
    dt = time.time() - t0
    return model, dt, history, best_val


@torch.no_grad()
def evaluate_model(model, imgs, phi_true, S_true, device, batch_size=128):
    preds = []
    for i in range(0, len(imgs), batch_size):
        batch = torch.from_numpy(imgs[i:i + batch_size]).to(device)
        preds.append(model(batch).cpu().numpy())
    q_pred = np.concatenate(preds, axis=0)
    phi_pred, S_pred = q_to_phi_S(q_pred[:, 0], q_pred[:, 1])
    ang_err_deg = np.degrees(angular_error_mod_pi(phi_pred, phi_true))
    S_err = np.abs(S_pred - S_true)
    corr_S = float(np.corrcoef(S_pred, S_true)[0, 1])
    return {
        "mean_angular_error_deg": float(np.mean(ang_err_deg)),
        "median_angular_error_deg": float(np.median(ang_err_deg)),
        "mean_S_error": float(np.mean(S_err)),
        "corr_S_pred_true": corr_S,
    }


def evaluate_classical(fn, imgs, phi_true, S_true, **kwargs):
    S_preds, phi_preds = [], []
    for img in imgs:
        S_est, phi_est = fn(img[0], **kwargs)
        S_preds.append(S_est)
        phi_preds.append(phi_est)
    S_preds, phi_preds = np.array(S_preds), np.array(phi_preds)
    ang_err_deg = np.degrees(angular_error_mod_pi(phi_preds, phi_true))
    S_err = np.abs(S_preds - S_true)
    corr_S = float(np.corrcoef(S_preds, S_true)[0, 1])
    return {
        "mean_angular_error_deg": float(np.mean(ang_err_deg)),
        "median_angular_error_deg": float(np.median(ang_err_deg)),
        "mean_S_error": float(np.mean(S_err)),
        "corr_S_pred_true": corr_S,
    }


@torch.no_grad()
def rotation_consistency_test(model, device, n_objects=25, seed=SEED_EQUIV_PROBE,
                               angles_deg=(0, 15, 30, 37, 45, 60, 90, 123, 150, 173)):
    """Step 7: CLEAN re-rendering rotation-consistency test (fixes the Phase 1 confound)."""
    rng = np.random.default_rng(seed)
    errors = []
    per_angle = {a: [] for a in angles_deg}
    for _ in range(n_objects):
        phi0 = rng.uniform(0, np.pi)
        S0 = rng.uniform(0.3, 0.9)
        params = sample_object_params(rng, size=SIZE)
        img0, _ = render_object(phi0, S0, size=SIZE, rng=rng, **params)
        x0 = torch.from_numpy(img0[None, None]).to(device)
        q0 = model(x0)[0].cpu().numpy()

        for alpha_deg in angles_deg:
            alpha = np.deg2rad(alpha_deg)
            img_rot, _ = render_object(phi0 + alpha, S0, size=SIZE, rng=rng, **params)
            xr = torch.from_numpy(img_rot[None, None]).to(device)
            q_rot = model(xr)[0].cpu().numpy()

            q_expected = np.array(rotate_Q_analytic(q0[0], q0[1], alpha))
            err = q_normalized_error(q_rot, q_expected)
            errors.append(err)
            per_angle[alpha_deg].append(err)

    per_angle_mean = {str(a): float(np.mean(v)) for a, v in per_angle.items()}
    return float(np.mean(errors)), per_angle_mean


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[phase1r] device = {device}")

    # ---- Step 1-2: calibration ----
    print("[phase1r] running rendering-floor calibration...")
    calib = run_calibration()
    print(f"[phase1r] E_render_absolute = {calib['E_render_absolute_deg']:.3f} deg")
    print(f"[phase1r] E_render_rotation (clean re-render) = {calib['E_render_rotation_clean']:.4f}")
    print(f"[phase1r] E_render_rotation (OLD pixel-rotate) = {calib['E_render_rotation_pixel_domain_OLD_METHOD']:.4f} "
          f"(kept only to show the old confound -- not used to set thresholds)")

    # ---- Step 3: FREEZE thresholds, informed only by calibration ----
    equiv_margin_multiplier = 3.0
    equiv_threshold = max(equiv_margin_multiplier * calib["E_render_rotation_clean"], 0.05)
    thresholds = {
        "written_before_training": True,
        "phase": "Phase 1R - corrected synthetic benchmark replication",
        "calibration": calib,
        "criteria": {
            "mean_angular_error_deg_lt": {
                "threshold": 10.0,
                "applies_to": "p8m model, held-out test set",
                "justification": "carried over from Phase 1's suggested value; unrelated to the rendering-confound issue being fixed here"
            },
            "mean_S_error_lt": {
                "threshold": 0.15,
                "applies_to": "p8m model, held-out test set",
                "justification": "test S is drawn from U(0.2,0.95) (range 0.75); 0.15 is 20% of that range, a non-trivial but achievable bar"
            },
            "substantially_better_than_gradient_baseline": {
                "definition": "p8m mean angular error < 0.5 * naive-gradient-baseline mean angular error, both on the same held-out test set",
                "justification": "same structure as Phase 1's naive-baseline criterion"
            },
            "equivariance_error_relative_to_rendering_floor": {
                "threshold": equiv_threshold,
                "definition": f"p8m mean rotation-consistency error (E_model, clean re-render method) < {equiv_margin_multiplier} * E_render_rotation_clean (calibration floor), floored at an absolute minimum of 0.05",
                "calibration_value_used": calib["E_render_rotation_clean"],
                "justification": "this is the Step 8 requirement that the equivariance criterion be set relative to the benchmark's OWN intrinsic rendering/measurement noise floor, fixed BEFORE any model is trained or evaluated -- computed here purely from oracle-vs-oracle rendering calibration, with no model involved"
            },
            "clearly_better_than_nonequivariant_control": {
                "definition": "p8m E_model < 0.5 * CNN-control E_model (same rotation-consistency metric)",
                "justification": "Step 9 -- equivariance should provide a measurable rotation-consistency advantage over a same-capacity ordinary CNN trained on the same data"
            }
        },
        "decision_rule": "GO only if ALL FIVE criteria above pass. No threshold in this file is edited after results are computed.",
    }
    thresholds_path = os.path.join(BASE, "preregistered_thresholds.json")
    with open(thresholds_path, "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"[phase1r] FROZEN thresholds written to {thresholds_path}")
    print(f"[phase1r]   equivariance threshold = {equiv_threshold:.4f} "
          f"({equiv_margin_multiplier}x calibration floor of {calib['E_render_rotation_clean']:.4f}, floored at 0.05)")

    # ---- Step 4: train/val/test data ----
    print("[phase1r] generating train/val/test datasets...")
    train_imgs, train_q, train_phi, train_S = generate_split(900, SEED_TRAIN)
    val_imgs, val_q, val_phi, val_S = generate_split(200, SEED_VAL)
    test_imgs, test_q, test_phi, test_S = generate_split(300, SEED_TEST)
    print(f"[phase1r] train={len(train_imgs)} val={len(val_imgs)} test={len(test_imgs)}")

    # ---- Step 5: train p8m and CNN control, early stopping ----
    print("[phase1r] training p8m nematic model (early stopping on val loss)...")
    net8, t8, hist8, bestval8 = train_with_early_stopping(
        NematicNet(group_N=8, c1=8, c2=8, in_channels=1),
        train_imgs, train_q, val_imgs, val_q, device,
    )
    print(f"[phase1r] p8m trained in {t8:.1f}s, best val_loss={bestval8:.5f}")

    print("[phase1r] training non-equivariant CNN control...")
    net_cnn, tcnn, histcnn, bestvalcnn = train_with_early_stopping(
        PlainCNN(in_channels=1, c1=64, c2=64),
        train_imgs, train_q, val_imgs, val_q, device,
    )
    print(f"[phase1r] CNN control trained in {tcnn:.1f}s, best val_loss={bestvalcnn:.5f}")

    n_params_p8m = sum(p.numel() for p in net8.parameters())
    n_params_cnn = sum(p.numel() for p in net_cnn.parameters())
    print(f"[phase1r] param counts: p8m={n_params_p8m}  CNN={n_params_cnn}")

    # ---- Step 6: evaluate all methods on test set ----
    print("[phase1r] evaluating all methods on held-out test set...")
    results = {}
    results["A_ground_truth_analytic"] = {"note": "trivial reference: analytic phi, S used to generate images; angular/S error is 0 by definition"}
    results["B_naive_gradient_baseline"] = evaluate_classical(naive_gradient_estimate, test_imgs, test_phi, test_S)
    results["C_structure_tensor_baseline"] = evaluate_classical(structure_tensor_estimate, test_imgs, test_phi, test_S, sigma=3.0)
    results["D_p8m_model"] = evaluate_model(net8, test_imgs, test_phi, test_S, device)
    results["E_nonequivariant_cnn_control"] = evaluate_model(net_cnn, test_imgs, test_phi, test_S, device)

    for k, v in results.items():
        if "mean_angular_error_deg" in v:
            print(f"    {k}: angular_MAE={v['mean_angular_error_deg']:.2f} deg  "
                  f"S_err={v['mean_S_error']:.4f}  corr_S={v['corr_S_pred_true']:.3f}")

    # ---- Step 7: rotation-consistency test (clean re-render) ----
    print("[phase1r] rotation-consistency test (p8m)...")
    equiv_p8m, equiv_p8m_per_angle = rotation_consistency_test(net8, device)
    print(f"[phase1r] p8m E_model = {equiv_p8m:.4f}")

    print("[phase1r] rotation-consistency test (CNN control)...")
    equiv_cnn, equiv_cnn_per_angle = rotation_consistency_test(net_cnn, device)
    print(f"[phase1r] CNN control E_model = {equiv_cnn:.4f}")

    # ---- Step 8: apply FROZEN decision rule ----
    crit = thresholds["criteria"]
    pass_angular = results["D_p8m_model"]["mean_angular_error_deg"] < crit["mean_angular_error_deg_lt"]["threshold"]
    pass_S = results["D_p8m_model"]["mean_S_error"] < crit["mean_S_error_lt"]["threshold"]
    pass_gradient = results["D_p8m_model"]["mean_angular_error_deg"] < 0.5 * results["B_naive_gradient_baseline"]["mean_angular_error_deg"]
    pass_equiv_floor = equiv_p8m < crit["equivariance_error_relative_to_rendering_floor"]["threshold"]
    pass_vs_control = equiv_p8m < 0.5 * equiv_cnn
    verdict = "GO" if (pass_angular and pass_S and pass_gradient and pass_equiv_floor and pass_vs_control) else "NO-GO"

    final = {
        "device": device,
        "n_train": len(train_imgs), "n_val": len(val_imgs), "n_test": len(test_imgs),
        "param_counts": {"p8m": n_params_p8m, "cnn_control": n_params_cnn},
        "training": {
            "p8m_epochs_run": len(hist8), "p8m_best_val_loss": bestval8, "p8m_train_time_sec": t8,
            "cnn_epochs_run": len(histcnn), "cnn_best_val_loss": bestvalcnn, "cnn_train_time_sec": tcnn,
        },
        "test_set_results": results,
        "rotation_consistency": {
            "p8m_E_model": equiv_p8m,
            "p8m_E_model_per_angle": equiv_p8m_per_angle,
            "cnn_control_E_model": equiv_cnn,
            "cnn_control_E_model_per_angle": equiv_cnn_per_angle,
            "E_render_rotation_floor_calibration": calib["E_render_rotation_clean"],
        },
        "criteria_results": {
            "mean_angular_error_lt_10deg": bool(pass_angular),
            "mean_S_error_lt_0.15": bool(pass_S),
            "substantially_better_than_gradient_baseline": bool(pass_gradient),
            "equivariance_error_lt_calibrated_threshold": bool(pass_equiv_floor),
            "clearly_better_than_nonequivariant_control": bool(pass_vs_control),
        },
        "VERDICT": verdict,
    }

    with open(os.path.join(BASE, "phase1r_results.json"), "w") as f:
        json.dump(final, f, indent=2)
    with open(os.path.join(BASE, "training_history.json"), "w") as f:
        json.dump({"p8m": hist8, "cnn_control": histcnn}, f, indent=2)

    torch.save(net8.state_dict(), os.path.join(CKPT_DIR, "p8m_phase1r.pt"))
    torch.save(net_cnn.state_dict(), os.path.join(CKPT_DIR, "cnn_control_phase1r.pt"))

    print("\n" + "=" * 60)
    print("PHASE 1R -- PRE-REGISTERED DECISION")
    print("=" * 60)
    for k, v in final["criteria_results"].items():
        print(f"  {k}: {v}")
    print(f"  VERDICT: {verdict}")
    print("=" * 60)

    return final


if __name__ == "__main__":
    main()
