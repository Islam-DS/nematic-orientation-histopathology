"""
Phase 1 synthetic benchmark. Reads (does not write) the pre-registered
thresholds in results_v2/synthetic/preregistered_thresholds.json -- that
file must already exist and is not edited by this script.

Trains a p8m physically-typed nematic regressor on synthetic ellipses with
exact (q1, q2) ground truth, evaluates held-out angular/magnitude error,
measures numerical equivariance error under CONTINUOUS rotation angles
(not just the 45 deg lattice), trains a p4m version as an architectural
sanity comparison, and computes the final GO/NO-GO verdict.

ROTATION CONVENTION (verified empirically, see notes below and in
synthetic_shapes.py): skimage.transform.rotate(img, angle=+alpha_deg)
rotates the image by +alpha in the same sense the model / escnn's group
elements use internally -- confirmed by matching model(rotated x) against
R(2alpha).model(x) at all 8 p8m lattice angles (errors 2-5%, consistent
with ordinary interpolation/boundary artifacts, not a sign error). This is
a DIFFERENT convention from skimage.draw.ellipse's `rotation` parameter
(handled separately in synthetic_shapes.py) -- the two skimage functions
do not share a rotation-sign convention, verified independently for each.
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from skimage.transform import rotate as sk_rotate

from nematic_math import phi_S_to_q, q_to_phi_S, rotate_Q_analytic, angular_error_mod_pi
from nematic_model import NematicNet
from synthetic_shapes import render_ellipse, make_dataset

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results_v2", "synthetic")
THRESHOLDS_PATH = os.path.join(RESULTS_DIR, "preregistered_thresholds.json")
SIZE = 64
MEAN_RADIUS = 16.0
SEED = 42


def load_thresholds():
    assert os.path.exists(THRESHOLDS_PATH), (
        "preregistered_thresholds.json must exist BEFORE running the benchmark -- "
        "see run_synthetic_benchmark.py docstring."
    )
    with open(THRESHOLDS_PATH) as f:
        return json.load(f)


def train_model(group_N, train_imgs, train_q, device, epochs=25, batch_size=64, lr=2e-3, seed=SEED):
    torch.manual_seed(seed)
    net = NematicNet(group_N=group_N, c1=8, c2=8, in_channels=1).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    x = torch.from_numpy(train_imgs).to(device)
    y = torch.from_numpy(train_q).to(device)
    n = x.shape[0]

    t0 = time.time()
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            pred = net(x[idx])
            loss = loss_fn(pred, y[idx])
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    [group_N={group_N}] epoch {epoch+1}/{epochs}  loss={total_loss/n:.5f}")
    dt = time.time() - t0
    print(f"    [group_N={group_N}] trained in {dt:.1f}s")
    net.eval()  # freeze BatchNorm running stats -- critical for consistent
                # equivariance behavior at inference time (see investigation
                # notes: train-mode per-batch statistics computed from a
                # single-image batch are numerically less stable than the
                # accumulated running statistics, and amplify boundary
                # effects into large apparent equivariance errors)
    return net, dt


@torch.no_grad()
def evaluate(net, imgs, q_true, phi_true, S_true, device, batch_size=128):
    preds = []
    for i in range(0, len(imgs), batch_size):
        batch = torch.from_numpy(imgs[i:i + batch_size]).to(device)
        preds.append(net(batch).cpu().numpy())
    q_pred = np.concatenate(preds, axis=0)
    phi_pred, S_pred = q_to_phi_S(q_pred[:, 0], q_pred[:, 1])
    ang_err_deg = np.degrees(angular_error_mod_pi(phi_pred, phi_true))
    S_err = np.abs(S_pred - S_true)
    return {
        "mean_angular_error_deg": float(np.mean(ang_err_deg)),
        "median_angular_error_deg": float(np.median(ang_err_deg)),
        "mean_S_error": float(np.mean(S_err)),
        "q_pred": q_pred,
        "phi_pred": phi_pred,
        "S_pred": S_pred,
        "ang_err_deg": ang_err_deg,
    }


@torch.no_grad()
def measure_equivariance_error(net, device, n_images=20, angles_deg=(0, 15, 30, 37, 45, 60, 90, 123, 150, 173)):
    rng = np.random.default_rng(SEED + 1)
    errors = []
    per_angle = {a: [] for a in angles_deg}
    for _ in range(n_images):
        phi0 = rng.uniform(0, np.pi)
        S0 = rng.uniform(0.3, 0.9)
        img, _ = render_ellipse(phi0, S0, size=SIZE, mean_radius=MEAN_RADIUS, noise_std=0.02, rng=rng)
        x0 = torch.from_numpy(img[None, None]).to(device)
        q0 = net(x0)[0].cpu().numpy()

        for alpha_deg in angles_deg:
            if alpha_deg == 0:
                rot_img = img
            else:
                rot_img = sk_rotate(img, angle=alpha_deg, mode='reflect', order=1).astype(np.float32)
            xr = torch.from_numpy(rot_img[None, None]).to(device)
            q_rot = net(xr)[0].cpu().numpy()

            alpha = np.deg2rad(alpha_deg)
            q_expected = np.array(rotate_Q_analytic(q0[0], q0[1], alpha))
            err = np.linalg.norm(q_rot - q_expected) / (np.linalg.norm(q_expected) + 1e-6)
            errors.append(err)
            per_angle[alpha_deg].append(err)

    per_angle_mean = {str(a): float(np.mean(v)) for a, v in per_angle.items()}
    return float(np.mean(errors)), per_angle_mean


def main():
    thresholds = load_thresholds()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[synthetic_benchmark] device = {device}")
    print(f"[synthetic_benchmark] loaded thresholds from {THRESHOLDS_PATH}")

    # ---- synthetic data ----
    rng_train_phis = np.linspace(0, np.pi, 60, endpoint=False)  # dense coverage, incl. non-45-multiples
    S_values_train = [0.3, 0.5, 0.7, 0.9]
    train_imgs, train_q, train_phi, train_S = make_dataset(
        rng_train_phis, S_values_train, size=SIZE, mean_radius=MEAN_RADIUS,
        noise_std=0.03, seed=SEED, repeats=2,
    )
    print(f"[synthetic_benchmark] train set: {len(train_imgs)} images")

    # held-out test angles: explicit mix of 45-multiples and non-multiples, per spec
    test_phis_deg = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 7, 52, 88, 133, 161]
    test_phis = np.deg2rad(test_phis_deg)
    S_values_test = [0.4, 0.6, 0.8]
    test_imgs, test_q, test_phi, test_S = make_dataset(
        test_phis, S_values_test, size=SIZE, mean_radius=MEAN_RADIUS,
        noise_std=0.03, seed=SEED + 100, repeats=3,
    )
    print(f"[synthetic_benchmark] held-out test set: {len(test_imgs)} images "
          f"({len(test_phis_deg)} distinct orientations, including non-45-multiples)")

    # ---- naive baseline ----
    mean_q = train_q.mean(axis=0)
    naive_phi, naive_S = q_to_phi_S(np.full(len(test_q), mean_q[0]), np.full(len(test_q), mean_q[1]))
    naive_ang_err_deg = np.degrees(angular_error_mod_pi(naive_phi, test_phi))
    naive_mean_ang_err = float(np.mean(naive_ang_err_deg))
    print(f"[synthetic_benchmark] naive baseline (predict mean train Q) "
          f"mean angular error = {naive_mean_ang_err:.2f} deg")

    # ---- p8m (primary model) ----
    print("[synthetic_benchmark] training p8m nematic model...")
    net8, train_time_8 = train_model(8, train_imgs, train_q, device)
    eval8 = evaluate(net8, test_imgs, test_q, test_phi, test_S, device)
    print(f"[synthetic_benchmark] p8m held-out: mean_angular_error={eval8['mean_angular_error_deg']:.2f} deg, "
          f"median={eval8['median_angular_error_deg']:.2f} deg, mean_S_error={eval8['mean_S_error']:.4f}")

    print("[synthetic_benchmark] measuring p8m equivariance error (continuous angles)...")
    equiv_err_8, equiv_per_angle_8 = measure_equivariance_error(net8, device)
    print(f"[synthetic_benchmark] p8m mean normalized equivariance error = {equiv_err_8:.4f}")
    for a, e in equiv_per_angle_8.items():
        print(f"    angle={a:>4} deg  equiv_err={e:.4f}")

    # ---- p4m (architectural sanity comparison; degenerate 1D irrep) ----
    print("[synthetic_benchmark] training p4m nematic model (degenerate 1D irrep, sanity comparison)...")
    net4, train_time_4 = train_model(4, train_imgs, train_q, device)
    eval4 = evaluate(net4, test_imgs, test_q, test_phi, test_S, device)
    print(f"[synthetic_benchmark] p4m held-out: mean_angular_error={eval4['mean_angular_error_deg']:.2f} deg "
          f"(expected to be poor -- p4m can only express phi in {{0, 90}} deg, see test_nematic_math.py #6)")

    # ---- decision ----
    crit = thresholds["criteria"]
    pass_angular = eval8["mean_angular_error_deg"] < crit["mean_angular_error_deg_lt"]["threshold"]
    pass_equiv = equiv_err_8 < crit["normalized_equivariance_error_lt"]["threshold"]
    pass_baseline = eval8["mean_angular_error_deg"] < 0.5 * naive_mean_ang_err
    verdict = "GO" if (pass_angular and pass_equiv and pass_baseline) else "NO-GO"

    summary = {
        "device": device,
        "n_train_images": int(len(train_imgs)),
        "n_test_images": int(len(test_imgs)),
        "n_test_orientations": len(test_phis_deg),
        "test_orientations_deg": test_phis_deg,
        "p8m": {
            "train_time_sec": train_time_8,
            "mean_angular_error_deg": eval8["mean_angular_error_deg"],
            "median_angular_error_deg": eval8["median_angular_error_deg"],
            "mean_S_error": eval8["mean_S_error"],
            "mean_normalized_equivariance_error": equiv_err_8,
            "equivariance_error_per_angle_deg": equiv_per_angle_8,
        },
        "p4m_sanity_comparison": {
            "train_time_sec": train_time_4,
            "mean_angular_error_deg": eval4["mean_angular_error_deg"],
            "median_angular_error_deg": eval4["median_angular_error_deg"],
            "mean_S_error": eval4["mean_S_error"],
            "note": "D4 irrep(1,2) is 1-dimensional (see test_nematic_math.py #6); "
                    "this model can only ever output q2=0, i.e. predict phi in {0, pi/2}",
        },
        "naive_baseline_mean_angular_error_deg": naive_mean_ang_err,
        "criteria_results": {
            "mean_angular_error_lt_10deg": bool(pass_angular),
            "normalized_equivariance_error_lt_0.05": bool(pass_equiv),
            "substantially_better_than_naive_baseline": bool(pass_baseline),
        },
        "VERDICT": verdict,
    }

    with open(os.path.join(RESULTS_DIR, "synthetic_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    np.savez(os.path.join(RESULTS_DIR, "eval_arrays.npz"),
              test_phi=test_phi, test_S=test_S,
              p8m_phi_pred=eval8["phi_pred"], p8m_S_pred=eval8["S_pred"], p8m_ang_err=eval8["ang_err_deg"],
              p4m_phi_pred=eval4["phi_pred"], p4m_S_pred=eval4["S_pred"], p4m_ang_err=eval4["ang_err_deg"])

    torch.save(net8.state_dict(), os.path.join(RESULTS_DIR, "..", "checkpoints", "p8m_synthetic.pt"))
    torch.save(net4.state_dict(), os.path.join(RESULTS_DIR, "..", "checkpoints", "p4m_synthetic.pt"))

    print("\n" + "=" * 60)
    print("PHASE 1 SYNTHETIC BENCHMARK -- PRE-REGISTERED DECISION")
    print("=" * 60)
    for k, v in summary["criteria_results"].items():
        print(f"  {k}: {v}")
    print(f"  VERDICT: {verdict}")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    os.makedirs(os.path.join(RESULTS_DIR, "..", "checkpoints"), exist_ok=True)
    main()
