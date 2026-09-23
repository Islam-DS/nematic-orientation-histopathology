"""
Unit tests for the nematic representation math (code_v2/nematic_math.py),
cross-checked directly against escnn's own group representation matrices
(not assumed from memory). Run:
    python code_v2/test_nematic_math.py
"""
import numpy as np

from nematic_math import (
    phi_S_to_q, q_to_phi_S, rotate_Q_analytic, angular_error_mod_pi,
)


def test_phi_S_q_roundtrip():
    print("[1] phi_S_to_q / q_to_phi_S round-trip...")
    rng = np.random.default_rng(0)
    phis = rng.uniform(0, np.pi, 200)
    Ss = rng.uniform(0.05, 0.99, 200)
    q1, q2 = phi_S_to_q(phis, Ss)
    phi_rec, S_rec = q_to_phi_S(q1, q2)
    ang_err = np.degrees(angular_error_mod_pi(phi_rec, phis))
    S_err = np.abs(S_rec - Ss)
    print(f"    max angular error = {ang_err.max():.4f} deg, max S error = {S_err.max():.6f}")
    assert ang_err.max() < 1e-3
    assert S_err.max() < 1e-6
    print("    PASS\n")


def test_head_tail_symmetry():
    print("[2] head-tail symmetry: phi and phi+pi give identical (q1,q2)...")
    rng = np.random.default_rng(1)
    phis = rng.uniform(0, np.pi, 100)
    Ss = rng.uniform(0.1, 0.9, 100)
    q1a, q2a = phi_S_to_q(phis, Ss)
    q1b, q2b = phi_S_to_q(phis + np.pi, Ss)
    max_diff = np.max(np.abs(q1a - q1b)) + np.max(np.abs(q2a - q2b))
    print(f"    max |Q(phi) - Q(phi+pi)| = {max_diff:.2e}")
    assert max_diff < 1e-10
    print("    PASS\n")


def test_rotate_Q_matches_escnn_p8m_irrep():
    print("[3] rotate_Q_analytic matches escnn's D8 irrep(1,2) representation matrices...")
    from escnn import gspaces
    g8 = gspaces.flipRot2dOnR2(N=8)
    group = g8.fibergroup
    irr = group.irrep(1, 2)
    assert irr.size == 2, f"expected a 2D irrep for p8m frequency 2, got size {irr.size}"

    rng = np.random.default_rng(2)
    q0 = rng.uniform(-1, 1, 2)

    max_err = 0.0
    for k in range(8):
        alpha = k * (2 * np.pi / 8)
        g = group.element((0, k))
        M = irr(g)  # escnn's own representation matrix
        q_escnn = M @ q0
        q_ours = np.array(rotate_Q_analytic(q0[0], q0[1], alpha))
        err = np.linalg.norm(q_escnn - q_ours)
        max_err = max(max_err, err)
    print(f"    max discrepancy vs escnn irrep matrices over 8 rotations = {max_err:.2e}")
    assert max_err < 1e-8
    print("    PASS\n")


def test_rotate_Q_group_closure_continuous():
    print("[4] rotate_Q_analytic satisfies group closure at arbitrary continuous angles "
          "(rotating by alpha then beta == rotating by alpha+beta)...")
    rng = np.random.default_rng(3)
    q0 = rng.uniform(-1, 1, 2)
    max_err = 0.0
    for _ in range(50):
        alpha, beta = rng.uniform(0, 2 * np.pi, 2)
        q_step = rotate_Q_analytic(*rotate_Q_analytic(q0[0], q0[1], alpha), beta)
        q_direct = rotate_Q_analytic(q0[0], q0[1], alpha + beta)
        err = np.linalg.norm(np.array(q_step) - np.array(q_direct))
        max_err = max(max_err, err)
    print(f"    max closure error over 50 random continuous angle pairs = {max_err:.2e}")
    assert max_err < 1e-8
    print("    PASS\n")


def test_rotate_Q_period_pi_for_doubled_angle():
    print("[5] rotate_Q_analytic: rotating by pi returns to the SAME (q1,q2) "
          "(since Q has period pi in the underlying angle, i.e. frequency 2)...")
    rng = np.random.default_rng(4)
    q0 = rng.uniform(-1, 1, 2)
    q_after_pi = np.array(rotate_Q_analytic(q0[0], q0[1], np.pi))
    err = np.linalg.norm(q_after_pi - q0)
    print(f"    |Q(rotate by pi) - Q(original)| = {err:.2e}")
    assert err < 1e-10
    print("    PASS\n")


def test_p4m_irrep_is_degenerate_1d():
    print("[6] sanity check: D4 (p4m) irrep(1,2) is only 1-DIMENSIONAL "
          "(confirms p4m cannot represent a genuine nematic 2-vector)...")
    from escnn import gspaces
    g4 = gspaces.flipRot2dOnR2(N=4)
    group4 = g4.fibergroup
    irr4 = group4.irrep(1, 2)
    print(f"    D4 irrep(1,2).size = {irr4.size} (expected 1, i.e. degenerate)")
    assert irr4.size == 1
    print("    PASS -- this is a REAL limitation of p4m, not a bug\n")


def test_flip_matches_escnn():
    print("[7] reflection behavior: (q1, q2) -> (q1, -q2) under the base flip, "
          "matches escnn's D8 irrep(1,2) flip matrix...")
    from escnn import gspaces
    g8 = gspaces.flipRot2dOnR2(N=8)
    group = g8.fibergroup
    irr = group.irrep(1, 2)
    flip_matrix = irr(group.element((1, 0)))
    expected = np.array([[1.0, 0.0], [0.0, -1.0]])
    err = np.max(np.abs(flip_matrix - expected))
    print(f"    max diff vs diag(1,-1) = {err:.2e}")
    assert err < 1e-8
    print("    PASS\n")


if __name__ == "__main__":
    test_phi_S_q_roundtrip()
    test_head_tail_symmetry()
    test_rotate_Q_matches_escnn_p8m_irrep()
    test_rotate_Q_group_closure_continuous()
    test_rotate_Q_period_pi_for_doubled_angle()
    test_p4m_irrep_is_degenerate_1d()
    test_flip_matches_escnn()
    print("ALL TESTS PASSED")
