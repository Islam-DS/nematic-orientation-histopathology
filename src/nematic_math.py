"""
v2 core math: explicit physically-typed nematic (director) representation.

WHY SECOND HARMONIC / WHY MOD PI:
A nematic director describes an axis of local anisotropy (e.g. the long
axis of an elongated gland) with NO preferred head/tail -- phi and phi+pi
describe the identical physical axis. A representation that is well
defined on this head-tail-symmetric space must be invariant under
phi -> phi + pi. Mapping phi to the point (cos(2*phi), sin(2*phi)) on the
unit circle achieves exactly this: doubling the angle before taking
cos/sin makes phi and phi+pi map to the SAME point, so the representation
is automatically head-tail symmetric, and it is smooth (no wraparound
discontinuity) everywhere except at S=0 (isotropic point, where phi is
undefined anyway). This is also why phi must never be regressed directly
with ordinary MSE: MSE on a periodic (mod pi) quantity is discontinuous at
the wrap point (e.g. predicting 179 deg for a target of 1 deg is a near-
perfect answer physically, but MSE on the raw angle scores it as a huge
error). Regressing (q1, q2) = S*(cos 2phi, sin 2phi) with ordinary
Euclidean MSE sidesteps this entirely.

GROUP-THEORETIC GROUNDING (verified against installed escnn 1.0.11, see
audit notes in results_v2/preregistration/):
For the p8m group (escnn: gspaces.flipRot2dOnR2(N=8), fiber group D8,
order 16), the irreducible representation group.irrep(1, 2) is a genuine
2-dimensional representation whose matrix for a rotation by alpha is
exactly the 2D rotation-by-2*alpha matrix R(2*alpha), and whose matrix for
the base reflection is diag(1, -1). This was confirmed numerically
(see test_nematic_math.py) directly against escnn's own representation
matrices -- not assumed from memory. This representation is therefore
EXACTLY the correct output type for a rotation+reflection-equivariant
nematic field: Q(R_alpha x) = R(2*alpha) Q(x).

By contrast, for p4m (escnn: flipRot2dOnR2(N=4), fiber group D4, order 8),
group.irrep(1, 2) is only 1-DIMENSIONAL (values +-1: rotation by 90 deg
maps to -1, by 180 deg to +1). D4's rotation subgroup has only 4 elements,
so frequency 2 sits exactly at the Nyquist limit and the representation
degenerates -- there is no way to build a genuine (q1, q2) 2D director
output for p4m. This is a rigorous, group-theoretic explanation of why the
original pilot's post-hoc Q extraction from p4m regular-representation
channels could only ever recover a binary (0 deg / 90 deg) director, which
is exactly what was observed empirically in the original pilot's rotation
sanity check.
"""
import numpy as np
import torch


def phi_S_to_q(phi, S):
    """(phi radians, S in [0,1]) -> (q1, q2) = S*(cos 2phi, sin 2phi)."""
    q1 = S * np.cos(2 * phi)
    q2 = S * np.sin(2 * phi)
    return q1, q2


def q_to_phi_S(q1, q2, eps=1e-8):
    """(q1, q2) -> (phi mod pi, S). Inverse of phi_S_to_q."""
    S = np.sqrt(q1 ** 2 + q2 ** 2)
    phi = 0.5 * np.arctan2(q2, q1)
    phi = np.mod(phi, np.pi)
    return phi, S


def rotate_Q_analytic(q1, q2, alpha):
    """
    Ground-truth transformation law for the nematic representation under a
    rotation of the underlying image by `alpha` (radians):
        Q(R_alpha x) = R(2*alpha) Q(x)
    i.e. (q1, q2) rotate by TWICE the image rotation angle. This is the
    defining property of the frequency-2 (second-harmonic) representation
    and is what group.irrep(1, 2) of D8 implements (verified in
    test_nematic_math.py). Used both to generate synthetic ground truth
    for rotated images and to compute the numerical equivariance error of
    a trained model.
    """
    c, s = np.cos(2 * alpha), np.sin(2 * alpha)
    q1_rot = c * q1 - s * q2
    q2_rot = s * q1 + c * q2
    return q1_rot, q2_rot


def rotate_Q_analytic_torch(q, alpha):
    """
    Same as rotate_Q_analytic but for a torch tensor q of shape (..., 2)
    and a scalar or (...,) tensor alpha (radians). Used inside the
    numerical equivariance-error check on model predictions.
    """
    c, s = torch.cos(2 * alpha), torch.sin(2 * alpha)
    q1, q2 = q[..., 0], q[..., 1]
    q1_rot = c * q1 - s * q2
    q2_rot = s * q1 + c * q2
    return torch.stack([q1_rot, q2_rot], dim=-1)


def angular_error_mod_pi(phi_pred, phi_true):
    """Smallest difference between two angles taken mod pi, in radians."""
    d = np.abs(phi_pred - phi_true) % np.pi
    d = np.minimum(d, np.pi - d)
    return d


def normalized_equivariance_error(q_transformed_input, q_expected, eps=1e-8):
    """
    ||model(rotated_x) - R(2alpha) . model(x)|| / ||R(2alpha) . model(x)||
    averaged over the batch. This is 0 for a perfectly equivariant model
    and O(1) for a model with no rotation structure at all.
    q_transformed_input, q_expected: (N, 2) numpy arrays.
    """
    num = np.linalg.norm(q_transformed_input - q_expected, axis=-1)
    den = np.linalg.norm(q_expected, axis=-1) + eps
    return np.mean(num / den)
