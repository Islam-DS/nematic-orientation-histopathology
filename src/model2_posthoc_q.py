"""
STAGE 9: torch-differentiable port of the post-hoc Q construction used for
Model 2 (code/model.py::P4MOrientationNet), so Model 2 can be trained
end-to-end against the same anatomical Q targets as the other baselines.

code/math_utils.py::Q_from_orientation_response (numpy) is NOT modified;
this reimplements the identical formula in torch for backprop, using the
SAME convention already established there and in the original pilot:
filter index 0 (unless a principled reason to average -- none introduced
here), theta_k = rot_idx * 90 deg (from model.orientation_angles()).

This is exactly the "POST-HOC / UN-TYPED" construction the master prompt
requires Model 2 to keep using: r_k -> theta_k is an ARBITRARY, assumed
mapping (which is why the channel-permutation null test, applied later in
Stage 9, is meaningful for this model and NOT for Model 3).
"""
import numpy as np
import torch

FILTER_INDEX = 0  # established convention, reused unchanged from the original pilot


def posthoc_q_dense(orientation_features, thetas, eps=1e-8):
    """
    orientation_features: (B, c2, 8, H2, W2) raw regular-rep channels
      (Model 2's return_features=True output).
    thetas: (8,) numpy array of angles in radians (model.orientation_angles()).
    Returns q_dense: (B, 2, H2, W2) -- (q1, q2) via the second-harmonic
    projection, differentiable w.r.t. orientation_features.
    """
    r = orientation_features[:, FILTER_INDEX]  # (B, 8, H2, W2)
    r = torch.clamp(r, min=0)  # post-ReLU features should already be >=0; guard anyway (same as code/experiments.py)

    theta_t = torch.as_tensor(thetas, dtype=torch.float32, device=r.device).view(1, 8, 1, 1)
    cos_2t = torch.cos(2 * theta_t)
    sin_2t = torch.sin(2 * theta_t)

    denom = r.sum(dim=1, keepdim=True) + eps  # (B, 1, H2, W2)
    q1 = (r * cos_2t).sum(dim=1, keepdim=True) / denom
    q2 = (r * sin_2t).sum(dim=1, keepdim=True) / denom
    return torch.cat([q1, q2], dim=1)  # (B, 2, H2, W2)


def permute_channel_angle_mapping(orientation_features, perm_indices):
    """
    Returns a copy of orientation_features with its 8 channels permuted
    according to perm_indices (a permutation of range(8)) -- used only by
    the Model 2 channel-permutation null test (Stage 9), never during
    ordinary training/inference.
    """
    return orientation_features[:, :, perm_indices, :, :]
