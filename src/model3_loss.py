"""
STAGE 7: Model 3 loss. Implementation only -- not used for training in
this stage (no training occurs in Stage 7).

L_total = L_class + lambda_Q * L_Q

- L_class: standard cross-entropy on the per-patch benign/malignant label
  (reuses the existing, unmodified GlaS grade-label parsing from
  code/data_prep.py -- not re-derived here).
- L_Q: masked MSE between predicted (q1,q2) and the Stage 4 anatomical
  target's (q1,q2), evaluated ONLY at locations the Stage 4 pipeline
  marked valid (docs_v2/phase2_target_spec.md's existing, UNMODIFIED
  border/size/anisotropy rules). Low-confidence/invalid locations
  contribute nothing to L_Q -- not zero-filled-and-averaged-in, which
  would silently treat "no information" as "target is exactly (0,0)".
- lambda_Q = 1.0, a FIXED, explicitly documented placeholder coefficient.
  Not tuned, not pathology-dependent, not test-set-dependent (Stage 7
  instruction #8). Any future change to this value would itself need to
  be documented and justified independently of any observed result.

RESOLUTION ALIGNMENT (documented per Stage 7 instruction #9): Model 3's Q
head outputs at (H/2, W/2) -- the shared representation's spatial
resolution after block1's one stride-2 pool (see model3_typed_equivariant.py).
The Stage 4 anatomical target field is at native (H, W) patch resolution.
To compare them, the target is downsampled to (H/2, W/2) via non-overlapping
2x2 average pooling of q1/q2, and the corresponding validity mask cell is
marked valid ONLY IF ALL 4 of its constituent native-resolution pixels were
valid (a conservative "all-4-valid" rule -- chosen specifically to avoid
blending a valid gland-interior pixel with an adjacent invalid/background
pixel into an ambiguous, partially-fabricated downsampled target value).
This is a data-alignment convention for the loss only; it does NOT modify
the Stage 4 target files or validity rules themselves.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

LAMBDA_Q = 1.0  # fixed, documented, not tuned -- see module docstring


def downsample_target_for_loss(q1, q2, valid, factor=2):
    """
    q1, q2, valid: (B, H, W) tensors (or numpy-derived torch tensors) at
    native patch resolution. Returns (q1_ds, q2_ds, valid_ds) at
    (H//factor, W//factor), using the all-4-valid conservative rule
    documented in the module docstring.
    """
    B, H, W = q1.shape
    assert H % factor == 0 and W % factor == 0, "patch dimensions must be divisible by the downsample factor"

    q1_ds = F.avg_pool2d(q1.unsqueeze(1), kernel_size=factor, stride=factor).squeeze(1)
    q2_ds = F.avg_pool2d(q2.unsqueeze(1), kernel_size=factor, stride=factor).squeeze(1)

    valid_f = valid.float().unsqueeze(1)
    valid_count = F.avg_pool2d(valid_f, kernel_size=factor, stride=factor).squeeze(1)  # fraction valid in each block
    valid_ds = valid_count >= 0.999  # "all 4 valid" (allowing for float rounding)

    return q1_ds, q2_ds, valid_ds


def masked_q_loss(q_pred, q1_target_ds, q2_target_ds, valid_ds, eps=1e-8):
    """
    q_pred: (B, 2, H2, W2) model output (q1, q2).
    q1_target_ds, q2_target_ds, valid_ds: (B, H2, W2), already resolution-matched.

    Returns a scalar loss: mean squared Euclidean (q1,q2) error over valid
    locations only, averaged first within each patch (so patches with more
    valid pixels don't dominate purely by pixel count) then across the batch.
    Patches with ZERO valid pixels contribute 0 to the batch loss and are
    excluded from the batch average (documented, not silently averaged in
    as if they were perfect predictions).
    """
    q1_pred, q2_pred = q_pred[:, 0], q_pred[:, 1]
    sq_err = (q1_pred - q1_target_ds) ** 2 + (q2_pred - q2_target_ds) ** 2  # (B, H2, W2)

    valid_f = valid_ds.float()
    per_patch_valid_count = valid_f.sum(dim=(1, 2))  # (B,)
    per_patch_sum_err = (sq_err * valid_f).sum(dim=(1, 2))  # (B,)

    has_valid = per_patch_valid_count > 0
    per_patch_mean_err = torch.zeros_like(per_patch_sum_err)
    per_patch_mean_err[has_valid] = per_patch_sum_err[has_valid] / (per_patch_valid_count[has_valid] + eps)

    if has_valid.sum() == 0:
        return torch.tensor(0.0, device=q_pred.device, requires_grad=q_pred.requires_grad)
    return per_patch_mean_err[has_valid].mean()


def model3_total_loss(logits, labels, q_pred, q1_target, q2_target, valid_target, lambda_Q=LAMBDA_Q):
    """
    logits: (B, n_classes), labels: (B,) long.
    q_pred: (B, 2, H2, W2).
    q1_target, q2_target, valid_target: (B, H, W) at NATIVE patch resolution
    (as produced directly by the Stage 4 field files) -- downsampled inside
    this function via downsample_target_for_loss, matching q_pred's H2, W2.

    Returns (total_loss, dict of individual components, separately logged
    per Stage 7 instruction #8).
    """
    L_class = F.cross_entropy(logits, labels)

    factor = q1_target.shape[-1] // q_pred.shape[-1]
    q1_ds, q2_ds, valid_ds = downsample_target_for_loss(q1_target, q2_target, valid_target, factor=factor)
    L_Q = masked_q_loss(q_pred, q1_ds, q2_ds, valid_ds)

    total = L_class + lambda_Q * L_Q
    return total, {"L_class": L_class.item(), "L_Q": L_Q.item(), "lambda_Q": lambda_Q, "L_total": total.item()}
