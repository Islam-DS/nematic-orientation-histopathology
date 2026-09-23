"""
Deterministic (non-learned) orientation estimators for Phase 1R, used both
as the rendering-noise-floor oracle and as scientific baselines B and C.

C. structure_tensor_estimate mirrors the verified formula from
   code/math_utils.py::classical_structure_tensor_order (same row/col axis
   handling, same fiber/gradient-vs-structure convention -- re-derived
   here rather than imported, to keep code_v2 self-contained per the
   "isolated v2 structure" requirement; the formula itself is unchanged).

B. naive_gradient_estimate is a DELIBERATELY WEAK baseline: it averages
   the raw (frequency-1) gradient VECTOR directly, rather than the
   structure tensor's squared/outer-product (frequency-2) quantities. For
   a head-tail-symmetric object (e.g. a symmetric ellipse), opposite-side
   gradient vectors point in opposite directions and cancel in a plain
   vector average, making this estimator unstable/near-degenerate by
   construction. This is included specifically to demonstrate WHY a
   second-harmonic formulation (structure tensor, or the learned Q head)
   is necessary at all -- not because anyone would seriously use it.
"""
import numpy as np
from skimage.feature import structure_tensor
from skimage.filters import sobel_h, sobel_v


def structure_tensor_estimate(patch_gray, sigma=2.0, eps=1e-8):
    """Same formula as code/math_utils.py::classical_structure_tensor_order."""
    patch_gray = np.asarray(patch_gray, dtype=np.float64)
    Arr, Arc, Acc = structure_tensor(patch_gray, sigma=sigma, order='rc')
    cy, cx = Arr.shape[0] // 2, Arr.shape[1] // 2
    Arr_c, Arc_c, Acc_c = Arr[cy, cx], Arc[cy, cx], Acc[cy, cx]
    Sxx, Syy, Sxy = Acc_c, Arr_c, Arc_c
    S = np.sqrt((Sxx - Syy) ** 2 + 4 * Sxy ** 2) / (Sxx + Syy + eps)
    phi = 0.5 * np.arctan2(-2 * Sxy, Syy - Sxx)
    phi = np.mod(phi, np.pi)
    return S, phi


def naive_gradient_estimate(patch_gray, eps=1e-8):
    """
    Deliberately weak baseline B: plain vector-average of Sobel gradients
    (frequency-1), folded into a mod-pi angle only at the end (there is no
    principled reason this folding should recover the true director -- see
    module docstring). Expected to perform poorly for symmetric objects.
    """
    patch_gray = np.asarray(patch_gray, dtype=np.float64)
    gx = sobel_v(patch_gray)  # d/dcol
    gy = sobel_h(patch_gray)  # d/drow
    mean_gx, mean_gy = gx.mean(), gy.mean()
    S_proxy = np.sqrt(mean_gx ** 2 + mean_gy ** 2) / (np.mean(np.sqrt(gx ** 2 + gy ** 2)) + eps)
    phi = np.mod(np.arctan2(mean_gy, mean_gx), np.pi)
    return float(np.clip(S_proxy, 0, 1)), float(phi)
