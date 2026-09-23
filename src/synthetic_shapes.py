"""
Synthetic elongated-object generator for Phase 1 benchmark, with EXACT
ground-truth nematic (q1, q2) by construction (no estimation involved).

S is defined as the eccentricity-like anisotropy measure
    S = (major_axis - minor_axis) / (major_axis + minor_axis)
of the generating ellipse -- a standard bounded-in-[0,1) anisotropy
measure. This is a controlled synthetic parameter, not something we
recover from the image: we CHOOSE (phi, S), compute the implied ellipse
semi-axes, render the ellipse, and the (phi, S) we chose is the exact
ground truth for that image.
"""
import numpy as np
from skimage.draw import ellipse as sk_ellipse

from nematic_math import phi_S_to_q


def semi_axes_from_S(S, area=None, mean_radius=18.0):
    """
    Given S in [0, 1), return (a, b) semi-axes with (a-b)/(a+b) = S and a
    fixed characteristic size (mean_radius = (a+b)/2 by default, optionally
    rescaled to hit a target area).
    """
    a = mean_radius * (1 + S)
    b = mean_radius * (1 - S)
    if area is not None:
        cur_area = np.pi * a * b
        scale = np.sqrt(area / cur_area)
        a, b = a * scale, b * scale
    return a, b


def render_ellipse(phi, S, size=64, mean_radius=16.0, noise_std=0.0, rng=None):
    """
    Render a single filled ellipse, long axis at angle `phi` (radians,
    standard math convention, counterclockwise from x-axis), anisotropy S.
    Returns a (size, size) float32 image in [0, 1] (grayscale; caller can
    stack to 3ch if needed) and the exact (q1, q2) ground truth.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    img = np.zeros((size, size), dtype=np.float32)
    a, b = semi_axes_from_S(S, mean_radius=mean_radius)
    cy, cx = size / 2.0, size / 2.0

    # skimage.draw.ellipse rotation parameter is measured clockwise-as-
    # displayed (row axis points down) -- same convention verified
    # empirically for skimage.transform.rotate in code/test_math.py.
    # We invert here so `phi` passed in matches the standard math
    # (counterclockwise, x-axis) convention used throughout nematic_math.py.
    rr, cc = sk_ellipse(cy, cx, b, a, shape=img.shape, rotation=-phi)
    img[rr, cc] = 1.0

    if noise_std > 0:
        img = img + rng.normal(0, noise_std, img.shape).astype(np.float32)
        img = np.clip(img, 0, 1)

    q1, q2 = phi_S_to_q(phi, S)
    return img, (q1, q2)


def make_dataset(phis, S_values, size=64, mean_radius=16.0, noise_std=0.02, seed=0, repeats=1):
    """
    Cartesian product of `phis` (radians) x `S_values`, `repeats` noisy
    draws each. Returns (images (N,1,size,size) float32, q (N,2) float32,
    phi (N,) float32, S (N,) float32).
    """
    rng = np.random.default_rng(seed)
    imgs, qs, phis_out, Ss_out = [], [], [], []
    for phi in phis:
        for S in S_values:
            for _ in range(repeats):
                img, (q1, q2) = render_ellipse(phi, S, size=size, mean_radius=mean_radius,
                                                noise_std=noise_std, rng=rng)
                imgs.append(img[None, :, :])
                qs.append([q1, q2])
                phis_out.append(phi)
                Ss_out.append(S)
    return (np.stack(imgs).astype(np.float32),
            np.array(qs, dtype=np.float32),
            np.array(phis_out, dtype=np.float32),
            np.array(Ss_out, dtype=np.float32))
