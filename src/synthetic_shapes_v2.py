"""
Phase 1R rendering pipeline. Fixes the confound identified in Phase 1:
skimage.transform.rotate on an already-rasterized hard-binary image
introduces bilinear-interpolation blur that is near-zero at 90-degree
multiples and non-zero elsewhere -- i.e. blur was CORRELATED with test
angle, and a near-perfectly-fit model reacted to the blur, not to genuine
non-equivariance.

Fix: every image (whatever its orientation) goes through the IDENTICAL
render pipeline -- supersample -> draw analytic ellipse -> anti-aliased
downsample -> optional blur/contrast/noise. No image is ever produced by
pixel-domain rotation of another raster; a "rotated" sample is generated
by calling the SAME renderer with phi -> phi + alpha (all other object
parameters held fixed). This makes angle just another rendering parameter,
not a source of differential image-quality artifacts.

We additionally keep the OLD (pixel-domain rotate) path available, purely
so Phase 1R can quantify how much worse it is than the fix (see
run_phase1r_benchmark.py's rendering-floor calibration) -- direct evidence
that the Phase 1 failure was a measurement artifact, not reused to produce
any reported model metric.
"""
import numpy as np
from skimage.draw import ellipse as sk_ellipse
from skimage.transform import resize as sk_resize
from skimage.transform import rotate as sk_rotate
from skimage.filters import gaussian as sk_gaussian

from nematic_math import phi_S_to_q

SUPERSAMPLE = 4


def render_object(phi, S, size=128, mean_radius=28.0, dx=0.0, dy=0.0,
                   contrast=1.0, brightness=0.0, blur_sigma=0.0, noise_std=0.0,
                   supersample=SUPERSAMPLE, rng=None):
    """
    Render one anisotropic object. ALL orientations go through this exact
    same procedure (supersample -> analytic draw -> anti-aliased downsample
    -> photometric/noise perturbation) -- angle is just a parameter, never
    special-cased.

    phi: orientation, radians, standard math convention (see verification
         in synthetic_shapes.py / Phase 1 for the sk_ellipse rotation-sign
         convention, reused identically here).
    S: anisotropy in [0, 1), defines semi-axes exactly as in Phase 1.
    dx, dy: object center offset from image center, in FINAL-resolution
         pixel units.
    contrast, brightness: img = clip(img*contrast + brightness, 0, 1)
    blur_sigma: additional Gaussian blur (final-resolution pixel units)
         applied on top of the anti-aliased downsampling.
    noise_std: i.i.d. Gaussian pixel noise added after all of the above.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    hi = size * supersample
    img_hi = np.zeros((hi, hi), dtype=np.float32)

    a = mean_radius * (1 + S) * supersample
    b = mean_radius * (1 - S) * supersample
    cy = hi / 2.0 + dy * supersample
    cx = hi / 2.0 + dx * supersample

    # rotation sign convention verified in synthetic_shapes.py (Phase 1):
    # sk_ellipse(rotation=R) gives measured angle = -R mod pi, so pass -phi
    # to get the standard math (ccw, x-axis) convention used throughout.
    rr, cc = sk_ellipse(cy, cx, b, a, shape=(hi, hi), rotation=-phi)
    valid = (rr >= 0) & (rr < hi) & (cc >= 0) & (cc < hi)
    img_hi[rr[valid], cc[valid]] = 1.0

    # anti-aliased downsample (this IS the anti-aliasing step -- avoids
    # ever having a "clean binary" raster at the working resolution)
    img = sk_resize(img_hi, (size, size), anti_aliasing=True, order=1)

    if blur_sigma > 0:
        img = sk_gaussian(img, sigma=blur_sigma)

    img = img * contrast + brightness
    img = np.clip(img, 0, 1)

    if noise_std > 0:
        img = img + rng.normal(0, noise_std, img.shape).astype(np.float32)
        img = np.clip(img, 0, 1)

    q1, q2 = phi_S_to_q(phi, S)
    return img.astype(np.float32), (q1, q2)


def sample_object_params(rng, size=128,
                          mean_radius_range=(18.0, 34.0),
                          dx_dy_range=(-14.0, 14.0),
                          contrast_range=(0.6, 1.0),
                          brightness_range=(0.0, 0.15),
                          blur_sigma_range=(0.0, 1.2),
                          noise_std_range=(0.0, 0.04)):
    """
    Randomize the nuisance parameters (everything except phi, S, which the
    caller controls explicitly). Ranges chosen to be visibly perturbed but
    keep the object mostly within frame for a 128x128 canvas (max
    mean_radius*2 ~ 68px plus offset up to 14px stays within bounds) --
    physically: variation in object size/contrast/blur/noise/position
    analogous to real imaging variability, documented here rather than
    tuned per-experiment.
    """
    return dict(
        mean_radius=rng.uniform(*mean_radius_range),
        dx=rng.uniform(*dx_dy_range),
        dy=rng.uniform(*dx_dy_range),
        contrast=rng.uniform(*contrast_range),
        brightness=rng.uniform(*brightness_range),
        blur_sigma=rng.uniform(*blur_sigma_range),
        noise_std=rng.uniform(*noise_std_range),
    )


def pixel_domain_rotate(img, alpha_deg):
    """
    OLD (Phase 1) rotation method: rotate an already-rasterized image via
    interpolation. Kept only for the Phase 1R rendering-floor calibration,
    to quantify how much worse this is than re-rendering (see
    run_phase1r_benchmark.py). Sign convention (positive angle = same
    rotation sense the model/escnn use internally) verified in Phase 1.
    """
    return sk_rotate(img, angle=alpha_deg, mode='reflect', order=1).astype(np.float32)
