"""
Target-generation utilities only (no training). Builds an "annotation-
derived anatomical orientation target" (NOT a "ground truth tissue
orientation field" -- see docs_v2/phase2_target_spec.md) from GlaS gland
instance-segmentation masks.

See docs_v2/phase2_target_math.md for the full mathematical derivation and
docs_v2/phase2_target_spec.md for the design decisions (aggregation rule,
exclusion rules, edge cases) this module implements.
"""
import sys
import os

import numpy as np
import cv2
from skimage.measure import label as cc_label

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
from data_prep import extract_patches  # noqa: E402  -- reuse ONLY, not modified

from nematic_math import phi_S_to_q, rotate_Q_analytic  # noqa: E402

MIN_GLAND_PIXELS = 20          # below this, PCA is numerically unreliable (near-degenerate covariance)
BORDER_FRAC_THRESHOLD = 0.02   # fraction of a gland's own pixels lying on the image border
                                # (data-driven choice: see docs_v2/phase2_target_spec.md section 2 --
                                # 65% of instances touch the border at all, but only ~22% exceed this
                                # fractional threshold; a binary "touches at all" rule was rejected as
                                # excluding an unreasonably large majority of the dataset for often-mild truncation)
DEFAULT_S_THRESHOLD = 0.15     # near-isotropic exclusion bar (see spec doc for derivation)


def gland_covariance_pca(rows, cols):
    """
    rows, cols: 1D arrays of a single gland's pixel coordinates (in the
    FULL, unpatched image). Returns (phi_cov, S_cov, lambda1, lambda2).
    x = col, y = row (standard math convention, verified in
    docs_v2/phase2_target_math.md section 1).
    """
    x = cols.astype(np.float64)
    y = rows.astype(np.float64)
    coords = np.stack([x, y], axis=0)
    C = np.cov(coords)
    if C.ndim == 0:  # single pixel: np.cov degenerates
        C = np.zeros((2, 2))
    eigvals, eigvecs = np.linalg.eigh(C)  # ascending order
    lambda2, lambda1 = eigvals[0], eigvals[1]
    v1 = eigvecs[:, 1]
    phi_cov = np.mod(np.arctan2(v1[1], v1[0]), np.pi)
    S_cov = (lambda1 - lambda2) / (lambda1 + lambda2 + 1e-8)
    return float(phi_cov), float(np.clip(S_cov, 0, 1)), float(lambda1), float(lambda2)


def gland_contour_ellipse(mask):
    """
    mask: 2D boolean array, single gland (full image or a tight crop -- the
    fit only depends on the mask's own boundary contour). Returns
    (phi_contour, S_contour) or (None, None) if a contour/ellipse could not
    be fit (e.g. too few boundary points).

    Angle convention (verified empirically, see module docstring history /
    docs_v2/phase2_target_math.md): cv2.fitEllipse's returned angle `theta`
    satisfies theta = (phi + 90 deg) mod 180 deg in our (x=col, y=row)
    convention, so phi_contour = (theta - 90 deg) mod pi.
    Axis-length convention: cv2.fitEllipse's returned (MA, ma) tuple is
    NOT (major, minor) despite the names -- verified empirically that the
    FIRST value is the smaller (minor) axis and the SECOND is the larger
    (major) axis for this OpenCV version. We sort explicitly rather than
    trust the names.
    """
    mask_u8 = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, None
    c = max(contours, key=cv2.contourArea)
    if len(c) < 5:  # cv2.fitEllipse requires >= 5 points
        return None, None
    (_, _), (ax0, ax1), theta_deg = cv2.fitEllipse(c)
    minor, major = sorted([ax0, ax1])  # do not trust the (MA, ma) naming -- verified empirically
    if major + minor < 1e-6:
        return None, None
    S_contour = (major - minor) / (major + minor + 1e-8)
    phi_contour = np.mod(np.deg2rad(theta_deg - 90.0), np.pi)
    return float(phi_contour), float(np.clip(S_contour, 0, 1))


def analyze_glands(labeled_mask, min_pixels=MIN_GLAND_PIXELS,
                    border_frac_threshold=BORDER_FRAC_THRESHOLD):
    """
    labeled_mask: 2D int array, 0=background, 1..N=gland instance ids
    (full, unpatched image -- Addendum 1 compliance).

    Returns a list of per-gland dicts with both anisotropy measures,
    border-truncation info, and validity flags. No pixels are assigned an
    invented angle here -- this only computes properties; assignment /
    exclusion happens in build_full_image_field.
    """
    H, W = labeled_mask.shape
    ids = np.unique(labeled_mask)
    ids = ids[ids > 0]

    results = []
    for gid in ids:
        mask = labeled_mask == gid
        rows, cols = np.nonzero(mask)
        npix = len(rows)

        border_pixels = (mask[0, :].sum() + mask[-1, :].sum() +
                          mask[:, 0].sum() + mask[:, -1].sum())
        border_frac = border_pixels / npix if npix > 0 else 1.0
        border_truncated = border_frac >= border_frac_threshold

        too_small = npix < min_pixels

        # fragmentation: an instance id whose binary mask has more than one
        # connected component (same method as Stage 2/3's dataset-level check,
        # applied here per-instance). INFORMATIONAL low-confidence flag only --
        # per Stage 4 instruction, this does NOT by itself cause exclusion
        # (no new hard rule invented for it); it is tracked and reported.
        cc = cc_label(mask)
        n_components = int(cc.max())
        fragmented = n_components > 1

        entry = {
            "gland_id": int(gid), "n_pixels": int(npix),
            "border_pixels": int(border_pixels), "border_frac": float(border_frac),
            "border_truncated": bool(border_truncated),
            "too_small": bool(too_small),
            "n_connected_components": n_components,
            "fragmented": bool(fragmented),
        }

        if too_small:
            entry.update(phi_cov=None, S_cov=None, phi_contour=None, S_contour=None,
                          near_isotropic=None, numerically_unstable=False, valid=False)
            results.append(entry)
            continue

        phi_cov, S_cov, lambda1, lambda2 = gland_covariance_pca(rows, cols)
        phi_contour, S_contour = gland_contour_ellipse(mask)

        # numerical-stability check: NaN/Inf in the eigendecomposition-derived
        # quantities (can arise from near-degenerate, near-collinear pixel sets)
        numerically_unstable = bool(
            not np.isfinite(phi_cov) or not np.isfinite(S_cov) or
            not np.isfinite(lambda1) or not np.isfinite(lambda2)
        )
        near_isotropic = bool(np.isfinite(S_cov) and S_cov < DEFAULT_S_THRESHOLD) if S_cov is not None else None

        entry.update(
            phi_cov=phi_cov, S_cov=S_cov, lambda1=lambda1, lambda2=lambda2,
            phi_contour=phi_contour, S_contour=S_contour,
            near_isotropic=near_isotropic, numerically_unstable=numerically_unstable,
        )
        results.append(entry)

    return results


def build_full_image_field(labeled_mask, gland_props, S_choice="cov",
                            S_threshold=DEFAULT_S_THRESHOLD, exclude_border_truncated=True):
    """
    Dense PIXEL-LEVEL target field at full image resolution. Each pixel is
    assigned the (q1, q2) of the gland it belongs to ONLY if that gland
    passes all confidence rules; otherwise the pixel is marked invalid
    (q1=q2=0, valid=False) -- never assigned an invented angle. Background
    pixels are always invalid. See docs_v2/phase2_target_spec.md for why
    per-pixel exact-label assignment was chosen over distance-weighted /
    kernel-smoothed alternatives.

    S_choice: "cov" or "contour" -- which anisotropy measure gates validity
    and scales the target. The two are compared, not combined, per the
    "choose one before training" requirement.
    """
    H, W = labeled_mask.shape
    q1_map = np.zeros((H, W), dtype=np.float32)
    q2_map = np.zeros((H, W), dtype=np.float32)
    S_map = np.zeros((H, W), dtype=np.float32)
    phi_map = np.zeros((H, W), dtype=np.float32)
    valid_map = np.zeros((H, W), dtype=bool)

    for entry in gland_props:
        if entry["too_small"]:
            continue
        if exclude_border_truncated and entry["border_truncated"]:
            continue
        phi = entry[f"phi_{S_choice}"]
        S = entry[f"S_{S_choice}"]
        if phi is None or S is None:
            continue
        if S < S_threshold:
            continue

        q1, q2 = phi_S_to_q(phi, S)
        mask = labeled_mask == entry["gland_id"]
        q1_map[mask] = q1
        q2_map[mask] = q2
        S_map[mask] = S
        phi_map[mask] = phi
        valid_map[mask] = True

    return {"q1": q1_map, "q2": q2_map, "S": S_map, "phi": phi_map, "valid": valid_map}


def crop_field_to_patches(field, image_shape, patch_size=128, stride=96):
    """
    Crops a full-image field dict (as returned by build_full_image_field)
    into the SAME patch grid used by code/data_prep.py::extract_patches, so
    patch indices line up 1:1 with the RGB image patches produced by the
    reused (unmodified) loader. Returns a list of per-patch dicts (same
    keys as `field`, cropped) plus the (y, x) top-left offset of each patch.
    """
    H, W = image_shape[:2]
    patches = []
    for y in range(0, max(1, H - patch_size + 1), stride):
        for x in range(0, max(1, W - patch_size + 1), stride):
            if y + patch_size > H or x + patch_size > W:
                continue
            crop = {k: v[y:y + patch_size, x:x + patch_size] for k, v in field.items()}
            patches.append({"offset": (y, x), **crop})
    return patches
