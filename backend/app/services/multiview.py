"""Multi-view image generation for manuscript OCR.

Produces a small, bounded set of image variants from one crop so that
the recognition engine can be tried on the variant most likely to yield
a better reading.  Zero extra API calls during variant *generation* — the
caller decides which (if any) variants to send downstream.

Variants
--------
1. **enhanced_rgb** — CLAHE-on-L* + bilateral + unsharp (the current default)
2. **highcontrast_gray** — aggressive CLAHE on grayscale + strong unsharp
3. **binarized** — Sauvola-binarized back to RGB for the vision model

The module also exposes a lightweight image-quality estimator so the
caller can rank variants without any OCR call.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from PIL import Image, ImageFilter

_CLAHE_HC_CLIP = 5.0      # aggressive CLAHE for high-contrast variant
_CLAHE_HC_GRID = 8
_SHARPEN_HC_RADIUS = 2
_SHARPEN_HC_PERCENT = 180
_SHARPEN_HC_THRESH = 2
_SAUVOLA_WINDOW = 25
_SAUVOLA_K = 0.2


@dataclass(frozen=True)
class ImageVariant:
    """One image rendering of an OCR crop."""
    label: str              # human-readable name for logging
    image: Image.Image      # PIL RGB image
    quality_score: float    # 0-1 estimated sharpness / contrast quality


def generate_variants(image: Image.Image) -> list[ImageVariant]:
    """Generate <=3 image variants from *image* (must be RGB).

    Returns a list sorted by *quality_score* descending (best first).
    """
    variants: list[ImageVariant] = []

    # Variant 1: the input itself (assumed already CLAHE-enhanced)
    variants.append(ImageVariant(
        label="enhanced_rgb",
        image=image,
        quality_score=_image_quality_score(image),
    ))

    try:
        import cv2
    except ImportError:
        return variants   # can't build other variants without OpenCV

    arr = np.array(image.convert("RGB"), dtype=np.uint8)

    # Variant 2: high-contrast grayscale (back in RGB for the vision model)
    gray = np.array(image.convert("L"), dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=_CLAHE_HC_CLIP, tileGridSize=(_CLAHE_HC_GRID, _CLAHE_HC_GRID))
    hc = clahe.apply(gray)
    hc_rgb = cv2.cvtColor(hc, cv2.COLOR_GRAY2RGB)
    hc_pil = Image.fromarray(hc_rgb, mode="RGB")
    hc_pil = hc_pil.filter(ImageFilter.UnsharpMask(
        radius=_SHARPEN_HC_RADIUS,
        percent=_SHARPEN_HC_PERCENT,
        threshold=_SHARPEN_HC_THRESH,
    ))
    variants.append(ImageVariant(
        label="highcontrast_gray",
        image=hc_pil,
        quality_score=_image_quality_score(hc_pil),
    ))

    # Variant 3: Sauvola-binarized (back in RGB)
    binarized = _sauvola(gray, _SAUVOLA_WINDOW, _SAUVOLA_K)
    bin_rgb = cv2.cvtColor(binarized, cv2.COLOR_GRAY2RGB)
    bin_pil = Image.fromarray(bin_rgb, mode="RGB")
    variants.append(ImageVariant(
        label="binarized",
        image=bin_pil,
        quality_score=_image_quality_score(bin_pil),
    ))

    variants.sort(key=lambda v: v.quality_score, reverse=True)
    return variants


def pick_retry_variant(
    variants: list[ImageVariant],
    already_used_label: str,
) -> ImageVariant | None:
    """Return the best variant whose label is *not* already_used_label.

    Returns None if no alternative is available.
    """
    for v in variants:
        if v.label != already_used_label:
            return v
    return None


# ── Image quality estimator ──────────────────────────────────────────


def _image_quality_score(image: Image.Image) -> float:
    """Fast, lightweight quality estimator.

    Combines:
      - Laplacian variance (sharpness/focus)
      - Michelson contrast
      - Edge density
    into a single 0-1 score.  Higher = crisper text strokes.
    """
    try:
        import cv2
    except ImportError:
        return 0.5  # neutral if no OpenCV

    gray = np.array(image.convert("L"), dtype=np.uint8)

    # Sharpness: variance of Laplacian
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness = min(1.0, laplacian_var / 2000.0)

    # Contrast: Michelson
    lo, hi = float(np.percentile(gray, 5)), float(np.percentile(gray, 95))
    contrast = (hi - lo) / max(hi + lo, 1.0)

    # Edge density
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.count_nonzero(edges)) / max(1, gray.size)

    return 0.45 * sharpness + 0.30 * contrast + 0.25 * min(1.0, edge_density * 10.0)


# ── Sauvola binarization (self-contained) ────────────────────────────


def _sauvola(gray: np.ndarray, window: int, k: float) -> np.ndarray:
    """Sauvola local-adaptive binarization."""
    import cv2

    if window % 2 == 0:
        window += 1
    gf = gray.astype(np.float64)
    mean = cv2.blur(gf, (window, window))
    mean_sq = cv2.blur(gf * gf, (window, window))
    var = np.clip(mean_sq - mean * mean, 0, None)
    std = np.sqrt(var)
    threshold = mean * (1.0 + k * (std / 128.0 - 1.0))
    return np.where(gf > threshold, 255, 0).astype(np.uint8)
