"""
src/preprocess.py — normalization and background correction

IMPORTANT: These functions are for SEGMENTATION INPUT only.
Never use normalized images for intensity measurement.
Always apply masks back to the raw uint16 data for quantification.
"""
import numpy as np
from scipy.ndimage import uniform_filter


def normalize_for_segmentation(img: np.ndarray,
                                low_pct: float = 1.0,
                                high_pct: float = 99.0) -> np.ndarray:
    """
    Percentile-stretch a single channel to float32 [0, 1].
    This is what gets fed to Cellpose/Omnipose — NOT used for measurement.

    Args:
        img: 2D uint16 array (single channel)
        low_pct: lower percentile to clip (removes very dim background)
        high_pct: upper percentile to clip (removes hot pixels)
    """
    lo = np.percentile(img, low_pct)
    hi = np.percentile(img, high_pct)
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)


def estimate_background(img: np.ndarray,
                         masks: np.ndarray,
                         method: str = 'median_outside') -> float:
    """
    Estimate per-image background intensity for a single channel.

    Args:
        img: 2D uint16 array (raw, not normalized)
        masks: 2D integer label mask (0 = background)
        method: 'median_outside' uses median of all background pixels
    Returns:
        background intensity estimate (scalar)
    """
    if method == 'median_outside':
        bg_pixels = img[masks == 0].astype(float)
        return float(np.median(bg_pixels)) if len(bg_pixels) > 0 else 0.0
    raise ValueError(f"Unknown background method: {method}")


def local_background_ring(img: np.ndarray,
                           cell_mask: np.ndarray,
                           ring_width: int = 5) -> float:
    """
    Estimate background for a single cell using a dilation ring around its mask.
    More accurate than global background when autofluorescence is spatially variable.

    Args:
        img: 2D uint16 array (full image, raw)
        cell_mask: binary mask for one cell (True = cell pixels)
        ring_width: pixels to dilate outward for background ring
    Returns:
        median intensity in the ring (float)
    """
    from scipy.ndimage import binary_dilation
    dilated = binary_dilation(cell_mask,
                              iterations=ring_width)
    ring = dilated & ~cell_mask
    ring_pixels = img[ring].astype(float)
    return float(np.median(ring_pixels)) if len(ring_pixels) > 0 else 0.0
