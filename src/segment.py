"""
src/segment.py — run Cellpose-SAM and apply post-processing filters
"""
import numpy as np
from cellpose import models


def run_cellpose_sam(bfp_norm: np.ndarray,
                     diameter: float | None = None,
                     flow_threshold: float = 0.4,
                     cellprob_threshold: float = 0.0,
                     gpu: bool = True) -> np.ndarray:
    """
    Run Cellpose-SAM on a normalised BFP tile.

    Args:
        bfp_norm: 2D float32 array, values in [0, 1] (from normalize_for_segmentation)
        diameter: expected cell diameter in pixels (None = auto-estimate)
        flow_threshold: lower = more masks kept (0.4 is default)
        cellprob_threshold: lower = more masks kept (0.0 is default)
        gpu: use GPU if available
    Returns:
        2D integer label mask (0 = background, 1..N = individual cells)
    """
    # Cellpose v4 defaults to cpsam; `model_type` is ignored in v4.0.1+.
    model = models.CellposeModel(gpu=gpu)
    masks, flows, styles = model.eval(
        bfp_norm,
        diameter=diameter,
        channels=[0, 0],   # [0,0] = grayscale, no nuclear channel
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        normalize=False,   # we already normalised
    )
    return masks


def size_filter(masks: np.ndarray,
                min_area: int = 8,
                max_area: int = 300) -> np.ndarray:
    """
    Remove objects outside the expected bacterial size range.

    Typical rod-shaped bacterium at 100x Airyscan: ~15–80 px²
    Fecal debris tends to be larger (>200 px²) or smaller (<8 px²)

    Args:
        min_area: minimum cell area in pixels (removes debris/noise)
        max_area: maximum cell area in pixels (removes host cell fragments)
    """
    from skimage.measure import regionprops
    filtered = np.zeros_like(masks)
    new_id = 1
    for region in regionprops(masks):
        if min_area <= region.area <= max_area:
            filtered[masks == region.label] = new_id
            new_id += 1
    return filtered


def spectral_filter(masks: np.ndarray,
                    bfp: np.ndarray,
                    gfp: np.ndarray,
                    rfp: np.ndarray,
                    max_autofl_score: float = 0.6) -> np.ndarray:
    """
    Remove objects that look like fecal autofluorescence.

    Autofluorescence signature: high signal simultaneously in GFP AND RFP
    relative to BFP. Genuine bacteria (BFP+): GFP and RFP are low or
    correlated with a specific biological state, not both elevated.

    autofluorescence score = min(norm_gfp, norm_rfp) / max(norm_bfp, 1)

    Objects with score > max_autofl_score are likely fecal debris.

    Args:
        masks: integer label mask from size_filter
        bfp, gfp, rfp: raw uint16 2D arrays (full image)
        max_autofl_score: objects above this are removed (tune on your data)
    """
    from skimage.measure import regionprops
    filtered = np.zeros_like(masks)
    new_id = 1
    for region in regionprops(masks):
        coords = tuple(np.array(region.coords).T)
        mean_bfp = float(bfp[coords].mean())
        mean_gfp = float(gfp[coords].mean())
        mean_rfp = float(rfp[coords].mean())
        autofl_score = min(mean_gfp, mean_rfp) / max(mean_bfp, 1.0)
        if autofl_score <= max_autofl_score:
            filtered[masks == region.label] = new_id
            new_id += 1
    return filtered


def sytox_exclusion(masks: np.ndarray,
                     sytox: np.ndarray,
                     sytox_threshold: int | None = None,
                     max_overlap: float = 0.25) -> np.ndarray:
    """
    Remove bacterial mask candidates that substantially overlap with host nuclei.

    Host nuclei = large, bright SYTOX+ objects. A bacterial object sitting
    within or on a host nucleus is likely a false positive (or not a bacterium).

    Args:
        masks: integer label mask
        sytox: raw uint16 SYTOX channel (2D)
        sytox_threshold: pixel intensity threshold for SYTOX+ classification.
                         If None, uses p90 of the image.
        max_overlap: fraction of cell mask overlapping SYTOX+ area; above
                     this threshold the cell is removed.
    """
    from skimage.measure import regionprops
    if sytox_threshold is None:
        sytox_threshold = float(np.percentile(sytox, 90))
    sytox_binary = sytox > sytox_threshold
    filtered = np.zeros_like(masks)
    new_id = 1
    for region in regionprops(masks):
        coords = tuple(np.array(region.coords).T)
        overlap = float(sytox_binary[coords].mean())
        if overlap <= max_overlap:
            filtered[masks == region.label] = new_id
            new_id += 1
    return filtered
