"""
src/measure.py — extract per-cell intensities from raw 16-bit data

This is the only place in the pipeline that touches the raw uint16 values
for the purpose of generating biological numbers.
"""
import numpy as np
import pandas as pd
from skimage.measure import regionprops


def extract_intensities(masks: np.ndarray,
                         stack: np.ndarray,
                         image_id: str,
                         local_bg: bool = False,
                         ring_width: int = 5) -> pd.DataFrame:
    """
    For every segmented cell, extract mean intensity in all 4 channels
    from the raw 16-bit stack. Optionally subtract local background.

    Args:
        masks: 2D integer label mask (0=background, 1..N=cells)
        stack: (4, H, W) uint16 array — raw 16-bit, NOT normalised
        image_id: string identifier for this image (used in output table)
        local_bg: if True, subtract per-cell ring background per channel
        ring_width: ring width for local background estimation (pixels)

    Returns:
        DataFrame with one row per cell, columns:
            image_id, cell_id, area,
            bfp_raw, gfp_raw, rfp_raw, sytox_raw,
            bfp_bg, gfp_bg, rfp_bg,          (background estimates)
            bfp, gfp, rfp,                    (background-subtracted)
            gfp_over_bfp, rfp_over_bfp        (normalised ratios)
    """
    from scipy.ndimage import binary_dilation

    bfp_ch, gfp_ch, rfp_ch, sytox_ch = stack[0], stack[1], stack[2], stack[3]
    records = []

    for region in regionprops(masks):
        row_coords = region.coords[:, 0]
        col_coords = region.coords[:, 1]

        mean_bfp_raw  = float(bfp_ch[row_coords, col_coords].mean())
        mean_gfp_raw  = float(gfp_ch[row_coords, col_coords].mean())
        mean_rfp_raw  = float(rfp_ch[row_coords, col_coords].mean())
        mean_sytox    = float(sytox_ch[row_coords, col_coords].mean())

        bfp_bg = gfp_bg = rfp_bg = 0.0
        if local_bg:
            cell_binary = masks == region.label
            dilated = binary_dilation(cell_binary, iterations=ring_width)
            ring = dilated & ~cell_binary
            if ring.any():
                bfp_bg  = float(bfp_ch[ring].mean())
                gfp_bg  = float(gfp_ch[ring].mean())
                rfp_bg  = float(rfp_ch[ring].mean())

        bfp_corr = max(mean_bfp_raw - bfp_bg, 0.0)
        gfp_corr = max(mean_gfp_raw - gfp_bg, 0.0)
        rfp_corr = max(mean_rfp_raw - rfp_bg, 0.0)

        gfp_over_bfp = gfp_corr / bfp_corr if bfp_corr > 0 else np.nan
        rfp_over_bfp = rfp_corr / bfp_corr if bfp_corr > 0 else np.nan

        records.append({
            'image_id':     image_id,
            'cell_id':      region.label,
            'area_px':      region.area,
            'centroid_y':   region.centroid[0],
            'centroid_x':   region.centroid[1],
            'bfp_raw':      mean_bfp_raw,
            'gfp_raw':      mean_gfp_raw,
            'rfp_raw':      mean_rfp_raw,
            'sytox_raw':    mean_sytox,
            'bfp_bg':       bfp_bg,
            'gfp_bg':       gfp_bg,
            'rfp_bg':       rfp_bg,
            'bfp':          bfp_corr,
            'gfp':          gfp_corr,
            'rfp':          rfp_corr,
            'gfp_over_bfp': gfp_over_bfp,
            'rfp_over_bfp': rfp_over_bfp,
        })

    return pd.DataFrame(records)
