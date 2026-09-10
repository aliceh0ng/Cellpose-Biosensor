"""
src/io.py — loading and saving images and masks
"""
import numpy as np
import tifffile
from pathlib import Path


def load_stack(path: str | Path) -> np.ndarray:
    """
    Load a 16-bit 4-channel Zeiss Airyscan stack.

    Expected input: (4, H, W) uint16 TIFF
    Returns: (4, H, W) uint16 numpy array

    Channels by convention:
        [0] BFP  — constitutive (segmentation channel)
        [1] GFP  — osmotic stress reporter
        [2] RFP  — oxidative stress reporter
        [3] SYTOX — host nuclear stain (exclusion channel)
    """
    img = tifffile.imread(str(path))
    if img.ndim == 3 and img.shape[0] == 4:
        pass  # already (C, H, W)
    elif img.ndim == 4:
        # (Z, C, H, W) — take first Z slice or max-project
        img = img.max(axis=0)
    else:
        raise ValueError(f"Unexpected image shape {img.shape} in {path}")

    if img.dtype != np.uint16:
        raise TypeError(f"Expected uint16, got {img.dtype} in {path}. "
                        "Use the 16-bit export from Zen, not the RGB preview.")
    return img


def get_bfp(stack: np.ndarray) -> np.ndarray:
    """Return the BFP channel (channel 0) as a 2D uint16 array."""
    return stack[0]


def save_masks(masks: np.ndarray, path: str | Path) -> None:
    """Save integer label mask as 16-bit TIFF. Each cell = unique integer ID."""
    tifffile.imwrite(str(path), masks.astype(np.uint16),
                     compression='lzw', photometric='minisblack')


def split_image(img: np.ndarray,
                n_rows: int = 3,
                n_cols: int = 3) -> list[dict]:
    """
    Split a 2D image into n_rows × n_cols equal patches.

    Patches are sized by np.linspace so edge patches absorb any remainder
    pixels — no padding, no overlap.

    Returns a list of dicts with keys:
        'tile'       : 2D crop (actual patch array)
        'y0', 'x0'  : top-left corner in original image
        'h', 'w'    : patch height and width
        'row', 'col': grid position (0-indexed)
    """
    H, W = img.shape[:2]
    row_edges = np.linspace(0, H, n_rows + 1, dtype=int)
    col_edges = np.linspace(0, W, n_cols + 1, dtype=int)
    patches = []
    for r in range(n_rows):
        for c in range(n_cols):
            y0, y1 = int(row_edges[r]), int(row_edges[r + 1])
            x0, x1 = int(col_edges[c]), int(col_edges[c + 1])
            patches.append({
                'tile': img[y0:y1, x0:x1],
                'y0': y0, 'x0': x0,
                'h': y1 - y0, 'w': x1 - x0,
                'row': r, 'col': c,
            })
    return patches


def stitch_masks(patches: list[dict], H: int, W: int) -> np.ndarray:
    """
    Reassemble patch masks into a full-image label array.
    Cell IDs are remapped to be globally unique.
    Cells that fall on a patch boundary will be split — this is expected.
    """
    full = np.zeros((H, W), dtype=np.int32)
    offset = 0
    for p in patches:
        y0, x0, h, w = p['y0'], p['x0'], p['h'], p['w']
        m = p['mask'][:h, :w].astype(np.int32)
        full[y0:y0+h, x0:x0+w] = np.where(m > 0, m + offset, 0)
        offset += int(m.max())
    return full
