#!/usr/bin/env python3
"""
scripts/fix_seg_npy.py
Convert _seg.npy files saved by the old inference script (plain np.save dict)
into the format the Cellpose GUI expects (masks_flows_to_seg).

Run once after transferring patches from Windows:
    conda activate cellpose
    python scripts/fix_seg_npy.py
"""

import sys
from pathlib import Path

import numpy as np
from cellpose import utils

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PATCHES_DIR  = PROJECT_ROOT / 'data' / 'patches' 


def is_old_format(seg: np.ndarray) -> bool:
    if seg.dtype != object:
        return False
    d = seg.item()
    return isinstance(d, dict) and set(d.keys()) <= {'masks', 'img'}


def convert(seg_path: Path) -> bool:
    seg = np.load(str(seg_path), allow_pickle=True)
    if not is_old_format(seg):
        return False

    d    = seg.item()
    img  = d['img']
    masks = d['masks']

    outlines = masks * utils.masks_to_outlines(masks)
    dat = {
        'outlines':  outlines.astype(np.uint16),
        'masks':     masks.astype(np.uint16),
        'chan_choose': [0, 0],
        'ismanual':  np.zeros(max(int(masks.max()), 1), bool),
        'filename':  str(seg_path.with_suffix('.tif')),
        'flows':     [],
        'diameter':  np.nan,
    }
    np.save(str(seg_path), dat, allow_pickle=True)
    return True


def main():
    seg_files = sorted(PATCHES_DIR.rglob('*_seg.npy'))
    if not seg_files:
        print('No _seg.npy files found under', PATCHES_DIR)
        sys.exit(0)

    converted = 0
    skipped   = 0
    for p in seg_files:
        if convert(p):
            print(f'  converted  {p.relative_to(PROJECT_ROOT)}')
            converted += 1
        else:
            print(f'  skip       {p.relative_to(PROJECT_ROOT)}  (already correct format)')
            skipped += 1

    print(f'\nDone. {converted} converted, {skipped} skipped.')


if __name__ == '__main__':
    main()
