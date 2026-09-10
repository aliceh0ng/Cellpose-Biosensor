#!/usr/bin/env python3
"""
scripts/inference.py
Run Cellpose-SAM on image stacks to produce annotation-ready BFP patches
and initial masks. Normalisation is applied per-patch (Cellpose default).

For each image in the target split:
  - Load uint16 stack
  - Split BFP channel into 3×3 patches
  - Run cpsam with diameter=43 px and normalize=True (= per-patch auto-adjust)
  - Save raw uint16 BFP patch  →  data/patches/v1/<stem>/patch_r{r}_c{c}.tif
  - Save initial cpsam mask    →  data/patches/v1/<stem>/patch_r{r}_c{c}_seg.npy

See inference_norm.py for global normalisation + 4-channel patch variant.

Usage
-----
# Test on one image first (recommended before submitting the full job)
python scripts/inference.py --single FILENAME.tif

# Annotate training images (most common use)
python scripts/inference.py --split train

# All images (baseline comparison)
python scripts/inference.py --split all
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
from cellpose import models, utils as cputils

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RAW_DIR     = PROJECT_ROOT / 'data' / 'raw' / '16bit'
PATCHES_DIR = PROJECT_ROOT / 'data' / 'patches' / 'v1'
SPLITS_FILE = PROJECT_ROOT / 'data' / 'splits.json'

# Segmentation parameters
N_ROWS          = 3
N_COLS          = 3
PIXEL_SIZE_UM   = 0.035
DIAMETER_PX     = round(1.5 / PIXEL_SIZE_UM)   # 43 px  (1.5 µm bacteria short axis)
FLOW_THRESH     = 0.4
CELLPROB_THRESH = 0.0
# normalize=True (Cellpose default) applies internal percentile normalisation
# before inference — equivalent to "auto-adjust saturation" in the Cellpose GUI.

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


def load_stack(path: Path) -> np.ndarray:
    img = tifffile.imread(str(path))
    if img.ndim == 3 and img.shape[0] == 4:
        return img.astype(np.uint16)
    raise ValueError(f'Expected (4, H, W) uint16, got {img.shape} in {path.name}')


def split_bfp(bfp: np.ndarray, n_rows: int, n_cols: int) -> list[dict]:
    H, W = bfp.shape
    row_edges = np.linspace(0, H, n_rows + 1, dtype=int)
    col_edges = np.linspace(0, W, n_cols + 1, dtype=int)
    patches = []
    for r in range(n_rows):
        for c in range(n_cols):
            y0, y1 = int(row_edges[r]), int(row_edges[r + 1])
            x0, x1 = int(col_edges[c]), int(col_edges[c + 1])
            patches.append({
                'tile': bfp[y0:y1, x0:x1],
                'y0': y0, 'x0': x0,
                'h': y1 - y0, 'w': x1 - x0,
                'row': r, 'col': c,
            })
    return patches


def process_stack(stack_path: Path, model, n_rows: int, n_cols: int) -> None:
    stem    = stack_path.stem
    out_dir = PATCHES_DIR / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = list(out_dir.glob('patch_r*_c*_seg.npy'))
    if existing:
        log.info(f'  Skip {stem} — {len(existing)} existing _seg.npy files found')
        return

    log.info(f'  Loading {stack_path.name}')
    stack   = load_stack(stack_path)
    patches = split_bfp(stack[0], n_rows, n_cols)
    log.info(f'  Stack {stack.shape}  →  {n_rows}×{n_cols} = {len(patches)} patches')

    for p in patches:
        r, c     = p['row'], p['col']
        bfp_path = out_dir / f'patch_r{r}_c{c}.tif'
        seg_path = out_dir / f'patch_r{r}_c{c}_seg.npy'

        # Raw uint16 BFP patch — opened in Cellpose GUI for annotation
        tifffile.imwrite(str(bfp_path), p['tile'].astype(np.uint16),
                         compression='lzw', photometric='minisblack')

        # Inference — normalize=True (default) = auto-adjust saturation
        t0 = time.time()
        masks, _, _ = model.eval(
            p['tile'].astype(np.float32),
            diameter=DIAMETER_PX,
            channels=[0, 0],
            flow_threshold=FLOW_THRESH,
            cellprob_threshold=CELLPROB_THRESH,
        )
        elapsed = time.time() - t0

        # Save as _seg.npy in the format Cellpose GUI expects
        outlines = masks * cputils.masks_to_outlines(masks)
        dat = {
            'outlines':    outlines.astype(np.uint16),
            'masks':       masks.astype(np.uint16),
            'chan_choose':  [0, 0],
            'ismanual':    np.zeros(max(int(masks.max()), 1), bool),
            'filename':    str(bfp_path),
            'flows':       [],
            'diameter':    np.nan,
        }
        np.save(str(seg_path), dat, allow_pickle=True)
        log.info(f'    r{r}c{c}  {p["h"]}×{p["w"]} px  {int(masks.max())} cells  '
                 f'{elapsed:.1f}s')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--split',  choices=['train', 'val', 'test', 'all'],
                       help='Process all images in this split')
    group.add_argument('--single', metavar='FILENAME',
                       help='Process one image (for testing)')
    parser.add_argument('--n-rows', type=int, default=N_ROWS,
                        help=f'Grid rows (default: {N_ROWS})')
    parser.add_argument('--n-cols', type=int, default=N_COLS,
                        help=f'Grid cols (default: {N_COLS})')
    args = parser.parse_args()

    log.info(f'diameter={DIAMETER_PX} px  grid={args.n_rows}×{args.n_cols}  '
             f'normalize=per-patch  flow_threshold={FLOW_THRESH}')
    log.info('Loading Cellpose-SAM (gpu=True)...')
    model = models.CellposeModel(gpu=True)

    if args.single:
        path = RAW_DIR / args.single
        if not path.exists():
            raise FileNotFoundError(path)
        log.info(f'\n[1/1] {args.single}')
        process_stack(path, model, args.n_rows, args.n_cols)
    else:
        if not SPLITS_FILE.exists():
            raise FileNotFoundError(
                f'{SPLITS_FILE} not found — run scripts/make_splits.py first')
        with open(SPLITS_FILE) as f:
            splits = json.load(f)
        names = (splits['train'] + splits['val'] + splits['test']
                 if args.split == 'all' else splits[args.split])
        log.info(f'Processing {len(names)} images  (split={args.split!r})')
        for i, name in enumerate(names, 1):
            log.info(f'\n[{i}/{len(names)}] {name}')
            process_stack(RAW_DIR / name, model, args.n_rows, args.n_cols)

    log.info('\nDone.')


if __name__ == '__main__':
    main()
