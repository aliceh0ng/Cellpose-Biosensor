#!/usr/bin/env python3
"""
scripts/inference_norm.py
Run Cellpose-SAM on image stacks with global BFP normalisation and
4-channel patch export for annotation with napari + Cellpose GUI.

Differences from inference.py:
  - BFP normalised globally (p1/p99 on full image) before splitting —
    all patches share the same intensity scale
  - Patch TIFs contain all 4 channels (BFP, GFP, RFP, SYTOX) as (4, H, W)
    uint16, so napari can show all channels while annotating
  - Patches saved to data/patches/v2/!! Change this if running again!!

For each image in the target split:
  - Load uint16 stack
  - Normalise BFP globally, split into N_ROWS×N_COLS patches
  - Run cpsam with diameter=43 px and normalize=False
  - Save 4-channel uint16 patch  →  data/patches/v2/<stem>/patch_r{r}_c{c}.tif
  - Save initial cpsam mask      →  data/patches/v2/<stem>/patch_r{r}_c{c}_seg.npy

Skips any stack that already has _seg.npy files in its output directory, so re-running is safe.

Inputs:                                                   
- data/raw/16bit/<stem>.tif — raw (4, H, W) uint16 stack                                                             
- data/splits.json — needed when using --split            
                                        
Outputs (per image, under data/patches/v2/<stem>/):
- patch_r{r}_c{c}.tif — all 4 channels uint16, ImageJ-compatible                                                     
- patch_r{r}_c{c}_seg.npy — initial cpsam masks in Cellpose GUI format

Usage
-----
# Test on one image first
python scripts/inference_norm.py --single FILENAME.tif

# Annotate training images
python scripts/inference_norm.py --split train

# All images
python scripts/inference_norm.py --split all

# Optional: adjust grid size (default: 3x3)
python scripts/inference_norm.py --split train --n-rows 5 --n-cols 5

# Optional: adjust batch size for GPU inference (default: 4, reduce if OOM)
python scripts/inference_norm.py --split train --batch-size 2
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

from src.preprocess import normalize_for_segmentation

RAW_DIR     = PROJECT_ROOT / 'data' / 'raw' / '16bit'
PATCHES_DIR = PROJECT_ROOT / 'data' / 'patches' / 'v2'
SPLITS_FILE = PROJECT_ROOT / 'data' / 'splits.json'

# default is 3x3 grid!
N_ROWS          = 3
N_COLS          = 3

PIXEL_SIZE_UM   = 0.035
DIAMETER_PX     = round(1.5 / PIXEL_SIZE_UM)   # 43 px
FLOW_THRESH     = 0.4
CELLPROB_THRESH = 0.0

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


def split_image(img: np.ndarray, n_rows: int, n_cols: int) -> list[dict]:
    H, W = img.shape[-2], img.shape[-1]
    row_edges = np.linspace(0, H, n_rows + 1, dtype=int)
    col_edges = np.linspace(0, W, n_cols + 1, dtype=int)
    patches = []
    for r in range(n_rows):
        for c in range(n_cols):
            y0, y1 = int(row_edges[r]), int(row_edges[r + 1])
            x0, x1 = int(col_edges[c]), int(col_edges[c + 1])
            patches.append({
                'y0': y0, 'x0': x0,
                'h': y1 - y0, 'w': x1 - x0,
                'row': r, 'col': c,
            })
    return patches


def process_stack(stack_path: Path, model, n_rows: int, n_cols: int,
                  batch_size: int) -> None:
    stem    = stack_path.stem
    out_dir = PATCHES_DIR / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = list(out_dir.glob('patch_r*_c*_seg.npy'))
    if existing:
        log.info(f'  Skip {stem} — {len(existing)} existing _seg.npy files found')
        return

    log.info(f'  Loading {stack_path.name}')
    stack    = load_stack(stack_path)

    bfp_norm = normalize_for_segmentation(stack[0].astype(np.float32))
    patches  = split_image(bfp_norm, n_rows, n_cols)
    log.info(f'  Stack {stack.shape}  global BFP norm  →  '
             f'{n_rows}×{n_cols} = {len(patches)} patches')

    # Batch inference — all patches in one model.eval() call
    bfp_patches = [bfp_norm[p['y0']:p['y0']+p['h'], p['x0']:p['x0']+p['w']]
                   for p in patches]
    t0 = time.time()
    masks_list, _, _ = model.eval(
        bfp_patches,
        diameter=DIAMETER_PX,
        channels=[0, 0],
        normalize=False,
        batch_size=batch_size,
        flow_threshold=FLOW_THRESH,
        cellprob_threshold=CELLPROB_THRESH,
    )
    log.info(f'  Inference done in {time.time()-t0:.1f}s  '
             f'({len(patches)} patches, batch_size={batch_size})')

    for p, masks in zip(patches, masks_list):
        r, c = p['row'], p['col']
        y0, x0, h, w = p['y0'], p['x0'], p['h'], p['w']

        patch_path = out_dir / f'patch_r{r}_c{c}.tif'
        seg_path   = out_dir / f'patch_r{r}_c{c}_seg.npy'

        patch_4ch = stack[:, y0:y0+h, x0:x0+w]
        tifffile.imwrite(str(patch_path), patch_4ch,
                         imagej=True,
                         metadata={'axes': 'CYX',
                                   'Labels': 'BFP\\GFP\\RFP\\SYTOX'})

        outlines = masks * cputils.masks_to_outlines(masks)
        dat = {
            'outlines':    outlines.astype(np.uint16),
            'masks':       masks.astype(np.uint16),
            'chan_choose': [0, 0],
            'ismanual':    np.zeros(max(int(masks.max()), 1), bool),
            'filename':    str(patch_path),
            'flows':       [],
            'diameter':    np.nan,
        }
        np.save(str(seg_path), dat, allow_pickle=True)
        log.info(f'    r{r}c{c}  {h}×{w} px  {int(masks.max())} cells')


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
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Patches per GPU batch (default: 4, reduce if OOM)')
    args = parser.parse_args()

    log.info(f'diameter={DIAMETER_PX} px  grid={args.n_rows}×{args.n_cols}  '
             f'batch_size={args.batch_size}  normalize=global  '
             f'flow_threshold={FLOW_THRESH}')
    log.info('Loading Cellpose-SAM (gpu=True)...')
    model = models.CellposeModel(gpu=True)

    if args.single:
        path = RAW_DIR / args.single
        if not path.exists():
            raise FileNotFoundError(path)
        log.info(f'\n[1/1] {args.single}')
        process_stack(path, model, args.n_rows, args.n_cols, args.batch_size)
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
            process_stack(RAW_DIR / name, model, args.n_rows, args.n_cols,
                          args.batch_size)

    log.info('\nDone.')


if __name__ == '__main__':
    main()
