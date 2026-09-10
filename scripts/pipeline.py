#!/usr/bin/env python3
"""
scripts/pipeline.py
Full segmentation and measurement pipeline for whole image stacks.

For each image: normalize BFP → segment (fine-tuned or base cpsam) →
size filter → measure → CSV.

Reads model path from models/finetuned/latest_model_path.txt if available;
falls back to base cpsam.

Skips images whose CSV already exists — delete to reprocess.

Usage
-----
# Run all .tif files in data/raw/16bit_all/
python scripts/pipeline.py

# Test on one image first
python scripts/pipeline.py --single FILENAME.tif
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # non-interactive backend for script use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cellpose import models

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.io         import load_stack, save_masks
from src.preprocess import normalize_for_segmentation
from src.segment    import size_filter
from src.measure    import extract_intensities

RAW_DIR     = PROJECT_ROOT / 'data' / 'raw' / '16bit_all'
RESULTS_DIR = PROJECT_ROOT / 'data' / 'results'
MASKS_DIR   = PROJECT_ROOT / 'data' / 'processed'

FIGURES_DIR = PROJECT_ROOT / 'figures' / 'qc'

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MASKS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Parameters — keep in sync with notebook 06
PIXEL_SIZE_UM   = 0.035
DIAMETER_PX     = round(1.5 / PIXEL_SIZE_UM)   # 43 px
FLOW_THRESH     = 0.4
CELLPROB_THRESH = 0.0
BATCH_SIZE      = 8    # tiles per GPU batch

MIN_AREA = 100    # px² — smaller = noise
MAX_AREA = 2000   # px² — larger = debris / host cell fragment

LOCAL_BG   = False  # set True to subtract per-cell annular ring background
RING_WIDTH = 5

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)


def load_model():
    path_file = PROJECT_ROOT / 'models' / 'finetuned' / 'latest_model_path.txt'
    if path_file.exists():
        model_path = Path(path_file.read_text().strip())
        if model_path.exists():
            log.info(f'Fine-tuned model: {model_path.name}')
            return models.CellposeModel(gpu=True, pretrained_model=str(model_path))
    log.info('No fine-tuned model found — using base cpsam')
    return models.CellposeModel(gpu=True)


def process_stack(stack_path: Path, model) -> None:
    stem          = stack_path.stem
    csv_out       = RESULTS_DIR / f'{stem}.csv'
    mask_raw_out  = MASKS_DIR   / f'{stem}_masks_raw.tif'
    mask_out      = MASKS_DIR   / f'{stem}_masks.tif'

    if csv_out.exists():
        log.info(f'  Skip {stem} (CSV exists — delete to reprocess)')
        return

    log.info(f'  Loading...')
    stack = load_stack(stack_path)
    log.info(f'  {stack.shape}')

    bfp_norm = normalize_for_segmentation(stack[0])

    log.info(f'  Segmenting...')
    t0 = time.time()
    masks_raw, _, _ = model.eval(
        bfp_norm,
        diameter=DIAMETER_PX,
        channels=[0, 0],
        normalize=False,
        flow_threshold=FLOW_THRESH,
        cellprob_threshold=CELLPROB_THRESH,
        batch_size=BATCH_SIZE,
    )
    log.info(f'  {masks_raw.max()} cells  ({time.time()-t0:.1f}s)')

    n0 = int(masks_raw.max())
    masks_final = size_filter(masks_raw, min_area=MIN_AREA, max_area=MAX_AREA)
    n1 = int(masks_final.max())
    log.info(f'  Size filter: {n0} → {n1} (removed {n0-n1})')

    save_masks(masks_raw,   mask_raw_out)
    save_masks(masks_final, mask_out)

    log.info(f'  Measuring intensities...')
    df = extract_intensities(masks_final, stack, image_id=stem,
                             local_bg=LOCAL_BG, ring_width=RING_WIDTH)
    df.to_csv(csv_out, index=False)
    log.info(f'  Saved {len(df)} cells → {csv_out.name}')

    _save_figures(stem, stack, masks_final, n0, n1, df)

    return {'stem': stem, 'n_raw': n0, 'n_size_filter': n1}


def _save_figures(stem, stack, masks_final, n0, n1, df):
    bg_label = 'bg-subtracted' if LOCAL_BG else 'raw'

    # 1 — masks overview (BFP thumbnail + contours)
    THUMB = 8
    thumb_norm = normalize_for_segmentation(stack[0][::THUMB, ::THUMB], low_pct=1, high_pct=99.9)
    thumb_mask = masks_final[::THUMB, ::THUMB]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')
    for ax, mask, title in zip(axes,
                                [np.zeros_like(thumb_mask), thumb_mask],
                                ['BFP (no masks)', f'BFP + {n1} cells']):
        ax.imshow(thumb_norm, cmap='gray', vmin=0, vmax=1)
        if mask.max() > 0:
            ax.contour(mask, levels=np.arange(0.5, mask.max() + 0.5),
                       colors='red', linewidths=0.3)
        ax.set_title(title, color='black', fontsize=10)
        ax.axis('off')
    fig.suptitle(f'{stem} (1:{THUMB} thumbnail)', color='black', fontsize=10)
    plt.tight_layout(pad=0.3)
    plt.savefig(FIGURES_DIR / f'{stem}_masks_overview.png', dpi=120,
                bbox_inches='tight', facecolor='white')
    plt.close()

    # 3 — intensity QC histograms
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor='white')
    axes[0].hist(df['bfp'], bins=60, color='#486dad', alpha=0.85)
    axes[0].set_title(f'BFP ({bg_label})', color='black')
    axes[0].set_xlabel('Intensity (ADU)', color='#aaa')
    axes[0].set_ylabel('Cells', color='#aaa')
    axes[1].hist(df['gfp_over_bfp'].dropna(), bins=60, color='#479965', alpha=0.85)
    axes[1].set_title('GFP / BFP ratio', color='black')
    axes[1].set_xlabel('Ratio', color='#aaa')
    axes[2].hist(df['rfp_over_bfp'].dropna(), bins=60, color='#ca5d5d', alpha=0.85)
    axes[2].set_title('RFP / BFP ratio', color='black')
    axes[2].set_xlabel('Ratio', color='#aaa')
    for ax in axes:
        ax.tick_params(colors='#aaa')
        for sp in ax.spines.values():
            sp.set_edgecolor('#333')
    fig.suptitle(f'{stem} — {len(df)} cells', color='black', fontsize=10)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'{stem}_intensity_qc.svg', bbox_inches='tight', facecolor='white')
    plt.close()

    log.info(f'  Figures saved → figures/qc/{stem}_*.svg/png')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--single', metavar='FILENAME',
                        help='Process one file; omit to run all .tif files in RAW_DIR')
    args = parser.parse_args()

    model = load_model()
    log.info(f'diameter={DIAMETER_PX} px  local_bg={LOCAL_BG}')

    filter_stats = []

    if args.single:
        path = RAW_DIR / args.single
        if not path.exists():
            raise FileNotFoundError(path)
        log.info(f'\n[1/1] {args.single}')
        stats = process_stack(path, model)
        if stats:
            filter_stats.append(stats)
    else:
        paths = sorted(RAW_DIR.glob('*.tif'))
        if not paths:
            raise FileNotFoundError(f'No .tif files found in {RAW_DIR}')
        log.info(f'Processing {len(paths)} images in {RAW_DIR}')
        for i, path in enumerate(paths, 1):
            log.info(f'\n[{i}/{len(paths)}] {path.name}')
            stats = process_stack(path, model)
            if stats:
                filter_stats.append(stats)

    if filter_stats:
        filter_csv = RESULTS_DIR / 'filter_steps.csv'
        new_df = pd.DataFrame(filter_stats)
        if filter_csv.exists():
            existing = pd.read_csv(filter_csv)
            combined = pd.concat([existing, new_df]).drop_duplicates(subset='stem', keep='last')
        else:
            combined = new_df
        combined.to_csv(filter_csv, index=False)
        log.info(f'\nFilter stats saved → {filter_csv}  ({len(combined)} images total)')

    log.info('\nAll done.')


if __name__ == '__main__':
    main()
