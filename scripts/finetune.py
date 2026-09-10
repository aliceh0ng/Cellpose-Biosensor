#!/usr/bin/env python3
"""
scripts/finetune.py
Fine-tune Cellpose-SAM on human-annotated patches from the stack-level
training split.

Expects (for each training image):
  data/patches/<version>/<stem>/patch_r{r}_c{c}.tif       — BFP patch (2D or 4-channel)
  data/patches/<version>/<stem>/patch_r{r}_c{c}_seg.npy   — Cellpose GUI annotation

Inputs:
  data/splits.json                     — stack-level train / val / test split
  data/annotations.json                — which training stacks are annotated

Outputs (per run):
  models/finetuned/<run-name>/models/cpsam_<run-name>   — fine-tuned weights
  models/finetuned/<run-name>/run_info.json             — metadata + AP scores
  models/finetuned/<run-name>/figures/training_curves.png
  models/finetuned/<run-name>/figures/val_comparison.png
  models/finetuned/latest_model_path.txt                — updated to this run

Usage
-----
python scripts/finetune.py --run-name 5stacks
python scripts/finetune.py --run-name 10stacks_v2 --patches-dir data/patches/v2
"""

import argparse
import json
import logging
import random
import sys
from datetime import date
from pathlib import Path

import numpy as np
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from aquarel import load_theme
from cellpose import models, train, metrics, utils

PROJECT_ROOT   = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SPLITS_FILE      = PROJECT_ROOT / 'data' / 'splits.json'
ANNOTATIONS_FILE = PROJECT_ROOT / 'data' / 'annotations.json'
RUNS_DIR         = PROJECT_ROOT / 'models' / 'finetuned'

# Hyperparameters (official Cellpose-SAM colab, can change if needed for later versions)
N_EPOCHS      = 100 
LEARNING_RATE = 1e-5 
WEIGHT_DECAY  = 0.1 
BATCH_SIZE    = 1
VAL_FRAC      = 0.2 
MIN_CELLS     = 0
MODEL_NAME    = None  # set to run_name in main()
AQUAREL_THEME = 'boxy_light'
DIAMETER_PX   = round(1.5 / 0.035)   # 43 px
SEED          = 42

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)


def load_annotated_pairs(image_names: list[str], patches_dir: Path) -> list[dict]:
    pairs = []
    for name in image_names:
        stem    = Path(name).stem
        img_dir = patches_dir / stem
        if not img_dir.exists():
            log.warning(f'No patches found for {stem} — run inference.py first')
            continue
        for bfp_path in sorted(img_dir.glob('patch_r*_c*.tif')):
            if '_masks' in bfp_path.name:
                continue
            seg_path = bfp_path.parent / (bfp_path.stem + '_seg.npy')
            if not seg_path.exists():
                continue
            img = tifffile.imread(str(bfp_path)).astype(np.float32)
            if img.ndim == 3:   # multichannel (C, H, W) — use BFP channel only
                img = img[0]
            dat     = np.load(str(seg_path), allow_pickle=True).item()
            ismanual = dat.get('ismanual')
            if ismanual is None:
                log.warning(f'  Warning {stem}/{bfp_path.stem} (no ismanual field)')
            elif np.asarray(ismanual).sum() <= 0:
                log.warning(f'  Warning {stem}/{bfp_path.stem} (ismanual.sum() == 0)')
            mask    = dat['masks'].astype(np.int32)
            n_cells = int(mask.max())
            if n_cells == 0:
                log.warning(f'  Empty patch {stem}/{bfp_path.stem} — including as negative '
                            f'example (verify annotation is intentionally empty)')
            elif n_cells < 3:
                log.warning(f'  Sparse patch {stem}/{bfp_path.stem} ({n_cells} cells) — '
                            f'including but verify no missed annotations')
            pairs.append({
                'stack': stem,
                'name':  f'{stem}/{bfp_path.stem}',
                'img':   img, 'mask': mask, 'n_cells': n_cells,
            })
    return pairs


def load_annotated_train_names(splits: dict) -> list[str]:
    if not ANNOTATIONS_FILE.exists():
        raise FileNotFoundError(
            f'{ANNOTATIONS_FILE} not found — create it with an "annotated_train" list.')
    with open(ANNOTATIONS_FILE) as f:
        annotations = json.load(f)
    annotated_train = annotations.get('annotated_train')
    if not isinstance(annotated_train, list) or not annotated_train:
        raise RuntimeError(
            f'{ANNOTATIONS_FILE} must contain a non-empty "annotated_train" list.')
    # Normalize both manifests to stack stems so annotations.json can list
    # either full filenames (e.g. "..._stack.tif") or stem-only names.
    train_names = {Path(n).stem for n in splits['train']}
    annotated_train = [Path(n).stem for n in annotated_train]
    invalid = sorted(n for n in annotated_train if n not in train_names)
    if invalid:
        raise RuntimeError(
            f'annotations.json lists stacks not in splits["train"]: {invalid}')
    return sorted(set(annotated_train))


def save_training_curves(train_losses, val_losses, run_dir: Path) -> None:
    theme = load_theme(AQUAREL_THEME)
    theme.apply()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label='train')
    ax.plot(val_losses,   label='val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Fine-tuning loss')
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / 'figures' / 'training_curves.png', dpi=150)
    plt.close(fig)
    plt.rcdefaults()


def save_loss_arrays(train_losses, val_losses, run_dir: Path) -> None:
    train_losses = np.asarray(train_losses, dtype=float)
    val_losses   = np.asarray(val_losses, dtype=float)

    np.savez(run_dir / 'losses.npz',
             train_losses=train_losses,
             val_losses=val_losses)

    losses_json = {
        'train_losses': train_losses.tolist(),
        'val_losses': val_losses.tolist(),
    }
    (run_dir / 'losses.json').write_text(json.dumps(losses_json, indent=2))


def save_val_comparison(val_imgs, val_masks, pred_base, pred_ft,
                        val_pairs, run_dir: Path) -> None:
    n_show = min(3, len(val_imgs))
    fig, axes = plt.subplots(n_show, 3, figsize=(12, 4 * n_show))
    if n_show == 1:
        axes = axes[np.newaxis, :]

    def norm(img):
        p1, p99 = np.percentile(img, [1, 99])
        return np.clip((img - p1) / (p99 - p1 + 1e-6), 0, 1)

    def show_mask(ax, img_norm, mask):
        ax.imshow(img_norm, cmap='gray', vmin=0, vmax=1)
        if mask.max() > 0:
            ax.imshow(mask, cmap='tab20b', alpha=0.45, interpolation='nearest')

    for i in range(n_show):
        img_n = norm(val_imgs[i])
        show_mask(axes[i, 0], img_n, val_masks[i])
        show_mask(axes[i, 1], img_n, pred_base[i])
        show_mask(axes[i, 2], img_n, pred_ft[i])
        axes[i, 0].text(-0.12, 0.5, val_pairs[i]['name'].split('/')[-1],
                        transform=axes[i, 0].transAxes,
                        rotation=90, va='center', ha='right', fontsize=10, clip_on=False)

    axes[0, 0].set_title('ground truth')
    axes[0, 1].set_title('base cpsam')
    axes[0, 2].set_title('fine-tuned')
    for ax in axes.flat:
        ax.axis('off')

    fig.tight_layout()
    fig.subplots_adjust(left=0.16)
    fig.savefig(run_dir / 'figures' / 'val_comparison.png', dpi=100)
    plt.close(fig)


def save_ap_bars(ap_base_per_patch, ap_ft_per_patch, val_pairs, run_dir: Path) -> None:
    names = [p['name'].split('/')[-1] for p in val_pairs]
    x = np.arange(len(names))
    w = 0.35

    theme = load_theme(AQUAREL_THEME)
    theme.apply()
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.2), 4))
    ax.bar(x - w/2, ap_base_per_patch, w, label='base cpsam')
    ax.bar(x + w/2, ap_ft_per_patch,   w, label='fine-tuned')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('AP @ IoU 0.5')
    ax.set_ylim(0, 1)
    ax.set_title('Per-patch AP@0.5')
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / 'figures' / 'ap_bars.png', dpi=150)
    plt.close(fig)
    plt.rcdefaults()


def save_cell_counts(val_masks, pred_base, pred_ft, val_pairs, run_dir: Path) -> None:
    names  = [p['name'].split('/')[-1] for p in val_pairs]
    gt     = [int(m.max()) for m in val_masks]
    base   = [int(m.max()) for m in pred_base]
    ft     = [int(m.max()) for m in pred_ft]
    x = np.arange(len(names))
    w = 0.25

    theme = load_theme(AQUAREL_THEME)
    theme.apply()
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.2), 4))
    ax.bar(x - w,   gt,   w, label='ground truth')
    ax.bar(x,       base, w, label='base cpsam')
    ax.bar(x + w,   ft,   w, label='fine-tuned')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Cell count')
    ax.set_title('Cell count per patch')
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / 'figures' / 'cell_counts.png', dpi=150)
    plt.close(fig)
    plt.rcdefaults()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-name', required=True,
                        help='Name for this fine-tuning run, e.g. "5stacks"')
    parser.add_argument('--patches-dir', default=None,
                        help='Path to patches directory (default: data/patches/v1)')
    args = parser.parse_args()
    run_name = args.run_name
    patches_dir = Path(args.patches_dir) if args.patches_dir \
                  else PROJECT_ROOT / 'data' / 'patches' / 'v1'
    model_name = f'cpsam_{run_name}'

    run_dir = RUNS_DIR / run_name
    fig_dir = run_dir / 'figures'
    run_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(exist_ok=True)

    if not SPLITS_FILE.exists():
        raise FileNotFoundError(
            f'{SPLITS_FILE} not found — run scripts/make_splits.py first')
    with open(SPLITS_FILE) as f:
        splits = json.load(f)

    annotated_train = load_annotated_train_names(splits)
    log.info(f'Run: {run_name}')
    log.info(f'Using {len(annotated_train)} annotated training stack(s): '
             f'{annotated_train}')
    log.info(f'Patches dir: {patches_dir}')
    pairs = load_annotated_pairs(annotated_train, patches_dir)
    n_cells_total = sum(p['n_cells'] for p in pairs)
    log.info(f'  {len(pairs)} annotated patches  |  {n_cells_total} total cells')

    assert len(pairs) >= 3, \
        f'Only {len(pairs)} annotated patches. Annotate more before fine-tuning.'

    stacks = sorted({p['stack'] for p in pairs})
    if len(stacks) < 3:
        raise RuntimeError(
            'Need annotations from at least 3 stacks. '
            f'Found {len(stacks)}: {stacks}')

    rng = random.Random(SEED)
    shuffled_stacks = stacks[:]
    rng.shuffle(shuffled_stacks)
    n_val_stacks = max(1, round(len(shuffled_stacks) * VAL_FRAC))
    val_stacks   = set(shuffled_stacks[:n_val_stacks])
    train_stacks = set(shuffled_stacks[n_val_stacks:])

    train_pairs = [p for p in pairs if p['stack'] in train_stacks]
    val_pairs   = [p for p in pairs if p['stack'] in val_stacks]

    train_imgs  = [p['img']  for p in train_pairs]
    train_masks = [p['mask'] for p in train_pairs]
    val_imgs    = [p['img']  for p in val_pairs]
    val_masks   = [p['mask'] for p in val_pairs]

    log.info(f'  Train stacks ({len(train_stacks)}): {sorted(train_stacks)}')
    log.info(f'  Val stacks   ({len(val_stacks)}): {sorted(val_stacks)}')
    log.info(f'  Train patches ({len(train_pairs)}): {[p["name"] for p in train_pairs]}')
    log.info(f'  Val patches   ({len(val_pairs)}):   {[p["name"] for p in val_pairs]}')

    log.info('Loading cpsam...')
    model = models.CellposeModel(gpu=True)

    log.info(f'Fine-tuning:  lr={LEARNING_RATE}  wd={WEIGHT_DECAY}  '
             f'batch={BATCH_SIZE}  epochs={N_EPOCHS}  '
             f'nimg_per_epoch={max(2, len(train_imgs))}')

    saved_path, train_losses, val_losses = train.train_seg(
        model.net,
        train_data=train_imgs,
        train_labels=train_masks,
        test_data=val_imgs,
        test_labels=val_masks,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        nimg_per_epoch=max(2, len(train_imgs)),
        model_name=model_name,
        save_path=str(run_dir),
    )

    saved_path = Path(saved_path)
    log.info(f'Model saved → {saved_path}')

    latest_file = RUNS_DIR / 'latest_model_path.txt'
    latest_file.write_text(str(saved_path))
    log.info(f'latest_model_path.txt → {latest_file}')

    eval_kw    = dict(diameter=DIAMETER_PX, channels=[0, 0],
                      flow_threshold=0.4, cellprob_threshold=0.0)
    model_ft   = models.CellposeModel(gpu=True, pretrained_model=str(saved_path))
    model_base = models.CellposeModel(gpu=True)
    pred_base  = [model_base.eval(img, **eval_kw)[0] for img in val_imgs]
    pred_ft    = [model_ft.eval(img,   **eval_kw)[0] for img in val_imgs]
    ap_base_per = metrics.average_precision(val_masks, pred_base)[0][:, 0]
    ap_ft_per   = metrics.average_precision(val_masks, pred_ft)[0][:, 0]
    ap_base     = float(ap_base_per.mean())
    ap_ft       = float(ap_ft_per.mean())
    log.info(f'AP@IoU0.5 — base cpsam: {ap_base:.3f}  fine-tuned: {ap_ft:.3f}')

    run_info = {
        'run_name':        run_name,
        'date':            str(date.today()),
        'annotated_stacks': annotated_train,
        'train_stacks':    sorted(train_stacks),
        'val_stacks':      sorted(val_stacks),
        'n_train_patches': len(train_pairs),
        'n_val_patches':   len(val_pairs),
        'n_cells_total':   n_cells_total,
        'n_epochs':        N_EPOCHS,
        'ap_base':         round(ap_base, 4),
        'ap_finetuned':    round(ap_ft, 4),
        'model_path':      str(saved_path),
        'losses_npz':      str(run_dir / 'losses.npz'),
        'losses_json':     str(run_dir / 'losses.json'),
    }
    info_path = run_dir / 'run_info.json'
    info_path.write_text(json.dumps(run_info, indent=2))
    log.info(f'Run info → {info_path}')

    log.info('Saving figures...')
    save_loss_arrays(train_losses, val_losses, run_dir)
    save_training_curves(train_losses, val_losses, run_dir)
    save_val_comparison(val_imgs, val_masks, pred_base, pred_ft, val_pairs, run_dir)
    save_ap_bars(ap_base_per, ap_ft_per, val_pairs, run_dir)
    save_cell_counts(val_masks, pred_base, pred_ft, val_pairs, run_dir)
    log.info(f'Figures → {fig_dir}')


if __name__ == '__main__':
    main()
