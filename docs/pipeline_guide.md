# Cellpose-SAM Pipeline Guide

How to run the segmentation pipeline on a new image stack, and how to fine-tune Cellpose-SAM on your own annotated data.

**Stack:** Cellpose 4.1.1 · conda env: `cellpose` · macOS / Windows / Linux

---

## Overview

This pipeline segments individual bacterial cells in Zeiss Airyscan confocal images of mouse colon tissue sections, then measures per-cell fluorescence in four channels (BFP, GFP, RFP, SYTOX) to quantify osmotic and oxidative stress reporter activity.

### Image format

| Property | Value |
| --- | --- |
| File format | TIFF, `(4, 6323, 6344)` uint16 stacks exported from Zen |
| Pixel size | 0.035 µm/px |
| Channel 0 (C1) | BFP — constitutive reporter → **segmentation input** |
| Channel 1 (C2) | GFP — osmotic-stress promoter |
| Channel 2 (C3) | RFP — oxidative-stress promoter |
| Channel 3 (C4) | SYTOX far-red — host nuclear stain (exclusion only) |

### Pipeline at a glance

```
raw uint16 TIFF
  → Normalise BFP (p1–p99)
  → Cellpose-SAM (diameter = 43 px)
  → Size filter (100–2000 px²)
  → Measure intensities from raw uint16 stack
  → CSV + mask TIFFs
```

> ⚠️ **Critical invariant:** Segmentation uses the *normalised* BFP image. Intensity measurement always uses the *raw uint16 stack*. These two paths must never be swapped.

---

## Environment setup

Two separate conda environments — keep them separate, they conflict.

### cellpose (main environment)

Used for fine-tuning, inference, and measurement.

```bash
# Create from scratch
conda env create -f environment.yml
conda activate cellpose
pip install -e .

# Verify GPU
python -c "import torch; print(torch.backends.mps.is_available())"  # macOS
python -c "import torch; print(torch.cuda.is_available())"           # Windows/Linux
```

### omnipose (baseline comparison only)

CPU-only. Used only for `scripts/run_omnipose_baseline.py`.

```bash
conda activate omnipose
```

### Getting the fine-tuned model

Fine-tuned weights are too large for GitHub, so `models/` is gitignored in this repo. They're hosted separately on the Hugging Face Hub: [huggingface.co/aliceh0ng/cellpose-biosensor](https://huggingface.co/aliceh0ng/cellpose-biosensor).

After cloning, pull the model before running the pipeline:

```bash
conda activate cellpose
python scripts/download_model.py                # recommended run: 4stacks_5x5_norm
python scripts/download_model.py --run 5stacks   # or a specific run
```

This downloads `models/finetuned/<run>/` and updates `models/finetuned/latest_model_path.txt`, which the pipeline reads automatically. Skip this step to run with the base `cpsam` model instead.

### Adding your images

`data/raw/16bit_all/` doesn't exist yet after a fresh clone (raw images aren't checked into git) — create it and copy your stacks in:

```bash
mkdir -p data/raw/16bit_all
cp /path/to/your/*.tif data/raw/16bit_all/
```

**Required file format:**

| Property | Requirement |
| --- | --- |
| File type | TIFF, 16-bit unsigned integer (`uint16`) — use the 16-bit export from Zen, **not** the RGB preview/quick-look export |
| Shape | `(4, H, W)` — one z-plane, 4 channels first. `(Z, 4, H, W)` is also accepted (max-projected automatically) |
| Channel order | `[0]` BFP (constitutive → segmentation input) · `[1]` GFP (osmotic-stress reporter) · `[2]` RFP (oxidative-stress reporter) · `[3]` SYTOX (host nuclear stain, exclusion only) |
| Pixel size | 0.035 µm/px — if yours differs, update `DIAMETER_PX` (`round(1.5 / pixel_size)`) |

> ⚠️ `load_stack()` in `src/io.py` validates dtype and shape on load and raises a clear error (not a silent failure) if you accidentally pass the RGB export instead of the 16-bit one.

Treat `data/raw/` as read-only once populated — nothing in the pipeline writes there.

---

## Running the segmentation pipeline

Two options: the interactive Jupyter notebook (single image, QC) or the batch Python script (all images at once).

### Option A — Notebook: `notebooks/06_segmentation_pipeline.ipynb`

**Step 1 — Activate and open**

```bash
conda activate cellpose
jupyter lab
```

Open `notebooks/06_segmentation_pipeline.ipynb`.

**Step 2 — Set the image path and model**

In the configuration cell:

```python
STACK_PATH = PROJECT_ROOT / 'data/raw/16bit_all/YOUR_IMAGE.tif'

# Auto-loaded from latest_model_path.txt — set to None for base cpsam
MODEL_PATH = None

MIN_AREA = 100   # px² — remove objects smaller than this (noise)
MAX_AREA = 2000  # px² — remove objects larger than this (debris / host cells)
```

**Step 3 — Run all cells (Kernel → Restart and Run All)**

The notebook will:
- Load the image stack and model
- Normalise the BFP channel (p1–p99 percentile stretch)
- Run Cellpose-SAM on the full image at diameter = 43 px
- Apply the size filter (100–2000 px²)
- Save raw and filtered masks as TIFFs
- Measure per-cell intensities from the raw uint16 stack
- Save results as CSV
- Show QC figures (mask overview + intensity histograms)

> 💡 Check the mask overview: a 1:8 BFP thumbnail with red cell outlines. Zoom into dense regions to verify boundary quality.

---

### Option B — Script: `scripts/pipeline.py`

Functionally identical to the notebook. Skips images whose CSV already exists — delete the CSV to reprocess.

```bash
conda activate cellpose

# Test on a single image first
python scripts/pipeline.py --single 20260416-C3M2_Tcol_1.tif

# Run all .tif files in data/raw/16bit_all/
python scripts/pipeline.py
```

On a SLURM cluster (Sockeye):

```bash
sbatch jobs/pipeline.sh --split all
```

---

### Key parameters

| Parameter | Default | What it controls |
| --- | --- | --- |
| `DIAMETER_PX` | `43` | Expected cell diameter in pixels. Set as `round(1.5 / 0.035)`. Change if pixel size differs. |
| `MIN_AREA` | `100` | Minimum cell area (px²). Below this = noise. |
| `MAX_AREA` | `2000` | Maximum cell area (px²). Above this = debris / host cells. |
| `LOCAL_BG` | `False` | If `True`, subtracts a per-cell 5 px annular ring background before computing ratios. |
| `FLOW_THRESH` | `0.4` | Cellpose flow error threshold. Lower = stricter (fewer, cleaner cells). |
| `CELLPROB_THRESH` | `0.0` | Cell probability threshold. Raise to require higher confidence. |
| `MODEL_PATH` | auto | Fine-tuned model weights. Auto-loaded from `latest_model_path.txt`. Set `None` for base cpsam. |

---

### Outputs per image

| File | Description |
| --- | --- |
| `data/processed/<name>_masks_raw.tif` | Integer label mask before size filter |
| `data/processed/<name>_masks.tif` | Final label mask after filters |
| `data/results/<name>.csv` | Per-cell intensity measurements |
| `figures/qc/<name>_masks_overview.png` | BFP thumbnail + red cell outlines |
| `figures/qc/<name>_intensity_qc.svg` | BFP, GFP/BFP, RFP/BFP histograms |

The script also writes `data/results/filter_steps.csv` (raw vs filtered cell counts per image).

---

## Training your own model

### What fine-tuning does

Fine-tuning adapts the Cellpose-SAM (cpsam) foundation model to your specific images using a small number of hand-annotated examples. In practice, 5–9 annotated patches (~500×500 px tiles) is sufficient to meaningfully improve over the base model.

The fine-tuned model is saved to `models/finetuned/` and automatically picked up by the pipeline via `latest_model_path.txt`.

> Run the base cpsam on your images first. If segmentation looks reasonable, fine-tuning will likely improve boundary accuracy and reduce false positives.

---

### Step 1 — Export and annotate patches

**1a. Export patches**

Run `notebooks/01_tiling_preprocessing.ipynb` on your training image. This splits the image into a 3×3 grid and saves BFP patches to `data/annotations/images/`.

```bash
conda activate cellpose
jupyter nbconvert --to notebook --execute notebooks/01_tiling_preprocessing.ipynb
```

Outputs: `patch_r0_c0.tif` through `patch_r2_c2.tif` (9 patches)

**1b. Open Cellpose GUI**

```bash
conda activate cellpose
python -m cellpose
```

**1c. Annotate each patch**

1. Open a patch from `data/annotations/images/`
2. Set **Calibrate diameter** to ~43 px
3. Run **cpsam** as a starting point
4. Correct the masks:
    - Add missed cells (right-click drag to draw outline)
    - Delete false positives (select + Delete)
    - Fix boundaries where needed
5. Click **Save** → writes `patch_r?_c?_seg.npy` next to the image
6. Repeat for all 9 patches (minimum 5 before first training run)

> ⚠️ Annotation quality matters more than quantity. A few carefully corrected patches outperform many hastily annotated ones. Focus on regions where the base model performs worst.

---

### Step 2 — Fine-tune: `notebooks/04_finetuning.ipynb`

Recommended for local training on a single image.

**2a. Configure**

```python
N_EPOCHS      = 100    # increase to 200 if val loss still falling at 100
LEARNING_RATE = 1e-5
WEIGHT_DECAY  = 0.1
BATCH_SIZE    = 1
VAL_FRAC      = 0.2    # 20% of patches held out for validation
MODEL_NAME    = 'cpsam_finetuned'
```

**2b. Check annotation QC cell**

The notebook shows all annotated patches with red outlines before training begins. If a patch looks wrong, re-annotate it in the GUI first.

**2c. Run all cells**

Training uses MPS (macOS) or CUDA (Windows/Linux). 100 epochs takes ~5–15 minutes.

When done, the notebook:
- Saves weights to `models/finetuned/models/cpsam_finetuned`
- Records the path in `models/finetuned/latest_model_path.txt`
- Plots training/validation loss curves
- Shows base cpsam vs fine-tuned comparison on held-out patches (AP @ IoU 0.5)

> 💡 Training loss should decrease steadily. If val loss rises while train loss keeps falling, you are overfitting — reduce `N_EPOCHS` or annotate more patches.

---

### Step 2b — Fine-tune (script): `scripts/finetune.py`

Use this when training on multiple annotated stacks or on a cluster.

Prerequisites: `data/splits.json` (from `make_splits.py`) and `data/annotations.json`.

Example `data/annotations.json`:

```json
{
  "annotated_train": [
    "20260416-C3M2_Tcol_1.tif",
    "20260416-C3M3_Pcol_2.tif",
    "20260416-C4M3_Dcol_1.tif"
  ]
}
```

```bash
# Give the run a descriptive name
python scripts/finetune.py --run-name 5stacks

# With custom patches directory
python scripts/finetune.py --run-name 5stacks_v2 --patches-dir data/patches/v2

# On Sockeye (SLURM)
sbatch jobs/finetune.sh 5stacks
```

**Outputs per run:**

| File | Description |
| --- | --- |
| `models/finetuned/<run-name>/models/cpsam_<run-name>` | Fine-tuned weights |
| `models/finetuned/<run-name>/run_info.json` | Date, stacks used, AP scores |
| `models/finetuned/<run-name>/figures/training_curves.png` | Loss curves |
| `models/finetuned/<run-name>/figures/val_comparison.png` | Ground truth vs base vs fine-tuned |
| `models/finetuned/<run-name>/figures/ap_bars.png` | Per-patch AP@0.5 bars |
| `models/finetuned/latest_model_path.txt` | Updated to this run |

---

### Step 3 — Evaluate

Check `run_info.json`:

```json
{
  "ap_base":      0.512,
  "ap_finetuned": 0.681
}
```

- **AP @ IoU 0.5 > base cpsam** → proceed to the full pipeline
- **AP ≤ base cpsam** — try:
    - Annotate more patches, focusing on regions where the base model fails
    - Increase `N_EPOCHS` to 200 if val loss was still decreasing
    - Re-check annotation quality (mis-labels hurt more than fewer clean labels)

---

### Hyperparameters

| Parameter | Default | Notes |
| --- | --- | --- |
| `LEARNING_RATE` | `1e-5` | Conservative — avoids overwriting SAM backbone. Do not exceed `1e-4`. |
| `WEIGHT_DECAY` | `0.1` | L2 regularisation. Use as-is. |
| `BATCH_SIZE` | `1` | SAM is memory-heavy. 1 is standard. |
| `N_EPOCHS` | `100` | Increase to 200 if val loss still falling at epoch 100. |
| `VAL_FRAC` | `0.2` | Fraction of patches (notebook) or stacks (script) held out for validation. |

> **Cellpose 4.x API:** Use `train.train_seg(model.net, ...)`, not `model.train(...)`. The latter is a 2.x wrapper. `train_seg` returns a 3-tuple: `(model_path, train_losses, val_losses)`.

---

## Understanding the CSV output

Each image produces `data/results/<name>.csv` with one row per segmented cell.

| Column | Type | Description |
| --- | --- | --- |
| `image_id` | str | Stack filename stem |
| `cell_id` | int | Integer label from the mask (1-indexed) |
| `area_px` | int | Cell area in pixels |
| `centroid_y`, `centroid_x` | float | Cell centroid coordinates (pixels) |
| `bfp_raw`, `gfp_raw`, `rfp_raw`, `sytox_raw` | float | Mean pixel intensity per channel from raw uint16 data (ADU) |
| `bfp_bg`, `gfp_bg`, `rfp_bg` | float | Local background (5 px ring). Zero if `LOCAL_BG=False`. |
| `bfp`, `gfp`, `rfp` | float | Background-subtracted intensity (`raw - bg`, floored at 0) |
| `gfp_over_bfp` | float | GFP/BFP ratio. Osmotic stress reporter. `NaN` if BFP = 0. |
| `rfp_over_bfp` | float | RFP/BFP ratio. Oxidative stress reporter. `NaN` if BFP = 0. |

> ⚠️ GFP and RFP channels show signal even when reporters are OFF — this is fecal autofluorescence. Always background-subtract before interpreting ratios.

---

## Tips and common issues

### Pipeline

- **Slow segmentation** — verify MPS or CUDA is active before running.
- **"Skip — CSV exists"** — delete `data/results/<name>.csv` to reprocess an image.
- **Too many / too few cells** — raise `CELLPROB_THRESH` for stricter detection; adjust `FLOW_THRESH` for denser regions.
- **Size filter removing real cells** — check `df['area_px'].describe()` and widen `MIN_AREA`/`MAX_AREA`.

### Fine-tuning

- **AP does not improve** — most commonly caused by annotation errors. Re-examine validation patches in the GUI.
- **MPS autocast warnings during training** — cosmetic, safe to ignore.

### Display vs. measurement normalisation

| Use | Normalisation |
| --- | --- |
| Cellpose-SAM input | p1–p99 percentile stretch → float32 [0, 1] |
| QC thumbnail display | p1–p99.9 (avoids autofluorescence blobs compressing range) |
| Intensity measurement | **None — raw uint16 only** |
