# =============================================================================
# scripts/run_omnipose_baseline.py
# Environment: omnipose  (conda activate omnipose)
#
# Reads the same BFP crop saved by notebook 02, runs Omnipose bact_fluor_omni,
# saves masks for the comparison figure.
#
# Run from the project root:
#   conda activate omnipose
#   python scripts/run_omnipose_baseline.py
# =============================================================================

import numpy as np
import tifffile
import time
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path('/Users/alicehong/projects/cellpose-biosensor')
CROP_PATH     = PROJECT_ROOT / 'data/processed/comparison_patch_bfp.tif'
OUTPUT_PATH   = PROJECT_ROOT / 'data/processed/masks_bact_fluor_omni.tif'
PIXEL_SIZE_UM = 0.035  # µm/px — from Zen Image Properties → Scaling

(PROJECT_ROOT / 'data/processed').mkdir(parents=True, exist_ok=True)

# ── Load crop ─────────────────────────────────────────────────────────────────
crop_raw = tifffile.imread(CROP_PATH).astype(np.float32)
print(f'Crop loaded: {crop_raw.shape} max={crop_raw.max():.0f}')

# Pass raw intensities to Omnipose and use its default internal normalization,
# matching the Cellpose baseline comparison.

# ── Run Omnipose bact_fluor ───────────────────────────────────────────────────
from cellpose_omni import models as omni_models

print('\n── Omnipose bact_fluor_omni ──')
# Correct omnipose 1.x model name is 'bact_fluor_omni', not 'bact_fluor'.
# 'bact_fluor' silently falls back to 'cyto' (2-channel), causing a size mismatch.
model = omni_models.CellposeModel(model_type='bact_fluor_omni', gpu=False)
# Note: gpu=False on Apple Silicon because omnipose doesn't support MPS.

diameter_px = None
if PIXEL_SIZE_UM is not None:
    diameter_px = 1.5 / PIXEL_SIZE_UM
    print(f'diameter = {diameter_px:.1f} px')
else:
    print('diameter = None (auto)')

t0 = time.time()
masks, flows, styles = model.eval(
    crop_raw,
    channels=[0, 0],
    diameter=diameter_px,
    flow_threshold=0.0,   # omnipose default for bacteria
    mask_threshold=0.0,
    omni=True,            # use omnipose mask reconstruction
    cluster=False,
    verbose=False,
)
t_omni = time.time() - t0
n_cells = masks.max()

print(f'Omnipose bact_fluor_omni: {n_cells} cells in {t_omni:.1f}s')

tifffile.imwrite(OUTPUT_PATH, masks.astype(np.uint16), compression='lzw')
print(f'Saved → {OUTPUT_PATH}')
