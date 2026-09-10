"""Run legacy Cellpose 3 baseline models on the saved comparison patch.

Use this script in an environment with Cellpose 3.x installed, not the
Cellpose 4.x environment used for cpsam. In Cellpose 4.x, CP3 model names such
as ``cyto3`` and ``bact_fluor_cp3`` can fall back to the default cpsam model,
which makes the benchmark invalid.

Expected input, created by ``notebooks/02_cellpose_baseline.ipynb``:
    data/processed/comparison_patch_bfp.tif

Outputs:
    data/processed/masks_cyto3.tif
    data/processed/masks_bact_fluor_cp3.tif
"""

from pathlib import Path
import time

import numpy as np
import tifffile
from cellpose import models


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data/processed/comparison_patch_bfp.tif"
OUTPUTS = {
    "cyto3": PROJECT_ROOT / "data/processed/masks_cyto3.tif",
    "bact_fluor_cp3": PROJECT_ROOT / "data/processed/masks_bact_fluor_cp3.tif",
}

PIXEL_SIZE_UM = 0.035
DIAMETER_PX = 1.5 / PIXEL_SIZE_UM


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing {INPUT_PATH}. Run notebooks/02_cellpose_baseline.ipynb "
            "first to save the shared comparison patch."
        )

    image = tifffile.imread(INPUT_PATH).astype(np.float32)
    print(f"Loaded {INPUT_PATH} with shape={image.shape}, dtype={image.dtype}")
    print(f"Using Cellpose diameter={DIAMETER_PX:.1f} px")

    for model_name, output_path in OUTPUTS.items():
        print(f"\n-- Cellpose 3 {model_name} --")
        model = models.CellposeModel(model_type=model_name, gpu=True)

        t0 = time.time()
        masks, *_ = model.eval(
            image,
            diameter=DIAMETER_PX,
            channels=[0, 0],
            flow_threshold=0.4,
            cellprob_threshold=0.0,
        )
        elapsed = time.time() - t0

        output_path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(output_path, masks.astype(np.uint16), compression="lzw")
        print(f"{model_name}: {int(masks.max())} cells in {elapsed:.1f}s")
        print(f"Saved -> {output_path}")


if __name__ == "__main__":
    main()
