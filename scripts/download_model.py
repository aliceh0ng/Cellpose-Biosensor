"""Download a fine-tuned Cellpose-SAM model run from the Hugging Face Hub.

Pulls models/finetuned/<run>/ from https://huggingface.co/aliceh0ng/cellpose-biosensor
into the local models/ directory and updates latest_model_path.txt so
pipeline.py picks it up automatically.

Usage:
    python scripts/download_model.py                     # recommended run
    python scripts/download_model.py --run 5stacks        # a specific run
"""
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HF_REPO_ID = "aliceh0ng/cellpose-biosensor"
DEFAULT_RUN = "4stacks_5x5_norm"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=DEFAULT_RUN, help="Model run name on the HF repo")
    args = parser.parse_args()

    snapshot_download(
        repo_id=HF_REPO_ID,
        allow_patterns=[f"models/finetuned/{args.run}/*"],
        local_dir=PROJECT_ROOT,
    )

    model_dir = PROJECT_ROOT / "models" / "finetuned" / args.run / "models"
    model_files = [p for p in model_dir.iterdir() if p.is_file()] if model_dir.exists() else []
    if not model_files:
        raise SystemExit(f"No model weight file found under {model_dir}")
    model_path = model_files[0]

    latest_path_file = PROJECT_ROOT / "models" / "finetuned" / "latest_model_path.txt"
    latest_path_file.write_text(str(model_path))

    print(f"Downloaded: {model_path}")
    print(f"Updated:    {latest_path_file}")


if __name__ == "__main__":
    main()
