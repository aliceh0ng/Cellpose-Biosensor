#!/usr/bin/env python3
"""
scripts/make_splits.py
Randomly assign whole image stacks to train / val / test splits.

These are stack-level splits, not the internal patch-level train/validation
split used by Cellpose during fine-tuning. In the current workflow:
  - `train` stacks are eligible for annotation and fine-tuning
  - `val` stacks are reserved for external evaluation / pipeline checks
  - `test` stacks are held out for final evaluation

Run once on Sockeye (or locally), then review data/splits.json before
running any inference or pipeline scripts.
Edit splits.json freely — it is the single source of truth for all scripts.
"""
import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR      = PROJECT_ROOT / 'data' / 'raw' / '16bit'

VAL_FRAC  = 0.15   # ~15 % reserved as a stack-level validation split
TEST_FRAC = 0.15   # ~15 % completely held out as a stack-level test split
SEED      = 42

stacks = sorted(f.name for f in RAW_DIR.glob('*.tif'))
if not stacks:
    raise FileNotFoundError(f'No TIFFs found in {RAW_DIR}')

print(f'Found {len(stacks)} stacks')

rng = random.Random(SEED)
shuffled = stacks[:]
rng.shuffle(shuffled)

n        = len(shuffled)
n_test   = max(1, round(n * TEST_FRAC))
n_val    = max(1, round(n * VAL_FRAC))
n_train  = n - n_test - n_val

splits = {
    'train': sorted(shuffled[:n_train]),
    'val':   sorted(shuffled[n_train:n_train + n_val]),
    'test':  sorted(shuffled[n_train + n_val:]),
}

out = PROJECT_ROOT / 'data' / 'splits.json'
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, 'w') as f:
    json.dump(splits, f, indent=2)

print(f'\nSplit (seed={SEED}):')
for name, lst in splits.items():
    print(f'  {name:5s} ({len(lst):2d}): {lst}')
print(f'\nSaved → {out}')
print('Review splits.json - edit freely before running inference.')
