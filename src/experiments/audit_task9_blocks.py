
import sys
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import config as cfg


BLOCK_SIZE = 5000  # metres

with rasterio.open(cfg.output_path("district_mask")) as ds:
    district = ds.read(1) == 1
    transform = ds.transform

with rasterio.open(cfg.output_path("features_valid_mask")) as ds:
    features_valid = ds.read(1) > 0

with rasterio.open(cfg.output_path("ksdma_modelled_mask")) as ds:
    modelled = ds.read(1) > 0

with rasterio.open(cfg.output_path("ksdma_100yr_90m")) as ds:
    target = ds.read(1)
    target_nodata = ds.nodata

eligible = district & features_valid & modelled
if target_nodata is not None:
    eligible &= target != target_nodata
eligible &= np.isfinite(target)

rows, cols = np.where(eligible)

x = transform.c + (cols + 0.5) * transform.a
y = transform.f + (rows + 0.5) * transform.e

block_x = np.floor(x / BLOCK_SIZE).astype(int)
block_y = np.floor(y / BLOCK_SIZE).astype(int)

block_ids = np.array([
    f"{bx}_{by}" for bx, by in zip(block_x, block_y)
])

unique, counts = np.unique(block_ids, return_counts=True)

print("Eligible target cells:", len(rows))
print("Occupied 5 km blocks:", len(unique))
print("\nCells per block:")

for block, count in sorted(
    zip(unique, counts), key=lambda item: item[1], reverse=True
):
    print(f"{block}: {count}")

print("\nSmallest block:", int(counts.min()))
print("Largest block:", int(counts.max()))
print("Mean cells per block:", round(float(counts.mean()), 1))

