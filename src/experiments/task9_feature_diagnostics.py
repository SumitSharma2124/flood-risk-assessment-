

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import config as cfg

FEATURE_BANDS = [1, 2, 3, 6, 7, 8, 9, 10, 11, 12]

FEATURE_NAMES = [
    "elevation", "slope", "twi", "distance_waterway",
    "trees", "cropland", "builtup", "water",
    "wetland", "mangrove"
]

with rasterio.open(cfg.output_path("features_stack")) as ds:
    X = ds.read(FEATURE_BANDS).astype(np.float32)

with rasterio.open(cfg.output_path("ksdma_100yr_90m")) as ds:
    y = ds.read(1).astype(np.float32)

with rasterio.open(cfg.PROCESSED_DIR / "task9_spatial_split.tif") as ds:
    split = ds.read(1)

with rasterio.open(cfg.output_path("features_valid_mask")) as ds:
    feature_mask = ds.read(1) > 0

with rasterio.open(cfg.output_path("ksdma_modelled_mask")) as ds:
    modelled = ds.read(1) > 0

with rasterio.open(cfg.output_path("district_mask")) as ds:
    district = ds.read(1) == 1

valid = (
    district & feature_mask & modelled
    & np.isfinite(y) & (y != -9999)
    & np.all(np.isfinite(X), axis=0)
    & (split > 0)
)

X_flat = X[:, valid].T
y_flat = y[valid]
split_flat = split[valid]

df = pd.DataFrame(X_flat, columns=FEATURE_NAMES)
df["target"] = y_flat

print("\n=== FEATURE-TARGET PEARSON CORRELATION ===")
print(df.corr(method="pearson")["target"].drop("target").sort_values())

print("\n=== FEATURE-TARGET SPEARMAN CORRELATION ===")
print(df.corr(method="spearman")["target"].drop("target").sort_values())

print("\n=== CORRELATIONS BY SPLIT ===")

for code, name in [(1, "Train"), (2, "Validation"), (3, "Test")]:
    subset = df.loc[split_flat == code]

    print(f"\n{name}: {len(subset)} cells")
    print(
        subset.corr(method="spearman")["target"]
        .drop("target")
        .sort_values()
    )

print("\n=== FEATURE SUMMARY ===")
print(df.describe().T[["mean", "std", "min", "max"]])
