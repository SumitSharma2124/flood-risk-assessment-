
from pathlib import Path
import json

import numpy as np
import rasterio


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
OUTPUT = ROOT / "outputs" / "tables"

FEATURE_BANDS = [1, 2, 3, 6, 7, 8, 9, 10, 11, 12]

FEATURE_NAMES = [
    "elevation",
    "slope",
    "twi",
    "distance_waterway",
    "trees",
    "cropland",
    "builtup",
    "water",
    "wetland",
    "mangrove",
]


def main():
    with rasterio.open(DATA / "features_stack.tif") as src:
        features = src.read(FEATURE_BANDS).astype(np.float64)

    with rasterio.open(DATA / "task9_spatial_split.tif") as src:
        split = src.read(1)

    with rasterio.open(DATA / "district_mask.tif") as src:
        district = src.read(1) == 1

    with rasterio.open(DATA / "features_valid_mask.tif") as src:
        feature_valid = src.read(1) > 0

    with rasterio.open(DATA / "ksdma_modelled_mask.tif") as src:
        target_valid = src.read(1) > 0

    with rasterio.open(DATA / "ksdma_100yr_90m.tif") as src:
        target = src.read(1)
        if src.nodata is not None:
            target_valid &= target != src.nodata

    train_mask = (
        (split == 1)
        & district
        & feature_valid
        & target_valid
        & np.isfinite(target)
    )

    if not np.any(train_mask):
        raise RuntimeError("No eligible training cells found.")

    stats = {}

    print("TASK 10 — TRAINING FEATURE NORMALISATION")
    print("=" * 60)
    print("Training cells:", int(train_mask.sum()))

    for i, name in enumerate(FEATURE_NAMES):
        values = features[i][train_mask]
        values = values[np.isfinite(values)]

        if len(values) == 0:
            raise RuntimeError(f"No valid values for {name}")

        mean = float(values.mean())
        std = float(values.std())

        if std == 0:
            raise RuntimeError(f"Zero standard deviation for {name}")

        stats[name] = {
            "mean": mean,
            "std": std,
            "count": int(len(values)),
        }

        print(
            f"{name:<22} "
            f"mean={mean:>12.5f} "
            f"std={std:>12.5f}"
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT / "10_unet_feature_normalisation.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
