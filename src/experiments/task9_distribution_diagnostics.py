
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

SPLITS = {1: "train", 2: "validation", 3: "test"}


def load_raster(key):
    with rasterio.open(cfg.output_path(key)) as ds:
        return ds.read(1)


def main():
    with rasterio.open(cfg.output_path("features_stack")) as ds:
        features = ds.read(FEATURE_BANDS).astype(np.float32)

    target = load_raster("ksdma_100yr_90m")
    feature_mask = load_raster("features_valid_mask") > 0
    modelled = load_raster("ksdma_modelled_mask") > 0
    district = load_raster("district_mask") == 1
    split = load_raster("district_mask")  # replaced below

    split_path = cfg.PROCESSED_DIR / "task9_spatial_split.tif"
    with rasterio.open(split_path) as ds:
        split = ds.read(1)

    valid = (
        district & feature_mask & modelled
        & np.isfinite(target) & (target != -9999)
        & np.all(np.isfinite(features), axis=0)
        & np.isin(split, [1, 2, 3])
    )

    X = features[:, valid].T
    y = target[valid]
    split_values = split[valid]

    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["target"] = y
    df["split"] = [SPLITS[int(v)] for v in split_values]

    print("\n=== SAMPLE COUNTS ===")
    print(df["split"].value_counts())

    # Compare distributions using descriptive statistics.
    print("\n=== FEATURE AND TARGET DISTRIBUTIONS ===")

    for name in ["train", "validation", "test"]:
        subset = df[df["split"] == name]

        print(f"\n--- {name.upper()} ---")
        print(
            subset[FEATURE_NAMES + ["target"]]
            .describe(percentiles=[0.10, 0.25, 0.50, 0.75, 0.90])
            .loc[["mean", "std", "10%", "50%", "90%"]]
            .round(3)
            .to_string()
        )

    # Compare means and standard deviations across splits.
    print("\n=== MEAN DIFFERENCES FROM TRAINING ===")

    train_means = df[df.split == "train"][FEATURE_NAMES + ["target"]].mean()
    train_stds = df[df.split == "train"][FEATURE_NAMES + ["target"]].std()

    rows = []

    for name in ["validation", "test"]:
        subset = df[df.split == name]
        means = subset[FEATURE_NAMES + ["target"]].mean()
        stds = subset[FEATURE_NAMES + ["target"]].std()

        for col in FEATURE_NAMES + ["target"]:
            pooled_std = np.sqrt(
                (train_stds[col] ** 2 + stds[col] ** 2) / 2
            )

            smd = (
                (means[col] - train_means[col]) / pooled_std
                if pooled_std > 0 else 0
            )

            rows.append({
                "split": name,
                "variable": col,
                "train_mean": train_means[col],
                "split_mean": means[col],
                "standardised_mean_difference": smd,
            })

    comparison = pd.DataFrame(rows)

    print(
        comparison.round(3).to_string(index=False)
    )

    print("\n=== TARGET DISTRIBUTION BY SPLIT ===")

    for name in ["train", "validation", "test"]:
        values = df.loc[df.split == name, "target"]

        print(
            f"{name}: "
            f"min={values.min():.2f}, "
            f"max={values.max():.2f}, "
            f"mean={values.mean():.2f}, "
            f"median={values.median():.2f}, "
            f"std={values.std():.2f}"
        )


if __name__ == "__main__":
    main()
