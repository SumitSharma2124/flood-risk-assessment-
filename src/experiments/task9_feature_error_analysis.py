
"""
Task 9 — Feature-wise XGBoost validation error analysis.

Investigates relationships between model errors and input features.

Uses validation cells only.
Does not retrain models or access the test set.

Run from project root:
    python src/experiments/task9_feature_error_analysis.py
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import rasterio


# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT / "src"))
import config as cfg

RASTER_DIR = ROOT / "outputs" / "rasters"
TABLE_DIR = ROOT / "outputs" / "tables"

TABLE_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_PATH = Path(cfg.output_path("features_stack"))
if not FEATURE_PATH.is_absolute():
    FEATURE_PATH = ROOT / FEATURE_PATH

RESIDUAL_PATH = RASTER_DIR / "09_xgb_validation_residuals.tif"
ERROR_PATH = RASTER_DIR / "09_xgb_validation_absolute_error.tif"
SPLIT_PATH = ROOT / "data" / "processed" / "task9_spatial_split.tif"


# --------------------------------------------------
# FEATURE DEFINITIONS
# --------------------------------------------------

# Band numbers are 1-based, matching the original feature stack.
FEATURE_BANDS = {
    "elevation": 1,
    "slope": 2,
    "TWI": 3,
    "distance_waterway": 6,
    "trees": 7,
    "cropland": 8,
    "builtup": 9,
    "water": 10,
    "wetland": 11,
    "mangrove": 12,
}


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def read_single_band(path):
    if not path.exists():
        raise FileNotFoundError(f"Raster not found: {path}")

    with rasterio.open(path) as src:
        array = src.read(1).astype(np.float64)
        nodata = src.nodata
        profile = src.profile.copy()

    if nodata is not None:
        array[array == nodata] = np.nan

    array[~np.isfinite(array)] = np.nan

    return array, profile


def pearson_correlation(x, y):
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def spearman_correlation(x, y):
    if len(x) < 2:
        return np.nan

    x_rank = pd.Series(x).rank().to_numpy()
    y_rank = pd.Series(y).rank().to_numpy()

    return pearson_correlation(x_rank, y_rank)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("Loading feature stack...")

    if not FEATURE_PATH.exists():
        raise FileNotFoundError(
            f"Feature stack not found: {FEATURE_PATH}"
        )

    with rasterio.open(FEATURE_PATH) as src:
        features = src.read().astype(np.float64)
        feature_profile = src.profile.copy()
        feature_transform = src.transform
        feature_crs = src.crs

    print("Loading validation diagnostics...")

    residual, residual_profile = read_single_band(RESIDUAL_PATH)
    absolute_error, error_profile = read_single_band(ERROR_PATH)
    split, split_profile = read_single_band(SPLIT_PATH)

    # --------------------------------------------------
    # GRID VALIDATION
    # --------------------------------------------------

    expected_shape = features.shape[1:]

    for name, array in [
        ("Residual", residual),
        ("Absolute error", absolute_error),
        ("Split", split),
    ]:
        if array.shape != expected_shape:
            raise ValueError(
                f"{name} raster shape {array.shape} "
                f"does not match feature stack {expected_shape}"
            )

    for name, profile in [
        ("Residual", residual_profile),
        ("Absolute error", error_profile),
        ("Split", split_profile),
    ]:
        if profile["transform"] != feature_transform:
            raise ValueError(f"{name} raster transform does not match.")

        if profile["crs"] != feature_crs:
            raise ValueError(f"{name} raster CRS does not match.")

    # --------------------------------------------------
    # VALIDATION MASK
    # --------------------------------------------------

    # Split values:
    # 0 = excluded
    # 1 = train
    # 2 = validation
    # 3 = test

    validation_mask = (
        (split == 2)
        & np.isfinite(residual)
        & np.isfinite(absolute_error)
        & (absolute_error >= 0)
    )

    rows, cols = np.where(validation_mask)

    if len(rows) == 0:
        raise RuntimeError("No valid validation cells found.")

    print(f"Validation cells available: {len(rows):,}")

    # --------------------------------------------------
    # BUILD VALIDATION DATAFRAME
    # --------------------------------------------------

    data = {
        "residual": residual[rows, cols],
        "absolute_error": absolute_error[rows, cols],
    }

    for feature_name, band in FEATURE_BANDS.items():
        data[feature_name] = features[band - 1, rows, cols]

    df = pd.DataFrame(data)

    # Remove cells with invalid feature values.
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    if len(df) == 0:
        raise RuntimeError("No complete validation feature records found.")

    print(f"Complete validation records: {len(df):,}")

    # --------------------------------------------------
    # 1. FEATURE-ERROR CORRELATIONS
    # --------------------------------------------------

    correlation_rows = []

    for feature_name in FEATURE_BANDS:

        x = df[feature_name].to_numpy()
        abs_error = df["absolute_error"].to_numpy()
        signed_error = df["residual"].to_numpy()

        correlation_rows.append({
            "feature": feature_name,
            "pearson_abs_error": pearson_correlation(x, abs_error),
            "spearman_abs_error": spearman_correlation(x, abs_error),
            "pearson_signed_residual": pearson_correlation(x, signed_error),
            "spearman_signed_residual": spearman_correlation(x, signed_error),
        })

    correlation_df = pd.DataFrame(correlation_rows)

    correlation_df["abs_spearman_error"] = (
        correlation_df["spearman_abs_error"].abs()
    )

    correlation_df = correlation_df.sort_values(
        "abs_spearman_error",
        ascending=False,
    )

    correlation_path = (
        TABLE_DIR / "09_xgb_feature_error_correlations.csv"
    )

    correlation_df.to_csv(correlation_path, index=False)

    # --------------------------------------------------
    # 2. ERROR BY FEATURE QUANTILE
    # --------------------------------------------------

    group_rows = []

    for feature_name in FEATURE_BANDS:

        temp = df[[feature_name, "residual", "absolute_error"]].copy()

        try:
            temp["feature_group"] = pd.qcut(
                temp[feature_name],
                q=5,
                labels=False,
                duplicates="drop",
            )
        except ValueError:
            print(
                f"Skipping quantile analysis for {feature_name}: "
                "insufficient unique values."
            )
            continue

        grouped = temp.groupby("feature_group", observed=True)

        for group_id, group in grouped:

            group_rows.append({
                "feature": feature_name,
                "quantile_group": int(group_id) + 1,
                "cells": len(group),
                "feature_min": group[feature_name].min(),
                "feature_max": group[feature_name].max(),
                "feature_mean": group[feature_name].mean(),
                "MAE": group["absolute_error"].mean(),
                "RMSE": np.sqrt(
                    np.mean(group["residual"].to_numpy() ** 2)
                ),
                "mean_residual": group["residual"].mean(),
            })

    group_df = pd.DataFrame(group_rows)

    group_path = (
        TABLE_DIR / "09_xgb_error_by_feature_quantile.csv"
    )

    group_df.to_csv(group_path, index=False)

    # --------------------------------------------------
    # 3. SUMMARY
    # --------------------------------------------------

    summary = pd.DataFrame([{
        "validation_cells": len(df),
        "overall_MAE": df["absolute_error"].mean(),
        "overall_RMSE": np.sqrt(
            np.mean(df["residual"].to_numpy() ** 2)
        ),
        "overall_mean_residual": df["residual"].mean(),
        "underprediction_cells": int((df["residual"] > 0).sum()),
        "overprediction_cells": int((df["residual"] < 0).sum()),
    }])

    summary_path = (
        TABLE_DIR / "09_xgb_feature_error_summary.csv"
    )

    summary.to_csv(summary_path, index=False)

    # --------------------------------------------------
    # RESULTS
    # --------------------------------------------------

    print("\n=== OVERALL VALIDATION SUMMARY ===")
    print(summary.to_string(index=False))

    print("\n=== FEATURE CORRELATION WITH ABSOLUTE ERROR ===")
    print(
        correlation_df[
            [
                "feature",
                "pearson_abs_error",
                "spearman_abs_error",
                "spearman_signed_residual",
            ]
        ].to_string(index=False)
    )

    print("\n=== ERROR BY FEATURE QUANTILE ===")

    for feature_name in FEATURE_BANDS:
        print(f"\n--- {feature_name} ---")

        subset = group_df[
            group_df["feature"] == feature_name
        ]

        print(
            subset[
                [
                    "quantile_group",
                    "cells",
                    "feature_min",
                    "feature_max",
                    "MAE",
                    "RMSE",
                    "mean_residual",
                ]
            ].to_string(index=False)
        )

    print("\nSaved outputs:")
    print(correlation_path)
    print(group_path)
    print(summary_path)

    print("\nOnly validation data was used.")
    print("No model was retrained.")
    print("Test data was not accessed.")


if __name__ == "__main__":
    main()
