
"""
Task 9 — XGBoost spatial residual diagnostics.

Uses validation cells only. Does not access the test set.

Run from project root:
    python src/experiments/task9_xgb_residual_diagnostics.py
"""

from pathlib import Path
import logging

import numpy as np
import pandas as pd
import rasterio
import xgboost as xgb

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import config as cfg


logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("task9_residuals")

ROOT = Path.cwd()

MODEL_PATH = ROOT / "models" / "task9_xgboost.json"

OUTPUT_DIR = ROOT / "outputs"
RASTER_DIR = OUTPUT_DIR / "rasters"
TABLE_DIR = OUTPUT_DIR / "tables"

RASTER_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

# Feature order must match Task 9 baseline training.
FEATURE_BANDS = [1, 2, 3, 6, 7, 8, 9, 10, 11, 12]


def get_raster_path(key):
    path = Path(cfg.output_path(key))
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Missing raster for config key '{key}': {path}")
    return path


def read_raster(key):
    path = get_raster_path(key)
    with rasterio.open(path) as src:
        return src.read(), src.profile.copy(), src.nodata


def read_single_band(key):
    data, profile, nodata = read_raster(key)
    return data[0], profile, nodata


def save_raster(path, array, profile, nodata=-9999.0):
    out_profile = profile.copy()
    out_profile.update(
        count=1,
        dtype="float32",
        nodata=nodata,
        compress="deflate",
        predictor=3,
    )

    output = np.full(array.shape, nodata, dtype=np.float32)
    valid = np.isfinite(array)
    output[valid] = array[valid].astype(np.float32)

    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(output, 1)


def morans_i_4_neighbour(residual_grid, valid_mask):
    """
    Global Moran's I using horizontal and vertical adjacency.
    Only pairs where both cells belong to validation are included.
    """
    values = residual_grid.astype(np.float64)
    valid = valid_mask & np.isfinite(values)

    if valid.sum() < 2:
        return np.nan, 0

    centred = np.zeros_like(values)
    centred[valid] = values[valid] - values[valid].mean()

    numerator_edges = 0.0
    edge_count = 0

    # Horizontal neighbours
    pair = valid[:, :-1] & valid[:, 1:]
    numerator_edges += np.sum(centred[:, :-1][pair] * centred[:, 1:][pair])
    edge_count += int(pair.sum())

    # Vertical neighbours
    pair = valid[:-1, :] & valid[1:, :]
    numerator_edges += np.sum(centred[:-1, :][pair] * centred[1:, :][pair])
    edge_count += int(pair.sum())

    denominator = np.sum(centred[valid] ** 2)

    if edge_count == 0 or denominator == 0:
        return np.nan, edge_count

    n = int(valid.sum())

    # Each undirected edge represents two directed neighbour relationships.
    moran_i = (n / edge_count) * (numerator_edges / denominator)

    return float(moran_i), edge_count


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    log.info("Loading feature stack...")
    features, profile, _ = read_raster("features_stack")

    log.info("Loading validation split and target...")
    split_path = ROOT / "data" / "processed" / "task9_spatial_split.tif"

    if not split_path.exists():
        raise FileNotFoundError(f"Split raster not found: {split_path}")

    with rasterio.open(split_path) as src:
        split = src.read(1)
    target, _, target_nodata = read_single_band("ksdma_100yr_90m")

    district, _, _ = read_single_band("district_mask")
    feature_mask, _, _ = read_single_band("features_valid_mask")
    modelled_mask, _, _ = read_single_band("ksdma_modelled_mask")

    # Validation cells only. Test cells (split == 3) are never selected.
    valid = (
        (split == 2)
        & (district > 0)
        & (feature_mask > 0)
        & (modelled_mask > 0)
        & np.isfinite(target)
    )

    if target_nodata is not None:
        valid &= target != target_nodata

    rows, cols = np.where(valid)

    if len(rows) == 0:
        raise RuntimeError("No eligible validation cells found.")

    log.info("Eligible validation cells: %s", f"{len(rows):,}")

    # Build feature matrix in the same order used during training.
    X = np.column_stack([
        features[band - 1, rows, cols]
        for band in FEATURE_BANDS
    ]).astype(np.float32)

    y_true = target[rows, cols].astype(np.float64)

    finite_features = np.isfinite(X).all(axis=1)
    rows = rows[finite_features]
    cols = cols[finite_features]
    X = X[finite_features]
    y_true = y_true[finite_features]

    log.info("Loading original-scale XGBoost model...")
    model = xgb.XGBRegressor()
    model.load_model(str(MODEL_PATH))

    log.info("Predicting validation cells...")
    y_pred = model.predict(X).astype(np.float64)

    # Positive residual = underprediction.
    # Negative residual = overprediction.
    residual = y_true - y_pred
    absolute_error = np.abs(residual)

    mae = float(np.mean(absolute_error))
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    r2 = float(
        1 - np.sum(residual ** 2) /
        np.sum((y_true - np.mean(y_true)) ** 2)
    )

    log.info("Validation MAE: %.4f", mae)
    log.info("Validation RMSE: %.4f", rmse)
    log.info("Validation R²: %.4f", r2)

    # Create full-grid arrays, with NoData outside validation cells.
    residual_grid = np.full(target.shape, np.nan, dtype=np.float32)
    prediction_grid = np.full(target.shape, np.nan, dtype=np.float32)
    abs_error_grid = np.full(target.shape, np.nan, dtype=np.float32)

    residual_grid[rows, cols] = residual
    prediction_grid[rows, cols] = y_pred
    abs_error_grid[rows, cols] = absolute_error

    save_raster(
        RASTER_DIR / "09_xgb_validation_residuals.tif",
        residual_grid,
        profile,
    )
    save_raster(
        RASTER_DIR / "09_xgb_validation_predictions.tif",
        prediction_grid,
        profile,
    )
    save_raster(
        RASTER_DIR / "09_xgb_validation_absolute_error.tif",
        abs_error_grid,
        profile,
    )

    # 1. Error distribution by actual target-value quantile.
    df = pd.DataFrame({
        "actual": y_true,
        "predicted": y_pred,
        "residual": residual,
        "absolute_error": absolute_error,
    })

    df["target_group"] = pd.qcut(
        df["actual"],
        q=5,
        labels=["Q1_low", "Q2", "Q3", "Q4", "Q5_high"],
        duplicates="drop",
    )

    quantile_summary = (
        df.groupby("target_group", observed=True)
        .agg(
            cells=("actual", "size"),
            actual_mean=("actual", "mean"),
            predicted_mean=("predicted", "mean"),
            mean_residual=("residual", "mean"),
            MAE=("absolute_error", "mean"),
            RMSE=("residual", lambda x: np.sqrt(np.mean(x ** 2))),
        )
        .reset_index()
    )

    quantile_summary.to_csv(
        TABLE_DIR / "09_xgb_residuals_by_target_quantile.csv",
        index=False,
    )

    # 2. Spatial autocorrelation of validation residuals.
    validation_mask = np.zeros(target.shape, dtype=bool)
    validation_mask[rows, cols] = True

    moran_i, neighbour_edges = morans_i_4_neighbour(
        residual_grid,
        validation_mask,
    )

    # 3. Approximate 5 km grid-block error summary.
    # At 90 m resolution, 56 pixels are approximately 5 km.
    block_size = 56

    block_df = pd.DataFrame({
        "block_row": rows // block_size,
        "block_col": cols // block_size,
        "residual": residual,
        "absolute_error": absolute_error,
    })

    block_summary = (
        block_df.groupby(["block_row", "block_col"])
        .agg(
            validation_cells=("residual", "size"),
            mean_residual=("residual", "mean"),
            MAE=("absolute_error", "mean"),
            RMSE=("residual", lambda x: np.sqrt(np.mean(x ** 2))),
        )
        .reset_index()
    )

    block_summary.to_csv(
        TABLE_DIR / "09_xgb_residuals_by_block.csv",
        index=False,
    )

    # Overall summary.
    summary = pd.DataFrame([{
        "validation_cells": len(y_true),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "mean_residual": float(np.mean(residual)),
        "median_absolute_error": float(np.median(absolute_error)),
        "underprediction_cells": int(np.sum(residual > 0)),
        "overprediction_cells": int(np.sum(residual < 0)),
        "moran_i_4_neighbour": moran_i,
        "neighbour_edges": neighbour_edges,
    }])

    summary.to_csv(
        TABLE_DIR / "09_xgb_residual_diagnostics_summary.csv",
        index=False,
    )

    print("\n=== VALIDATION RESIDUAL SUMMARY ===")
    print(summary.to_string(index=False))

    print("\n=== ERROR BY TARGET QUANTILE ===")
    print(quantile_summary.to_string(index=False))

    print("\n=== WORST 10 APPROXIMATE 5 KM BLOCKS ===")
    print(
        block_summary.sort_values("MAE", ascending=False)
        .head(10)
        .to_string(index=False)
    )

    print("\nSaved diagnostic outputs:")
    print(" - outputs/rasters/09_xgb_validation_residuals.tif")
    print(" - outputs/rasters/09_xgb_validation_predictions.tif")
    print(" - outputs/rasters/09_xgb_validation_absolute_error.tif")
    print(" - outputs/tables/09_xgb_residual_diagnostics_summary.csv")
    print(" - outputs/tables/09_xgb_residuals_by_target_quantile.csv")
    print(" - outputs/tables/09_xgb_residuals_by_block.csv")
    print("\nTest data was not used.")


if __name__ == "__main__":
    main()
