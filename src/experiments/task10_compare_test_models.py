
"""
Task 10 — Final RF vs XGBoost vs U-Net Test Comparison

Run from project root:
    python src/experiments/task10_compare_test_models.py

Evaluates all three saved models on the exact same Task 9 test cells.

No model is retrained.
No test data is used for model selection or tuning.
The target raster is the common ground truth.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import xgboost as xgb


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

TABLE_DIR = ROOT / "outputs" / "tables"
FIGURE_DIR = ROOT / "outputs" / "figures"

RF_MODEL = MODEL_DIR / "task9_random_forest.joblib"
XGB_MODEL = MODEL_DIR / "task9_xgboost.json"

FEATURES_FILE = DATA_DIR / "features_stack.tif"
TARGET_FILE = DATA_DIR / "ksdma_100yr_90m.tif"
FEATURE_MASK_FILE = DATA_DIR / "features_valid_mask.tif"
TARGET_MASK_FILE = DATA_DIR / "ksdma_modelled_mask.tif"
SPLIT_FILE = DATA_DIR / "task9_spatial_split.tif"

UNET_PREDICTIONS_FILE = (
    TABLE_DIR / "10_unet_test_predictions.csv"
)

PREDICTIONS_FILE = (
    TABLE_DIR / "10_all_models_test_predictions.csv"
)

METRICS_FILE = (
    TABLE_DIR / "10_all_models_test_metrics.json"
)

QUANTILE_FILE = (
    TABLE_DIR / "10_all_models_test_error_by_quantile.csv"
)

COMPARISON_PLOT = (
    FIGURE_DIR / "10_all_models_test_metrics.png"
)

QUANTILE_PLOT = (
    FIGURE_DIR / "10_all_models_test_quantile_mae.png"
)

EXPECTED_TEST_CELLS = 10767

# 1-based feature-stack band indices.
SELECTED_BANDS = [
    1, 2, 3, 6, 7, 8, 9, 10, 11, 12
]

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

N_FEATURES = len(FEATURE_NAMES)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)

logger = logging.getLogger("task10_comparison")


# ============================================================
# 2. RASTER UTILITIES
# ============================================================

def read_raster(path: Path):

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with rasterio.open(path) as src:

        array = src.read()

        metadata = {
            "height": src.height,
            "width": src.width,
            "transform": src.transform,
            "crs": src.crs,
            "nodata": src.nodata,
            "count": src.count,
        }

    return array, metadata


def verify_grid(reference, other, name):

    if reference["height"] != other["height"]:
        raise ValueError(f"{name}: height mismatch.")

    if reference["width"] != other["width"]:
        raise ValueError(f"{name}: width mismatch.")

    if reference["transform"] != other["transform"]:
        raise ValueError(f"{name}: transform mismatch.")

    if reference["crs"] != other["crs"]:
        raise ValueError(f"{name}: CRS mismatch.")


# ============================================================
# 3. LOAD COMMON TEST DATA
# ============================================================

def load_test_data():

    logger.info("Loading raster data...")

    features, feature_meta = read_raster(FEATURES_FILE)
    target, target_meta = read_raster(TARGET_FILE)

    feature_mask, feature_mask_meta = read_raster(
        FEATURE_MASK_FILE
    )

    target_mask, target_mask_meta = read_raster(
        TARGET_MASK_FILE
    )

    split, split_meta = read_raster(SPLIT_FILE)

    verify_grid(feature_meta, target_meta, "Target raster")
    verify_grid(feature_meta, feature_mask_meta, "Feature mask")
    verify_grid(feature_meta, target_mask_meta, "Target mask")
    verify_grid(feature_meta, split_meta, "Spatial split")

    if feature_meta["count"] < max(SELECTED_BANDS):
        raise ValueError("Feature stack is missing selected bands.")

    if any(
        array.shape[0] != 1
        for array in [target, feature_mask, target_mask, split]
    ):
        raise ValueError("Expected single-band target and mask rasters.")

    # Select the same 10 bands used during baseline training.
    features = features[
        np.asarray(SELECTED_BANDS) - 1
    ].astype(np.float32)

    target = target[0].astype(np.float64)
    feature_mask = feature_mask[0]
    target_mask = target_mask[0]
    split = split[0]

    # Reconstruct Task 9 test eligibility.
    eligible = (
        (split == 3)
        & (feature_mask == 1)
        & (target_mask == 1)
        & np.isfinite(target)
    )

    target_nodata = target_meta["nodata"]

    if target_nodata is not None:
        eligible &= target != target_nodata

    feature_nodata = feature_meta["nodata"]

    for band in features:

        eligible &= np.isfinite(band)

        if feature_nodata is not None:
            eligible &= band != feature_nodata

    rows, cols = np.where(eligible)

    if len(rows) != EXPECTED_TEST_CELLS:
        raise ValueError(
            f"Expected {EXPECTED_TEST_CELLS} test cells, "
            f"but reconstructed {len(rows)}. "
            "Check raster masks and spatial split."
        )

    # Shape: (10767, 10)
    X = features[:, rows, cols].T

    # Common ground truth from target raster.
    y = target[rows, cols]

    if X.shape != (EXPECTED_TEST_CELLS, N_FEATURES):
        raise ValueError(f"Unexpected feature shape: {X.shape}")

    if not np.isfinite(X).all():
        raise ValueError("Feature matrix contains non-finite values.")

    if not np.isfinite(y).all():
        raise ValueError("Target values contain non-finite values.")

    logger.info("Test cells: %d", len(y))
    logger.info("Feature matrix: %s", X.shape)
    logger.info("Target vector: %s", y.shape)

    return X, y, rows, cols


# ============================================================
# 4. LOAD AND ALIGN U-NET PREDICTIONS
# ============================================================

def load_unet_predictions(rows, cols, raster_actual):

    if not UNET_PREDICTIONS_FILE.exists():
        raise FileNotFoundError(UNET_PREDICTIONS_FILE)

    logger.info("Loading U-Net test predictions...")

    df = pd.read_csv(UNET_PREDICTIONS_FILE)

    required = {"row", "col", "actual", "predicted"}

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"U-Net CSV is missing columns: {missing}"
        )

    if len(df) != EXPECTED_TEST_CELLS:
        raise ValueError(
            f"Expected {EXPECTED_TEST_CELLS} U-Net predictions, "
            f"found {len(df)}."
        )

    if df.duplicated(["row", "col"]).any():
        raise ValueError("Duplicate coordinates in U-Net CSV.")

    # Use coordinates to align U-Net predictions to raster order.
    df["row"] = df["row"].astype(int)
    df["col"] = df["col"].astype(int)

    df = df.set_index(["row", "col"])

    expected_coordinates = list(
        zip(rows.tolist(), cols.tolist())
    )

    actual_coordinates = set(df.index.tolist())

    if set(expected_coordinates) != actual_coordinates:
        raise ValueError(
            "U-Net coordinates do not match the Task 9 test cells."
        )

    aligned = df.loc[expected_coordinates]

    unet_actual = aligned["actual"].to_numpy(
        dtype=np.float64
    )

    unet_predictions = aligned["predicted"].to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(unet_predictions).all():
        raise ValueError(
            "U-Net predictions contain NaN or infinite values."
        )

    # --------------------------------------------------------
    # Diagnose actual-value mismatch.
    # Do not use CSV actual values as the common ground truth.
    # --------------------------------------------------------

    differences = np.abs(unet_actual - raster_actual)

    logger.info("U-Net CSV actual vs target raster comparison:")
    logger.info(
        "Maximum absolute difference: %.10f",
        differences.max(),
    )
    logger.info(
        "Mean absolute difference: %.10f",
        differences.mean(),
    )
    logger.info(
        "Median absolute difference: %.10f",
        np.median(differences),
    )
    logger.info(
        "Cells differing by more than 1: %d",
        np.sum(differences > 1),
    )

    # Save diagnostic information for investigation.
    mismatch_summary = {
        "max_absolute_difference": float(differences.max()),
        "mean_absolute_difference": float(differences.mean()),
        "median_absolute_difference": float(
            np.median(differences)
        ),
        "cells_difference_gt_0.00001": int(
            np.sum(differences > 0.00001)
        ),
        "cells_difference_gt_0.01": int(
            np.sum(differences > 0.01)
        ),
        "cells_difference_gt_1": int(
            np.sum(differences > 1)
        ),
    }

    logger.info(
        "U-Net predictions aligned to raster coordinates."
    )

    # Return predictions only.
    # All model metrics will use raster_actual.
    return unet_predictions, mismatch_summary


# ============================================================
# 5. RANDOM FOREST
# ============================================================

def predict_random_forest(X):

    if not RF_MODEL.exists():
        raise FileNotFoundError(RF_MODEL)

    logger.info("Loading Random Forest...")

    model = joblib.load(RF_MODEL)

    expected = getattr(model, "n_features_in_", None)

    if expected is not None and expected != N_FEATURES:
        raise ValueError(
            f"RF expects {expected} features, received {N_FEATURES}."
        )

    logger.info("Predicting with Random Forest...")

    predictions = np.asarray(
        model.predict(X),
        dtype=np.float64,
    ).reshape(-1)

    if len(predictions) != len(X):
        raise ValueError("RF prediction count mismatch.")

    if not np.isfinite(predictions).all():
        raise ValueError("RF predictions contain non-finite values.")

    return predictions


# ============================================================
# 6. XGBOOST
# ============================================================

def predict_xgboost(X):

    if not XGB_MODEL.exists():
        raise FileNotFoundError(XGB_MODEL)

    logger.info("Loading XGBoost...")

    model = xgb.XGBRegressor()
    model.load_model(str(XGB_MODEL))

    expected = getattr(model, "n_features_in_", None)

    if expected is not None and expected != N_FEATURES:
        raise ValueError(
            f"XGBoost expects {expected} features, "
            f"received {N_FEATURES}."
        )

    logger.info("Predicting with XGBoost...")

    booster = model.get_booster()

    try:
        best_iteration = booster.best_iteration

        predictions = model.predict(
            X,
            iteration_range=(0, best_iteration + 1),
        )

        logger.info(
            "Using XGBoost best iteration: %d",
            best_iteration,
        )

    except (AttributeError, xgb.core.XGBoostError):
        predictions = model.predict(X)

        logger.info(
            "Using all available XGBoost trees."
        )

    predictions = np.asarray(
        predictions,
        dtype=np.float64,
    ).reshape(-1)

    if len(predictions) != len(X):
        raise ValueError("XGBoost prediction count mismatch.")

    if not np.isfinite(predictions).all():
        raise ValueError("XGBoost predictions contain non-finite values.")

    return predictions


# ============================================================
# 7. METRICS
# ============================================================

def calculate_metrics(actual, predicted):

    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)

    residual = predicted - actual

    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual ** 2)))

    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))

    r2 = (
        float(1 - ss_res / ss_tot)
        if ss_tot > 0
        else None
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "mean_actual": float(actual.mean()),
        "mean_predicted": float(predicted.mean()),
        "actual_std": float(actual.std()),
        "predicted_std": float(predicted.std()),
        "mean_residual": float(residual.mean()),
        "median_absolute_error": float(
            np.median(np.abs(residual))
        ),
        "underprediction_cells": int(
            np.sum(predicted < actual)
        ),
        "overprediction_cells": int(
            np.sum(predicted > actual)
        ),
        "test_cells": int(len(actual)),
    }


# ============================================================
# 8. QUANTILE ANALYSIS
# ============================================================

def create_quantile_table(df):

    result = df.copy()

    result["quantile"] = pd.qcut(
        result["actual"],
        q=5,
        labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
        duplicates="drop",
    )

    records = []

    model_names = [
        "random_forest",
        "xgboost",
        "unet",
    ]

    for quantile, group in result.groupby(
        "quantile",
        observed=True,
    ):

        actual = group["actual"].to_numpy()

        record = {
            "quantile": str(quantile),
            "cells": int(len(group)),
            "mean_actual": float(actual.mean()),
        }

        for model_name in model_names:

            predicted = group[
                f"{model_name}_predicted"
            ].to_numpy()

            residual = predicted - actual

            record[f"{model_name}_MAE"] = float(
                np.mean(np.abs(residual))
            )

            record[f"{model_name}_RMSE"] = float(
                np.sqrt(np.mean(residual ** 2))
            )

            record[f"{model_name}_mean_predicted"] = float(
                predicted.mean()
            )

            record[f"{model_name}_mean_residual"] = float(
                residual.mean()
            )

        records.append(record)

    return pd.DataFrame(records)


# ============================================================
# 9. PLOTS
# ============================================================

def create_plots(metrics, quantile_df):

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    names = [
        "Random Forest",
        "XGBoost",
        "U-Net",
    ]

    keys = [
        "random_forest",
        "xgboost",
        "unet",
    ]

    mae = [metrics[k]["MAE"] for k in keys]
    rmse = [metrics[k]["RMSE"] for k in keys]
    r2 = [metrics[k]["R2"] for k in keys]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
    )

    axes[0].bar(names, mae)
    axes[0].set_title("Test MAE")
    axes[0].set_ylabel("MAE")

    axes[1].bar(names, rmse)
    axes[1].set_title("Test RMSE")
    axes[1].set_ylabel("RMSE")

    axes[2].bar(names, r2)
    axes[2].set_title("Test R²")
    axes[2].set_ylabel("R²")

    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
        ax.tick_params(axis="x", rotation=15)

    fig.suptitle(
        "Final Test Comparison: RF vs XGBoost vs U-Net"
    )

    fig.tight_layout()
    fig.savefig(COMPARISON_PLOT, dpi=200)
    plt.close(fig)

    # Quantile-wise MAE.
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(quantile_df))
    width = 0.25

    ax.bar(
        x - width,
        quantile_df["random_forest_MAE"],
        width,
        label="Random Forest",
    )

    ax.bar(
        x,
        quantile_df["xgboost_MAE"],
        width,
        label="XGBoost",
    )

    ax.bar(
        x + width,
        quantile_df["unet_MAE"],
        width,
        label="U-Net",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(quantile_df["quantile"])

    ax.set_xlabel("Actual hazard quantile")
    ax.set_ylabel("MAE")
    ax.set_title("Test MAE by Hazard Quantile")

    ax.legend()
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(QUANTILE_PLOT, dpi=200)
    plt.close(fig)


# ============================================================
# 10. MAIN
# ============================================================

def main():

    logger.info("=" * 65)
    logger.info("FINAL THREE-MODEL TEST COMPARISON")
    logger.info("=" * 65)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # Common test dataset.
    X, actual, rows, cols = load_test_data()

    # Existing U-Net predictions, aligned by coordinates.
    unet_predictions, unet_actual_diagnostic = (
        load_unet_predictions(
            rows,
            cols,
            actual,
        )
    )

    # Baseline predictions.
    rf_predictions = predict_random_forest(X)
    xgb_predictions = predict_xgboost(X)

    # --------------------------------------------------------
    # Common prediction table
    # --------------------------------------------------------

    df = pd.DataFrame({
        "row": rows,
        "col": cols,
        "actual": actual,
        "random_forest_predicted": rf_predictions,
        "xgboost_predicted": xgb_predictions,
        "unet_predicted": unet_predictions,
    })

    for name in [
        "random_forest",
        "xgboost",
        "unet",
    ]:

        df[f"{name}_residual"] = (
            df[f"{name}_predicted"] - df["actual"]
        )

        df[f"{name}_absolute_error"] = (
            df[f"{name}_residual"].abs()
        )

    df.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    metrics = {
        "random_forest": calculate_metrics(
            actual,
            rf_predictions,
        ),

        "xgboost": calculate_metrics(
            actual,
            xgb_predictions,
        ),

        "unet": calculate_metrics(
            actual,
            unet_predictions,
        ),
    }

    logger.info("-" * 65)
    logger.info("OVERALL TEST METRICS")
    logger.info("-" * 65)

    for name, values in metrics.items():

        logger.info(
            "%s | MAE: %.4f | RMSE: %.4f | R2: %.6f",
            name.upper(),
            values["MAE"],
            values["RMSE"],
            values["R2"],
        )

    # --------------------------------------------------------
    # Quantile analysis
    # --------------------------------------------------------

    quantile_df = create_quantile_table(df)

    quantile_df.to_csv(
        QUANTILE_FILE,
        index=False,
    )

    logger.info("-" * 65)
    logger.info("ERROR BY ACTUAL HAZARD QUANTILE")
    logger.info("-" * 65)

    for _, row in quantile_df.iterrows():

        logger.info(
            "%s | RF MAE: %.2f | XGB MAE: %.2f | U-Net MAE: %.2f",
            row["quantile"],
            row["random_forest_MAE"],
            row["xgboost_MAE"],
            row["unet_MAE"],
        )

    # --------------------------------------------------------
    # Best metric results
    # --------------------------------------------------------

    best_mae = min(
        metrics,
        key=lambda k: metrics[k]["MAE"],
    )

    best_rmse = min(
        metrics,
        key=lambda k: metrics[k]["RMSE"],
    )

    best_r2 = max(
        metrics,
        key=lambda k: metrics[k]["R2"],
    )

    summary = {
        "task": "Final RF, XGBoost and U-Net test comparison",
        "test_cells": int(len(actual)),
        "feature_names": FEATURE_NAMES,
        "selected_bands_1_based": SELECTED_BANDS,
        "spatial_split": "Task 9 test split, value 3",
        "ground_truth": "ksdma_100yr_90m.tif",
        "all_models_evaluated_on_same_cells": True,
        "test_used_for_training_or_tuning": False,
        "unet_actual_value_diagnostic": unet_actual_diagnostic,
        "metrics": metrics,
        "best_model_by_MAE": best_mae,
        "best_model_by_RMSE": best_rmse,
        "best_model_by_R2": best_r2,
    }

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(summary, file, indent=4)

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    create_plots(metrics, quantile_df)

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    logger.info("-" * 65)
    logger.info("BEST RESULTS")
    logger.info("Lowest MAE: %s", best_mae.upper())
    logger.info("Lowest RMSE: %s", best_rmse.upper())
    logger.info("Highest R2: %s", best_r2.upper())

    logger.info("-" * 65)
    logger.info("OUTPUT FILES")

    for path in [
        PREDICTIONS_FILE,
        METRICS_FILE,
        QUANTILE_FILE,
        COMPARISON_PLOT,
        QUANTILE_PLOT,
    ]:
        logger.info("%s", path)

    logger.info("=" * 65)
    logger.info(
        "COMPARISON COMPLETED ON %d IDENTICAL TEST CELLS",
        len(actual),
    )
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
