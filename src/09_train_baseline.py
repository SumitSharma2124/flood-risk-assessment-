
"""
Task 9: Train Random Forest and XGBoost regression baselines.

Run from project root:
    python src/09_train_baseline.py

Optional:
    python src/09_train_baseline.py --overwrite
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import rasterio

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg

from xgboost import XGBRegressor


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)
logger = logging.getLogger("task9_baseline")

SEED = 42

# 1-based raster band numbers:
# 1 elevation, 2 slope, 3 TWI, 6 waterway distance,
# 7 trees, 8 cropland, 9 built-up, 10 water,
# 11 wetland, 12 mangrove.
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

SPLIT_CODES = {
    "train": 1,
    "validation": 2,
    "test": 3,
}


def read_inputs():
    """Load aligned rasters and construct the eligible modelling dataset."""

    paths = {
        "features": cfg.output_path("features_stack"),
        "feature_mask": cfg.output_path("features_valid_mask"),
        "target": cfg.output_path("ksdma_100yr_90m"),
        "modelled_mask": cfg.output_path("ksdma_modelled_mask"),
        "district": cfg.output_path("district_mask"),
        "split": cfg.PROCESSED_DIR / "task9_spatial_split.tif",
    }

    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}")

    with rasterio.open(paths["features"]) as ds:
        if ds.count < max(FEATURE_BANDS):
            raise ValueError(
                f"Feature stack has {ds.count} bands; "
                f"band {max(FEATURE_BANDS)} is required."
            )

        feature_stack = ds.read(FEATURE_BANDS).astype(np.float32)
        feature_nodata = ds.nodata
        reference = (ds.height, ds.width, ds.crs, ds.transform)

    def read_single(path):
        with rasterio.open(path) as ds:
            current = (ds.height, ds.width, ds.crs, ds.transform)
            if current != reference:
                raise ValueError(f"Raster grid mismatch: {path}")
            return ds.read(1), ds.nodata

    feature_mask, _ = read_single(paths["feature_mask"])
    target, target_nodata = read_single(paths["target"])
    modelled_mask, _ = read_single(paths["modelled_mask"])
    district, _ = read_single(paths["district"])
    split, _ = read_single(paths["split"])

    # Only cells with real KSDMA target values are eligible.
    eligible = (
        (district == 1)
        & (feature_mask > 0)
        & (modelled_mask > 0)
        & np.isfinite(target)
    )

    if target_nodata is not None:
        eligible &= target != target_nodata

    if feature_nodata is not None:
        eligible &= np.all(feature_stack != feature_nodata, axis=0)

    eligible &= np.all(np.isfinite(feature_stack), axis=0)

    # Every eligible cell must belong to exactly one saved split.
    if np.any(~np.isin(split[eligible], [1, 2, 3])):
        raise ValueError("Some eligible cells have no valid split assignment.")

    X = feature_stack[:, eligible].T
    y = target[eligible].astype(np.float32)
    split_values = split[eligible]

    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("Non-finite values found in modelling dataset.")

    data = {}

    for name, code in SPLIT_CODES.items():
        mask = split_values == code
        data[name] = (X[mask], y[mask])

    logger.info("Eligible cells: %s", len(y))
    logger.info("Input features: %s", len(FEATURE_NAMES))

    for name, (x_part, y_part) in data.items():
        logger.info("%s cells: %s", name, len(y_part))

    if len(data["train"][1]) == 0 or len(data["validation"][1]) == 0:
        raise ValueError("Training or validation split is empty.")

    return data


def calculate_metrics(y_true, predictions):
    rmse = float(np.sqrt(mean_squared_error(y_true, predictions)))

    return {
        "MAE": float(mean_absolute_error(y_true, predictions)),
        "RMSE": rmse,
        "R2": float(r2_score(y_true, predictions)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing Task 9 model outputs."
    )
    args = parser.parse_args()

    output_keys = [
        "rf_model",
        "xgb_model",
        "baseline_metrics",
        "baseline_metadata",
    ]
    output_paths = [cfg.output_path(key) for key in output_keys]

    existing = [path for path in output_paths if path.exists()]

    if existing and not args.overwrite:
        formatted = "\n".join(str(path) for path in existing)
        raise FileExistsError(
            "Task 9 outputs already exist. "
            "Use --overwrite only if you intend to replace them:\n"
            + formatted
        )

    data = read_inputs()

    X_train, y_train = data["train"]
    X_val, y_val = data["validation"]

    results = []

    # ---------------------------------------------------------
    # 1. Random Forest regression
    # ---------------------------------------------------------
    logger.info("Training Random Forest...")

    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=2,
        max_features=1.0,
        bootstrap=True,
        random_state=SEED,
        n_jobs=-1,
    )

    rf.fit(X_train, y_train)

    rf_predictions = rf.predict(X_val)
    rf_metrics = calculate_metrics(y_val, rf_predictions)

    rf_metrics["model"] = "Random Forest"
    results.append(rf_metrics)

    joblib.dump(rf, cfg.output_path("rf_model"))

    logger.info("Random Forest validation metrics: %s", rf_metrics)

    # ---------------------------------------------------------
    # 2. XGBoost regression
    # ---------------------------------------------------------
    logger.info("Training XGBoost...")

    xgb = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        reg_alpha=0.0,
        tree_method="hist",
        early_stopping_rounds=50,
        random_state=SEED,
        n_jobs=-1,
    )

    xgb.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )

    xgb_predictions = xgb.predict(X_val)
    xgb_metrics = calculate_metrics(y_val, xgb_predictions)

    xgb_metrics["model"] = "XGBoost"
    results.append(xgb_metrics)

    xgb.save_model(str(cfg.output_path("xgb_model")))

    logger.info("XGBoost best iteration: %s", xgb.best_iteration)
    logger.info("XGBoost validation metrics: %s", xgb_metrics)

    # ---------------------------------------------------------
    # 3. Save validation comparison
    # ---------------------------------------------------------
    metrics_df = pd.DataFrame(results)[
        ["model", "MAE", "RMSE", "R2"]
    ]

    metrics_df.to_csv(
        cfg.output_path("baseline_metrics"),
        index=False
    )

    metadata = {
        "district": cfg.DISTRICT_NAME,
        "scenario": cfg.SCENARIO_NAME,
        "target": "KSDMA 100-year hazard index",
        "task": "continuous regression",
        "seed": SEED,
        "feature_names": FEATURE_NAMES,
        "feature_bands_1_based": FEATURE_BANDS,
        "train_cells": len(y_train),
        "validation_cells": len(y_val),
        "test_cells": len(data["test"][1]),
        "split_raster": str(
            cfg.PROCESSED_DIR / "task9_spatial_split.tif"
        ),
        "xgboost_best_iteration": int(xgb.best_iteration),
        "validation_metrics": results,
    }

    with open(
        cfg.output_path("baseline_metadata"),
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(metadata, f, indent=4)

    print("\n=== VALIDATION COMPARISON ===")
    print(metrics_df.to_string(index=False))

    print("\nSaved models:")
    print(cfg.output_path("rf_model"))
    print(cfg.output_path("xgb_model"))

    print("\nSaved metrics:")
    print(cfg.output_path("baseline_metrics"))

    print("\nTraining completed. Test data was not used.")


if __name__ == "__main__":
    main()
