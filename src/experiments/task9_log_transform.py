
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import config as cfg

SEED = 42
FEATURE_BANDS = [1, 2, 3, 6, 7, 8, 9, 10, 11, 12]


def evaluate(y, pred):
    return {
        "MAE": mean_absolute_error(y, pred),
        "RMSE": np.sqrt(mean_squared_error(y, pred)),
        "R2": r2_score(y, pred),
    }


def main():
    with rasterio.open(cfg.output_path("features_stack")) as ds:
        X = ds.read(FEATURE_BANDS).astype(np.float32)

    with rasterio.open(cfg.output_path("ksdma_100yr_90m")) as ds:
        y = ds.read(1).astype(np.float32)

    with rasterio.open(cfg.PROCESSED_DIR / "task9_spatial_split.tif") as ds:
        split = ds.read(1)

    with rasterio.open(cfg.output_path("district_mask")) as ds:
        district = ds.read(1) == 1

    with rasterio.open(cfg.output_path("features_valid_mask")) as ds:
        feature_mask = ds.read(1) > 0

    with rasterio.open(cfg.output_path("ksdma_modelled_mask")) as ds:
        modelled = ds.read(1) > 0

    valid = (
        district & feature_mask & modelled
        & np.isfinite(y) & (y != -9999)
        & np.all(np.isfinite(X), axis=0)
        & np.isin(split, [1, 2, 3])
    )

    X_data = X[:, valid].T
    y_data = y[valid]
    split_data = split[valid]

    X_train = X_data[split_data == 1]
    y_train = y_data[split_data == 1]

    X_val = X_data[split_data == 2]
    y_val = y_data[split_data == 2]

    if np.any(y_train < 0) or np.any(y_val < 0):
        raise ValueError("log1p experiment requires nonnegative targets.")

    y_train_log = np.log1p(y_train)

    results = []

    # Random Forest on transformed target
    print("Training log-transformed Random Forest...")

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

    rf.fit(X_train, y_train_log)

    rf_log_predictions = rf.predict(X_val)
    rf_predictions = np.maximum(np.expm1(rf_log_predictions), 0)

    results.append({
        "model": "Random Forest log1p",
        **evaluate(y_val, rf_predictions)
    })

    # XGBoost on transformed target
    print("Training log-transformed XGBoost...")

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
        y_train_log,
        eval_set=[(X_val, np.log1p(y_val))],
        verbose=100,
    )

    xgb_log_predictions = xgb.predict(X_val)
    xgb_predictions = np.maximum(np.expm1(xgb_log_predictions), 0)

    results.append({
        "model": "XGBoost log1p",
        **evaluate(y_val, xgb_predictions)
    })

    # Compare against existing original-scale models.
    rf_original = joblib.load(cfg.output_path("rf_model"))
    rf_original_pred = rf_original.predict(X_val)

    xgb_original = XGBRegressor()
    xgb_original.load_model(str(cfg.output_path("xgb_model")))
    xgb_original_pred = xgb_original.predict(X_val)

    results.extend([
        {
            "model": "Random Forest original",
            **evaluate(y_val, rf_original_pred)
        },
        {
            "model": "XGBoost original",
            **evaluate(y_val, xgb_original_pred)
        }
    ])

    result_df = pd.DataFrame(results)
    result_df = result_df[["model", "MAE", "RMSE", "R2"]]

    print("\n=== ORIGINAL VS LOG1P VALIDATION RESULTS ===")
    print(result_df.round(4).to_string(index=False))

    output = cfg.TABLES_DIR / "09_log_transform_comparison.csv"
    result_df.to_csv(output, index=False)

    print("\nSaved:", output)
    print("Test data was not used.")


if __name__ == "__main__":
    main()
