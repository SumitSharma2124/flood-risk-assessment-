
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import config as cfg

FEATURE_BANDS = [1, 2, 3, 6, 7, 8, 9, 10, 11, 12]


def metrics(y_true, predictions):
    return {
        "MAE": mean_absolute_error(y_true, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_true, predictions)),
        "R2": r2_score(y_true, predictions),
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

    print("Training cells:", len(y_train))
    print("Validation cells:", len(y_val))
    print("Training target mean:", y_train.mean())

    results = []

    # 1. Mean predictor
    mean_model = DummyRegressor(strategy="mean")
    mean_model.fit(X_train, y_train)

    mean_predictions = mean_model.predict(X_val)
    results.append({
        "model": "Training Mean",
        **metrics(y_val, mean_predictions)
    })

    # 2. Linear regression
    linear_model = make_pipeline(
        StandardScaler(),
        LinearRegression()
    )

    linear_model.fit(X_train, y_train)

    linear_predictions = linear_model.predict(X_val)
    results.append({
        "model": "Linear Regression",
        **metrics(y_val, linear_predictions)
    })

    # 3. Existing Random Forest
    rf_path = cfg.output_path("rf_model")
    rf_model = joblib.load(rf_path)

    rf_predictions = rf_model.predict(X_val)
    results.append({
        "model": "Random Forest",
        **metrics(y_val, rf_predictions)
    })

    # 4. Existing XGBoost
    from xgboost import XGBRegressor

    xgb_path = cfg.output_path("xgb_model")
    xgb_model = XGBRegressor()
    xgb_model.load_model(str(xgb_path))

    xgb_predictions = xgb_model.predict(X_val)
    results.append({
        "model": "XGBoost",
        **metrics(y_val, xgb_predictions)
    })

    results_df = pd.DataFrame(results)

    print("\n=== VALIDATION BASELINE COMPARISON ===")
    print(results_df.round(4).to_string(index=False))

    output = cfg.TABLES_DIR / "09_simple_baseline_comparison.csv"
    results_df.to_csv(output, index=False)

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
