
"""
Task 9 — SHAP interpretation of the original-scale XGBoost model.

Explains predictions for a deterministic sample of validation cells.
Does not retrain the model or access the test set.

Run from project root:
    python src/experiments/task9_xgb_shap.py
"""

from pathlib import Path
import sys
import logging

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import shap
import xgboost as xgb


# --------------------------------------------------
# PATHS AND CONFIG
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT / "src"))
import config as cfg

MODEL_PATH = ROOT / "models" / "task9_xgboost.json"
SPLIT_PATH = ROOT / "data" / "processed" / "task9_spatial_split.tif"

OUTPUT_DIR = ROOT / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)
log = logging.getLogger("task9_shap")


# --------------------------------------------------
# FEATURE DEFINITIONS
# --------------------------------------------------

# Must match the feature order used during baseline training.
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

FEATURE_NAMES = list(FEATURE_BANDS.keys())

RANDOM_SEED = 42
MAX_EXPLANATION_SAMPLES = 2000


# --------------------------------------------------
# RASTER HELPERS
# --------------------------------------------------

def read_config_stack(key):
    path = Path(cfg.output_path(key))

    if not path.is_absolute():
        path = ROOT / path

    if not path.exists():
        raise FileNotFoundError(f"Raster not found: {path}")

    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs

    return data, profile, transform, crs


def read_single_band(path):
    if not path.exists():
        raise FileNotFoundError(f"Raster not found: {path}")

    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float64)
        nodata = src.nodata
        transform = src.transform
        crs = src.crs

    if nodata is not None:
        data[data == nodata] = np.nan

    data[~np.isfinite(data)] = np.nan

    return data, transform, crs


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    # 1. Check required files.
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"XGBoost model not found: {MODEL_PATH}")

    if not SPLIT_PATH.exists():
        raise FileNotFoundError(f"Split raster not found: {SPLIT_PATH}")

    log.info("Loading feature stack...")

    features, profile, transform, crs = read_config_stack(
        "features_stack"
    )

    log.info("Loading geographic split...")

    split, split_transform, split_crs = read_single_band(SPLIT_PATH)

    if split.shape != features.shape[1:]:
        raise ValueError("Feature stack and split raster shapes differ.")

    if transform != split_transform or crs != split_crs:
        raise ValueError("Feature stack and split raster grids differ.")

    # 2. Select validation cells only.
    validation_mask = split == 2

    rows, cols = np.where(validation_mask)

    if len(rows) == 0:
        raise RuntimeError("No validation cells found.")

    # Extract features in exact training order.
    X_all = np.column_stack([
        features[band - 1, rows, cols]
        for band in FEATURE_BANDS.values()
    ]).astype(np.float32)

    # Exclude invalid feature records.
    finite = np.isfinite(X_all).all(axis=1)

    X_all = X_all[finite]

    log.info(
        "Eligible validation cells: %s",
        f"{len(X_all):,}",
    )

    # 3. Deterministic sample for SHAP.
    rng = np.random.default_rng(RANDOM_SEED)

    sample_size = min(
        MAX_EXPLANATION_SAMPLES,
        len(X_all),
    )

    selected = rng.choice(
        len(X_all),
        size=sample_size,
        replace=False,
    )

    X_sample = X_all[selected]

    X_df = pd.DataFrame(
        X_sample,
        columns=FEATURE_NAMES,
    )

    log.info(
        "SHAP explanation sample: %s cells",
        f"{sample_size:,}",
    )

    # 4. Load existing original-scale XGBoost model.
    log.info("Loading original-scale XGBoost model...")

    model = xgb.XGBRegressor()
    model.load_model(str(MODEL_PATH))

    # 5. Calculate SHAP values.
    log.info("Creating TreeExplainer...")

    explainer = shap.TreeExplainer(model)

    log.info("Calculating SHAP values...")

    shap_values = explainer.shap_values(X_df)

    # Handle possible output shape variations.
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3 and shap_values.shape[-1] == 1:
        shap_values = shap_values[:, :, 0]

    if shap_values.shape != X_df.shape:
        raise ValueError(
            f"Unexpected SHAP shape: {shap_values.shape}. "
            f"Expected: {X_df.shape}"
        )

    log.info("SHAP calculation complete.")

    # 6. Calculate global feature importance.
    mean_abs_shap = np.mean(
        np.abs(shap_values),
        axis=0,
    )

    mean_signed_shap = np.mean(
        shap_values,
        axis=0,
    )

    importance_df = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "mean_absolute_SHAP": mean_abs_shap,
        "mean_signed_SHAP": mean_signed_shap,
    })

    importance_df = importance_df.sort_values(
        "mean_absolute_SHAP",
        ascending=False,
    ).reset_index(drop=True)

    importance_df.insert(
        0,
        "rank",
        np.arange(1, len(importance_df) + 1),
    )

    importance_path = (
        TABLE_DIR / "09_xgb_shap_feature_importance.csv"
    )

    importance_df.to_csv(
        importance_path,
        index=False,
    )

    # 7. Beeswarm plot.
    log.info("Generating SHAP beeswarm plot...")

    explanation = shap.Explanation(
        values=shap_values,
        base_values=np.repeat(
            float(np.asarray(explainer.expected_value).reshape(-1)[0]),
            sample_size,
        ),
        data=X_df.to_numpy(),
        feature_names=FEATURE_NAMES,
    )

    plt.figure(figsize=(11, 8))

    shap.plots.beeswarm(
        explanation,
        max_display=10,
        show=False,
    )

    plt.title(
        "XGBoost SHAP Summary — Validation Sample"
    )

    plt.tight_layout()

    beeswarm_path = (
        FIGURE_DIR / "09_xgb_shap_beeswarm.png"
    )

    plt.savefig(
        beeswarm_path,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close("all")

    # 8. SHAP bar plot.
    log.info("Generating SHAP importance plot...")

    plt.figure(figsize=(10, 7))

    shap.plots.bar(
        explanation,
        max_display=10,
        show=False,
    )

    plt.title(
        "XGBoost Mean Absolute SHAP Importance"
    )

    plt.tight_layout()

    bar_path = (
        FIGURE_DIR / "09_xgb_shap_importance.png"
    )

    plt.savefig(
        bar_path,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close("all")

    # 9. Dependence plots for top four features.
    top_features = importance_df["feature"].head(4).tolist()

    log.info("Generating dependence plots...")

    for feature_name in top_features:

        plt.figure(figsize=(8, 6))

        shap.plots.scatter(
            explanation[:, feature_name],
            color=explanation,
            show=False,
        )

        plt.title(
            f"SHAP Dependence — {feature_name}"
        )

        plt.tight_layout()

        output_path = (
            FIGURE_DIR /
            f"09_xgb_shap_dependence_{feature_name}.png"
        )

        plt.savefig(
            output_path,
            dpi=250,
            bbox_inches="tight",
        )

        plt.close("all")

    # 10. Print results.
    print("\n=== SHAP FEATURE IMPORTANCE ===")

    print(
        importance_df[
            [
                "rank",
                "feature",
                "mean_absolute_SHAP",
                "mean_signed_SHAP",
            ]
        ].to_string(index=False)
    )

    print("\nSaved outputs:")
    print(importance_path)
    print(beeswarm_path)
    print(bar_path)

    for feature_name in top_features:
        print(
            FIGURE_DIR /
            f"09_xgb_shap_dependence_{feature_name}.png"
        )

    print("\nSHAP sample size:", sample_size)
    print("Only validation features were explained.")
    print("No model was retrained.")
    print("Test data was not accessed.")


if __name__ == "__main__":
    main()
