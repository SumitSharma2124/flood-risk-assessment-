
"""
Task 9 — Random Forest SHAP Analysis (Optimised).

- 500 deterministic validation samples
- All 10 original features
- Existing trained Random Forest model
- No retraining
- No test-set access

Run from project root:
    python src/experiments/task9_rf_shap.py
"""

from pathlib import Path
import sys
import logging
import time

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import shap
import joblib


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT / "src"))
import config as cfg

MODEL_PATH = ROOT / "models" / "task9_random_forest.joblib"
SPLIT_PATH = ROOT / "data" / "processed" / "task9_spatial_split.tif"

OUTPUT_DIR = ROOT / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

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
MAX_EXPLANATION_SAMPLES = 500

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)

log = logging.getLogger("task9_rf_shap")


# --------------------------------------------------
# RASTER HELPERS
# --------------------------------------------------

def read_feature_stack():

    path = Path(cfg.output_path("features_stack"))

    if not path.is_absolute():
        path = ROOT / path

    if not path.exists():
        raise FileNotFoundError(path)

    with rasterio.open(path) as src:
        features = src.read().astype(np.float32)
        transform = src.transform
        crs = src.crs

    return features, transform, crs


def read_split():

    if not SPLIT_PATH.exists():
        raise FileNotFoundError(SPLIT_PATH)

    with rasterio.open(SPLIT_PATH) as src:
        split = src.read(1)
        transform = src.transform
        crs = src.crs

    return split, transform, crs


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    start_time = time.time()

    # 1. Check files.
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Random Forest model missing: {MODEL_PATH}"
        )

    log.info("Loading feature stack...")

    features, feature_transform, feature_crs = (
        read_feature_stack()
    )

    log.info("Loading geographic split...")

    split, split_transform, split_crs = read_split()

    if split.shape != features.shape[1:]:
        raise ValueError("Raster shapes do not match.")

    if (
        feature_transform != split_transform
        or feature_crs != split_crs
    ):
        raise ValueError("Raster grids do not match.")

    # 2. Select validation cells only.
    log.info("Selecting validation cells...")

    rows, cols = np.where(split == 2)

    X_all = np.column_stack([
        features[band - 1, rows, cols]
        for band in FEATURE_BANDS.values()
    ]).astype(np.float32)

    finite = np.isfinite(X_all).all(axis=1)
    X_all = X_all[finite]

    log.info(
        "Eligible validation cells: %s",
        f"{len(X_all):,}",
    )

    # 3. Deterministic sampling.
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

    X_df = pd.DataFrame(
        X_all[selected],
        columns=FEATURE_NAMES,
    )

    log.info("SHAP sample size: %s", sample_size)

    # 4. Load trained model.
    log.info("Loading Random Forest model...")

    model = joblib.load(MODEL_PATH)

    # 5. Create explainer.
    log.info("Creating TreeExplainer...")

    explainer = shap.TreeExplainer(
        model,
        feature_perturbation="tree_path_dependent",
    )

    # 6. Calculate SHAP.
    log.info("Calculating SHAP values...")

    shap_values = explainer.shap_values(X_df)

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3 and shap_values.shape[-1] == 1:
        shap_values = shap_values[:, :, 0]

    if shap_values.shape != X_df.shape:
        raise ValueError(
            f"Unexpected SHAP shape: {shap_values.shape}"
        )

    log.info(
        "SHAP calculation completed in %.1f seconds.",
        time.time() - start_time,
    )

    # 7. Feature importance.
    log.info("Calculating feature importance...")

    importance_df = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "mean_absolute_SHAP": np.mean(
            np.abs(shap_values),
            axis=0,
        ),
        "mean_signed_SHAP": np.mean(
            shap_values,
            axis=0,
        ),
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
        TABLE_DIR / "09_rf_shap_feature_importance.csv"
    )

    importance_df.to_csv(
        importance_path,
        index=False,
    )

    # 8. SHAP explanation object.
    expected_value = np.asarray(
        explainer.expected_value
    ).reshape(-1)[0]

    explanation = shap.Explanation(
        values=shap_values,
        base_values=np.repeat(
            float(expected_value),
            sample_size,
        ),
        data=X_df.to_numpy(),
        feature_names=FEATURE_NAMES,
    )

    # 9. Beeswarm plot.
    log.info("Generating beeswarm plot...")

    shap.plots.beeswarm(
        explanation,
        max_display=10,
        show=False,
    )

    plt.title("Random Forest SHAP Summary — Validation Sample")
    plt.tight_layout()

    beeswarm_path = (
        FIGURE_DIR / "09_rf_shap_beeswarm.png"
    )

    plt.savefig(
        beeswarm_path,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close("all")

    # 10. Importance plot.
    log.info("Generating importance plot...")

    shap.plots.bar(
        explanation,
        max_display=10,
        show=False,
    )

    plt.title("Random Forest Mean Absolute SHAP Importance")
    plt.tight_layout()

    bar_path = (
        FIGURE_DIR / "09_rf_shap_importance.png"
    )

    plt.savefig(
        bar_path,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close("all")

    # 11. Dependence plots for top four features.
    top_features = importance_df["feature"].head(4).tolist()

    for feature_name in top_features:

        log.info("Generating dependence plot: %s", feature_name)

        shap.plots.scatter(
            explanation[:, feature_name],
            color=explanation,
            show=False,
        )

        plt.title(
            f"Random Forest SHAP Dependence — {feature_name}"
        )

        plt.tight_layout()

        output_path = (
            FIGURE_DIR /
            f"09_rf_shap_dependence_{feature_name}.png"
        )

        plt.savefig(
            output_path,
            dpi=250,
            bbox_inches="tight",
        )

        plt.close("all")

    # 12. Final results.
    print("\n=== RANDOM FOREST SHAP IMPORTANCE ===")

    print(
        importance_df.to_string(index=False)
    )

    print("\nSaved files:")
    print(importance_path)
    print(beeswarm_path)
    print(bar_path)

    for feature_name in top_features:
        print(
            FIGURE_DIR /
            f"09_rf_shap_dependence_{feature_name}.png"
        )

    print(f"\nTotal runtime: {time.time() - start_time:.1f} seconds")
    print(f"SHAP samples: {sample_size}")
    print("Test data was not accessed.")
    print("Model was not retrained.")


if __name__ == "__main__":
    main()
