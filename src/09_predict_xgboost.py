
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import rasterio
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg

FEATURE_BANDS = [1, 2, 3, 6, 7, 8, 9, 10, 11, 12]

FEATURE_NAMES = [
    "elevation", "slope", "twi", "distance_waterway",
    "trees", "cropland", "builtup", "water",
    "wetland", "mangrove",
]

MODEL_FILE = cfg.output_path("xgb_model")
FEATURE_FILE = cfg.output_path("features_stack")
FEATURE_MASK_FILE = cfg.output_path("features_valid_mask")
DISTRICT_FILE = cfg.output_path("district_mask")

OUTPUT_DIR = ROOT / "outputs" / "rasters"
OUTPUT_FILE = OUTPUT_DIR / "09_xgb_full_district_prediction.tif"
METADATA_FILE = ROOT / "outputs" / "tables" / "09_xgb_full_district_prediction.json"

NODATA = -9999.0

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("xgb_prediction")


def main():
    for path in [
        MODEL_FILE, FEATURE_FILE, FEATURE_MASK_FILE, DISTRICT_FILE
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    if OUTPUT_FILE.exists() or METADATA_FILE.exists():
        raise FileExistsError(
            "Prediction output already exists. "
            "Move or rename it before rerunning."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading aligned input rasters...")

    with rasterio.open(FEATURE_FILE) as src:
        features = src.read(FEATURE_BANDS).astype(np.float32)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        height, width = src.height, src.width
        feature_nodata = src.nodata

    def read_aligned(path):
        with rasterio.open(path) as src:
            if (
                src.height != height
                or src.width != width
                or src.transform != transform
                or src.crs != crs
            ):
                raise ValueError(f"Grid mismatch: {path}")
            return src.read(1)

    feature_mask = read_aligned(FEATURE_MASK_FILE)
    district = read_aligned(DISTRICT_FILE)

    finite = np.all(np.isfinite(features), axis=0)

    if feature_nodata is not None:
        finite &= np.all(features != feature_nodata, axis=0)

    valid = (district == 1) & (feature_mask > 0) & finite

    if not np.any(valid):
        raise RuntimeError("No valid district cells available.")

    logger.info("Valid prediction cells: %d", int(valid.sum()))
    logger.info("Loading saved XGBoost model: %s", MODEL_FILE)

    model = XGBRegressor()
    model.load_model(str(MODEL_FILE))

    X = features[:, valid].T

    if X.shape[1] != len(FEATURE_NAMES):
        raise ValueError("Unexpected number of input features.")

    logger.info("Predicting %d cells...", len(X))

    predictions = model.predict(X).astype(np.float32)

    if not np.isfinite(predictions).all():
        raise RuntimeError("Model produced non-finite predictions.")

    output = np.full((height, width), NODATA, dtype=np.float32)
    output[valid] = predictions

    profile.update(
        driver="GTiff",
        count=1,
        dtype="float32",
        nodata=NODATA,
        compress="deflate",
        predictor=3,
    )

    with rasterio.open(OUTPUT_FILE, "w", **profile) as dst:
        dst.write(output, 1)
        dst.set_band_description(1, "XGBoost predicted KSDMA hazard index")
        dst.update_tags(
            model="XGBoost",
            district=cfg.DISTRICT_NAME,
            scenario=cfg.SCENARIO_NAME,
            feature_names=",".join(FEATURE_NAMES),
            prediction_domain="District cells with valid input features",
            caveat=(
                "Predictions outside the KSDMA modelled mask are "
                "extrapolations beyond the supervised target domain."
            ),
        )

    metadata = {
        "model": "XGBoost",
        "model_file": str(MODEL_FILE),
        "output_file": str(OUTPUT_FILE),
        "district": cfg.DISTRICT_NAME,
        "scenario": cfg.SCENARIO_NAME,
        "features": FEATURE_NAMES,
        "feature_bands_1_based": FEATURE_BANDS,
        "valid_prediction_cells": int(valid.sum()),
        "prediction_min": float(predictions.min()),
        "prediction_max": float(predictions.max()),
        "prediction_mean": float(predictions.mean()),
        "nodata": NODATA,
        "note": (
            "Predictions outside the KSDMA modelled mask are "
            "extrapolations, not independently validated hazard values."
        ),
    }

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    logger.info("XGBoost prediction completed.")
    logger.info("Raster: %s", OUTPUT_FILE)
    logger.info("Metadata: %s", METADATA_FILE)


if __name__ == "__main__":
    main()
