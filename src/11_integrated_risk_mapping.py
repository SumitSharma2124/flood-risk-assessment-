
"""
Task 11: Integrated flood risk mapping.

Risk = Normalised Hazard x Exposure x Vulnerability

All hazard surfaces use the KSDMA modelled-cell min/max
as a common reference scale.

Run from project root:
    python src/11_integrated_risk_mapping.py
"""

from pathlib import Path
import json
import logging

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]

INPUTS = {
    "ksdma": ROOT / "data/processed/ksdma_100yr_90m.tif",
    "ksdma_mask": ROOT / "data/processed/ksdma_modelled_mask.tif",
    "xgb": ROOT / "outputs/rasters/09_xgb_full_district_prediction.tif",
    "unet": ROOT / "outputs/rasters/10_unet_full_district_prediction.tif",
    "exposure": ROOT / "data/processed/exposure_score.tif",
    "vulnerability": ROOT / "data/processed/vulnerability_score.tif",
}

OUTPUT_DIR = ROOT / "outputs/rasters"
TABLE_DIR = ROOT / "outputs/tables"

OUTPUTS = {
    "ksdma": OUTPUT_DIR / "11_ksdma_reference_risk.tif",
    "xgb": OUTPUT_DIR / "11_xgb_integrated_risk.tif",
    "unet": OUTPUT_DIR / "11_unet_integrated_risk.tif",
}

NODATA = -9999.0

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)
log = logging.getLogger("task11")


def read_raster(path):
    with rasterio.open(path) as ds:
        data = ds.read(1).astype(np.float32)
        profile = ds.profile.copy()
        valid = ds.read_masks(1) > 0

        if ds.nodata is not None:
            valid &= data != ds.nodata

        valid &= np.isfinite(data)

    return data, valid, profile


def check_alignment(reference, profile, name):
    checks = (
        profile["height"] == reference["height"]
        and profile["width"] == reference["width"]
        and profile["crs"] == reference["crs"]
        and profile["transform"] == reference["transform"]
    )

    if not checks:
        raise ValueError(f"{name} is not aligned with the reference grid.")

    log.info("%s grid alignment: PASS", name)


def save_raster(path, array, profile):
    out_profile = profile.copy()
    out_profile.update(
        driver="GTiff",
        count=1,
        dtype="float32",
        nodata=NODATA,
        compress="deflate",
        predictor=3,
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )

    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(array.astype(np.float32), 1)


def raster_stats(array, valid):
    values = array[valid]

    return {
        "valid_cells": int(values.size),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading input rasters...")

    loaded = {}
    reference_profile = None

    for name, path in INPUTS.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing input: {path}")

        data, valid, profile = read_raster(path)

        if reference_profile is None:
            reference_profile = profile
        else:
            check_alignment(reference_profile, profile, name)

        loaded[name] = {
            "data": data,
            "valid": valid,
        }

        log.info("%s loaded: %d valid cells", name, int(valid.sum()))

    # Extract arrays.
    ksdma = loaded["ksdma"]["data"]
    ksdma_valid = loaded["ksdma"]["valid"]

    modelled_mask = (
        loaded["ksdma_mask"]["data"] > 0
    ) & loaded["ksdma_mask"]["valid"]

    xgb = loaded["xgb"]["data"]
    xgb_valid = loaded["xgb"]["valid"]

    unet = loaded["unet"]["data"]
    unet_valid = loaded["unet"]["valid"]

    exposure = loaded["exposure"]["data"]
    exposure_valid = loaded["exposure"]["valid"]

    vulnerability = loaded["vulnerability"]["data"]
    vulnerability_valid = loaded["vulnerability"]["valid"]

    # Common district domain.
    district_valid = (
        exposure_valid
        & vulnerability_valid
        & xgb_valid
        & unet_valid
    )

    # KSDMA reference domain.
    reference_valid = (
        district_valid
        & modelled_mask
        & ksdma_valid
    )

    if not np.any(reference_valid):
        raise ValueError("No valid KSDMA reference cells found.")

    log.info(
        "KSDMA reference cells: %d",
        int(reference_valid.sum())
    )
    log.info(
        "Full district cells: %d",
        int(district_valid.sum())
    )

    # Common hazard normalization reference.
    reference_values = ksdma[reference_valid]
    hazard_min = float(reference_values.min())
    hazard_max = float(reference_values.max())

    if hazard_max <= hazard_min:
        raise ValueError("Invalid KSDMA hazard reference range.")

    log.info(
        "KSDMA hazard reference range: %.4f to %.4f",
        hazard_min,
        hazard_max
    )

    def normalize_hazard(values):
        normalized = (values - hazard_min) / (hazard_max - hazard_min)
        return np.clip(normalized, 0.0, 1.0).astype(np.float32)

    # Normalize all three hazard surfaces using identical limits.
    ksdma_norm = normalize_hazard(ksdma)
    xgb_norm = normalize_hazard(xgb)
    unet_norm = normalize_hazard(unet)

    # Exposure and vulnerability are already relative scores.
    # Clip only to protect against minor floating-point deviations.
    exposure = np.clip(exposure, 0.0, 1.0)
    vulnerability = np.clip(vulnerability, 0.0, 1.0)

    # Integrated risk.
    ksdma_risk = np.full(ksdma.shape, NODATA, dtype=np.float32)
    xgb_risk = np.full(xgb.shape, NODATA, dtype=np.float32)
    unet_risk = np.full(unet.shape, NODATA, dtype=np.float32)

    ksdma_risk[reference_valid] = (
        ksdma_norm[reference_valid]
        * exposure[reference_valid]
        * vulnerability[reference_valid]
    )

    xgb_risk[district_valid] = (
        xgb_norm[district_valid]
        * exposure[district_valid]
        * vulnerability[district_valid]
    )

    unet_risk[district_valid] = (
        unet_norm[district_valid]
        * exposure[district_valid]
        * vulnerability[district_valid]
    )

    # Save output rasters.
    save_raster(OUTPUTS["ksdma"], ksdma_risk, reference_profile)
    save_raster(OUTPUTS["xgb"], xgb_risk, reference_profile)
    save_raster(OUTPUTS["unet"], unet_risk, reference_profile)

    log.info("Saved KSDMA reference risk: %s", OUTPUTS["ksdma"])
    log.info("Saved XGBoost risk: %s", OUTPUTS["xgb"])
    log.info("Saved U-Net risk: %s", OUTPUTS["unet"])

    # Count predictions clipped by common reference range.
    xgb_clipped = int(
        np.count_nonzero(
            district_valid
            & ((xgb < hazard_min) | (xgb > hazard_max))
        )
    )
    unet_clipped = int(
        np.count_nonzero(
            district_valid
            & ((unet < hazard_min) | (unet > hazard_max))
        )
    )

    outside_reference = district_valid & ~reference_valid

    report = {
        "method": "Hazard x Exposure x Vulnerability",
        "district": "Alappuzha",
        "scenario": "100yr",
        "hazard_normalisation": {
            "method": "Min-max using KSDMA modelled cells",
            "reference_minimum": hazard_min,
            "reference_maximum": hazard_max,
            "clipping": "Values outside reference range clipped to 0-1",
        },
        "valid_cells": {
            "full_district": int(district_valid.sum()),
            "ksdma_reference": int(reference_valid.sum()),
            "outside_ksdma_reference": int(outside_reference.sum()),
        },
        "clipped_prediction_cells": {
            "xgboost": xgb_clipped,
            "unet": unet_clipped,
        },
        "risk_statistics": {
            "ksdma": raster_stats(ksdma_risk, reference_valid),
            "xgboost": raster_stats(xgb_risk, district_valid),
            "unet": raster_stats(unet_risk, district_valid),
        },
        "outputs": {
            name: str(path)
            for name, path in OUTPUTS.items()
        },
        "limitations": [
            "KSDMA hazard values are treated as a relative index; units are unconfirmed.",
            "XGBoost and U-Net predictions outside the KSDMA modelled mask are extrapolations.",
            "Risk scores are relative indices, not flood probabilities or expected losses.",
            "Full-district model risk statistics include extrapolated cells.",
            "For direct model comparison, use the common KSDMA reference domain.",
        ],
    }

    report_path = TABLE_DIR / "11_integrated_risk_mapping_summary.json"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    log.info("Summary: %s", report_path)
    log.info("Task 11 risk mapping completed.")


if __name__ == "__main__":
    main()
