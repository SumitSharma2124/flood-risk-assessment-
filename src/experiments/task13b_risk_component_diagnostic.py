
"""
Task 13B: Risk component diagnostic.

Compares individual hazard, exposure, vulnerability and integrated
risk surfaces against the observed 2018 Sentinel-1 flood extent.

Uses the same KSDMA modelled-cell common domain as Task 13.
Does not modify existing rasters, models or Task 13 outputs.

Run from project root:
    python src/experiments/task13b_risk_component_diagnostic.py
"""

from pathlib import Path
import json
import logging

import numpy as np
import pandas as pd
import rasterio

from sklearn.metrics import roc_auc_score, average_precision_score


ROOT = Path(__file__).resolve().parents[2]

INPUTS = {
    "ksdma_hazard": ROOT / "data/processed/ksdma_100yr_90m.tif",
    "ksdma_mask": ROOT / "data/processed/ksdma_modelled_mask.tif",

    "xgb_hazard": ROOT / "outputs/rasters/09_xgb_full_district_prediction.tif",
    "unet_hazard": ROOT / "outputs/rasters/10_unet_full_district_prediction.tif",

    "exposure": ROOT / "data/processed/exposure_score.tif",
    "vulnerability": ROOT / "data/processed/vulnerability_score.tif",

    "ksdma_risk": ROOT / "outputs/rasters/11_ksdma_reference_risk.tif",
    "xgb_risk": ROOT / "outputs/rasters/11_xgb_integrated_risk.tif",
    "unet_risk": ROOT / "outputs/rasters/11_unet_integrated_risk.tif",

    "flood_fraction": ROOT / "data/processed/12_2018_flood_fraction_90m.tif",
}

OUTPUT_DIR = ROOT / "outputs/tables"

FLOOD_THRESHOLDS = [0.10, 0.25, 0.50, 0.75]
PRIMARY_THRESHOLD = 0.50

NODATA = -9999.0

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)
log = logging.getLogger("task13b")


def read_raster(path):
    with rasterio.open(path) as ds:
        data = ds.read(1).astype(np.float32)
        valid = ds.read_masks(1) > 0
        profile = ds.profile.copy()

        if ds.nodata is not None:
            valid &= data != ds.nodata

        valid &= np.isfinite(data)

    return data, valid, profile


def check_alignment(reference, profile, name):
    aligned = (
        profile["height"] == reference["height"]
        and profile["width"] == reference["width"]
        and profile["crs"] == reference["crs"]
        and profile["transform"] == reference["transform"]
    )

    if not aligned:
        raise ValueError(
            f"{name} is not aligned with the reference raster."
        )


def top_prevalence_capture(scores, labels):
    """
    Select the highest-scoring cells equal in number to observed
    flooded cells. This is a ranking diagnostic, not a probability
    threshold or calibrated classification.
    """
    positive_count = int(labels.sum())

    if positive_count == 0:
        return np.nan

    selected_indices = np.argpartition(
        scores,
        -positive_count
    )[-positive_count:]

    captured = int(labels[selected_indices].sum())

    return captured / positive_count


def calculate_metrics(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.uint8)

    if not np.all(np.isfinite(scores)):
        raise ValueError("Non-finite values found in diagnostic scores.")

    flooded = labels == 1
    non_flooded = labels == 0

    flooded_mean = float(scores[flooded].mean())
    non_flooded_mean = float(scores[non_flooded].mean())

    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(
            average_precision_score(labels, scores)
        ),
        "top_prevalence_capture": float(
            top_prevalence_capture(scores, labels)
        ),
        "flooded_mean_score": flooded_mean,
        "non_flooded_mean_score": non_flooded_mean,
        "mean_score_difference": flooded_mean - non_flooded_mean,
        "flooded_median_score": float(np.median(scores[flooded])),
        "non_flooded_median_score": float(
            np.median(scores[non_flooded])
        ),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading diagnostic rasters...")

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

        log.info(
            "%s loaded: %d valid cells",
            name,
            int(valid.sum())
        )

    # Extract arrays.
    ksdma = loaded["ksdma_hazard"]["data"]
    ksdma_valid = loaded["ksdma_hazard"]["valid"]

    modelled_mask = (
        (loaded["ksdma_mask"]["data"] > 0)
        & loaded["ksdma_mask"]["valid"]
    )

    xgb = loaded["xgb_hazard"]["data"]
    unet = loaded["unet_hazard"]["data"]

    exposure = loaded["exposure"]["data"]
    vulnerability = loaded["vulnerability"]["data"]

    ksdma_risk = loaded["ksdma_risk"]["data"]
    xgb_risk = loaded["xgb_risk"]["data"]
    unet_risk = loaded["unet_risk"]["data"]

    flood_fraction = loaded["flood_fraction"]["data"]

    # Construct exactly the common comparison domain:
    # all required data valid AND KSDMA modelled cells.
    common_valid = modelled_mask

    for name, item in loaded.items():
        if name == "ksdma_mask":
            continue
        common_valid &= item["valid"]

    common_count = int(common_valid.sum())

    if common_count == 0:
        raise ValueError("No common validation cells found.")

    log.info("Common validation cells: %d", common_count)

    # Normalize hazards using the KSDMA reference range,
    # exactly as in Task 11 and Task 13.
    reference_values = ksdma[common_valid]
    hazard_min = float(reference_values.min())
    hazard_max = float(reference_values.max())

    if hazard_max <= hazard_min:
        raise ValueError("Invalid KSDMA hazard reference range.")

    def normalize_hazard(values):
        return np.clip(
            (values - hazard_min) / (hazard_max - hazard_min),
            0.0,
            1.0
        ).astype(np.float32)

    # Higher score means higher hazard/exposure/vulnerability/risk.
    surfaces = {
        "KSDMA hazard": normalize_hazard(ksdma),
        "XGBoost hazard": normalize_hazard(xgb),
        "U-Net hazard": normalize_hazard(unet),

        "Exposure": np.clip(exposure, 0.0, 1.0),
        "Vulnerability": np.clip(vulnerability, 0.0, 1.0),

        "KSDMA integrated risk": ksdma_risk,
        "XGBoost integrated risk": xgb_risk,
        "U-Net integrated risk": unet_risk,
    }

    # Extract only common-domain values.
    surface_values = {
        name: values[common_valid].astype(np.float64)
        for name, values in surfaces.items()
    }

    flood_values = flood_fraction[common_valid].astype(np.float64)

    # Check score ranges and basic distributions.
    distribution_rows = []

    for name, scores in surface_values.items():
        distribution_rows.append({
            "surface": name,
            "minimum": float(scores.min()),
            "maximum": float(scores.max()),
            "mean": float(scores.mean()),
            "median": float(np.median(scores)),
            "std": float(scores.std()),
        })

    distribution_df = pd.DataFrame(distribution_rows)

    # Calculate metrics at all four observed flood thresholds.
    metric_rows = []
    primary_component_rows = []

    for threshold in FLOOD_THRESHOLDS:
        labels = (flood_values >= threshold).astype(np.uint8)

        positive_count = int(labels.sum())
        prevalence = float(labels.mean())

        log.info(
            "Flood threshold %.2f: %d positive cells (%.2f%%)",
            threshold,
            positive_count,
            prevalence * 100
        )

        for name, scores in surface_values.items():
            metrics = calculate_metrics(scores, labels)

            metric_rows.append({
                "flood_fraction_threshold": threshold,
                "surface": name,
                "observed_flood_cells": positive_count,
                "observed_flood_prevalence": prevalence,
                **metrics,
            })

            if threshold == PRIMARY_THRESHOLD:
                primary_component_rows.append({
                    "surface": name,
                    "observed_flood_cells": positive_count,
                    "observed_flood_prevalence": prevalence,
                    **metrics,
                })

    metrics_df = pd.DataFrame(metric_rows)
    primary_df = pd.DataFrame(primary_component_rows)

    # Save outputs separately from Task 13.
    metrics_path = (
        OUTPUT_DIR / "13b_risk_component_metrics.csv"
    )
    primary_path = (
        OUTPUT_DIR / "13b_primary_component_comparison.csv"
    )
    distribution_path = (
        OUTPUT_DIR / "13b_surface_distributions.csv"
    )

    metrics_df.to_csv(metrics_path, index=False)
    primary_df.to_csv(primary_path, index=False)
    distribution_df.to_csv(distribution_path, index=False)

    # JSON summary.
    report = {
        "task": "Task 13B - Risk component diagnostic",
        "district": "Alappuzha",
        "scenario": "100yr",
        "observed_event": "2018 Sentinel-1 flood extent",
        "common_validation_cells": common_count,
        "common_domain": (
            "KSDMA modelled cells with all required rasters valid"
        ),
        "hazard_normalisation": {
            "method": "KSDMA common-domain min-max",
            "minimum": hazard_min,
            "maximum": hazard_max,
        },
        "primary_flood_fraction_threshold": PRIMARY_THRESHOLD,
        "thresholds_tested": FLOOD_THRESHOLDS,
        "primary_threshold_results": primary_component_rows,
        "surface_distributions": distribution_rows,
        "limitations": [
            "2018 observed flood extent is not equivalent to a "
            "100-year return-period hazard map.",
            "Metrics measure spatial agreement with the observed "
            "2018 event.",
            "Top-prevalence capture is a ranking diagnostic, not "
            "a calibrated probability threshold.",
            "Exposure and vulnerability are relative indices.",
            "The analysis diagnoses spatial association and does "
            "not establish causal relationships.",
            "Risk scores are relative indices, not probabilities "
            "or expected losses.",
        ],
        "outputs": {
            "all_threshold_metrics": str(metrics_path),
            "primary_threshold_comparison": str(primary_path),
            "surface_distributions": str(distribution_path),
        },
    }

    summary_path = (
        OUTPUT_DIR / "13b_risk_component_summary.json"
    )

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    log.info("Saved: %s", metrics_path)
    log.info("Saved: %s", primary_path)
    log.info("Saved: %s", distribution_path)
    log.info("Saved: %s", summary_path)

    log.info("Task 13B completed successfully.")


if __name__ == "__main__":
    main()
