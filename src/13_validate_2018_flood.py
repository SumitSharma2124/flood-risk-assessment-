
"""
Task 13: Validate hazard and integrated risk maps against
the observed 2018 Sentinel-1 flood extent.

Run from project root:
    python src/13_validate_2018_flood.py
"""

from pathlib import Path
import json
import logging

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]

INPUTS = {
    "ksdma_hazard": ROOT / "data/processed/ksdma_100yr_90m.tif",
    "ksdma_mask": ROOT / "data/processed/ksdma_modelled_mask.tif",
    "xgb_hazard": ROOT / "outputs/rasters/09_xgb_full_district_prediction.tif",
    "unet_hazard": ROOT / "outputs/rasters/10_unet_full_district_prediction.tif",
    "ksdma_risk": ROOT / "outputs/rasters/11_ksdma_reference_risk.tif",
    "xgb_risk": ROOT / "outputs/rasters/11_xgb_integrated_risk.tif",
    "unet_risk": ROOT / "outputs/rasters/11_unet_integrated_risk.tif",
    "flood_fraction": ROOT / "data/processed/12_2018_flood_fraction_90m.tif",
}

TABLE_DIR = ROOT / "outputs/tables"
FIGURE_DIR = ROOT / "outputs/figures"

FLOOD_THRESHOLDS = [0.10, 0.25, 0.50, 0.75]
FIXED_HAZARD_THRESHOLD = 0.50

NODATA = -9999.0

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)
log = logging.getLogger("task13")


def read_raster(path):
    with rasterio.open(path) as ds:
        data = ds.read(1).astype(np.float64)
        valid = ds.read_masks(1) > 0

        if ds.nodata is not None:
            valid &= data != ds.nodata

        valid &= np.isfinite(data)

        return data, valid, ds.profile.copy()


def check_alignment(reference_profile, profile, name):
    checks = (
        profile["height"] == reference_profile["height"]
        and profile["width"] == reference_profile["width"]
        and profile["crs"] == reference_profile["crs"]
        and profile["transform"] == reference_profile["transform"]
    )

    if not checks:
        raise ValueError(f"{name} is not aligned with reference grid.")

    log.info("%s alignment: PASS", name)


def safe_metrics(y_true, scores):
    """Continuous-score metrics."""
    if np.unique(y_true).size < 2:
        return {
            "roc_auc": None,
            "average_precision": None,
        }

    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(
            average_precision_score(y_true, scores)
        ),
    }


def binary_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    return {
        "precision": float(precision_score(
            y_true, y_pred, zero_division=0
        )),
        "recall": float(recall_score(
            y_true, y_pred, zero_division=0
        )),
        "f1": float(f1_score(
            y_true, y_pred, zero_division=0
        )),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
    }


def top_prevalence_metrics(y_true, scores):
    """
    Select the highest-scoring cells equal in number to the
    observed flooded cells. This is a spatial capture measure,
    not an independently calibrated prediction threshold.
    """
    flooded_count = int(y_true.sum())
    total = len(y_true)

    if flooded_count == 0:
        return {
            "selected_cells": 0,
            "selected_fraction": 0.0,
            "flood_capture_rate": None,
            "precision": None,
            "recall": None,
            "f1": None,
        }

    selected_count = flooded_count
    selected_indices = np.argpartition(
        scores, total - selected_count
    )[total - selected_count:]

    predicted = np.zeros(total, dtype=bool)
    predicted[selected_indices] = True

    captured = int(np.count_nonzero(predicted & (y_true == 1)))

    return {
        "selected_cells": selected_count,
        "selected_fraction": float(selected_count / total),
        "flood_capture_rate": float(captured / flooded_count),
        "precision": float(captured / selected_count),
        "recall": float(captured / flooded_count),
        "f1": float(
            2 * captured / (selected_count + flooded_count)
        ),
    }


def main():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading validation rasters...")

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

    # Read arrays.
    ksdma = loaded["ksdma_hazard"]["data"]
    xgb = loaded["xgb_hazard"]["data"]
    unet = loaded["unet_hazard"]["data"]

    ksdma_mask = (
        loaded["ksdma_mask"]["valid"]
        & (loaded["ksdma_mask"]["data"] > 0.5)
    )

    ksdma_risk = loaded["ksdma_risk"]["data"]
    xgb_risk = loaded["xgb_risk"]["data"]
    unet_risk = loaded["unet_risk"]["data"]

    flood_fraction = loaded["flood_fraction"]["data"]

    # Common domain: all three KSDMA/model prediction/risk
    # surfaces and the observed flood fraction must be valid.
    common_valid = (
        ksdma_mask
        & loaded["ksdma_hazard"]["valid"]
        & loaded["xgb_hazard"]["valid"]
        & loaded["unet_hazard"]["valid"]
        & loaded["ksdma_risk"]["valid"]
        & loaded["xgb_risk"]["valid"]
        & loaded["unet_risk"]["valid"]
        & loaded["flood_fraction"]["valid"]
    )

    if not np.any(common_valid):
        raise ValueError("No common validation cells.")

    log.info("Common validation cells: %d", int(common_valid.sum()))

    # KSDMA reference normalization, same as Task 11.
    reference_values = ksdma[common_valid]
    hazard_min = float(reference_values.min())
    hazard_max = float(reference_values.max())

    if hazard_max <= hazard_min:
        raise ValueError("Invalid KSDMA hazard range.")

    def normalize(values):
        return np.clip(
            (values - hazard_min) / (hazard_max - hazard_min),
            0.0,
            1.0,
        )

    scores = {
        "KSDMA": normalize(ksdma[common_valid]),
        "XGBoost": normalize(xgb[common_valid]),
        "U-Net": normalize(unet[common_valid]),
    }

    risk_scores = {
        "KSDMA": ksdma_risk[common_valid],
        "XGBoost": xgb_risk[common_valid],
        "U-Net": unet_risk[common_valid],
    }

    fractions = flood_fraction[common_valid]

    hazard_rows = []
    risk_rows = []
    binary_rows = []

    # Evaluate each observed flood-fraction threshold.
    for flood_threshold in FLOOD_THRESHOLDS:
        y_true = (fractions >= flood_threshold).astype(np.uint8)

        positive_count = int(y_true.sum())
        prevalence = float(y_true.mean())

        log.info(
            "Flood threshold %.2f: %d positives (%.2f%%)",
            flood_threshold,
            positive_count,
            prevalence * 100,
        )

        if positive_count == 0 or positive_count == len(y_true):
            log.warning(
                "Skipping threshold %.2f: only one class present.",
                flood_threshold,
            )
            continue

        for model_name in scores:
            hazard_score = scores[model_name]
            risk_score = risk_scores[model_name]

            # Hazard ranking.
            h_metrics = safe_metrics(y_true, hazard_score)
            h_capture = top_prevalence_metrics(
                y_true, hazard_score
            )

            hazard_rows.append({
                "flood_fraction_threshold": flood_threshold,
                "model": model_name,
                "observed_flood_cells": positive_count,
                "observed_flood_prevalence": prevalence,
                **h_metrics,
                **{
                    f"top_prevalence_{k}": v
                    for k, v in h_capture.items()
                },
            })

            # Risk ranking.
            r_metrics = safe_metrics(y_true, risk_score)
            r_capture = top_prevalence_metrics(
                y_true, risk_score
            )

            risk_rows.append({
                "flood_fraction_threshold": flood_threshold,
                "model": model_name,
                "observed_flood_cells": positive_count,
                "observed_flood_prevalence": prevalence,
                **r_metrics,
                **{
                    f"top_prevalence_{k}": v
                    for k, v in r_capture.items()
                },
            })

            # Fixed normalized hazard threshold.
            predicted = (
                hazard_score >= FIXED_HAZARD_THRESHOLD
            ).astype(np.uint8)

            binary_rows.append({
                "flood_fraction_threshold": flood_threshold,
                "model": model_name,
                "hazard_score_threshold": FIXED_HAZARD_THRESHOLD,
                **binary_metrics(y_true, predicted),
            })

    hazard_df = pd.DataFrame(hazard_rows)
    risk_df = pd.DataFrame(risk_rows)
    binary_df = pd.DataFrame(binary_rows)

    hazard_path = TABLE_DIR / "13_hazard_validation_metrics.csv"
    risk_path = TABLE_DIR / "13_risk_validation_metrics.csv"
    binary_path = TABLE_DIR / "13_hazard_binary_metrics.csv"

    hazard_df.to_csv(hazard_path, index=False)
    risk_df.to_csv(risk_path, index=False)
    binary_df.to_csv(binary_path, index=False)

    # Main threshold summary.
    main_threshold = 0.50

    main_hazard = hazard_df[
        hazard_df["flood_fraction_threshold"] == main_threshold
    ]
    main_risk = risk_df[
        risk_df["flood_fraction_threshold"] == main_threshold
    ]

    # Plot ROC-AUC sensitivity.
    fig, ax = plt.subplots(figsize=(9, 5))

    for model_name in scores:
        subset = hazard_df[hazard_df["model"] == model_name]
        ax.plot(
            subset["flood_fraction_threshold"],
            subset["roc_auc"],
            marker="o",
            label=model_name,
        )

    ax.set_title("2018 Flood Validation: Hazard ROC-AUC Sensitivity")
    ax.set_xlabel("Observed flood fraction threshold")
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    roc_path = FIGURE_DIR / "13_hazard_roc_auc_sensitivity.png"
    fig.savefig(roc_path, dpi=200)
    plt.close(fig)

    # Plot risk ROC-AUC sensitivity.
    fig, ax = plt.subplots(figsize=(9, 5))

    for model_name in risk_scores:
        subset = risk_df[risk_df["model"] == model_name]
        ax.plot(
            subset["flood_fraction_threshold"],
            subset["roc_auc"],
            marker="o",
            label=model_name,
        )

    ax.set_title("2018 Flood Validation: Risk ROC-AUC Sensitivity")
    ax.set_xlabel("Observed flood fraction threshold")
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    risk_roc_path = FIGURE_DIR / "13_risk_roc_auc_sensitivity.png"
    fig.savefig(risk_roc_path, dpi=200)
    plt.close(fig)

    # Save compact JSON summary.
    summary = {
        "district": "Alappuzha",
        "scenario": "100yr",
        "observed_event": "2018 Sentinel-1 flood extent",
        "common_validation_cells": int(common_valid.sum()),
        "common_domain": "KSDMA modelled cells with all hazard, risk and flood data valid",
        "flood_fraction_thresholds": FLOOD_THRESHOLDS,
        "primary_flood_fraction_threshold": main_threshold,
        "hazard_fixed_score_threshold": FIXED_HAZARD_THRESHOLD,
        "hazard_normalisation": {
            "method": "KSDMA common-domain min-max",
            "minimum": hazard_min,
            "maximum": hazard_max,
        },
        "primary_threshold_hazard_metrics": (
            main_hazard.to_dict(orient="records")
        ),
        "primary_threshold_risk_metrics": (
            main_risk.to_dict(orient="records")
        ),
        "limitations": [
            "2018 observed flood extent is not equivalent to a 100-year return-period hazard map.",
            "Metrics measure spatial agreement with the observed 2018 event.",
            "The common KSDMA modelled domain is used for all model comparisons.",
            "Top-prevalence capture selects the same number of high-scoring cells as observed flooded cells; it is not a calibrated probability threshold.",
            "Hazard binary metrics use a fixed normalized score threshold of 0.5.",
            "Risk scores are relative indices, not probabilities or expected losses.",
        ],
        "outputs": {
            "hazard_metrics_csv": str(hazard_path),
            "risk_metrics_csv": str(risk_path),
            "hazard_binary_metrics_csv": str(binary_path),
            "hazard_roc_plot": str(roc_path),
            "risk_roc_plot": str(risk_roc_path),
        },
    }

    summary_path = TABLE_DIR / "13_2018_validation_summary.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    log.info("Hazard metrics: %s", hazard_path)
    log.info("Risk metrics: %s", risk_path)
    log.info("Binary metrics: %s", binary_path)
    log.info("Summary: %s", summary_path)
    log.info("Task 13 validation completed.")


if __name__ == "__main__":
    main()
