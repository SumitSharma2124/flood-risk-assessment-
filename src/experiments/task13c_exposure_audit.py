
"""
Task 13C: Exposure component audit.

Compares population, building and facility components between
observed flooded and non-flooded cells.

Uses the same 71,799-cell KSDMA validation domain.
Does not modify existing rasters or configuration.

Run from project root:
    python src/experiments/task13c_exposure_audit.py
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
    "district_mask": ROOT / "data/processed/district_mask.tif",
    "ksdma_mask": ROOT / "data/processed/ksdma_modelled_mask.tif",

    "population_count": ROOT / "data/processed/exposure_population_count.tif",
    "population_scaled": ROOT / "data/processed/exposure_pop.tif",

    "building_count": ROOT / "data/processed/exposure_buildings_count.tif",
    "building_area": ROOT / "data/processed/exposure_buildings_area.tif",

    "facility_raw": ROOT / "data/processed/exposure_facilities_score.tif",
    "exposure_final": ROOT / "data/processed/exposure_score.tif",

    "flood_fraction": ROOT / "data/processed/12_2018_flood_fraction_90m.tif",
}

OUTPUT_DIR = ROOT / "outputs/tables"

FLOOD_THRESHOLD = 0.50

# Must match the current config.py.
WEIGHTS = {
    "population": 0.50,
    "buildings": 0.30,
    "facilities": 0.20,
}

BUILDING_METRIC = "count"
CAP_PERCENTILE = 99
SCALE_LOG = False

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)
log = logging.getLogger("task13c")


def read_raster(path):
    with rasterio.open(path) as ds:
        data = ds.read(1).astype(np.float32)
        valid = ds.read_masks(1) > 0

        if ds.nodata is not None:
            valid &= data != ds.nodata

        valid &= np.isfinite(data)

        profile = ds.profile.copy()

    return data, valid, profile


def check_alignment(reference, profile, name):
    aligned = (
        reference["height"] == profile["height"]
        and reference["width"] == profile["width"]
        and reference["crs"] == profile["crs"]
        and reference["transform"] == profile["transform"]
    )

    if not aligned:
        raise ValueError(f"{name} is not aligned with reference grid.")



def robust_scale(values, valid_mask, upper_percentile=99, log1p=False):
    data = np.asarray(values, dtype="float64")

    finite = np.isfinite(data) & (data != -9999.0)
    domain_values = data[valid_mask & finite]

    if not domain_values.size:
        raise ValueError("Cannot scale: no finite values inside domain")

    transformed = data.copy()

    if log1p:
        transformed = np.log1p(np.maximum(transformed, 0.0))

    selected = transformed[valid_mask & finite]

    lower = float(np.percentile(selected, 0.0))
    upper = float(np.percentile(selected, upper_percentile))

    if upper <= lower:
        upper = lower + 1.0

    scaled = (
        np.clip(transformed, lower, upper) - lower
    ) / (upper - lower)

    output = np.full(data.shape, -9999.0, dtype=np.float64)
    output[valid_mask] = np.where(
        finite[valid_mask],
        scaled[valid_mask],
        0.0
    )

    return output.astype(np.float32), {
        "lower_value": lower,
        "upper_value": upper,
        "valid_values": int(domain_values.size),
    }


def component_stats(values, labels):
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.uint8)

    flooded = values[labels == 1]
    non_flooded = values[labels == 0]

    return {
        "flooded_mean": float(flooded.mean()),
        "non_flooded_mean": float(non_flooded.mean()),
        "mean_difference": float(flooded.mean() - non_flooded.mean()),

        "flooded_median": float(np.median(flooded)),
        "non_flooded_median": float(np.median(non_flooded)),

        "flooded_p95": float(np.percentile(flooded, 95)),
        "non_flooded_p95": float(np.percentile(non_flooded, 95)),

        "flooded_zero_fraction": float((flooded == 0).mean()),
        "non_flooded_zero_fraction": float((non_flooded == 0).mean()),

        "roc_auc": float(roc_auc_score(labels, values)),
        "average_precision": float(
            average_precision_score(labels, values)
        ),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    loaded = {}
    reference = None

    log.info("Loading exposure inputs...")

    for name, path in INPUTS.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing input: {path}")

        data, valid, profile = read_raster(path)

        if reference is None:
            reference = profile
        else:
            check_alignment(reference, profile, name)

        loaded[name] = {
            "data": data,
            "valid": valid,
        }

        log.info("%s: %d valid cells", name, int(valid.sum()))

    district = (
        (loaded["district_mask"]["data"] == 1)
        & loaded["district_mask"]["valid"]
    )

    ksdma = (
        (loaded["ksdma_mask"]["data"] > 0)
        & loaded["ksdma_mask"]["valid"]
    )

    # All component rasters must be valid.
    required = [
        "population_count",
        "population_scaled",
        "building_count",
        "building_area",
        "facility_raw",
        "exposure_final",
        "flood_fraction",
    ]

    common = district & ksdma

    for name in required:
        common &= loaded[name]["valid"]

    count = int(common.sum())

    if count != 71799:
        log.warning(
            "Common domain contains %d cells; expected 71799. "
            "Check input masks and alignment.",
            count
        )

    log.info("Common audit domain: %d cells", count)

    flood_fraction = loaded["flood_fraction"]["data"][common]

    labels = (
        flood_fraction >= FLOOD_THRESHOLD
    ).astype(np.uint8)

    log.info(
        "Flooded cells: %d | Non-flooded cells: %d",
        int(labels.sum()),
        int((labels == 0).sum())
    )

    # Extract component arrays on the common domain.
    population_count = loaded["population_count"]["data"][common]
    population_scaled = loaded["population_scaled"]["data"][common]

    building_count = loaded["building_count"]["data"][common]
    building_area = loaded["building_area"]["data"][common]

    facility_raw = loaded["facility_raw"]["data"][common]
    exposure_final = loaded["exposure_final"]["data"][common]

    # Reproduce scaling for an independent comparison.
    district_mask = district

    building_source = (
        loaded["building_count"]["data"]
        if BUILDING_METRIC == "count"
        else loaded["building_area"]["data"]
    )

    building_recalculated, building_scale_info = robust_scale(
        building_source,
        district_mask,
        upper_percentile=CAP_PERCENTILE,
        log1p=SCALE_LOG,
    )

    facility_recalculated, facility_scale_info = robust_scale(
        loaded["facility_raw"]["data"],
        district_mask,
        upper_percentile=CAP_PERCENTILE,
        log1p=False,
    )

    building_recalculated = building_recalculated[common]
    facility_recalculated = facility_recalculated[common]

    # Reconstruct exposure from the saved population layer and
    # independently recalculated building/facility components.
    exposure_reconstructed = (
        WEIGHTS["population"] * population_scaled
        + WEIGHTS["buildings"] * building_recalculated
        + WEIGHTS["facilities"] * facility_recalculated
    )

    exposure_reconstructed = np.clip(
        exposure_reconstructed, 0.0, 1.0
    )

    surfaces = {
        "Population count": population_count,
        "Population scaled": population_scaled,
        "Building count": building_count,
        "Building area": building_area,
        "Facility raw": facility_raw,
        "Facility scaled": facility_recalculated,
        "Building scaled": building_recalculated,
        "Exposure final": exposure_final,
        "Exposure reconstructed": exposure_reconstructed,
    }

    rows = []

    for name, values in surfaces.items():
        stats = component_stats(values, labels)

        rows.append({
            "component": name,
            **stats,
        })

    comparison = pd.DataFrame(rows)

    # Check reconstruction against the existing exposure raster.
    reconstruction_error = (
        exposure_reconstructed - exposure_final
    )

    reconstruction = {
        "mean_absolute_difference": float(
            np.mean(np.abs(reconstruction_error))
        ),
        "maximum_absolute_difference": float(
            np.max(np.abs(reconstruction_error))
        ),
        "pearson_correlation": float(
            np.corrcoef(
                exposure_reconstructed,
                exposure_final
            )[0, 1]
        ),
    }

    # Weighted component contributions.
    contributions = {
        "population": WEIGHTS["population"] * population_scaled,
        "buildings": WEIGHTS["buildings"] * building_recalculated,
        "facilities": WEIGHTS["facilities"] * facility_recalculated,
    }

    contribution_rows = []

    for name, values in contributions.items():
        contribution_rows.append({
            "component": name,
            "weight": WEIGHTS[name],
            "flooded_mean_contribution": float(
                values[labels == 1].mean()
            ),
            "non_flooded_mean_contribution": float(
                values[labels == 0].mean()
            ),
            "difference": float(
                values[labels == 1].mean()
                - values[labels == 0].mean()
            ),
        })

    contribution_df = pd.DataFrame(contribution_rows)

    comparison_path = OUTPUT_DIR / "13c_exposure_component_metrics.csv"
    contribution_path = OUTPUT_DIR / "13c_exposure_weighted_contributions.csv"
    summary_path = OUTPUT_DIR / "13c_exposure_audit_summary.json"

    comparison.to_csv(comparison_path, index=False)
    contribution_df.to_csv(contribution_path, index=False)

    report = {
        "task": "Task 13C - Exposure component audit",
        "district": "Alappuzha",
        "scenario": "100yr",
        "common_validation_cells": count,
        "flood_fraction_threshold": FLOOD_THRESHOLD,
        "flooded_cells": int(labels.sum()),
        "non_flooded_cells": int((labels == 0).sum()),

        "exposure_configuration": {
            "weights": WEIGHTS,
            "building_metric": BUILDING_METRIC,
            "cap_percentile": CAP_PERCENTILE,
            "log_scaling": SCALE_LOG,
        },

        "recalculated_scaling": {
            "building": building_scale_info,
            "facility": facility_scale_info,
        },

        "reconstruction_check": reconstruction,

        "weighted_component_contributions": contribution_rows,

        "outputs": {
            "component_metrics": str(comparison_path),
            "weighted_contributions": str(contribution_path),
        },

        "limitations": [
            "The audit uses 2018 observed flood extent, not a "
            "100-year flood event.",
            "ROC-AUC and Average Precision describe spatial "
            "association, not causal effects.",
            "Current buildings and facilities may not represent "
            "their exact 2018 locations.",
            "The independent scaling reconstruction must be "
            "checked against the project's actual utils.robust_scale "
            "implementation before drawing conclusions.",
        ],
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    log.info("Saved: %s", comparison_path)
    log.info("Saved: %s", contribution_path)
    log.info("Saved: %s", summary_path)
    log.info("Task 13C completed.")


if __name__ == "__main__":
    main()
