
"""
Task 13D - Land-cover-stratified exposure audit.

Run from the project root:
    python src/experiments/task13d_landcover_exposure_audit.py

Uses the common KSDMA modelled domain and 2018 flood-fraction threshold 0.50.
Does not modify any existing raster or model.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import rasterio
from sklearn.metrics import roc_auc_score

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

import config as cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)
log = logging.getLogger("task13d")


CLASS_NAMES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}

FLOOD_THRESHOLD = 0.50


def read_raster(path):
    """Read one raster as float64, converting NoData to NaN."""
    with rasterio.open(path) as src:
        arr = src.read(1, masked=True).astype("float64").filled(np.nan)
        profile = {
            "shape": src.shape,
            "transform": src.transform,
            "crs": src.crs,
        }
    return arr, profile


def check_alignment(reference, candidate, name):
    if (
        reference["shape"] != candidate["shape"]
        or reference["transform"] != candidate["transform"]
        or reference["crs"] != candidate["crs"]
    ):
        raise ValueError(f"Grid mismatch: {name} is not aligned.")


def safe_mean(values):
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else None


def safe_median(values):
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else None


def safe_zero_fraction(values):
    values = values[np.isfinite(values)]
    return float(np.mean(values <= 0)) if values.size else None


def main():
    paths = {
        "landcover": cfg.output_path("landcover_mode"),
        "ksdma_mask": cfg.output_path("ksdma_modelled_mask"),
        "flood_fraction": cfg.PROCESSED_DIR / "12_2018_flood_fraction_90m.tif",
        "exposure": cfg.output_path("exposure_score"),
        "population_count": cfg.output_path("exposure_population_count"),
        "building_count": cfg.output_path("exposure_buildings_count"),
        "facility_score_raw": cfg.output_path("exposure_facilities_score_raw"),
    }

    arrays = {}
    reference = None

    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name} raster: {path}")

        arrays[name], profile = read_raster(path)

        if reference is None:
            reference = profile
        else:
            check_alignment(reference, profile, name)

        log.info("Loaded %-18s %s", name, path.name)

    landcover = arrays["landcover"]
    ksdma_mask = arrays["ksdma_mask"]
    flood_fraction = arrays["flood_fraction"]
    exposure = arrays["exposure"]

    population = arrays["population_count"]
    buildings = arrays["building_count"]
    facilities = arrays["facility_score_raw"]

    # Same common domain used by Tasks 13 and 13C.
    common = (
        np.isfinite(ksdma_mask)
        & (ksdma_mask > 0)
        & np.isfinite(flood_fraction)
        & np.isfinite(landcover)
        & np.isfinite(exposure)
    )

    # Require valid component values for component-specific comparisons.
    common &= (
        np.isfinite(population)
        & np.isfinite(buildings)
        & np.isfinite(facilities)
    )

    flooded = flood_fraction >= FLOOD_THRESHOLD
    non_flooded = ~flooded

    domain_count = int(common.sum())
    if domain_count == 0:
        raise ValueError("No valid cells in the common audit domain.")

    log.info("Common audit domain: %s cells", f"{domain_count:,}")
    log.info(
        "Flood threshold %.2f: %s flooded, %s non-flooded",
        FLOOD_THRESHOLD,
        f"{int((common & flooded).sum()):,}",
        f"{int((common & non_flooded).sum()):,}",
    )

    rows = []

    present_classes = sorted(
        int(v)
        for v in np.unique(landcover[common])
        if np.isfinite(v)
    )

    for class_value in present_classes:
        class_mask = common & (landcover == class_value)
        class_flooded = class_mask & flooded
        class_non_flooded = class_mask & non_flooded

        n_total = int(class_mask.sum())
        n_flooded = int(class_flooded.sum())
        n_non_flooded = int(class_non_flooded.sum())

        exp_flooded = exposure[class_flooded]
        exp_non_flooded = exposure[class_non_flooded]

        # AUC is only defined when both flooded and non-flooded cells exist.
        if n_flooded > 0 and n_non_flooded > 0:
            auc = float(
                roc_auc_score(
                    flooded[class_mask].astype("uint8"),
                    exposure[class_mask],
                )
            )
        else:
            auc = None

        rows.append({
            "landcover_code": class_value,
            "landcover_class": CLASS_NAMES.get(
                class_value, f"Unknown class {class_value}"
            ),
            "cells": n_total,
            "share_of_common_domain_pct": 100 * n_total / domain_count,
            "flooded_cells": n_flooded,
            "non_flooded_cells": n_non_flooded,
            "flooded_share_within_class_pct": (
                100 * n_flooded / n_total if n_total else None
            ),
            "mean_2018_flood_fraction": safe_mean(
                flood_fraction[class_mask]
            ),
            "exposure_flooded_mean": safe_mean(exp_flooded),
            "exposure_non_flooded_mean": safe_mean(exp_non_flooded),
            "exposure_mean_difference_flooded_minus_non": (
                safe_mean(exp_flooded) - safe_mean(exp_non_flooded)
                if n_flooded and n_non_flooded else None
            ),
            "exposure_flooded_median": safe_median(exp_flooded),
            "exposure_non_flooded_median": safe_median(exp_non_flooded),
            "exposure_flooded_zero_fraction": safe_zero_fraction(
                exp_flooded
            ),
            "exposure_non_flooded_zero_fraction": safe_zero_fraction(
                exp_non_flooded
            ),
            "exposure_auc_within_class": auc,
            "population_count_flooded_mean": safe_mean(
                population[class_flooded]
            ),
            "population_count_non_flooded_mean": safe_mean(
                population[class_non_flooded]
            ),
            "building_count_flooded_mean": safe_mean(
                buildings[class_flooded]
            ),
            "building_count_non_flooded_mean": safe_mean(
                buildings[class_non_flooded]
            ),
            "facility_score_flooded_mean": safe_mean(
                facilities[class_flooded]
            ),
            "facility_score_non_flooded_mean": safe_mean(
                facilities[class_non_flooded]
            ),
        })

    df = pd.DataFrame(rows).sort_values("landcover_code")

    cfg.TABLES_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = cfg.TABLES_DIR / "13d_landcover_exposure_audit.csv"
    json_path = cfg.TABLES_DIR / "13d_landcover_exposure_audit_summary.json"

    df.to_csv(csv_path, index=False)

    summary = {
        "task": "13D - Land-cover-stratified exposure audit",
        "district": cfg.DISTRICT_NAME,
        "flood_fraction_threshold": FLOOD_THRESHOLD,
        "common_domain_cells": domain_count,
        "flooded_cells": int((common & flooded).sum()),
        "non_flooded_cells": int((common & non_flooded).sum()),
        "landcover_valid_cells": domain_count,
        "landcover_classes_present": len(present_classes),
        "landcover_class_codes": present_classes,
        "interpretation_note": (
            "Exposure measures people and assets potentially affected; "
            "it is not a flood-likelihood score. Differences by land-cover "
            "class are descriptive and do not establish causation."
        ),
        "outputs": {
            "csv": str(csv_path.relative_to(cfg.PROJECT_ROOT)),
            "summary_json": str(json_path.relative_to(cfg.PROJECT_ROOT)),
        },
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log.info("Saved: %s", csv_path)
    log.info("Saved: %s", json_path)
    log.info("Task 13D completed.")


if __name__ == "__main__":
    main()
