
"""
Task 13E - Exposure source coverage audit.

Run from project root:
    python src/experiments/task13e_source_coverage_audit.py

Descriptive audit only. Does not modify existing rasters or weights.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from sklearn.metrics import roc_auc_score

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

import config as cfg

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("task13e")

FLOOD_THRESHOLD = 0.50

LANDCOVER_NAMES = {
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


def read_raster(path):
    with rasterio.open(path) as src:
        data = src.read(1, masked=True).astype("float64").filled(np.nan)
        profile = {
            "shape": src.shape,
            "transform": src.transform,
            "crs": src.crs,
        }
    return data, profile


def check_alignment(reference, other, name):
    if (
        reference["shape"] != other["shape"]
        or reference["transform"] != other["transform"]
        or reference["crs"] != other["crs"]
    ):
        raise ValueError(f"{name} is not aligned with the reference grid.")


def stats(values):
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "cells": 0,
            "zero_pct": None,
            "nonzero_pct": None,
            "mean": None,
            "median": None,
            "p95": None,
        }

    return {
        "cells": int(values.size),
        "zero_pct": float(np.mean(values <= 0) * 100),
        "nonzero_pct": float(np.mean(values > 0) * 100),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
    }


def main():
    paths = {
        "landcover": cfg.output_path("landcover_mode"),
        "ksdma_mask": cfg.output_path("ksdma_modelled_mask"),
        "flood_fraction": cfg.PROCESSED_DIR / "12_2018_flood_fraction_90m.tif",
        "population": cfg.output_path("exposure_population_count"),
        "buildings": cfg.output_path("exposure_buildings_count"),
        "facilities": cfg.output_path("exposure_facilities_score_raw"),
        "exposure": cfg.output_path("exposure_score"),
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

        log.info("Loaded %s", path.name)

    landcover = arrays["landcover"]
    ksdma = arrays["ksdma_mask"]
    flood_fraction = arrays["flood_fraction"]

    valid_domain = (
        np.isfinite(ksdma)
        & (ksdma > 0)
        & np.isfinite(flood_fraction)
        & np.isfinite(landcover)
    )

    flooded = flood_fraction >= FLOOD_THRESHOLD
    non_flooded = ~flooded

    components = {
        "population": arrays["population"],
        "buildings": arrays["buildings"],
        "facilities": arrays["facilities"],
        "final_exposure": arrays["exposure"],
    }

    rows = []
    overall = {}

    for name, data in components.items():
        valid = valid_domain & np.isfinite(data)
        missing_count = int((valid_domain & ~np.isfinite(data)).sum())

        flooded_values = data[valid & flooded]
        non_flooded_values = data[valid & non_flooded]

        auc = None
        if flooded_values.size and non_flooded_values.size:
            auc = float(
                roc_auc_score(
                    flooded[valid].astype("uint8"),
                    data[valid],
                )
            )

        overall[name] = {
            "valid_cells": int(valid.sum()),
            "missing_cells_in_common_domain": missing_count,
            "missing_pct_in_common_domain": (
                100 * missing_count / int(valid_domain.sum())
            ),
            "flooded": stats(flooded_values),
            "non_flooded": stats(non_flooded_values),
            "flooded_vs_non_flooded_auc": auc,
        }

        for class_code in sorted(
            int(v) for v in np.unique(landcover[valid_domain])
        ):
            class_mask = valid & (landcover == class_code)
            fmask = class_mask & flooded
            nmask = class_mask & non_flooded

            fvals = data[fmask]
            nvals = data[nmask]

            class_auc = None
            if fvals.size and nvals.size:
                class_auc = float(
                    roc_auc_score(
                        flooded[class_mask].astype("uint8"),
                        data[class_mask],
                    )
                )

            fs = stats(fvals)
            ns = stats(nvals)

            rows.append({
                "component": name,
                "landcover_code": class_code,
                "landcover_class": LANDCOVER_NAMES.get(
                    class_code, f"Unknown {class_code}"
                ),
                "flooded_cells": fs["cells"],
                "non_flooded_cells": ns["cells"],
                "flooded_zero_pct": fs["zero_pct"],
                "non_flooded_zero_pct": ns["zero_pct"],
                "flooded_nonzero_pct": fs["nonzero_pct"],
                "non_flooded_nonzero_pct": ns["nonzero_pct"],
                "flooded_mean": fs["mean"],
                "non_flooded_mean": ns["mean"],
                "flooded_median": fs["median"],
                "non_flooded_median": ns["median"],
                "flooded_p95": fs["p95"],
                "non_flooded_p95": ns["p95"],
                "auc_within_landcover": class_auc,
            })

    df = pd.DataFrame(rows)

    cfg.TABLES_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = cfg.TABLES_DIR / "13e_source_coverage_by_landcover.csv"
    json_path = cfg.TABLES_DIR / "13e_source_coverage_summary.json"

    df.to_csv(csv_path, index=False)

    summary = {
        "task": "13E - Exposure source coverage audit",
        "district": cfg.DISTRICT_NAME,
        "flood_fraction_threshold": FLOOD_THRESHOLD,
        "common_domain_cells": int(valid_domain.sum()),
        "flooded_cells": int((valid_domain & flooded).sum()),
        "non_flooded_cells": int((valid_domain & non_flooded).sum()),
        "component_results": overall,
        "limitations": [
            "Zero values indicate no recorded rasterised count or score; "
            "they do not prove that source data are complete.",
            "This audit cannot independently establish OSM or Google "
            "building/facility completeness.",
            "Population, building and facility layers have different "
            "source dates and collection methods.",
            "AUC is descriptive association with the 2018 flood mask, "
            "not a measure of source-data completeness.",
        ],
        "outputs": {
            "csv": str(csv_path.relative_to(cfg.PROJECT_ROOT)),
            "summary_json": str(json_path.relative_to(cfg.PROJECT_ROOT)),
        },
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log.info("Saved %s", csv_path)
    log.info("Saved %s", json_path)
    log.info("Task 13E completed.")


if __name__ == "__main__":
    main()
