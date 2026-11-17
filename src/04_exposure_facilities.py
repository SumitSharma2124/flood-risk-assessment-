"""Task 5: critical-facility exposure on the common 90 m grid.

Run after Task 3::

    python src/04_exposure_facilities.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

from src import config as cfg
from src import utils

LOGGER = utils.get_logger("exposure_facilities")

BASE_CATEGORIES = {
    "hospital": "hospital",
    "fire_station": "fire_station",
    "police": "police",
    "substation": "substation",
    "plant": "power_plant",
    "school": "school",
    "college": "college",
    "clinic": "clinic",
    "water_point": "water_point",
}


def _require_inputs() -> None:
    required = {
        "facilities_clip": cfg.output_path("facilities_clip"),
        "district_mask": cfg.output_path("district_mask"),
        "grid": cfg.GRID_PATH,
    }
    missing = [
        f"{name}: {path}"
        for name, path in required.items()
        if not Path(path).exists()
    ]
    if missing:
        formatted = "\n".join(f"  - {item}" for item in missing)
        raise FileNotFoundError(f"Missing Task 5 input(s):\n{formatted}")


def _mask_domain(array: np.ndarray, district: np.ndarray) -> np.ndarray:
    result = np.full(array.shape, cfg.NODATA, dtype="float32")
    result[district] = np.asarray(array, dtype="float32")[district]
    return result


def _text_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series("", index=frame.index, dtype="object")
    return frame[column].fillna("").astype(str).str.lower().str.strip()


def _categorize(facilities):
    facility_type = _text_column(facilities, "facility_type")
    man_made = _text_column(facilities, "man_made")
    natural = _text_column(facilities, "natural")
    landuse = _text_column(facilities, "landuse")

    category = facility_type.map(BASE_CATEGORIES)

    unknown = facility_type == "unknown"
    category.loc[unknown & (man_made == "pumping_station")] = "pump_house"
    category.loc[unknown & (man_made == "water_tower")] = "water_tower"

    excluded_mask = (
        facility_type.isin(["dam", "water"])
        | (landuse == "reservoir")
        | (natural == "water")
        | category.isna()
    )
    category.loc[excluded_mask] = pd.NA

    valid_categories = set(cfg.FACILITY_CATEGORIES)
    category = category.where(category.isin(valid_categories))

    return category.astype("object"), {
        "facility_type": facility_type,
        "man_made": man_made,
        "natural": natural,
        "landuse": landuse,
        "excluded": excluded_mask,
    }


def _prepare_facilities() -> tuple[object, dict]:
    facilities = utils.read_vector(cfg.output_path("facilities_clip"))
    facilities = facilities.copy()

    category, fields = _categorize(facilities)
    facilities["_category"] = category
    facilities["_source_category"] = fields["facility_type"]
    facilities["_man_made"] = fields["man_made"]
    facilities["_natural"] = fields["natural"]
    facilities["_landuse"] = fields["landuse"]

    excluded = facilities.loc[fields["excluded"]].copy()
    exposed = facilities.loc[~fields["excluded"]].copy()

    explained_exclusion = (
        fields["facility_type"].isin(["dam", "water"])
        | (fields["landuse"] == "reservoir")
    )
    excluded_reasons = {
        "dam": int((fields["facility_type"] == "dam").sum()),
        "water": int((fields["facility_type"] == "water").sum()),
        "reservoir_landuse": int((fields["landuse"] == "reservoir").sum()),
        "unmapped": int((fields["excluded"] & ~explained_exclusion).sum()),
    }

    centroids = utils.polygon_centroids(exposed, crs=cfg.CRS_WORKING)
    centroids = centroids.reset_index(drop=True)

    district_geometry = utils.load_district(
        crs=cfg.CRS_WORKING,
        as_geometry=True,
    )
    centroids = centroids.loc[
        centroids.within(district_geometry)
    ].reset_index(drop=True)
    inside_count = int(len(centroids))

    deduped, dedup_info = utils.deduplicate_points(
        centroids,
        cfg.FACILITY_DEDUP_RADIUS_M,
        group_col="_category",
    )

    source_counts = exposed["_category"].value_counts().reindex(
        cfg.FACILITY_CATEGORIES,
        fill_value=0,
    )
    final_counts = deduped["_category"].value_counts().reindex(
        cfg.FACILITY_CATEGORIES,
        fill_value=0,
    )

    stats = {
        "raw_features": int(len(facilities)),
        "excluded_features": int(len(excluded)),
        "exposed_features": int(len(exposed)),
        "inside_district": inside_count,
        "after_deduplication": int(len(deduped)),
        "deduplication": dedup_info,
        "excluded_reasons": excluded_reasons,
        "category_counts_source": {
            str(key): int(value) for key, value in source_counts.items()
        },
        "category_counts": {
            str(key): int(value) for key, value in final_counts.items()
        },
        "category_order": list(cfg.FACILITY_CATEGORIES),
        "category_weights": dict(cfg.FACILITY_CATEGORY_WEIGHTS),
        "dedup_radius_m": float(cfg.FACILITY_DEDUP_RADIUS_M),
        "smoothing_radius_m": float(cfg.FACILITY_SMOOTH_RADIUS_M),
        "coverage": {
            "excluded_fraction": (
                float(len(excluded) / len(facilities)) if len(facilities) else 0.0
            ),
            "inside_fraction_of_exposed": (
                float(inside_count / len(exposed)) if len(exposed) else 0.0
            ),
        },
    }
    return deduped, stats


def main() -> None:
    cfg.check_input_paths()
    _require_inputs()

    district_raw, _, _ = utils.read_raster(cfg.output_path("district_mask"))
    district = np.asarray(district_raw) == 1

    facilities, prep_stats = _prepare_facilities()
    if facilities.empty:
        raise RuntimeError("No facilities left after filtering")

    category_layers = []
    score_raw = np.zeros(district.shape, dtype="float64")
    for category in cfg.FACILITY_CATEGORIES:
        subset = facilities.loc[facilities["_category"] == category]
        layer, _ = utils.points_to_grid(
            subset,
            template=cfg.GRID_PATH,
            dtype="float32",
        )
        layer = _mask_domain(layer, district)
        category_layers.append(layer)
        score_raw[district] += (
            layer[district].astype("float64")
            * float(cfg.FACILITY_CATEGORY_WEIGHTS[category])
        )

    counts = np.stack(category_layers).astype("float32")
    utils.write_multiband_raster(
        counts,
        cfg.output_path("exposure_facilities_counts"),
        cfg.GRID_PATH,
        band_names=list(cfg.FACILITY_CATEGORIES),
        nodata=cfg.NODATA,
        dtype="float32",
    )

    score_raw = _mask_domain(score_raw, district)
    utils.write_raster(
        score_raw,
        cfg.output_path("exposure_facilities_score_raw"),
        cfg.GRID_PATH,
        nodata=cfg.NODATA,
        dtype="float32",
    )

    score_smooth, _ = utils.smooth_raster(
        score_raw,
        cfg.FACILITY_SMOOTH_RADIUS_M,
        cfg.GRID_PATH,
        valid_mask=district,
        nodata=cfg.NODATA,
        dtype="float32",
    )
    score_smooth = _mask_domain(score_smooth, district)
    utils.write_raster(
        score_smooth,
        cfg.output_path("exposure_facilities_score"),
        cfg.GRID_PATH,
        nodata=cfg.NODATA,
        dtype="float32",
    )

    raw_values = score_raw[district]
    smooth_values = score_smooth[district]
    stats = {
        **prep_stats,
        "grid": {
            "height": int(district.shape[0]),
            "width": int(district.shape[1]),
        },
        "score_raw": {
            "min": float(raw_values.min()),
            "max": float(raw_values.max()),
            "mean": float(raw_values.mean()),
            "nonzero_cells": int((raw_values > 0).sum()),
        },
        "score_smoothed": {
            "min": float(smooth_values.min()),
            "max": float(smooth_values.max()),
            "mean": float(smooth_values.mean()),
            "nonzero_cells": int((smooth_values > 0).sum()),
        },
        "checks": {
            "has_facilities": bool(prep_stats["after_deduplication"] > 0),
            "score_nonzero": bool((smooth_values > 0).any()),
        },
        "caveat": (
            "Facilities are sparse and point-based; the 500 m smoothed score "
            "spreads local service access across nearby cells."
        ),
    }

    categories_path = cfg.output_path("exposure_facilities_categories")
    categories_path.write_text(
        json.dumps(
            {
                "categories": list(cfg.FACILITY_CATEGORIES),
                "weights": dict(cfg.FACILITY_CATEGORY_WEIGHTS),
                "dedup_radius_m": float(cfg.FACILITY_DEDUP_RADIUS_M),
                "smoothing_radius_m": float(cfg.FACILITY_SMOOTH_RADIUS_M),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    stats_path = cfg.output_path("exposure_facilities_stats")
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    utils.assert_same_grid(
        cfg.output_path("exposure_facilities_score"),
        cfg.GRID_PATH,
    )

    LOGGER.info(
        "Facilities: %s raw -> %s inside district -> %s after %.0f m dedup",
        f"{prep_stats['raw_features']:,}",
        f"{prep_stats['inside_district']:,}",
        f"{prep_stats['after_deduplication']:,}",
        cfg.FACILITY_DEDUP_RADIUS_M,
    )
    LOGGER.info("Facility exposure complete")


if __name__ == "__main__":
    main()
