"""Task 5: building exposure on the common 90 m grid.

Run after Task 2 and Task 3::

    python src/04_exposure_buildings.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import geopandas as gpd

from src import config as cfg
from src import utils

LOGGER = utils.get_logger("exposure_buildings")


def _require_inputs() -> None:
    required = {
        "buildings_clip": cfg.output_path("buildings_clip"),
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


def _prepare_buildings() -> tuple[gpd.GeoDataFrame, dict]:
    buildings = utils.read_vector(cfg.output_path("buildings_clip"))
    raw_count = int(len(buildings))

    if "area_in_meters" in buildings.columns:
        area = buildings["area_in_meters"].astype("float64")
    else:
        area = buildings.geometry.area.astype("float64")
    buildings = buildings.assign(_area_m2=area)

    confidence = buildings["confidence"].astype("float64")
    buildings = buildings[
        confidence >= float(cfg.BUILDING_CONFIDENCE_THRESHOLD)
    ].copy()
    after_confidence = int(len(buildings))

    buildings = buildings[
        buildings["_area_m2"] >= float(cfg.MIN_BUILDING_FOOTPRINT_M2)
    ].copy()
    after_area = int(len(buildings))

    duplicate_columns = ["latitude", "longitude", "_area_m2"]
    if all(column in buildings.columns for column in duplicate_columns):
        buildings = buildings.drop_duplicates(subset=duplicate_columns).copy()
    else:
        buildings = buildings.drop_duplicates(subset=["geometry"]).copy()
    after_duplicates = int(len(buildings))

    centroids = utils.polygon_centroids(buildings, crs=cfg.CRS_WORKING)
    centroids = centroids.reset_index(drop=True)

    district_geometry = utils.load_district(
        crs=cfg.CRS_WORKING,
        as_geometry=True,
    )
    inside = centroids.within(district_geometry)
    centroids = centroids.loc[inside].reset_index(drop=True)

    stats = {
        "raw_features": raw_count,
        "after_confidence": after_confidence,
        "after_min_area": after_area,
        "after_duplicates": after_duplicates,
        "inside_district": int(len(centroids)),
        "confidence_removed": int(raw_count - after_confidence),
        "small_area_removed": int(after_confidence - after_area),
        "duplicate_removed": int(after_area - after_duplicates),
        "outside_district_removed": int(after_duplicates - len(centroids)),
        "confidence_threshold": float(cfg.BUILDING_CONFIDENCE_THRESHOLD),
        "minimum_footprint_m2": float(cfg.MIN_BUILDING_FOOTPRINT_M2),
        "duplicate_rule": "latitude + longitude + footprint area",
    }
    return centroids, stats


def main() -> None:
    cfg.check_input_paths()
    _require_inputs()

    district_raw, _, _ = utils.read_raster(cfg.output_path("district_mask"))
    district = np.asarray(district_raw) == 1

    centroids, prep_stats = _prepare_buildings()
    if centroids.empty:
        raise RuntimeError("No buildings left after filtering")

    counts, _ = utils.points_to_grid(
        centroids,
        template=cfg.GRID_PATH,
        dtype="float32",
    )
    areas, _ = utils.points_to_grid(
        centroids,
        template=cfg.GRID_PATH,
        values=centroids["_area_m2"].to_numpy(dtype="float64"),
        dtype="float32",
    )

    counts = _mask_domain(counts, district)
    areas = _mask_domain(areas, district)

    utils.write_raster(
        counts,
        cfg.output_path("exposure_buildings_count"),
        cfg.GRID_PATH,
        nodata=cfg.NODATA,
        dtype="float32",
    )
    utils.write_raster(
        areas,
        cfg.output_path("exposure_buildings_area"),
        cfg.GRID_PATH,
        nodata=cfg.NODATA,
        dtype="float32",
    )

    count_values = counts[district]
    area_values = areas[district]
    building_count = int(count_values.sum())
    building_area = float(area_values.sum())
    reference = float(cfg.BUILDING_COUNT_REFERENCE)
    ratio = building_count / reference if reference else 0.0

    stats = {
        "grid": {"height": int(counts.shape[0]), "width": int(counts.shape[1])},
        "preparation": prep_stats,
        "district_buildings_count": building_count,
        "district_buildings_area_m2": building_area,
        "district_buildings_area_km2": building_area / 1e6,
        "nonzero_cells": int((count_values > 0).sum()),
        "max_buildings_per_cell": float(count_values.max()),
        "mean_buildings_per_cell": float(count_values.mean()),
        "max_area_per_cell_m2": float(area_values.max()),
        "reference_count": reference,
        "count_ratio_to_reference": ratio,
        "checks": {
            "same_order_as_reference": bool(0.25 <= ratio <= 2.0),
            "all_inside_values_nonnegative": bool(
                np.isfinite(count_values).all() and (count_values >= 0).all()
            ),
        },
        "caveat": (
            "Google Open Buildings uses newer imagery than 2018 and may miss "
            "structures under dense tree cover."
        ),
    }
    stats_path = cfg.output_path("exposure_buildings_stats")
    utils.make_folder(stats_path.parent)
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    utils.assert_same_grid(
        cfg.output_path("exposure_buildings_count"),
        cfg.GRID_PATH,
    )
    utils.assert_same_grid(
        cfg.output_path("exposure_buildings_area"),
        cfg.GRID_PATH,
    )

    LOGGER.info(
        "Buildings: %s raw -> %s inside district; %.2f km2 footprint",
        f"{prep_stats['raw_features']:,}",
        f"{building_count:,}",
        building_area / 1e6,
    )
    if not stats["checks"]["same_order_as_reference"]:
        LOGGER.warning(
            "Building count %.0f is outside the expected order of magnitude",
            building_count,
        )
    LOGGER.info("Building exposure complete")


if __name__ == "__main__":
    main()
