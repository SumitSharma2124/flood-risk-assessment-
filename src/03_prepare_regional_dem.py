"""Task 4: build the regional DEM and burn OSM waterways into it.

Run after the Task 3 grid/aligned layers exist::

    python src/03_prepare_regional_dem.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.merge import merge
from rasterio.warp import transform_bounds
from shapely.geometry import box

from src import config as cfg
from src import utils

LOGGER = utils.get_logger("prepare_regional_dem")


def _regional_grid() -> dict:
    paths = [Path(path) for path in cfg.DEM_REGIONAL_TILES]
    missing = [path for path in paths if not path.exists()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing regional DEM tile(s):\n{formatted}")

    bounds = []
    for path in paths:
        with rasterio.open(path) as src:
            if src.crs is None:
                raise ValueError(f"DEM tile has no CRS: {path}")
            bounds.append(src.bounds)

    left = min(item.left for item in bounds)
    bottom = min(item.bottom for item in bounds)
    right = max(item.right for item in bounds)
    top = max(item.top for item in bounds)
    utm_bounds = transform_bounds(
        cfg.CRS_GEOGRAPHIC,
        cfg.CRS_WORKING,
        left,
        bottom,
        right,
        top,
        densify_pts=21,
    )

    grid = utils.build_grid(
        bounds=utm_bounds,
        resolution=cfg.GRID_RESOLUTION,
        crs=cfg.CRS_WORKING,
        buffer_m=0,
    )
    utils.save_grid(grid, cfg.output_path("regional_grid"))
    LOGGER.info(
        "Regional grid: %s x %s cells, bounds=%s",
        f"{grid['width']:,}",
        f"{grid['height']:,}",
        tuple(round(value, 1) for value in grid["bounds"]),
    )
    return grid


def _mosaic_dem() -> tuple[np.ndarray, object, object]:
    datasets = [rasterio.open(path) for path in cfg.DEM_REGIONAL_TILES]
    try:
        mosaic, transform = merge(
            datasets,
            nodata=np.nan,
            dtype="float32",
        )
        crs = datasets[0].crs
    finally:
        for dataset in datasets:
            dataset.close()
    return mosaic[0], transform, crs


def _fill_regional_edges(dem: np.ndarray) -> tuple[np.ndarray, int, float]:
    """Fill UTM edge wedges by copying the nearest valid DEM cell.

    The regional grid is a snapped UTM box around four geographic tiles, so its
    corners extend outside the source mosaic.  These cells are outside
    Alappuzha and are filled only so the flow router has a complete array.
    """
    from scipy.ndimage import distance_transform_edt

    valid = dem != cfg.NODATA
    if not valid.any():
        raise RuntimeError("Regional DEM has no valid cells")
    if valid.all():
        return dem, 0, 0.0

    distances, indices = distance_transform_edt(
        ~valid,
        return_distances=True,
        return_indices=True,
    )
    filled = dem[tuple(indices)]
    filled_count = int((~valid).sum())
    max_distance = float(distances[~valid].max())
    return filled, filled_count, max_distance


def _load_waterways(grid: dict) -> tuple[gpd.GeoDataFrame, int]:
    frames: list[gpd.GeoDataFrame] = []
    for district, path in cfg.WATERWAYS_BY_DISTRICT.items():
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Missing waterways for {district}: {path}")

        gdf = utils.read_vector(path)
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
        gdf = gdf[
            gdf.geom_type.isin(
                ["LineString", "MultiLineString", "Polygon", "MultiPolygon"]
            )
        ].copy()

        waterway = (
            gdf["waterway"].fillna("")
            if "waterway" in gdf.columns
            else pd.Series("", index=gdf.index)
        )
        natural = (
            gdf["natural"].fillna("")
            if "natural" in gdf.columns
            else pd.Series("", index=gdf.index)
        )
        water = (
            gdf["water"].notna()
            if "water" in gdf.columns
            else pd.Series(False, index=gdf.index)
        )
        relevant = (
            waterway.isin(cfg.WATERWAY_FLOW_TYPES)
            | (natural == "water")
            | water
        )
        gdf = gdf[relevant].copy()
        gdf["source_district"] = district
        frames.append(gdf)

    if not frames:
        raise RuntimeError("No waterway features found")

    combined = gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True),
        geometry="geometry",
        crs=frames[0].crs,
    )
    before = len(combined)
    if "id" in combined.columns:
        combined = combined.sort_values(
            ["source_district"],
            kind="stable",
        ).drop_duplicates(subset=["id"], keep="first")
    duplicates = before - len(combined)

    combined = combined.to_crs(cfg.CRS_WORKING)
    combined = gpd.clip(combined, box(*grid["bounds"]))
    combined = combined[combined.geometry.notna() & ~combined.geometry.is_empty]

    columns = [
        column
        for column in (
            "element",
            "id",
            "waterway",
            "name",
            "name:en",
            "water",
            "natural",
            "source_district",
            "geometry",
        )
        if column in combined.columns
    ]
    combined = combined[columns].copy()
    utils.write_vector(
        combined,
        cfg.output_path("waterways_regional"),
        layer="waterways",
    )
    LOGGER.info(
        "Waterways: %s features -> %s after dedupe/clip (%s duplicates), %s lines",
        f"{before:,}",
        f"{len(combined):,}",
        f"{duplicates:,}",
        f"{int(combined.geom_type.isin(['LineString', 'MultiLineString']).sum()):,}",
    )
    return combined, duplicates


def main() -> None:
    cfg.check_input_paths()
    grid = _regional_grid()

    mosaic, mosaic_transform, mosaic_crs = _mosaic_dem()
    dem, _ = utils.reproject_array(
        mosaic,
        mosaic_transform,
        mosaic_crs,
        grid,
        out_path=cfg.output_path("dem_regional_raw_90m"),
        method="average",
        src_nodata=np.nan,
        dst_nodata=cfg.NODATA,
        dtype="float32",
    )
    dem, filled_cells, max_fill_distance = _fill_regional_edges(dem)
    utils.write_raster(
        dem,
        cfg.output_path("dem_regional_raw_90m"),
        grid,
        nodata=cfg.NODATA,
        dtype="float32",
    )

    waterways, duplicates = _load_waterways(grid)
    lines = waterways[
        waterways.geom_type.isin(["LineString", "MultiLineString"])
        & waterways["waterway"].isin(cfg.WATERWAY_FLOW_TYPES)
    ].copy()
    waterway_mask, _ = utils.rasterize_streams(
        lines,
        template=grid,
        out_path=cfg.output_path("waterways_regional_grid"),
        all_touched=True,
    )
    waterway_cells = int((waterway_mask == 1).sum())
    if waterway_cells == 0:
        raise RuntimeError("Rasterised regional waterways are empty")

    burned = dem.copy()
    burned[waterway_mask == 1] -= cfg.WATERWAY_BURN_DEPTH_M
    utils.write_raster(
        burned,
        cfg.output_path("dem_regional_90m"),
        grid,
        nodata=cfg.NODATA,
        dtype="float32",
    )

    valid = dem != cfg.NODATA
    stats = {
        "regional_districts": list(cfg.REGIONAL_DISTRICT_NAMES),
        "dem_tiles": [Path(path).name for path in cfg.DEM_REGIONAL_TILES],
        "regional_grid": {
            "crs": str(grid["crs"]),
            "width": int(grid["width"]),
            "height": int(grid["height"]),
            "resolution": float(grid["transform"].a),
            "bounds": [float(value) for value in grid["bounds"]],
        },
        "edge_cells_filled": filled_cells,
        "edge_fill_max_distance_cells": max_fill_distance,
        "dem_min": float(dem[valid].min()),
        "dem_max": float(dem[valid].max()),
        "dem_mean": float(dem[valid].mean()),
        "waterway_features": int(len(waterways)),
        "waterway_duplicates_removed": int(duplicates),
        "waterway_lines": int(len(lines)),
        "waterway_cells": waterway_cells,
        "burn_depth_m": float(cfg.WATERWAY_BURN_DEPTH_M),
        "burned_min": float(burned[valid].min()),
        "burned_max": float(burned[valid].max()),
    }
    stats_path = cfg.output_path("regional_dem_stats")
    utils.make_folder(stats_path.parent)
    stats_path.write_text(
        json.dumps(stats, indent=2, default=str),
        encoding="utf-8",
    )
    LOGGER.info(
        "Regional DEM: %s x %s, elevation %.1f to %.1f m, %s waterway cells",
        f"{grid['width']:,}",
        f"{grid['height']:,}",
        stats["dem_min"],
        stats["dem_max"],
        f"{waterway_cells:,}",
    )
    LOGGER.info("Regional DEM preparation complete")


if __name__ == "__main__":
    main()
