"""Task 4: derive terrain, flow and drainage features.

Run after ``src/03_prepare_regional_dem.py``::

    python src/03_terrain_features.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from scipy.ndimage import distance_transform_edt

if not hasattr(np, "in1d"):
    np.in1d = np.isin

from pysheds.grid import Grid
from pysheds.sview import Raster

from src import config as cfg
from src import utils

LOGGER = utils.get_logger("terrain_features")


def _require_inputs() -> None:
    required = {
        "regional_grid": cfg.output_path("regional_grid"),
        "dem_regional_90m": cfg.output_path("dem_regional_90m"),
        "dem_regional_raw_90m": cfg.output_path("dem_regional_raw_90m"),
        "waterways_regional_grid": cfg.output_path("waterways_regional_grid"),
        "dem_30m": cfg.output_path("dem_30m"),
        "grid": cfg.GRID_PATH,
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not Path(path).exists()]
    if missing:
        formatted = "\n".join(f"  - {item}" for item in missing)
        raise FileNotFoundError(f"Missing Task 4 input(s):\n{formatted}")


def _crop_offsets() -> tuple[int, int, dict]:
    target = utils.load_grid(cfg.GRID_PATH)
    with rasterio.open(cfg.output_path("regional_grid")) as src:
        window = from_bounds(
            *target["bounds"],
            transform=src.transform,
        ).round_offsets().round_lengths()
    if int(window.width) != target["width"] or int(window.height) != target["height"]:
        raise RuntimeError(
            "Regional grid is not aligned with the Alappuzha grid: "
            f"{int(window.width)} x {int(window.height)} vs "
            f"{target['width']} x {target['height']}"
        )
    return int(window.row_off), int(window.col_off), target


def _crop(array: np.ndarray, row_off: int, col_off: int, grid: dict) -> np.ndarray:
    return np.asarray(array)[
        row_off : row_off + grid["height"],
        col_off : col_off + grid["width"],
    ]


def _condition_dem() -> tuple[Grid, Raster, Raster]:
    path = str(cfg.output_path("dem_regional_90m"))
    raw_path = str(cfg.output_path("dem_regional_raw_90m"))

    grid = Grid.from_raster(path)
    dem = grid.read_raster(path)
    raw_dem = grid.read_raster(raw_path)

    LOGGER.info("Filling pits and depressions in the regional DEM")
    dem = grid.fill_pits(dem)
    dem = grid.fill_depressions(dem)
    dem = grid.resolve_flats(dem, eps=cfg.RESOLVE_FLATS_EPS)
    return grid, dem, raw_dem


def _compute_hand(
    grid: Grid,
    fdir: Raster,
    raw_dem: Raster,
    drainage_mask: np.ndarray,
) -> tuple[np.ndarray, dict]:
    if not drainage_mask.any():
        raise RuntimeError("HAND drainage mask is empty")

    viewfinder = grid.viewfinder.copy()
    viewfinder.nodata = False
    mask = Raster(
        drainage_mask.astype(bool),
        viewfinder=viewfinder,
        metadata={"nodata": False},
    )

    LOGGER.info("Computing HAND from the regional flow directions")
    hand = np.asarray(
        grid.compute_hand(fdir, raw_dem, mask, routing="d8"),
        dtype="float32",
    )

    nan_cells = ~np.isfinite(hand)
    nan_count = int(nan_cells.sum())
    if nan_count:
        _, nearest = distance_transform_edt(
            ~drainage_mask,
            return_distances=True,
            return_indices=True,
        )
        raw = np.asarray(raw_dem, dtype="float32")
        fallback = raw - raw[tuple(nearest)]
        hand[nan_cells] = fallback[nan_cells]

    negative_count = int((hand < 0).sum())
    hand = np.maximum(hand, 0).astype("float32")

    stats = {
        "nan_cells_filled": nan_count,
        "negative_cells_clipped": negative_count,
        "drainage_cells": int(drainage_mask.sum()),
    }
    return hand, stats


def _slope_from_30m() -> tuple[np.ndarray, dict]:
    LOGGER.info("Averaging the 30 m DEM to 90 m and deriving slope")
    dem_90m, _ = utils.align_raster(
        cfg.output_path("dem_30m"),
        template=cfg.GRID_PATH,
        method="average",
        dtype="float32",
        nodata=cfg.NODATA,
    )
    valid = dem_90m != cfg.NODATA
    if not valid.any():
        raise RuntimeError("30 m DEM has no valid cells")

    if valid.all():
        filled = dem_90m
    else:
        _, indices = distance_transform_edt(
            ~valid,
            return_distances=True,
            return_indices=True,
        )
        filled = dem_90m[tuple(indices)]

    dz_dy, dz_dx = np.gradient(
        np.asarray(filled, dtype="float64"),
        float(cfg.GRID_RESOLUTION),
        float(cfg.GRID_RESOLUTION),
    )
    slope = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy))).astype("float32")
    slope[~valid] = cfg.NODATA
    return slope, {
        "valid_cells": int(valid.sum()),
        "void_cells": int((~valid).sum()),
    }


def _twi(accumulation: np.ndarray, slope: np.ndarray) -> np.ndarray:
    cell_size = float(cfg.GRID_RESOLUTION)
    cell_area = cell_size**2
    valid = (
        (accumulation != cfg.NODATA)
        & (slope != cfg.NODATA)
        & np.isfinite(accumulation)
        & np.isfinite(slope)
    )
    effective_slope = np.radians(
        np.maximum(slope, cfg.TWI_MIN_SLOPE_DEG)
    )
    specific_area = np.maximum(accumulation, 1.0) * cell_area / cell_size
    twi = np.log(specific_area / np.tan(effective_slope)).astype("float32")
    twi[~valid] = cfg.NODATA
    return twi


def _summary(array: np.ndarray) -> dict:
    values = np.asarray(array)
    valid = np.isfinite(values) & (values != cfg.NODATA)
    if not valid.any():
        return {
            "valid_cells": 0,
            "min": None,
            "max": None,
            "mean": None,
        }
    selected = values[valid].astype("float64")
    return {
        "valid_cells": int(valid.sum()),
        "min": float(selected.min()),
        "max": float(selected.max()),
        "mean": float(selected.mean()),
    }


def main() -> None:
    cfg.check_input_paths()
    _require_inputs()
    row_off, col_off, alap_grid = _crop_offsets()
    regional_grid = utils.load_grid(cfg.output_path("regional_grid"))

    grid, dem, raw_dem = _condition_dem()
    LOGGER.info("Routing flow across the regional DEM")
    fdir = grid.flowdir(dem, routing="d8")
    accumulation = grid.accumulation(fdir, routing="d8")

    fdir_array = np.asarray(fdir)
    accumulation_array = np.asarray(accumulation)
    pits = int((fdir_array == -2).sum())
    flats = int((fdir_array == -1).sum())
    LOGGER.info(
        "Flow routing: accumulation %.0f to %.0f cells, %s pits, %s flats",
        float(accumulation_array.min()),
        float(accumulation_array.max()),
        f"{pits:,}",
        f"{flats:,}",
    )

    waterway_mask, _, _ = utils.read_raster(
        cfg.output_path("waterways_regional_grid")
    )
    waterway_mask = np.asarray(waterway_mask) > 0

    drainage_mask = waterway_mask | (
        accumulation_array >= cfg.HAND_ACCUMULATION_THRESHOLD
    )
    hand, hand_stats = _compute_hand(grid, fdir, raw_dem, drainage_mask)

    streams = (
        waterway_mask
        | (accumulation_array >= cfg.STREAM_ACCUMULATION_THRESHOLD)
    ).astype("uint8")

    utils.write_raster(
        fdir_array.astype("int16"),
        cfg.output_path("flow_direction"),
        regional_grid["profile"],
        nodata=-9999,
        dtype="int16",
    )
    utils.write_raster(
        accumulation_array.astype("float32"),
        cfg.output_path("flow_accumulation"),
        regional_grid["profile"],
        nodata=cfg.NODATA,
        dtype="float32",
    )

    accumulation_alap = _crop(
        accumulation_array,
        row_off,
        col_off,
        alap_grid,
    ).astype("float32")
    log_accumulation = np.log1p(
        np.maximum(accumulation_alap, 0.0)
    ).astype("float32")
    waterway_alap = _crop(waterway_mask, row_off, col_off, alap_grid)
    streams_alap = _crop(streams, row_off, col_off, alap_grid)
    hand_alap = _crop(hand, row_off, col_off, alap_grid)

    slope, slope_stats = _slope_from_30m()
    twi = _twi(accumulation_alap, slope)

    utils.write_raster(
        log_accumulation,
        cfg.output_path("log_flow_accumulation"),
        alap_grid["profile"],
        nodata=cfg.NODATA,
        dtype="float32",
    )
    utils.write_raster(
        streams_alap,
        cfg.output_path("streams_grid"),
        alap_grid["profile"],
        nodata=0,
        dtype="uint8",
    )
    utils.write_raster(
        hand_alap,
        cfg.output_path("hand_grid"),
        alap_grid["profile"],
        nodata=cfg.NODATA,
        dtype="float32",
    )
    utils.write_raster(
        slope,
        cfg.output_path("slope_grid"),
        alap_grid["profile"],
        nodata=cfg.NODATA,
        dtype="float32",
    )
    utils.write_raster(
        twi,
        cfg.output_path("twi_grid"),
        alap_grid["profile"],
        nodata=cfg.NODATA,
        dtype="float32",
    )

    stats = {
        "regional_grid": {
            "height": regional_grid["height"],
            "width": regional_grid["width"],
            "bounds": [float(value) for value in regional_grid["bounds"]],
        },
        "alappuzha_grid": {
            "height": alap_grid["height"],
            "width": alap_grid["width"],
            "bounds": [float(value) for value in alap_grid["bounds"]],
        },
        "flow_routing": {
            "resolve_flats_eps": float(cfg.RESOLVE_FLATS_EPS),
            "pits": pits,
            "flats": flats,
            "accumulation_cells": _summary(accumulation_array),
        },
        "waterways": {
            "regional_cells": int(waterway_mask.sum()),
            "alappuzha_cells": int((waterway_alap == 1).sum()),
        },
        "streams": {
            "threshold_cells": int(cfg.STREAM_ACCUMULATION_THRESHOLD),
            "alappuzha_cells": int((streams_alap == 1).sum()),
        },
        "hand": {
            **hand_stats,
            "threshold_cells": int(cfg.HAND_ACCUMULATION_THRESHOLD),
            "alappuzha": _summary(hand_alap),
        },
        "slope": {
            **slope_stats,
            "alappuzha": _summary(slope),
        },
        "twi": {
            "minimum_slope_deg": float(cfg.TWI_MIN_SLOPE_DEG),
            "alappuzha": _summary(twi),
        },
        "log_flow_accumulation": {
            "alappuzha": _summary(log_accumulation),
        },
    }
    stats_path = cfg.output_path("terrain_stats")
    utils.make_folder(stats_path.parent)
    stats_path.write_text(
        json.dumps(stats, indent=2, default=str),
        encoding="utf-8",
    )

    LOGGER.info(
        "Alappuzha terrain: HAND %.1f to %.1f m; slope %.1f to %.1f deg; "
        "TWI %.2f to %.2f; streams %s cells",
        stats["hand"]["alappuzha"]["min"],
        stats["hand"]["alappuzha"]["max"],
        stats["slope"]["alappuzha"]["min"],
        stats["slope"]["alappuzha"]["max"],
        stats["twi"]["alappuzha"]["min"],
        stats["twi"]["alappuzha"]["max"],
        f"{stats['streams']['alappuzha_cells']:,}",
    )
    LOGGER.info("Terrain feature generation complete")


if __name__ == "__main__":
    main()
