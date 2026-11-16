"""Task 3: build the common 90 m grid and exact district mask.

Run from the project root::

    python src/02_build_grid.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
from shapely.geometry import box

from src import config as cfg
from src import utils

LOGGER = utils.get_logger("build_grid")


def main() -> None:
    cfg.check_input_paths()

    district_path = cfg.output_path("district")
    if not district_path.exists():
        raise FileNotFoundError(
            f"Missing Task 2 district file: {district_path}. "
            "Run src/01_clip_layers.py first."
        )

    district = utils.read_vector(district_path, layer="district")
    buffered = utils.read_vector(district_path, layer="district_buffered")
    if district.empty or buffered.empty:
        raise RuntimeError("District layers are empty")

    district_working = district.to_crs(cfg.CRS_WORKING)
    buffered_working = buffered.to_crs(cfg.CRS_WORKING)
    buffered_geom = buffered_working.geometry.union_all()

    grid = utils.build_grid(
        bounds=buffered_working.total_bounds,
        resolution=cfg.GRID_RESOLUTION,
        crs=cfg.CRS_WORKING,
        buffer_m=0,
    )

    grid_path = cfg.output_path("grid")
    utils.save_grid(grid, grid_path)

    mask, _ = utils.rasterize_vector(
        district_working,
        template=grid,
        out_path=cfg.output_path("district_mask"),
        burn_value=1,
        fill=0,
        dtype="float32",
        nodata=cfg.NODATA,
    )

    if not set(np.unique(mask)).issubset({0.0, 1.0}):
        raise RuntimeError("district_mask contains values other than 0 and 1")
    if mask.sum() == 0:
        raise RuntimeError("district_mask has no cells inside the district")

    grid_box = box(*grid["bounds"])
    if not grid_box.covers(buffered_geom):
        raise RuntimeError("Grid bounds do not cover the buffered district")

    utils.assert_same_grid(grid_path, cfg.output_path("district_mask"))

    mask_area_km2 = float(mask.sum()) * cfg.POPULATION_TARGET_CELL_AREA_M2 / 1e6
    district_area_km2 = float(district_working.geometry.union_all().area) / 1e6

    LOGGER.info(
        "Grid: %s x %s cells, %s m, bounds=%s",
        grid["height"],
        grid["width"],
        grid["resolution"],
        tuple(round(value, 2) for value in grid["bounds"]),
    )
    LOGGER.info(
        "District mask: %.1f km2 from %.1f km2 vector area",
        mask_area_km2,
        district_area_km2,
    )
    LOGGER.info("Grid written to %s", grid_path)
    LOGGER.info("District mask written to %s", cfg.output_path("district_mask"))


if __name__ == "__main__":
    main()
