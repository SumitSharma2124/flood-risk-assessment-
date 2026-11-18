from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from pysheds.grid import Grid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import config as cfg

# Reuse the exact HAND, TWI and cropping logic from the production script.
spec = importlib.util.spec_from_file_location(
    "terrain_features",
    ROOT / "src" / "03_terrain_features.py",
)
terrain = importlib.util.module_from_spec(spec)
spec.loader.exec_module(terrain)


def write_raster(array, path, profile, dtype="float32"):
    profile = profile.copy()
    profile.update(
        driver="GTiff",
        count=1,
        dtype=dtype,
        nodata=cfg.NODATA,
        compress="deflate",
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(dtype), 1)


def summary(array, mask):
    valid = mask & np.isfinite(array) & (array != cfg.NODATA)
    values = array[valid].astype("float64")

    if not len(values):
        return {"valid_cells": 0}

    return {
        "valid_cells": int(len(values)),
        "min": float(values.min()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(values.max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--burn-depth", type=float, required=True)
    args = parser.parse_args()

    depth = args.burn_depth
    if depth < 0:
        raise ValueError("Burn depth cannot be negative")

    out_dir = (
        cfg.PROJECT_ROOT
        / "outputs"
        / "sensitivity_task4"
        / f"burn_{depth:g}m"
    )

    if out_dir.exists():
        raise FileExistsError(
            f"{out_dir} already exists. "
            "Do not overwrite an existing experiment."
        )

    out_dir.mkdir(parents=True)

    raw_path = cfg.output_path("dem_regional_raw_90m")
    water_path = cfg.output_path("waterways_regional_grid")

    with rasterio.open(raw_path) as src:
        raw = src.read(1).astype("float32")
        profile = src.profile.copy()

    with rasterio.open(water_path) as src:
        water = src.read(1) == 1

    if raw.shape != water.shape:
        raise RuntimeError("Regional DEM and waterway raster shapes differ")

    valid = np.isfinite(raw) & (raw != cfg.NODATA)
    burned = raw.copy()
    burned[water & valid] -= depth

    # Store experiment-specific DEM; never touch the production DEM.
    burned_path = out_dir / "dem_burned.tif"
    write_raster(burned, burned_path, profile)

    print(f"Burn depth: {depth} m")
    print("Conditioning DEM and calculating D8 flow...")

    grid = Grid.from_raster(str(burned_path))
    dem = grid.read_raster(str(burned_path))
    raw_dem = grid.read_raster(str(raw_path))

    dem = grid.fill_pits(dem)
    dem = grid.fill_depressions(dem)
    dem = grid.resolve_flats(dem, eps=cfg.RESOLVE_FLATS_EPS)

    fdir = grid.flowdir(dem, routing="d8")
    accumulation = grid.accumulation(fdir, routing="d8")

    accumulation = np.asarray(accumulation, dtype="float32")
    fdir_array = np.asarray(fdir)

    print("Calculating HAND...")

    drainage = water | (
        accumulation >= cfg.HAND_ACCUMULATION_THRESHOLD
    )

    hand, hand_stats = terrain._compute_hand(
        grid,
        fdir,
        raw_dem,
        drainage,
    )

    # Crop regional outputs to the Alappuzha processing grid.
    row, col, alap_grid = terrain._crop_offsets()

    accumulation_alap = terrain._crop(
        accumulation, row, col, alap_grid
    ).astype("float32")

    hand_alap = terrain._crop(
        hand, row, col, alap_grid
    ).astype("float32")

    # Keep slope unchanged: only the burn depth is being tested here.
    with rasterio.open(cfg.output_path("slope_grid")) as src:
        slope = src.read(1).astype("float32")

    twi = terrain._twi(accumulation_alap, slope)

    with rasterio.open(cfg.output_path("district_mask")) as src:
        district = src.read(1) == 1

    # Save district-grid results in the experiment folder.
    outputs = {
        "flow_accumulation.tif": accumulation_alap,
        "hand.tif": hand_alap,
        "twi.tif": twi,
    }

    alap_profile = alap_grid["profile"]

    for filename, array in outputs.items():
        write_raster(
            array,
            out_dir / filename,
            alap_profile,
        )

    stats = {
        "burn_depth_m": depth,
        "district_cells": int(district.sum()),
        "waterway_cells_regional": int(water.sum()),
        "hand_fallback_regional_cells": hand_stats["nan_cells_filled"],
        "hand_negative_cells_clipped": hand_stats[
            "negative_cells_clipped"
        ],
        "hand_drainage_cells_regional": hand_stats["drainage_cells"],
        "flow_accumulation": summary(accumulation_alap, district),
        "hand": summary(hand_alap, district),
        "twi": summary(twi, district),
    }

    (out_dir / "stats.json").write_text(
        json.dumps(stats, indent=2),
        encoding="utf-8",
    )

    print("\nExperiment completed.")
    print(f"Output: {out_dir}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()