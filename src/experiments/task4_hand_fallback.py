
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt
from pysheds.grid import Grid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import config as cfg

spec = importlib.util.spec_from_file_location(
    "terrain_features",
    ROOT / "src" / "03_terrain_features.py",
)
terrain = importlib.util.module_from_spec(spec)
spec.loader.exec_module(terrain)


def main():
    out_dir = ROOT / "outputs" / "sensitivity_task4" / "hand_fallback"

    if out_dir.exists():
        raise FileExistsError(
            f"{out_dir} already exists. Do not overwrite experiment results."
        )

    out_dir.mkdir(parents=True)

    raw_path = cfg.output_path("dem_regional_raw_90m")
    burned_path = cfg.output_path("dem_regional_90m")
    water_path = cfg.output_path("waterways_regional_grid")
    mask_path = cfg.output_path("district_mask")

    # Load the exact production inputs.
    with rasterio.open(raw_path) as src:
        raw_array = src.read(1).astype("float32")
        raw_profile = src.profile.copy()

    with rasterio.open(burned_path) as src:
        burned_array = src.read(1).astype("float32")

    with rasterio.open(water_path) as src:
        water = src.read(1) > 0

    with rasterio.open(mask_path) as src:
        district = src.read(1) == 1
        district_profile = src.profile.copy()

    print("Loading production flow directions...")

    # Use the existing production flow-direction raster.
    grid = Grid.from_raster(str(burned_path))
    dem = grid.read_raster(str(burned_path))
    raw_dem = grid.read_raster(str(raw_path))

    fdir = grid.read_raster(
        str(cfg.output_path("flow_direction"))
    )

    accumulation = np.asarray(
        grid.read_raster(str(cfg.output_path("flow_accumulation"))),
        dtype="float32",
    )

    drainage = water | (
        accumulation >= cfg.HAND_ACCUMULATION_THRESHOLD
    )

    if raw_array.shape != drainage.shape:
        raise RuntimeError("Regional input grids do not match")

    print("Calculating HAND before fallback...")

    viewfinder = grid.viewfinder.copy()
    viewfinder.nodata = False

    from pysheds.sview import Raster

    drainage_raster = Raster(
        drainage.astype(bool),
        viewfinder=viewfinder,
        metadata={"nodata": False},
    )

    hand_raw = np.asarray(
        grid.compute_hand(
            fdir,
            raw_dem,
            drainage_raster,
            routing="d8",
        ),
        dtype="float32",
    )

    fallback_mask = ~np.isfinite(hand_raw)

    print("Applying the production fallback logic...")

    hand_final, stats = terrain._compute_hand(
        grid,
        fdir,
        raw_dem,
        drainage,
    )

    # Obtain the same Alappuzha crop used by production.
    row, col, alap_grid = terrain._crop_offsets()

    fallback_alap = terrain._crop(
        fallback_mask.astype("uint8"),
        row,
        col,
        alap_grid,
    ).astype(bool)

    hand_raw_alap = terrain._crop(
        hand_raw,
        row,
        col,
        alap_grid,
    ).astype("float32")

    hand_final_alap = terrain._crop(
        hand_final,
        row,
        col,
        alap_grid,
    ).astype("float32")

    if fallback_alap.shape != district.shape:
        raise RuntimeError("Fallback mask and district mask shapes differ")

    valid_district = district
    fallback_district = fallback_alap & valid_district
    normal_district = (~fallback_alap) & valid_district

    total = int(valid_district.sum())
    fallback_count = int(fallback_district.sum())
    normal_count = int(normal_district.sum())

    def describe(values):
        values = values[np.isfinite(values)]

        if len(values) == 0:
            return {"valid_cells": 0}

        return {
            "valid_cells": int(len(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    # Raw HAND values are non-finite for fallback cells.
    # Compare final HAND distributions for fallback and normal cells.
    fallback_values = hand_final_alap[fallback_district]
    normal_values = hand_final_alap[normal_district]

    stats_out = {
        "district_cells": total,
        "fallback_cells": fallback_count,
        "fallback_percentage": fallback_count / total * 100,
        "normal_hand_cells": normal_count,
        "normal_percentage": normal_count / total * 100,
        "regional_fallback_cells": stats["nan_cells_filled"],
        "regional_drainage_cells": stats["drainage_cells"],
        "regional_negative_cells_clipped": stats[
            "negative_cells_clipped"
        ],
        "fallback_hand_final": describe(fallback_values),
        "normal_hand_final": describe(normal_values),
    }

    # Save the district fallback mask for spatial inspection.
    profile = district_profile.copy()
    profile.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        nodata=255,
        compress="deflate",
    )

    with rasterio.open(out_dir / "fallback_mask.tif", "w", **profile) as dst:
        dst.write(
            np.where(valid_district, fallback_alap, 255).astype("uint8"),
            1,
        )

    (out_dir / "stats.json").write_text(
        json.dumps(stats_out, indent=2),
        encoding="utf-8",
    )

    print("\nHAND fallback analysis completed.")
    print(json.dumps(stats_out, indent=2))
    print(f"\nOutputs: {out_dir}")


if __name__ == "__main__":
    main()
