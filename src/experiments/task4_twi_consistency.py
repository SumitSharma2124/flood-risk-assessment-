
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import config as cfg


def load_terrain_module():
    """Load the existing production terrain functions."""
    spec = importlib.util.spec_from_file_location(
        "terrain_features",
        ROOT / "src" / "03_terrain_features.py",
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load terrain_features module")

    terrain = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(terrain)

    return terrain


def calculate_slope(dem, nodata, cell_size):
    """Calculate slope in degrees using the production gradient approach."""
    from scipy.ndimage import distance_transform_edt

    dem = dem.astype("float32").copy()

    valid = np.isfinite(dem) & (dem != nodata)

    if not valid.any():
        raise ValueError("No valid DEM cells")

    nearest = distance_transform_edt(
        ~valid,
        return_distances=False,
        return_indices=True,
    )

    filled = dem[tuple(nearest)]

    dy, dx = np.gradient(filled, cell_size, cell_size)

    slope = np.degrees(
        np.arctan(np.hypot(dx, dy))
    )

    slope[~valid] = nodata

    return slope.astype("float32")


def summary(values):
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


def main():
    out_dir = (
        ROOT
        / "outputs"
        / "sensitivity_task4"
        / "twi_consistency"
    )

    if out_dir.exists():
        raise FileExistsError(
            f"{out_dir} already exists. "
            "Do not overwrite existing experiment results."
        )

    terrain = load_terrain_module()

    # --------------------------------------------------
    # 1. Load production inputs
    # --------------------------------------------------

    dem_path = cfg.output_path("dem_regional_90m")
    accumulation_path = cfg.output_path("flow_accumulation")
    slope_path = cfg.output_path("slope_grid")
    twi_path = cfg.output_path("twi_grid")
    mask_path = cfg.output_path("district_mask")

    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype("float32")
        dem_nodata = src.nodata
        cell_size = abs(src.transform.a)

    with rasterio.open(accumulation_path) as src:
        accumulation_regional = src.read(1).astype("float32")

    with rasterio.open(slope_path) as src:
        current_slope = src.read(1).astype("float32")

    with rasterio.open(twi_path) as src:
        current_twi = src.read(1).astype("float32")

    with rasterio.open(mask_path) as src:
        district = src.read(1) == 1

    print("Loaded production inputs.")

    # --------------------------------------------------
    # 2. Calculate alternative slope on regional DEM
    # --------------------------------------------------

    print("Calculating slope from conditioned 90 m DEM...")

    alternative_slope_regional = calculate_slope(
        dem,
        dem_nodata,
        cell_size,
    )

    if alternative_slope_regional.shape != accumulation_regional.shape:
        raise RuntimeError(
            "Regional DEM and accumulation dimensions differ: "
            f"{alternative_slope_regional.shape} vs "
            f"{accumulation_regional.shape}"
        )

    # --------------------------------------------------
    # 3. Crop both regional layers to Alappuzha
    # --------------------------------------------------

    row, col, alap_grid = terrain._crop_offsets()

    alternative_slope = terrain._crop(
        alternative_slope_regional,
        row,
        col,
        alap_grid,
    ).astype("float32")

    accumulation_alap = terrain._crop(
        accumulation_regional,
        row,
        col,
        alap_grid,
    ).astype("float32")

    print("Cropped regional layers to Alappuzha.")

    # --------------------------------------------------
    # 4. Validate district grid dimensions
    # --------------------------------------------------

    expected_shape = district.shape

    if accumulation_alap.shape != expected_shape:
        raise RuntimeError(
            f"Accumulation grid mismatch: {accumulation_alap.shape} "
            f"vs {expected_shape}"
        )

    if alternative_slope.shape != expected_shape:
        raise RuntimeError(
            f"Alternative slope grid mismatch: "
            f"{alternative_slope.shape} vs {expected_shape}"
        )

    if current_slope.shape != expected_shape:
        raise RuntimeError(
            f"Current slope grid mismatch: "
            f"{current_slope.shape} vs {expected_shape}"
        )

    if current_twi.shape != expected_shape:
        raise RuntimeError(
            f"Current TWI grid mismatch: "
            f"{current_twi.shape} vs {expected_shape}"
        )

    print(f"Validated Alappuzha grid: {expected_shape}")

    # --------------------------------------------------
    # 5. Calculate alternative TWI
    # --------------------------------------------------

    print("Calculating alternative TWI...")

    alternative_twi = terrain._twi(
        accumulation_alap,
        alternative_slope,
    )

    # --------------------------------------------------
    # 6. Compare current and alternative TWI
    # --------------------------------------------------

    valid = (
        district
        & np.isfinite(current_twi)
        & np.isfinite(alternative_twi)
        & (current_twi != cfg.NODATA)
        & (alternative_twi != cfg.NODATA)
    )

    current_values = current_twi[valid].astype("float64")
    alternative_values = alternative_twi[valid].astype("float64")

    if len(current_values) == 0:
        raise RuntimeError("No valid overlapping district TWI cells")

    difference = alternative_values - current_values

    correlation = spearmanr(
        current_values,
        alternative_values,
    ).statistic

    results = {
        "methodology": {
            "current": (
                "Flow accumulation from conditioned 90 m DEM; "
                "slope from averaged original 30 m DEM"
            ),
            "alternative": (
                "Flow accumulation and slope from conditioned "
                "90 m DEM"
            ),
        },
        "valid_cells": int(len(current_values)),
        "current_twi": summary(current_values),
        "alternative_twi": summary(alternative_values),
        "mean_absolute_difference": float(
            np.mean(np.abs(difference))
        ),
        "rmse": float(
            np.sqrt(np.mean(difference ** 2))
        ),
        "spearman_correlation": float(correlation),
        "changed_cells_pct": float(
            np.mean(np.abs(difference) > 1e-6) * 100
        ),
        "cells_difference_over_1": float(
            np.mean(np.abs(difference) > 1) * 100
        ),
    }

    # --------------------------------------------------
    # 7. Save experiment outputs
    # --------------------------------------------------

    out_dir.mkdir(parents=True)

    profile = alap_grid["profile"].copy()

    profile.update(
        driver="GTiff",
        count=1,
        dtype="float32",
        nodata=cfg.NODATA,
        compress="deflate",
    )

    with rasterio.open(
        out_dir / "slope_conditioned_90m.tif",
        "w",
        **profile,
    ) as dst:
        dst.write(alternative_slope.astype("float32"), 1)

    with rasterio.open(
        out_dir / "twi_conditioned_90m.tif",
        "w",
        **profile,
    ) as dst:
        dst.write(alternative_twi.astype("float32"), 1)

    (out_dir / "stats.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("\nTWI consistency experiment completed.")
    print(json.dumps(results, indent=2))
    print(f"\nOutputs: {out_dir}")


if __name__ == "__main__":
    main()
