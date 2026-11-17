"""Task 4: distance to waterways and permanent water.

Run after ``src/03_prepare_regional_dem.py`` and Task 3::

    python src/03_waterway_distance.py
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

from src import config as cfg
from src import utils

LOGGER = utils.get_logger("waterway_distance")


def _require_inputs() -> None:
    required = {
        "waterways_regional": cfg.output_path("waterways_regional"),
        "landcover_frac": cfg.output_path("landcover_frac"),
        "district_mask": cfg.output_path("district_mask"),
        "grid": cfg.GRID_PATH,
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not Path(path).exists()]
    if missing:
        formatted = "\n".join(f"  - {item}" for item in missing)
        raise FileNotFoundError(f"Missing Task 4 input(s):\n{formatted}")


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


def _waterway_lines():
    waterways = utils.read_vector(cfg.output_path("waterways_regional"))
    waterways = waterways[
        waterways.geometry.notna() & ~waterways.geometry.is_empty
    ].copy()
    lines = waterways[
        waterways.geom_type.isin(["LineString", "MultiLineString"])
        & waterways["waterway"].isin(cfg.WATERWAY_FLOW_TYPES)
    ].copy()
    if lines.empty:
        raise RuntimeError("No line waterways found in the regional layer")
    return lines


def _permanent_water_mask() -> tuple[np.ndarray, int]:
    with rasterio.open(cfg.output_path("landcover_frac")) as src:
        descriptions = tuple(src.descriptions or ())
        band = descriptions.index("water") + 1 if "water" in descriptions else 4
        water_fraction = src.read(band)
    mask = (water_fraction >= 0.5) & (water_fraction != cfg.NODATA)
    return mask, band


def main() -> None:
    cfg.check_input_paths()
    _require_inputs()

    lines = _waterway_lines()
    LOGGER.info("Rasterising %s regional waterway lines", f"{len(lines):,}")
    water_mask, _ = utils.rasterize_streams(
        lines,
        template=cfg.GRID_PATH,
        out_path=cfg.output_path("waterways_grid"),
        all_touched=True,
    )
    water_mask = np.asarray(water_mask)
    water_cells = int((water_mask == 1).sum())
    if water_cells == 0:
        raise RuntimeError("Rasterised waterways are empty on the Alappuzha grid")

    LOGGER.info(
        "Waterways cover %s cells; computing distance",
        f"{water_cells:,}",
    )
    distance_water, _ = utils.distance_to_mask(
        water_mask,
        template=cfg.GRID_PATH,
        out_path=cfg.output_path("distance_waterways_grid"),
        mask_values=[1],
    )

    permanent_mask, permanent_band = _permanent_water_mask()
    permanent_cells = int(permanent_mask.sum())
    permanent_stats = None
    permanent_district_stats = None
    if permanent_cells:
        LOGGER.info(
            "Permanent water (land-cover band %s) covers %s cells",
            permanent_band,
            f"{permanent_cells:,}",
        )
        distance_permanent, _ = utils.distance_to_mask(
            permanent_mask,
            template=cfg.GRID_PATH,
            out_path=cfg.output_path("distance_permanent_water_grid"),
        )
        permanent_stats = _summary(distance_permanent)
    else:
        LOGGER.warning("No permanent-water cells found; skipping that distance layer")

    district_mask, _, _ = utils.read_raster(cfg.output_path("district_mask"))
    district = np.asarray(district_mask) == 1
    if permanent_stats is not None:
        permanent_district_stats = _summary(distance_permanent[district])

    stats = {
        "waterway_lines": int(len(lines)),
        "waterway_cells": water_cells,
        "distance_waterway": _summary(distance_water),
        "distance_waterway_district": _summary(distance_water[district]),
        "permanent_water_band": int(permanent_band),
        "permanent_water_threshold": 0.5,
        "permanent_water_cells": permanent_cells,
        "distance_permanent_water": permanent_stats,
        "distance_permanent_water_district": permanent_district_stats,
    }
    stats_path = cfg.output_path("waterway_distance_stats")
    utils.make_folder(stats_path.parent)
    stats_path.write_text(
        json.dumps(stats, indent=2, default=str),
        encoding="utf-8",
    )

    LOGGER.info(
        "Distance to waterways: %.0f to %.0f m (mean %.0f m)",
        stats["distance_waterway"]["min"],
        stats["distance_waterway"]["max"],
        stats["distance_waterway"]["mean"],
    )
    LOGGER.info("Waterway distance generation complete")


if __name__ == "__main__":
    main()
