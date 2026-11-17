"""Task 5: population exposure on the common 90 m grid.

Run after Task 3::

    python src/04_exposure_population.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from src import config as cfg
from src import utils

LOGGER = utils.get_logger("exposure_population")


def _require_inputs() -> None:
    required = {
        "pop2018_90m": cfg.output_path("pop2018_90m"),
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


def _clean_population(
    population: np.ndarray,
    district: np.ndarray,
    source_nodata: float | int | None,
) -> tuple[np.ndarray, dict]:
    finite = np.isfinite(population) & (population != cfg.NODATA)
    if source_nodata is not None:
        finite &= population != source_nodata

    clean = np.full(population.shape, cfg.NODATA, dtype="float32")
    clean[district] = np.where(
        finite[district],
        np.maximum(population[district], 0.0),
        0.0,
    )

    values = clean[district]
    return clean, {
        "district_cells": int(district.sum()),
        "district_population_total": float(values.sum()),
        "nonzero_cells": int((values > 0).sum()),
        "zero_cells": int((values == 0).sum()),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "source_nodata": source_nodata,
    }


def main() -> None:
    cfg.check_input_paths()
    _require_inputs()

    population, _, source_nodata = utils.read_raster(cfg.output_path("pop2018_90m"))
    population = np.asarray(population, dtype="float32")
    district_raw, _, _ = utils.read_raster(cfg.output_path("district_mask"))
    district = np.asarray(district_raw) == 1

    finite = np.isfinite(population) & (population != cfg.NODATA)
    if source_nodata is not None:
        finite &= population != source_nodata
    task3_total = float(population[district & finite].sum())

    clean, clean_stats = _clean_population(population, district, source_nodata)
    utils.write_raster(
        clean,
        cfg.output_path("exposure_population_count"),
        cfg.GRID_PATH,
        nodata=cfg.NODATA,
        dtype="float32",
    )

    scaled, scale_info = utils.robust_scale(
        clean,
        valid_mask=district,
        upper_percentile=cfg.EXPOSURE_CAP_PERCENTILE,
        log1p=cfg.EXPOSURE_SCALE_LOG,
        nodata=cfg.NODATA,
        dtype="float32",
    )
    utils.write_raster(
        scaled,
        cfg.output_path("exposure_pop"),
        cfg.GRID_PATH,
        nodata=cfg.NODATA,
        dtype="float32",
    )

    clean_total = clean_stats["district_population_total"]
    relative_to_task3 = (
        (clean_total - task3_total) / task3_total if task3_total else 0.0
    )
    census_total = float(cfg.ALAPPUZHA_POPULATION_2011)
    relative_to_census = (clean_total - census_total) / census_total

    stats = {
        "grid": {"height": int(clean.shape[0]), "width": int(clean.shape[1])},
        "source": str(cfg.output_path("pop2018_90m")),
        "task3_population_total": task3_total,
        "census_2011_population": census_total,
        "relative_to_task3": relative_to_task3,
        "relative_to_census_2011": relative_to_census,
        "cleaned": clean_stats,
        "scale": scale_info,
        "settings": {
            "cap_percentile": cfg.EXPOSURE_CAP_PERCENTILE,
            "log1p": cfg.EXPOSURE_SCALE_LOG,
        },
        "checks": {
            "matches_task3": bool(abs(relative_to_task3) <= 1e-4),
            "within_25_percent_of_census_2011": bool(abs(relative_to_census) <= 0.25),
        },
    }
    stats_path = cfg.output_path("exposure_population_stats")
    utils.make_folder(stats_path.parent)
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    utils.assert_same_grid(cfg.output_path("exposure_pop"), cfg.GRID_PATH)
    utils.assert_same_grid(cfg.output_path("exposure_population_count"), cfg.GRID_PATH)

    LOGGER.info(
        "Population: Task 3 %.0f -> cleaned %.0f (%.2f%%); scaled 0-1",
        task3_total,
        clean_total,
        relative_to_task3 * 100.0,
    )
    if abs(relative_to_census) > 0.25:
        LOGGER.warning(
            "Population is %.1f%% above the 2011 census total; see Task 1 note",
            relative_to_census * 100.0,
        )
    LOGGER.info("Population exposure complete")


if __name__ == "__main__":
    main()
