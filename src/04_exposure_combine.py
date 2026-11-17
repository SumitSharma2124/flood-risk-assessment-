"""Task 5: combine exposure layers, summarize exposure and draw an overview.

Run after the three Task 5 exposure scripts::

    python src/04_exposure_combine.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter

from src import config as cfg
from src import utils

LOGGER = utils.get_logger("exposure_combine")


def _require_inputs() -> None:
    keys = [
        "district_mask",
        "ksdma_modelled_mask",
        "flood2018_bin",
        "exposure_population_count",
        "exposure_pop",
        "exposure_buildings_count",
        "exposure_buildings_area",
        "exposure_facilities_score",
        "grid",
    ]
    missing = [
        f"{name}: {cfg.output_path(name)}"
        for name in keys
        if not Path(cfg.output_path(name)).exists()
    ]
    if missing:
        formatted = "\n".join(f"  - {item}" for item in missing)
        raise FileNotFoundError(f"Missing Task 5 input(s):\n{formatted}")


def _mask_domain(array: np.ndarray, district: np.ndarray) -> np.ndarray:
    result = np.full(array.shape, cfg.NODATA, dtype="float32")
    result[district] = np.asarray(array, dtype="float32")[district]
    return result


def _pearson_1km(
    first: np.ndarray,
    second: np.ndarray,
    district: np.ndarray,
) -> float | None:
    block = max(1, int(round(1000.0 / float(cfg.GRID_RESOLUTION))))
    first_smooth = uniform_filter(
        np.where(district, first, 0.0),
        size=block,
        mode="constant",
        cval=0.0,
    )
    second_smooth = uniform_filter(
        np.where(district, second, 0.0),
        size=block,
        mode="constant",
        cval=0.0,
    )
    x = first_smooth[district]
    y = second_smooth[district]
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return None
    x = x[valid]
    y = y[valid]
    if x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _summary_row(
    name: str,
    mask: np.ndarray,
    population_count: np.ndarray,
    building_count: np.ndarray,
    building_area: np.ndarray,
    facility_score: np.ndarray,
    exposure_score: np.ndarray,
    district_population: float,
    district_buildings: int,
) -> dict:
    cells = int(mask.sum())
    cell_area_km2 = float(cfg.GRID_RESOLUTION**2) / 1e6
    values = exposure_score[mask]
    population = float(population_count[mask].sum())
    buildings = int(building_count[mask].sum())

    return {
        "mask": name,
        "cells": cells,
        "area_km2": cells * cell_area_km2,
        "population": population,
        "population_share_of_district": (
            population / district_population if district_population else 0.0
        ),
        "buildings": buildings,
        "buildings_share_of_district": (
            buildings / district_buildings if district_buildings else 0.0
        ),
        "building_area_m2": float(building_area[mask].sum()),
        "facility_score_sum": float(facility_score[mask].sum()),
        "mean_exposure": float(values.mean()) if values.size else 0.0,
        "p95_exposure": float(np.percentile(values, 95)) if values.size else 0.0,
        "max_exposure": float(values.max()) if values.size else 0.0,
    }


def _plot_overview(
    population_count: np.ndarray,
    building_count: np.ndarray,
    facility_score: np.ndarray,
    exposure_score: np.ndarray,
    ksdma: np.ndarray,
    flood: np.ndarray,
    district: np.ndarray,
) -> None:
    district_gdf = utils.load_district(crs=cfg.CRS_WORKING)
    panels = [
        ("Population (count)", np.where(district, population_count, np.nan), "viridis"),
        ("Buildings (count)", np.where(district, building_count, np.nan), "magma"),
        (
            "Facilities (weighted, raw)",
            np.where(district, facility_score, np.nan),
            "Reds",
        ),
        (
            "Combined exposure (0-1)",
            np.where(district, exposure_score, np.nan),
            "inferno",
        ),
        ("KSDMA modelled mask", np.where(district, ksdma, np.nan), "Blues"),
        ("2018 flood mask", np.where(district, flood, np.nan), "Blues"),
    ]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 9),
        constrained_layout=True,
    )
    for axis, (title, data, cmap) in zip(axes.ravel(), panels):
        finite = data[np.isfinite(data)]
        if title.endswith("mask"):
            vmin, vmax = 0.0, 1.0
        elif finite.size:
            vmin, vmax = np.nanpercentile(finite, [2, 98])
        else:
            vmin, vmax = 0.0, 1.0
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmax = vmin + 1.0

        image = axis.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        district_gdf.boundary.plot(
            ax=axis,
            color="black",
            linewidth=0.35,
        )
        axis.set_title(title, fontsize=10)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_aspect("equal")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.02)

    fig.suptitle("Alappuzha 90 m exposure layers", fontsize=14)
    out_path = cfg.output_path("exposure_overview")
    utils.make_folder(out_path.parent)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cfg.check_input_paths()
    _require_inputs()

    district_raw, _, _ = utils.read_raster(cfg.output_path("district_mask"))
    district = np.asarray(district_raw) == 1

    ksdma_raw, _, _ = utils.read_raster(cfg.output_path("ksdma_modelled_mask"))
    ksdma = np.asarray(ksdma_raw) == 1

    flood_raw, _, _ = utils.read_raster(cfg.output_path("flood2018_bin"))
    flood = np.asarray(flood_raw) == 1

    population_count, _, _ = utils.read_raster(
        cfg.output_path("exposure_population_count")
    )
    population_scaled, _, _ = utils.read_raster(cfg.output_path("exposure_pop"))
    building_count, _, _ = utils.read_raster(
        cfg.output_path("exposure_buildings_count")
    )
    building_area, _, _ = utils.read_raster(
        cfg.output_path("exposure_buildings_area")
    )
    facility_raw, _, _ = utils.read_raster(
        cfg.output_path("exposure_facilities_score")
    )

    population_count = np.asarray(population_count, dtype="float32")
    population_scaled = np.asarray(population_scaled, dtype="float32")
    building_count = np.asarray(building_count, dtype="float32")
    building_area = np.asarray(building_area, dtype="float32")
    facility_raw = np.asarray(facility_raw, dtype="float32")

    for array in (
        population_count,
        population_scaled,
        building_count,
        building_area,
        facility_raw,
        ksdma,
        flood,
    ):
        if array.shape != district.shape:
            raise ValueError("Input rasters do not share the district grid")

    building_metric = str(cfg.EXPOSURE_BUILDING_METRIC).lower()
    if building_metric == "count":
        building_raw = building_count
    elif building_metric == "area":
        building_raw = building_area
    else:
        raise ValueError(
            f"Unknown EXPOSURE_BUILDING_METRIC: {cfg.EXPOSURE_BUILDING_METRIC}"
        )

    building_scaled, building_scale = utils.robust_scale(
        building_raw,
        valid_mask=district,
        upper_percentile=cfg.EXPOSURE_CAP_PERCENTILE,
        log1p=cfg.EXPOSURE_SCALE_LOG,
        nodata=cfg.NODATA,
        dtype="float32",
    )
    facility_scaled, facility_scale = utils.robust_scale(
        facility_raw,
        valid_mask=district,
        upper_percentile=cfg.EXPOSURE_CAP_PERCENTILE,
        log1p=False,
        nodata=cfg.NODATA,
        dtype="float32",
    )

    weights = dict(cfg.EXPOSURE_WEIGHTS)
    weight_sum = float(sum(weights.values()))
    if abs(weight_sum - 1.0) > 1e-6:
        LOGGER.warning("Exposure weights sum to %.4f, normalizing", weight_sum)
        weights = {key: value / weight_sum for key, value in weights.items()}

    exposure = (
        weights["population"] * np.where(district, population_scaled, 0.0)
        + weights["buildings"] * np.where(district, building_scaled, 0.0)
        + weights["facilities"] * np.where(district, facility_scaled, 0.0)
    )
    exposure = np.clip(exposure, 0.0, 1.0).astype("float32")
    exposure = _mask_domain(exposure, district)

    utils.write_raster(
        exposure,
        cfg.output_path("exposure_score"),
        cfg.GRID_PATH,
        nodata=cfg.NODATA,
        dtype="float32",
    )

    district_population = float(population_count[district].sum())
    district_buildings = int(building_count[district].sum())
    district_area = float(building_area[district].sum())
    summary_rows = [
        _summary_row(
            "district",
            district,
            population_count,
            building_count,
            building_area,
            facility_raw,
            exposure,
            district_population,
            district_buildings,
        ),
        _summary_row(
            "ksdma_modelled",
            district & ksdma,
            population_count,
            building_count,
            building_area,
            facility_raw,
            exposure,
            district_population,
            district_buildings,
        ),
        _summary_row(
            "flood_2018",
            district & flood,
            population_count,
            building_count,
            building_area,
            facility_raw,
            exposure,
            district_population,
            district_buildings,
        ),
    ]
    summary = pd.DataFrame(summary_rows)
    summary_path = cfg.output_path("exposure_summary")
    utils.make_folder(summary_path.parent)
    summary.to_csv(summary_path, index=False)

    exposure_values = exposure[district]
    population_building_correlation = _pearson_1km(
        population_count,
        building_count,
        district,
    )
    stats = {
        "grid": {
            "height": int(district.shape[0]),
            "width": int(district.shape[1]),
            "resolution_m": float(cfg.GRID_RESOLUTION),
        },
        "weights": weights,
        "building_metric": building_metric,
        "building_scale": building_scale,
        "facility_scale": facility_scale,
        "district": {
            "cells": int(district.sum()),
            "area_km2": float(district.sum() * cfg.GRID_RESOLUTION**2 / 1e6),
            "population": district_population,
            "buildings": district_buildings,
            "building_area_m2": district_area,
        },
        "exposure_score": {
            "min": float(exposure_values.min()),
            "max": float(exposure_values.max()),
            "mean": float(exposure_values.mean()),
            "p50": float(np.percentile(exposure_values, 50)),
            "p95": float(np.percentile(exposure_values, 95)),
            "nonzero_cells": int((exposure_values > 0).sum()),
            "zero_cells": int((exposure_values == 0).sum()),
        },
        "checks": {
            "same_grid": True,
            "district_cells_with_zero_population": int(
                (population_count[district] == 0).sum()
            ),
            "district_cells_without_buildings": int(
                (building_count[district] == 0).sum()
            ),
            "population_building_pearson_1km": population_building_correlation,
            "score_within_0_1": bool(
                np.isfinite(exposure_values).all()
                and (exposure_values >= 0).all()
                and (exposure_values <= 1).all()
            ),
        },
        "summary": summary_rows,
        "caveat": (
            "Population is 2018 WorldPop; buildings and facilities are current "
            "OSM/Google features. Exposure is a relative index, not a probability."
        ),
    }
    stats_path = cfg.output_path("exposure_combine_stats")
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    _plot_overview(
        population_count,
        building_count,
        facility_raw,
        exposure,
        ksdma.astype("float32"),
        flood.astype("float32"),
        district,
    )

    utils.assert_same_grid(cfg.output_path("exposure_score"), cfg.GRID_PATH)

    LOGGER.info(
        "Exposure score: min %.3f, mean %.3f, max %.3f; 1 km pop/building r=%s",
        stats["exposure_score"]["min"],
        stats["exposure_score"]["mean"],
        stats["exposure_score"]["max"],
        (
            "n/a"
            if population_building_correlation is None
            else f"{population_building_correlation:.2f}"
        ),
    )
    LOGGER.info("Exposure combination complete")


if __name__ == "__main__":
    main()
