"""Task 3: align all raster layers to the common 90 m grid.

Run after ``src/02_build_grid.py``::

    python src/02_align_rasters.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import mapping

from src import config as cfg
from src import utils

LOGGER = utils.get_logger("align_rasters")

GRID_OUTPUT_KEYS = [
    "grid",
    "district_mask",
    "dem_grid",
    "landcover_mode",
    "landcover_frac",
    "ksdma_100yr_90m",
    "ksdma_modelled_mask",
    "flood2018_frac",
    "flood2018_bin",
    "pop2018_90m",
]


def _align_dem(grid: dict) -> tuple[np.ndarray, dict]:
    dem, _ = utils.align_raster(
        cfg.DEM_RAW,
        template=grid,
        out_path=cfg.output_path("dem_grid"),
        method="average",
        dtype="float32",
        nodata=cfg.NODATA,
    )

    valid = dem != cfg.NODATA
    if not valid.all():
        raise RuntimeError(
            f"DEM has {int((~valid).sum())} void cells on the common grid"
        )

    edge = np.zeros(valid.shape, dtype=bool)
    edge[0, :] = True
    edge[-1, :] = True
    edge[:, 0] = True
    edge[:, -1] = True
    if not valid[edge].all():
        raise RuntimeError("DEM has voids along the grid edge")

    with rasterio.open(cfg.output_path("dem_clip")) as src:
        clipped = src.read(1)
        clipped_valid = utils.valid_mask(clipped, src.nodata)
    clipped_min = float(clipped[clipped_valid].min())
    clipped_max = float(clipped[clipped_valid].max())
    dem_min = float(dem[valid].min())
    dem_max = float(dem[valid].max())

    tolerance = max(5.0, (clipped_max - clipped_min) * 0.10)
    if (
        abs(dem_min - clipped_min) > tolerance
        or abs(dem_max - clipped_max) > tolerance
    ):
        raise RuntimeError(
            "DEM range does not match the clipped DEM: "
            f"({dem_min:.1f}, {dem_max:.1f}) vs "
            f"({clipped_min:.1f}, {clipped_max:.1f})"
        )

    LOGGER.info(
        "DEM 90 m: range %.1f to %.1f m; clipped range %.1f to %.1f m",
        dem_min,
        dem_max,
        clipped_min,
        clipped_max,
    )
    return dem, {
        "min": dem_min,
        "max": dem_max,
        "clipped_min": clipped_min,
        "clipped_max": clipped_max,
        "valid_fraction": float(valid.mean()),
        "edge_voids": int((~valid[edge]).sum()),
    }


def _align_landcover(
    grid: dict,
    district_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    source, src_transform, src_crs, src_nodata = utils.read_raster_window(
        cfg.LANDCOVER_RAW,
        grid,
    )
    if src_nodata is None:
        source_valid = source != 0
    else:
        source_valid = utils.valid_mask(source, src_nodata)

    mode, _ = utils.reproject_array(
        source,
        src_transform,
        src_crs,
        grid,
        out_path=cfg.output_path("landcover_mode"),
        method="mode",
        src_nodata=src_nodata,
        dst_nodata=cfg.NODATA,
        dtype="float32",
    )

    valid_fraction, _ = utils.reproject_array(
        source_valid.astype("float32"),
        src_transform,
        src_crs,
        grid,
        method="average",
        src_nodata=None,
        dst_nodata=0.0,
        dtype="float32",
    )

    class_values = list(cfg.LANDCOVER_CLASS_VALUES)
    class_fractions: list[np.ndarray] = []
    for class_value in class_values:
        source_class = (
            (source == class_value) & source_valid
        ).astype("float32")
        average, _ = utils.reproject_array(
            source_class,
            src_transform,
            src_crs,
            grid,
            method="average",
            src_nodata=None,
            dst_nodata=0.0,
            dtype="float32",
        )
        fraction = np.divide(
            average,
            valid_fraction,
            out=np.zeros_like(average),
            where=valid_fraction > 0,
        )
        class_fractions.append(fraction)

    all_fractions = np.stack(class_fractions, axis=0)
    good = valid_fraction > 0
    class_sums = all_fractions.sum(axis=0)
    normalized = np.zeros_like(all_fractions)
    np.divide(
        all_fractions,
        class_sums,
        out=normalized,
        where=good,
    )

    key_indices = [class_values.index(value) for value in cfg.LANDCOVER_KEY_CLASSES.values()]
    key_bands = normalized[key_indices]
    other = np.clip(1.0 - key_bands.sum(axis=0), 0.0, 1.0)
    bands = np.concatenate([key_bands, other[np.newaxis, :, :]], axis=0)

    band_sums = bands.sum(axis=0)
    np.divide(bands, band_sums, out=bands, where=good)
    bands[:, ~good] = cfg.NODATA

    band_names = [*cfg.LANDCOVER_KEY_CLASSES, "other"]
    utils.write_multiband_raster(
        bands,
        cfg.output_path("landcover_frac"),
        grid,
        band_names=band_names,
        nodata=cfg.NODATA,
        dtype="float32",
    )

    fraction_sums = bands.sum(axis=0)
    max_deviation = float(np.max(np.abs(fraction_sums[good] - 1.0)))
    if not np.isclose(max_deviation, 0.0, atol=1e-5):
        raise RuntimeError(
            f"Land cover fractions do not sum to 1 (max deviation {max_deviation})"
        )

    mode_valid = mode != cfg.NODATA
    mode_values = set(np.unique(mode[mode_valid]).tolist())
    if not mode_values.issubset(set(class_values)):
        raise RuntimeError(f"Unexpected land cover mode values: {mode_values}")

    inside = (district_mask == 1) & good
    means = {
        name: float(bands[index][inside].mean())
        for index, name in enumerate(band_names)
    }
    argmax = np.asarray(class_values)[np.argmax(normalized, axis=0)]
    mode_mismatch = int((mode_valid & (mode != argmax)).sum())

    LOGGER.info(
        "Land cover: valid %s cells; fractions sum to 1 (max deviation %.2g)",
        f"{int(good.sum()):,}",
        max_deviation,
    )
    LOGGER.info(
        "Land cover district means: trees %.1f%%, cropland %.1f%%, "
        "built-up %.1f%%, water %.1f%%, wetland %.1f%%, mangrove %.1f%%, "
        "other %.1f%%",
        means["trees"] * 100,
        means["cropland"] * 100,
        means["builtup"] * 100,
        means["water"] * 100,
        means["wetland"] * 100,
        means["mangrove"] * 100,
        means["other"] * 100,
    )
    return mode, bands, {
        "valid_cells": int(good.sum()),
        "fraction_sum_max_deviation": max_deviation,
        "mode_mismatch_cells": mode_mismatch,
        "district_means": means,
    }


def _align_ksdma(grid: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    source, src_transform, src_crs, src_nodata = utils.read_raster_window(
        cfg.KSDMA_SCENARIO_RAW,
        grid,
    )
    source_valid = utils.valid_mask(source, src_nodata)
    source_nan = np.where(source_valid, source, np.nan).astype("float32")

    ksdma, _ = utils.reproject_array(
        source_nan,
        src_transform,
        src_crs,
        grid,
        out_path=cfg.output_path("ksdma_100yr_90m"),
        method="nearest",
        src_nodata=np.nan,
        dst_nodata=cfg.NODATA,
        dtype="float32",
    )
    modelled = (ksdma != cfg.NODATA).astype("float32")
    utils.write_raster(
        modelled,
        cfg.output_path("ksdma_modelled_mask"),
        grid["profile"],
        nodata=cfg.NODATA,
        dtype="float32",
    )

    with rasterio.open(cfg.output_path("ksdma_100yr_clip")) as src:
        clipped = src.read(1)
        clipped_valid = utils.valid_mask(clipped, src.nodata)
        clipped_area = (
            utils.valid_area_m2(
                clipped,
                src.transform,
                src.crs,
                valid=clipped_valid,
            )
            / 1e6
        )

    cell_area_m2 = float(abs(grid["transform"].a * grid["transform"].e))
    target_area = float(modelled.sum()) * cell_area_m2 / 1e6
    relative = (target_area - clipped_area) / clipped_area
    if abs(relative) > 0.05:
        raise RuntimeError(
            "KSDMA modelled area changed by "
            f"{relative * 100:.1f}% ({clipped_area:.1f} -> {target_area:.1f} km2)"
        )

    valid = modelled == 1
    values = ksdma[valid]
    LOGGER.info(
        "KSDMA 100 yr: modelled %.1f km2 -> %.1f km2 (%+.1f%%); "
        "range %.2f to %.2f",
        clipped_area,
        target_area,
        relative * 100,
        float(values.min()),
        float(values.max()),
    )
    return ksdma, modelled, {
        "clipped_modelled_km2": clipped_area,
        "target_modelled_km2": target_area,
        "relative_change": relative,
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _align_flood(grid: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    source, src_transform, src_crs, src_nodata = utils.read_raster_window(
        cfg.FLOOD_2018_RAW,
        grid,
    )
    valid_source = (
        utils.valid_mask(source, src_nodata)
        if src_nodata is not None
        else np.ones(source.shape, dtype=bool)
    )
    flooded_source = (source == 1) & valid_source

    fraction, _ = utils.reproject_array(
        source.astype("float32"),
        src_transform,
        src_crs,
        grid,
        out_path=cfg.output_path("flood2018_frac"),
        method="average",
        src_nodata=src_nodata,
        dst_nodata=cfg.NODATA,
        dtype="float32",
    )

    binary = np.where(
        fraction == cfg.NODATA,
        cfg.NODATA,
        (fraction >= cfg.FLOOD_MASK_THRESHOLD).astype("float32"),
    )
    utils.write_raster(
        binary,
        cfg.output_path("flood2018_bin"),
        grid["profile"],
        nodata=cfg.NODATA,
        dtype="float32",
    )

    with rasterio.open(cfg.output_path("flood2018_clip")) as src:
        clipped = src.read(1)
        clipped_flooded = clipped == 1
        clipped_area = (
            utils.valid_area_m2(
                clipped,
                src.transform,
                src.crs,
                valid=clipped_flooded,
            )
            / 1e6
        )
    cell_area_m2 = float(abs(grid["transform"].a * grid["transform"].e))
    target_area = float((binary == 1).sum()) * cell_area_m2 / 1e6
    relative = (target_area - clipped_area) / clipped_area
    if abs(relative) > 0.10:
        raise RuntimeError(
            "Flooded area changed by "
            f"{relative * 100:.1f}% ({clipped_area:.1f} -> {target_area:.1f} km2)"
        )

    fraction_valid = fraction != cfg.NODATA
    if fraction_valid.any():
        fraction_min = float(fraction[fraction_valid].min())
        fraction_max = float(fraction[fraction_valid].max())
    else:
        fraction_min = fraction_max = float("nan")
    if fraction_min < 0 or fraction_max > 1:
        raise RuntimeError("Flood fraction is outside [0, 1]")
    if not set(np.unique(binary).tolist()).issubset({cfg.NODATA, 0.0, 1.0}):
        raise RuntimeError("Flood binary contains unexpected values")

    LOGGER.info(
        "Flood 2018: %.1f km2 -> %.1f km2 (%+.1f%%); fraction range %.3f to %.3f",
        clipped_area,
        target_area,
        relative * 100,
        fraction_min,
        fraction_max,
    )
    return fraction, binary, {
        "clipped_flooded_km2": clipped_area,
        "target_flooded_km2": target_area,
        "relative_change": relative,
        "fraction_min": fraction_min,
        "fraction_max": fraction_max,
        "source_flooded_cells": int(flooded_source.sum()),
        "target_flooded_cells": int((binary == 1).sum()),
    }


def _align_population(
    grid: dict,
    district_mask: np.ndarray,
    district,
) -> tuple[np.ndarray, dict]:
    counts, src_transform, src_crs, src_nodata = utils.read_raster_window(
        cfg.POPULATION_2018_RAW,
        grid,
    )
    district_source = district.to_crs(src_crs).geometry.union_all()

    population, stats = utils.population_to_grid(
        counts,
        src_transform,
        src_crs,
        grid,
        out_path=cfg.output_path("pop2018_90m"),
        source_nodata=src_nodata,
        district_geometry=district_source,
        district_mask=district_mask == 1,
    )

    with rasterio.open(cfg.output_path("pop2018_clip")) as src:
        clipped = src.read(1)
        clipped_valid = utils.valid_mask(clipped, src.nodata)
        clipped_inside = geometry_mask(
            [mapping(district.to_crs(src.crs).geometry.union_all())],
            out_shape=clipped.shape,
            transform=src.transform,
            invert=True,
            all_touched=False,
        )
        task2_total = float(clipped[clipped_inside & clipped_valid].sum())

    relative = (
        (stats["target_after_rescale"] - task2_total) / task2_total
        if task2_total
        else 0.0
    )
    if abs(relative) > 0.01:
        raise RuntimeError(
            "Population total differs from Task 2 by "
            f"{relative * 100:.2f}% ({task2_total:,.0f} -> "
            f"{stats['target_after_rescale']:,.0f})"
        )

    valid = population != cfg.NODATA
    if (population[valid] < 0).any():
        raise RuntimeError("Population contains negative values")

    LOGGER.info(
        "Population: Task 2 %.0f; rescale %.0f -> %.0f (%+.2f%%)",
        task2_total,
        stats["target_before_rescale"],
        stats["target_after_rescale"],
        relative * 100,
    )
    stats["task2_total"] = task2_total
    stats["relative_to_task2"] = relative
    stats["valid_fraction"] = float(valid.mean())
    return population, stats


def _align_dem_30m(grid: dict) -> dict:
    grid_30m = utils.build_grid(
        bounds=grid["bounds"],
        resolution=cfg.DEM_30M_RESOLUTION,
        crs=cfg.CRS_WORKING,
        buffer_m=0,
    )
    dem_30m, _ = utils.align_raster(
        cfg.DEM_RAW,
        template=grid_30m,
        out_path=cfg.output_path("dem_30m"),
        method="average",
        dtype="float32",
        nodata=cfg.NODATA,
    )
    valid = dem_30m != cfg.NODATA
    if not valid.all():
        raise RuntimeError("Optional 30 m DEM has voids")

    LOGGER.info(
        "Optional DEM 30 m: %s x %s cells, range %.1f to %.1f m",
        grid_30m["height"],
        grid_30m["width"],
        float(dem_30m[valid].min()),
        float(dem_30m[valid].max()),
    )
    return {
        "width": grid_30m["width"],
        "height": grid_30m["height"],
        "resolution": grid_30m["resolution"],
        "min": float(dem_30m[valid].min()),
        "max": float(dem_30m[valid].max()),
    }


def _same_grid_check() -> None:
    grid_path = cfg.output_path("grid")
    for key in GRID_OUTPUT_KEYS:
        if key == "grid":
            continue
        utils.assert_same_grid(grid_path, cfg.output_path(key))
    LOGGER.info(
        "Same-grid check passed for %s rasters",
        len(GRID_OUTPUT_KEYS) - 1,
    )


def _plot_panel(
    ax,
    path: Path,
    extent: tuple[float, float, float, float],
    district,
    *,
    band: int = 1,
    title: str,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    with rasterio.open(path) as src:
        data = src.read(band)
        nodata = src.nodata
    array = utils.nodata_to_nan(
        data,
        (nodata,) if nodata is not None else None,
    )
    finite = array[np.isfinite(array)]
    if vmin is None and finite.size:
        vmin = float(np.nanpercentile(finite, 2))
    if vmax is None and finite.size:
        vmax = float(np.nanpercentile(finite, 98))

    image = ax.imshow(
        array,
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    district.boundary.plot(ax=ax, color="black", linewidth=0.4)
    ax.set_title(title, fontsize=8)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.tick_params(labelsize=6)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.02)


def _make_overview(grid: dict, district) -> None:
    extent = (
        grid["bounds"][0],
        grid["bounds"][2],
        grid["bounds"][1],
        grid["bounds"][3],
    )
    district_working = district.to_crs(grid["crs"])
    panels = [
        ("district_mask", 1, "District mask", "Greys", 0.0, 1.0),
        ("dem_grid", 1, "DEM 90 m", "terrain", -10.0, 150.0),
        ("landcover_mode", 1, "Land cover mode", "tab20", 10.0, 100.0),
        ("landcover_frac", 1, "Tree fraction", "viridis", 0.0, 1.0),
        ("landcover_frac", 3, "Built-up fraction", "viridis", 0.0, 1.0),
        ("ksdma_100yr_90m", 1, "KSDMA 100 yr", "viridis", None, None),
        ("ksdma_modelled_mask", 1, "KSDMA modelled", "Greys", 0.0, 1.0),
        ("flood2018_frac", 1, "Flood 2018 fraction", "Blues", 0.0, 1.0),
        ("flood2018_bin", 1, "Flood 2018 binary", "Blues", 0.0, 1.0),
        ("pop2018_90m", 1, "Population 2018", "inferno", 0.0, None),
    ]

    ncols = 4
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 12), dpi=150)
    axes = np.atleast_1d(axes).ravel()

    for ax, (key, band, title, cmap, vmin, vmax) in zip(axes, panels):
        _plot_panel(
            ax,
            cfg.output_path(key),
            extent,
            district_working,
            band=band,
            title=title,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

    for ax in axes[len(panels) :]:
        ax.axis("off")

    fig.suptitle("Alappuzha aligned layers (Task 3)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path = cfg.output_path("aligned_overview")
    utils.make_folder(out_path.parent)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Overview figure written to %s", out_path)


def main() -> None:
    cfg.check_input_paths()

    grid_path = cfg.output_path("grid")
    mask_path = cfg.output_path("district_mask")
    if not grid_path.exists() or not mask_path.exists():
        raise FileNotFoundError(
            "Task 3 grid is missing. Run src/02_build_grid.py first."
        )

    grid = utils.load_grid(grid_path)
    district = utils.read_vector(cfg.output_path("district"), layer="district")
    with rasterio.open(mask_path) as src:
        district_mask = src.read(1)

    if district_mask.sum() == 0:
        raise RuntimeError("District mask is empty")

    stats = {
        "grid": {
            "crs": str(grid["crs"]),
            "transform": tuple(float(value) for value in grid["transform"])[:6],
            "width": int(grid["width"]),
            "height": int(grid["height"]),
            "bounds": tuple(float(value) for value in grid["bounds"]),
            "resolution": float(grid["transform"].a),
        },
        "district_mask_cells": int((district_mask == 1).sum()),
    }

    _, stats["dem"] = _align_dem(grid)
    _, _, stats["landcover"] = _align_landcover(grid, district_mask)
    _, _, stats["ksdma"] = _align_ksdma(grid)
    _, _, stats["flood"] = _align_flood(grid)
    _, stats["population"] = _align_population(grid, district_mask, district)

    _same_grid_check()
    stats["optional_dem_30m"] = _align_dem_30m(grid)

    missing = [
        key for key in GRID_OUTPUT_KEYS if not cfg.output_path(key).exists()
    ]
    if missing:
        raise RuntimeError(f"Missing Task 3 outputs: {missing}")

    stats_path = cfg.output_path("alignment_stats")
    utils.make_folder(stats_path.parent)
    stats_path.write_text(
        json.dumps(stats, indent=2, default=str),
        encoding="utf-8",
    )

    _make_overview(grid, district)
    LOGGER.info("Alignment stats written to %s", stats_path)
    LOGGER.info("Task 3 alignment complete")


if __name__ == "__main__":
    main()
