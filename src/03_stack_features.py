"""Task 4: stack terrain, drainage and land-cover features.

Run after ``src/03_terrain_features.py`` and ``src/03_waterway_distance.py``::

    python src/03_stack_features.py
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
import rasterio

from src import config as cfg
from src import utils

LOGGER = utils.get_logger("stack_features")

BAND_SPECS = [
    ("elevation", "dem_grid", "m", "Alappuzha 90 m DEM"),
    ("slope", "slope_grid", "degrees", "Slope from averaged 30 m DEM"),
    ("twi", "twi_grid", "index", "Topographic wetness index"),
    ("log_flow_accumulation", "log_flow_accumulation", "log(cells)", "log1p(D8 accumulation)"),
    ("hand", "hand_grid", "m", "Height above nearest drainage"),
    ("distance_waterway", "distance_waterways_grid", "m", "Distance to nearest waterway line"),
]


def _require_inputs() -> None:
    keys = [
        "grid",
        "district_mask",
        "dem_grid",
        "landcover_frac",
        "ksdma_modelled_mask",
        "slope_grid",
        "twi_grid",
        "log_flow_accumulation",
        "hand_grid",
        "distance_waterways_grid",
    ]
    missing = [
        f"{key}: {cfg.output_path(key)}"
        for key in keys
        if not Path(cfg.output_path(key)).exists()
    ]
    if missing:
        formatted = "\n".join(f"  - {item}" for item in missing)
        raise FileNotFoundError(f"Missing Task 4 input(s):\n{formatted}")


def _read_band(key: str) -> np.ndarray:
    data, _, _ = utils.read_raster(cfg.output_path(key))
    return np.asarray(data, dtype="float32")


def _load_features() -> tuple[list[tuple[str, np.ndarray, str, str]], dict]:
    features: list[tuple[str, np.ndarray, str, str]] = []
    for name, key, units, description in BAND_SPECS:
        features.append((name, _read_band(key), units, description))

    with rasterio.open(cfg.output_path("landcover_frac")) as src:
        descriptions = tuple(src.descriptions or ())
        for index, name in enumerate(cfg.LANDCOVER_KEY_CLASSES, start=1):
            band = descriptions.index(name) + 1 if name in descriptions else index
            fraction = np.asarray(src.read(band), dtype="float32")
            fraction[(fraction >= 0) & (fraction < 1e-6)] = 0.0
            features.append(
                (
                    name,
                    fraction,
                    "fraction",
                    f"ESA WorldCover {name} fraction",
                )
            )
    return features, {
        "landcover_bands": {
            name: int(descriptions.index(name) + 1) if name in descriptions else index
            for index, name in enumerate(cfg.LANDCOVER_KEY_CLASSES, start=1)
        }
    }


def _stats(values: np.ndarray) -> dict:
    selected = np.asarray(values, dtype="float64")
    selected = selected[np.isfinite(selected) & (selected != cfg.NODATA)]
    if not selected.size:
        return {
            "valid_cells": 0,
            "min": None,
            "max": None,
            "mean": None,
        }
    return {
        "valid_cells": int(selected.size),
        "min": float(selected.min()),
        "max": float(selected.max()),
        "mean": float(selected.mean()),
    }


def _ksdma_link_check(
    stack: np.ndarray,
    band_index: dict[str, int],
    valid: np.ndarray,
) -> dict:
    modelled_raw, _, _ = utils.read_raster(cfg.output_path("ksdma_modelled_mask"))
    modelled = (np.asarray(modelled_raw) == 1) & valid
    unmodelled = valid & ~modelled

    result = {
        "modelled_cells": int(modelled.sum()),
        "unmodelled_cells": int(unmodelled.sum()),
    }
    if not modelled.any() or not unmodelled.any():
        LOGGER.warning(
            "KSDMA link check skipped: modelled=%s, unmodelled=%s cells",
            f"{int(modelled.sum()):,}",
            f"{int(unmodelled.sum()):,}",
        )
        result["skipped"] = True
        return result

    for name in ("hand", "twi"):
        values = stack[band_index[name]]
        modelled_median = float(np.median(values[modelled]))
        unmodelled_median = float(np.median(values[unmodelled]))
        difference = modelled_median - unmodelled_median
        result[name] = {
            "modelled_median": modelled_median,
            "unmodelled_median": unmodelled_median,
            "difference": difference,
        }

    hand_difference = abs(result["hand"]["difference"])
    twi_difference = abs(result["twi"]["difference"])
    if hand_difference > 5.0 or twi_difference > 1.0:
        LOGGER.warning(
            "KSDMA modelled and unmodelled terrain differ: HAND %.2f m, TWI %.2f",
            result["hand"]["difference"],
            result["twi"]["difference"],
        )
    else:
        LOGGER.info(
            "KSDMA terrain check: HAND difference %.2f m, TWI difference %.2f",
            result["hand"]["difference"],
            result["twi"]["difference"],
        )

    result["skipped"] = False
    result["warn_hand_m"] = 5.0
    result["warn_twi"] = 1.0
    return result


def _plot_overview(
    stack: np.ndarray,
    valid: np.ndarray,
    band_names: list[str],
    units: list[str],
) -> None:
    cmaps = {
        "elevation": "terrain",
        "slope": "magma",
        "twi": "viridis",
        "log_flow_accumulation": "Blues",
        "hand": "viridis",
        "distance_waterway": "viridis_r",
        "trees": "Greens",
        "cropland": "YlGn",
        "builtup": "Reds",
        "water": "Blues",
        "wetland": "YlGnBu",
        "mangrove": "Greens",
    }

    district = utils.load_district(crs=cfg.CRS_WORKING)
    rows = 3
    cols = 4
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(16, 12),
        constrained_layout=True,
    )
    axes = axes.ravel()

    for index, name in enumerate(band_names):
        axis = axes[index]
        data = np.where(valid, stack[index], np.nan)
        finite = data[np.isfinite(data)]
        if finite.size:
            if name in cfg.LANDCOVER_KEY_CLASSES:
                vmin, vmax = 0.0, 1.0
            else:
                vmin, vmax = np.nanpercentile(finite, [2, 98])
                if not np.isfinite(vmin) or not np.isfinite(vmax):
                    vmin, vmax = float(finite.min()), float(finite.max())
                if vmin == vmax:
                    vmax = vmin + 1.0
        else:
            vmin, vmax = 0.0, 1.0

        image = axis.imshow(
            data,
            cmap=cmaps.get(name, "viridis"),
            vmin=vmin,
            vmax=vmax,
        )
        district.boundary.plot(
            ax=axis,
            color="black",
            linewidth=0.25,
        )
        axis.set_title(f"{name} ({units[index]})", fontsize=10)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_aspect("equal")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.02)

    for axis in axes[len(band_names) :]:
        axis.axis("off")

    fig.suptitle("Alappuzha 90 m feature stack", fontsize=14)
    out_path = cfg.output_path("features_overview")
    utils.make_folder(out_path.parent)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cfg.check_input_paths()
    _require_inputs()

    features, source_info = _load_features()
    band_names = [name for name, _, _, _ in features]
    units = [units for _, _, units, _ in features]
    stack = np.stack([array for _, array, _, _ in features], axis=0)

    feature_valid = np.ones(stack.shape[1:], dtype=bool)
    for _, array, _, _ in features:
        feature_valid &= np.isfinite(array) & (array != cfg.NODATA)

    district_raw, _, _ = utils.read_raster(cfg.output_path("district_mask"))
    district = np.asarray(district_raw) == 1
    valid = feature_valid & district

    stack[:, ~feature_valid] = cfg.NODATA

    band_index = {name: index for index, name in enumerate(band_names)}
    link_check = _ksdma_link_check(stack, band_index, valid)

    band_specs = [
        {
            "index": index + 1,
            "name": name,
            "units": units[index],
            "description": description,
        }
        for index, (name, _, _, description) in enumerate(features)
    ]
    bands_path = cfg.output_path("features_stack_bands")
    utils.make_folder(bands_path.parent)
    bands_path.write_text(
        json.dumps(band_specs, indent=2),
        encoding="utf-8",
    )

    utils.write_multiband_raster(
        stack,
        cfg.output_path("features_stack"),
        cfg.GRID_PATH,
        band_names=band_names,
        nodata=cfg.NODATA,
        dtype="float32",
    )
    utils.write_raster(
        valid.astype("uint8"),
        cfg.output_path("features_valid_mask"),
        cfg.GRID_PATH,
        nodata=0,
        dtype="uint8",
    )

    rows = []
    band_stats = {}
    for index, name in enumerate(band_names):
        band = stack[index]
        finite = np.isfinite(band) & (band != cfg.NODATA)
        values = band[valid]
        rows.append(
            {
                "band": name,
                "units": units[index],
                "min": float(values.min()) if values.size else np.nan,
                "max": float(values.max()) if values.size else np.nan,
                "mean": float(values.mean()) if values.size else np.nan,
                "percent_nan": float((~finite).mean() * 100.0),
                "valid_cells": int(values.size),
            }
        )
        band_stats[name] = {
            "full": _stats(band),
            "district": _stats(band[valid]),
        }

    summary_path = cfg.output_path("feature_summary")
    utils.make_folder(summary_path.parent)
    pd.DataFrame(rows).to_csv(summary_path, index=False, float_format="%.6g")

    terrain_stats_path = cfg.output_path("terrain_stats")
    waterway_stats_path = cfg.output_path("waterway_distance_stats")
    terrain_stats = (
        json.loads(terrain_stats_path.read_text(encoding="utf-8"))
        if terrain_stats_path.exists()
        else None
    )
    waterway_stats = (
        json.loads(waterway_stats_path.read_text(encoding="utf-8"))
        if waterway_stats_path.exists()
        else None
    )

    stats = {
        "grid": {
            "height": int(stack.shape[1]),
            "width": int(stack.shape[2]),
            "bands": int(stack.shape[0]),
        },
        "bands": band_specs,
        "valid_mask_cells": int(valid.sum()),
        "district_cells": int(district.sum()),
        "feature_valid_cells": int(feature_valid.sum()),
        "band_stats": band_stats,
        "ksdma_link_check": link_check,
        "landcover_bands": source_info["landcover_bands"],
        "terrain_stats": terrain_stats,
        "waterway_distance_stats": waterway_stats,
    }
    stats_path = cfg.output_path("feature_stats")
    stats_path.write_text(
        json.dumps(stats, indent=2, default=str),
        encoding="utf-8",
    )

    utils.assert_same_grid(
        cfg.output_path("features_stack"),
        cfg.GRID_PATH,
    )
    if valid.sum() == 0:
        raise RuntimeError("Feature valid mask is empty")

    _plot_overview(stack, valid, band_names, units)

    LOGGER.info(
        "Feature stack: %s bands, %s x %s; %s valid district cells",
        f"{stack.shape[0]:,}",
        f"{stack.shape[2]:,}",
        f"{stack.shape[1]:,}",
        f"{int(valid.sum()):,}",
    )
    LOGGER.info("Feature stacking complete")


if __name__ == "__main__":
    main()
