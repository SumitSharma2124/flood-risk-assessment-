"""Task 2: inspect every clipped layer and run data sanity checks.

Run after ``src/01_clip_layers.py``::

    python src/01_inspect_layers.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.warp import transform_bounds
from shapely.geometry import box, mapping

from src import config as cfg
from src import utils

LOGGER = utils.get_logger("inspect_layers")

RASTER_KEYS = [
    "dem_clip",
    "landcover_clip",
    "ksdma_100yr_clip",
    "flood2018_clip",
    "pop2018_clip",
]

VECTOR_KEYS = [
    "district",
    "roads_clip",
    "waterways_clip",
    "buildings_clip",
    "facilities_clip",
    "villages_clip",
]

SUMMARY_COLUMNS = [
    "file",
    "layer",
    "kind",
    "crs",
    "size",
    "bounds",
    "dtype",
    "nodata",
    "min",
    "max",
    "percent_valid",
    "notes",
]


def _bounds_text(bounds: tuple[float, float, float, float]) -> str:
    return ", ".join(f"{value:.6f}" for value in bounds)


def _raster_row(key: str) -> tuple[dict, dict]:
    path = cfg.output_path(key)
    with rasterio.open(path) as src:
        data = src.read(1)
        nodata = src.nodata
        valid = utils.valid_mask(data, (nodata,) if nodata is not None else None)
        values = data[valid]
        stats = {
            "path": path,
            "width": src.width,
            "height": src.height,
            "crs": src.crs,
            "bounds": tuple(src.bounds),
            "dtype": src.dtypes[0],
            "nodata": nodata,
            "valid_count": int(valid.sum()),
            "valid_percent": float(valid.mean() * 100.0),
            "min": float(values.min()) if values.size else None,
            "max": float(values.max()) if values.size else None,
            "data": data,
        }
        row = {
            "file": path.name,
            "layer": "",
            "kind": "raster",
            "crs": src.crs.to_string() if src.crs is not None else None,
            "size": f"{src.height} x {src.width}",
            "bounds": _bounds_text(tuple(src.bounds)),
            "dtype": src.dtypes[0],
            "nodata": nodata,
            "min": stats["min"],
            "max": stats["max"],
            "percent_valid": stats["valid_percent"],
            "notes": "",
        }
    return row, stats


def _vector_rows(key: str) -> tuple[list[dict], list[dict]]:
    path = cfg.output_path(key)
    rows: list[dict] = []
    stats: list[dict] = []
    for layer_name, geometry_type in utils.list_gpkg_layers(path):
        gdf = utils.read_vector(path, layer=layer_name)
        geometries = gdf.geometry
        valid_geometries = geometries.notna() & ~geometries.is_empty
        percent_valid = (
            float(geometries.is_valid[valid_geometries].mean() * 100.0)
            if valid_geometries.any()
            else 0.0
        )
        bounds = tuple(float(value) for value in gdf.total_bounds)
        row = {
            "file": path.name,
            "layer": layer_name,
            "kind": "vector",
            "crs": gdf.crs.to_string() if gdf.crs is not None else None,
            "size": len(gdf),
            "bounds": _bounds_text(bounds),
            "dtype": geometry_type,
            "nodata": None,
            "min": None,
            "max": None,
            "percent_valid": percent_valid,
            "notes": "",
        }
        rows.append(row)
        stats.append(
            {
                "path": path,
                "layer": layer_name,
                "gdf": gdf,
                "feature_count": len(gdf),
                "empty": gdf.empty,
                "crs": gdf.crs,
                "bounds": bounds,
                "geometry_type": geometry_type,
            }
        )
    return rows, stats


def _district_geometries() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    path = cfg.output_path("district")
    district = utils.read_vector(path, layer="district").to_crs(cfg.CRS_GEOGRAPHIC)
    buffered = utils.read_vector(path, layer="district_buffered").to_crs(
        cfg.CRS_GEOGRAPHIC
    )
    return district, buffered


def _check_rows(
    rows: list[dict],
    raster_stats: dict[str, dict],
    vector_stats: list[dict],
    district_geom,
    errors: list[str],
) -> None:
    for row in rows:
        label = f"{row['file']}:{row['layer']}" if row["layer"] else row["file"]

        if row["crs"] in (None, "", "None"):
            errors.append(f"{label} has no CRS")

        if row["kind"] == "raster":
            stats = raster_stats[label]
            if stats["width"] == 0 or stats["height"] == 0:
                errors.append(f"{label} is empty (zero-sized raster)")
                continue
            if stats["valid_count"] == 0:
                errors.append(f"{label} has no valid cells")
                continue
            bounds = stats["bounds"]
        else:
            stats = next(
                item
                for item in vector_stats
                if item["path"].name == row["file"] and item["layer"] == row["layer"]
            )
            if stats["empty"]:
                errors.append(f"{label} is empty")
                continue
            if row["crs"] is None:
                continue
            bounds = transform_bounds(
                stats["crs"],
                cfg.CRS_GEOGRAPHIC,
                *stats["bounds"],
                densify_pts=21,
            )

        if not box(*bounds).intersects(district_geom):
            errors.append(f"{label} lies outside the district")


def _population_check(
    district_geom,
    buffered_geom,
    rows: list[dict],
    errors: list[str],
) -> None:
    path = cfg.output_path("pop2018_clip")
    with rasterio.open(path) as src:
        data = src.read(1)
        valid = utils.valid_mask(data, src.nodata)
        inside = geometry_mask(
            [mapping(district_geom)],
            out_shape=data.shape,
            transform=src.transform,
            invert=True,
            all_touched=False,
        )
        buffered = geometry_mask(
            [mapping(buffered_geom)],
            out_shape=data.shape,
            transform=src.transform,
            invert=True,
            all_touched=False,
        )
        district_total = float(data[inside & valid].sum())
        buffered_total = float(data[buffered & valid].sum())

    expected = float(cfg.ALAPPUZHA_POPULATION_2011)
    relative = (district_total - expected) / expected
    LOGGER.info(
        "Population inside district: %.0f; buffered: %.0f; 2011 census: %.0f "
        "(%+.1f%%)",
        district_total,
        buffered_total,
        expected,
        relative * 100.0,
    )

    if not (0.70 * expected <= district_total <= 1.30 * expected):
        errors.append(
            "Population inside the district is "
            f"{district_total:,.0f}, not within 30% of the 2011 census "
            f"({expected:,.0f})"
        )
    if buffered_total <= district_total:
        errors.append(
            "Buffered population is not larger than the district population; "
            "check that the exact district mask is being used"
        )

    note = (
        f"district={district_total:,.0f}; buffered={buffered_total:,.0f}; "
        f"census2011={expected:,.0f}; delta={relative * 100.0:+.1f}%"
    )
    if abs(relative) > 0.10:
        note += " (WorldPop R2025A is above the census total)"
    for row in rows:
        if row["file"] == path.name:
            row["notes"] = note


def _dem_check(district_geom, rows: list[dict], errors: list[str]) -> None:
    path = cfg.output_path("dem_clip")
    with rasterio.open(path) as src:
        data = src.read(1)
        valid = utils.valid_mask(data, src.nodata)
        inside = geometry_mask(
            [mapping(district_geom)],
            out_shape=data.shape,
            transform=src.transform,
            invert=True,
            all_touched=False,
        )
        valid = valid & inside
        values = data[valid]
        dem_min = float(values.min())
        dem_max = float(values.max())

        columns = np.arange(src.width)
        x_centres = src.transform.c + (columns + 0.5) * src.transform.a
        west = x_centres < 76.45
        east = x_centres > 76.55
        west_mean = float(data[:, west][valid[:, west]].mean())
        east_mean = float(data[:, east][valid[:, east]].mean())

    LOGGER.info(
        "DEM: min %.1f m, max %.1f m, west mean %.1f m, east mean %.1f m",
        dem_min,
        dem_max,
        west_mean,
        east_mean,
    )

    if dem_min >= 5.0:
        errors.append(
            f"DEM minimum is {dem_min:.1f} m; Kuttanad is not near/below 0 m"
        )
    if dem_max <= 100.0:
        errors.append(
            f"DEM maximum is {dem_max:.1f} m; eastern Alappuzha does not rise"
        )
    if east_mean <= west_mean:
        errors.append(
            f"DEM east mean ({east_mean:.1f} m) is not above west mean "
            f"({west_mean:.1f} m)"
        )

    note = (
        f"min={dem_min:.1f} m; max={dem_max:.1f} m; "
        f"west_mean={west_mean:.1f} m; east_mean={east_mean:.1f} m"
    )
    for row in rows:
        if row["file"] == path.name:
            row["notes"] = note


def _ksdma_check(district_geom, rows: list[dict]) -> None:
    path = cfg.output_path("ksdma_100yr_clip")
    with rasterio.open(path) as src:
        data = src.read(1)
        valid = utils.valid_mask(data, src.nodata)
        inside = geometry_mask(
            [mapping(district_geom)],
            out_shape=data.shape,
            transform=src.transform,
            invert=True,
            all_touched=False,
        )
        inside_count = int(inside.sum())
        valid_inside = int((inside & valid).sum())
        share = valid_inside / inside_count * 100.0 if inside_count else 0.0
        values = data[inside & valid]

    LOGGER.info(
        "KSDMA 100 yr: %.2f%% valid inside the district, range %.2f to %.2f",
        share,
        float(values.min()) if values.size else float("nan"),
        float(values.max()) if values.size else float("nan"),
    )

    with rasterio.open(cfg.KSDMA_SCENARIO_RAW) as raw:
        raw_data = raw.read(1)
        raw_valid = utils.valid_mask(raw_data, raw.nodata)
        raw_share = float(raw_valid.mean() * 100.0)
    LOGGER.info("KSDMA 100 yr: %.2f%% valid in the full raw tile", raw_share)

    note = (
        f"full_tile_valid={raw_share:.2f}%; "
        f"inside_district_valid={share:.2f}%; "
        f"range={float(values.min()) if values.size else float('nan'):.2f}-"
        f"{float(values.max()) if values.size else float('nan'):.2f}"
    )
    if share >= 90.0:
        note += "; unusually high valid share"
    for row in rows:
        if row["file"] == path.name:
            row["notes"] = note


def _flood_check(district_geom, rows: list[dict]) -> None:
    path = cfg.output_path("flood2018_clip")
    with rasterio.open(path) as src:
        data = src.read(1)
        inside = geometry_mask(
            [mapping(district_geom)],
            out_shape=data.shape,
            transform=src.transform,
            invert=True,
            all_touched=False,
        )
        inside_count = int(inside.sum())
        flooded = int((inside & (data == 1)).sum())
        share = flooded / inside_count * 100.0 if inside_count else 0.0
        unique = np.unique(data[inside]).tolist()

    LOGGER.info(
        "Flood 2018: %.2f%% flooded inside the district; values=%s",
        share,
        unique,
    )

    note = f"flooded_inside={share:.2f}%; values={unique}"
    if share == 0.0:
        note += "; no flooded cells"
    elif share > 80.0:
        note += "; nearly uniform fill"
    for row in rows:
        if row["file"] == path.name:
            row["notes"] = note


def _landcover_check(district_geom, rows: list[dict]) -> None:
    path = cfg.output_path("landcover_clip")
    with rasterio.open(path) as src:
        data = src.read(1)
        valid = data != 0
        inside = geometry_mask(
            [mapping(district_geom)],
            out_shape=data.shape,
            transform=src.transform,
            invert=True,
            all_touched=False,
        )
        values = data[inside & valid]
        classes, counts = np.unique(values, return_counts=True)
        total = int(counts.sum())
        ranked = sorted(
            zip(classes.tolist(), counts.tolist()),
            key=lambda item: item[1],
            reverse=True,
        )

    top = [
        f"{cfg.ESA_WORLDCOVER_CLASSES.get(int(cls), str(cls))} "
        f"{count / total * 100.0:.1f}%"
        for cls, count in ranked[:5]
    ] if total else []
    LOGGER.info("Land cover inside district: %s", "; ".join(top))

    note = "top: " + "; ".join(top)
    for row in rows:
        if row["file"] == path.name:
            row["notes"] = note


def _count_checks(rows: list[dict], errors: list[str]) -> None:
    stats_path = cfg.output_path("clip_stats")
    if not stats_path.exists():
        errors.append("clip stats file is missing")
        return

    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    for key in ("roads_clip", "waterways_clip"):
        item = stats.get("vectors", {}).get(key, {})
        source = item.get("source_features")
        clipped = item.get("clipped_features")
        if source is None or clipped is None:
            errors.append(f"clip stats missing counts for {key}")
            continue
        if clipped > source:
            errors.append(f"{key} gained features during clipping")
        note = (
            f"source={source:,}; clipped={clipped:,}; "
            f"duplicates={item.get('duplicates_dropped', 0):,}"
        )
        if clipped == source:
            LOGGER.warning("%s kept all source features", key)
            note += "; source already within buffer"
        for row in rows:
            if row["file"] == cfg.output_path(key).name:
                row["notes"] = note

    buildings = stats.get("buildings", {})
    csv_rows = buildings.get("csv_rows")
    clipped = buildings.get("clipped_features")
    bbox_rows = buildings.get("bbox_rows")
    if None in (csv_rows, clipped):
        errors.append("clip stats missing building counts")
    elif clipped >= csv_rows:
        errors.append("buildings count did not drop from the CSV total")
    else:
        LOGGER.info(
            "Buildings: %s CSV rows -> %s in bbox -> %s clipped",
            f"{csv_rows:,}",
            f"{bbox_rows:,}",
            f"{clipped:,}",
        )
    for row in rows:
        if row["file"] == cfg.output_path("buildings_clip").name:
            row["notes"] = (
                f"csv={csv_rows:,}; bbox={bbox_rows:,}; clipped={clipped:,}"
                if None not in (csv_rows, bbox_rows, clipped)
                else ""
            )


def _plot_layer(ax, key: str, district_4326, buffered_4326, extent) -> None:
    path = cfg.output_path(key)
    if not path.exists():
        ax.set_title(f"{key}\nMISSING", fontsize=8)
        ax.axis("off")
        return

    if key in RASTER_KEYS:
        with rasterio.open(path) as src:
            max_pixels = 700
            scale = max(1, math.ceil(max(src.width, src.height) / max_pixels))
            out_shape = (
                max(1, src.height // scale),
                max(1, src.width // scale),
            )
            resampling = (
                Resampling.nearest
                if key in {"landcover_clip", "flood2018_clip"}
                else Resampling.average
            )
            data = src.read(1, out_shape=out_shape, resampling=resampling)
            nodata = src.nodata
            image_extent = tuple(src.bounds)

        array = utils.nodata_to_nan(data, (nodata,) if nodata is not None else None)
        if key == "dem_clip":
            cmap, vmin, vmax = "terrain", -10.0, 100.0
        elif key == "landcover_clip":
            cmap, vmin, vmax = "tab20", 0.0, 100.0
        elif key == "flood2018_clip":
            cmap, vmin, vmax = "Blues", 0.0, 1.0
        else:
            finite = array[np.isfinite(array)]
            vmax = float(np.percentile(finite, 99)) if finite.size else None
            vmin = float(np.percentile(finite, 1)) if finite.size else None
            cmap = "viridis"

        ax.imshow(
            array,
            extent=image_extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        title = key.replace("_clip", "").replace("_", " ")
    else:
        gdf = (
            utils.read_vector(path, layer="district")
            if key == "district"
            else utils.read_vector(path)
        )
        if key == "district":
            gdf.plot(ax=ax, color="#d9d9d9", edgecolor="black", linewidth=0.6)
            buffered_4326.boundary.plot(
                ax=ax,
                color="tab:red",
                linewidth=0.7,
                linestyle="--",
            )
        else:
            if key == "buildings_clip":
                centroids = gdf.geometry.representative_point()
                points = gpd.GeoSeries(
                    centroids,
                    crs=gdf.crs,
                ).to_crs(cfg.CRS_GEOGRAPHIC)
                points.plot(ax=ax, markersize=0.08, color="black")
            elif key in {"roads_clip", "waterways_clip"}:
                gdf.to_crs(cfg.CRS_GEOGRAPHIC).plot(
                    ax=ax,
                    color="tab:blue" if key == "roads_clip" else "tab:cyan",
                    linewidth=0.25,
                )
            elif key == "facilities_clip":
                gdf.to_crs(cfg.CRS_GEOGRAPHIC).plot(
                    ax=ax,
                    markersize=0.8,
                    color="tab:red",
                )
            elif key == "villages_clip":
                gdf.to_crs(cfg.CRS_GEOGRAPHIC).boundary.plot(
                    ax=ax,
                    color="tab:green",
                    linewidth=0.25,
                )
        title = key.replace("_clip", "").replace("_", " ")

    district_4326.boundary.plot(ax=ax, color="black", linewidth=0.5)
    ax.set_title(title, fontsize=8)
    ax.set_xlim(extent[0], extent[2])
    ax.set_ylim(extent[1], extent[3])
    ax.set_aspect("equal")
    ax.tick_params(labelsize=6)


def _make_overview(district_4326, buffered_4326, out_path: Path) -> None:
    plot_keys = [
        "district",
        "dem_clip",
        "landcover_clip",
        "ksdma_100yr_clip",
        "flood2018_clip",
        "pop2018_clip",
        "roads_clip",
        "waterways_clip",
        "buildings_clip",
        "facilities_clip",
        "villages_clip",
    ]
    extent = tuple(buffered_4326.total_bounds)
    ncols = 4
    nrows = math.ceil(len(plot_keys) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 10), dpi=150)
    axes = np.atleast_1d(axes).ravel()

    for ax, key in zip(axes, plot_keys):
        _plot_layer(ax, key, district_4326, buffered_4326, extent)

    for ax in axes[len(plot_keys) :]:
        ax.axis("off")

    fig.suptitle("Alappuzha clipped layers (Task 2)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    utils.make_folder(out_path.parent)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _check_vector_inside(
    vectors: list[dict],
    buffered_geom,
    errors: list[str],
) -> None:
    for stats in vectors:
        if stats["path"].name in {
            cfg.output_path("district").name,
            cfg.output_path("villages_clip").name,
        }:
            continue
        gdf = stats["gdf"]
        if gdf.empty or gdf.crs is None:
            continue
        if not gdf.intersects(buffered_geom).all():
            errors.append(
                f"{stats['path'].name}:{stats['layer']} has features outside "
                "the buffered district"
            )


def main() -> None:
    cfg.check_input_paths()

    errors: list[str] = []
    missing = [
        key
        for key in [*RASTER_KEYS, *VECTOR_KEYS]
        if not cfg.output_path(key).exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing Task 2 outputs: {missing}")

    district, buffered = _district_geometries()
    district_geom = district.geometry.union_all()
    buffered_geom = buffered.geometry.union_all()
    buffered_working = buffered.to_crs(cfg.CRS_WORKING).geometry.union_all()

    rows: list[dict] = []
    raster_stats: dict[str, dict] = {}
    vector_stats: list[dict] = []

    for key in RASTER_KEYS:
        row, stats = _raster_row(key)
        rows.append(row)
        label = row["file"]
        raster_stats[label] = stats

    for key in VECTOR_KEYS:
        new_rows, new_stats = _vector_rows(key)
        rows.extend(new_rows)
        vector_stats.extend(new_stats)

    _check_rows(rows, raster_stats, vector_stats, district_geom, errors)
    _check_vector_inside(vector_stats, buffered_working, errors)

    _population_check(district_geom, buffered_geom, rows, errors)
    _dem_check(district_geom, rows, errors)
    _ksdma_check(district_geom, rows)
    _flood_check(district_geom, rows)
    _landcover_check(district_geom, rows)
    _count_checks(rows, errors)

    summary_path = cfg.output_path("layer_summary")
    utils.make_folder(summary_path.parent)
    table = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    table.to_csv(summary_path, index=False)

    overview_path = cfg.output_path("layers_overview")
    _make_overview(district, buffered, overview_path)

    LOGGER.info("Layer summary written to %s", summary_path)
    LOGGER.info("Overview figure written to %s", overview_path)

    if errors:
        formatted = "\n".join(f"  - {message}" for message in errors)
        raise RuntimeError(f"Task 2 inspection failed:\n{formatted}")


if __name__ == "__main__":
    main()
