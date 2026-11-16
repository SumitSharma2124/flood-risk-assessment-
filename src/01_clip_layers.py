"""Task 2: clip every raw layer to Alappuzha plus a 1 km buffer.

Run from the project root::

    python src/01_clip_layers.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from rasterio.transform import array_bounds
from rasterio.warp import transform_bounds

from src import config as cfg
from src import utils

LOGGER = utils.get_logger("clip_layers")

TASK_OUTPUTS = [
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


def _remove_if_exists(path: str | Path) -> None:
    path = Path(path)
    if path.exists():
        path.unlink()


def _output_bounds(profile: dict) -> list[float]:
    height = int(profile["height"])
    width = int(profile["width"])
    return list(array_bounds(height, width, profile["transform"]))


def _save_district() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Save the exact and buffered district polygons to ``district.gpkg``."""
    district = utils.load_district(crs=cfg.CRS_WORKING)
    if district.empty:
        raise RuntimeError("District polygon is empty")

    district_geom = district.geometry.union_all()
    buffered_geom = utils.buffer_geometry(
        district_geom,
        cfg.DISTRICT_BUFFER_M,
        crs=cfg.CRS_WORKING,
    )
    buffered = gpd.GeoDataFrame(
        {
            cfg.DISTRICT_NAME_COLUMN: [cfg.DISTRICT_NAME],
            "buffer_m": [cfg.DISTRICT_BUFFER_M],
        },
        geometry=[buffered_geom],
        crs=cfg.CRS_WORKING,
    )

    out_path = cfg.output_path("district")
    _remove_if_exists(out_path)
    utils.write_vector(district, out_path, layer="district")
    buffered.to_file(
        out_path,
        layer="district_buffered",
        driver="GPKG",
        append=True,
    )

    LOGGER.info(
        "District saved: exact %.1f km2, buffered %.1f km2",
        district_geom.area / 1e6,
        buffered_geom.area / 1e6,
    )
    return district, buffered


def _buffered_bbox_wgs84(buffered: gpd.GeoDataFrame) -> tuple[float, float, float, float]:
    bounds = transform_bounds(
        cfg.CRS_WORKING,
        cfg.CRS_GEOGRAPHIC,
        *buffered.total_bounds,
        densify_pts=21,
    )
    minx, miny, maxx, maxy = (float(value) for value in bounds)
    LOGGER.info(
        "Buffered bounding box (EPSG:4326): %.6f, %.6f, %.6f, %.6f",
        minx,
        miny,
        maxx,
        maxy,
    )
    return minx, miny, maxx, maxy


def _record_raster(
    stats: dict,
    key: str,
    source: Path,
    data,
    profile: dict,
) -> None:
    stats["rasters"][key] = {
        "source": str(source),
        "output": str(cfg.output_path(key)),
        "shape": [int(data.shape[0]), int(data.shape[1])],
        "dtype": str(data.dtype),
        "nodata": profile.get("nodata"),
        "bounds_wgs84": _output_bounds(profile),
    }
    LOGGER.info(
        "%s: %s x %s, dtype=%s, nodata=%s",
        key,
        data.shape[0],
        data.shape[1],
        data.dtype,
        profile.get("nodata"),
    )


def _clip_raster(
    key: str,
    source: Path,
    bounds: tuple[float, float, float, float],
    stats: dict,
) -> None:
    out_path = cfg.output_path(key)
    _remove_if_exists(out_path)
    data, profile = utils.clip_raster_to_bounds(
        source,
        bounds,
        out_path,
        crs=cfg.CRS_GEOGRAPHIC,
    )
    _record_raster(stats, key, source, data, profile)


def _clip_dem(
    bounds: tuple[float, float, float, float],
    stats: dict,
) -> None:
    """Clip the DEM, mosaicking neighbouring tiles if the box crosses an edge."""
    from rasterio.merge import merge

    tiles = sorted(cfg.DEM_RAW.parent.glob("Copernicus_DSM_COG_30_*_DEM.tif"))
    covering: list[Path] = []
    intersecting: list[Path] = []

    for tile in tiles:
        with rasterio.open(tile) as src:
            tile_bounds = src.bounds
        if (
            tile_bounds.right < bounds[0]
            or tile_bounds.left > bounds[2]
            or tile_bounds.top < bounds[1]
            or tile_bounds.bottom > bounds[3]
        ):
            continue
        intersecting.append(tile)
        if (
            tile_bounds.left <= bounds[0]
            and tile_bounds.right >= bounds[2]
            and tile_bounds.bottom <= bounds[1]
            and tile_bounds.top >= bounds[3]
        ):
            covering.append(tile)

    out_path = cfg.output_path("dem_clip")
    _remove_if_exists(out_path)

    if covering:
        source = covering[0]
        LOGGER.info("DEM: box is fully covered by %s", source.name)
        data, profile = utils.clip_raster_to_bounds(
            source,
            bounds,
            out_path,
            crs=cfg.CRS_GEOGRAPHIC,
        )
    else:
        if not intersecting:
            raise RuntimeError("No DEM tile overlaps the buffered district")
        LOGGER.info(
            "DEM: box crosses tile edges; mosaicking %s",
            ", ".join(tile.name for tile in intersecting),
        )
        sources = [rasterio.open(tile) for tile in intersecting]
        try:
            mosaic, transform = merge(
                sources,
                bounds=bounds,
                nodata=sources[0].nodata,
            )
            profile = sources[0].profile.copy()
            profile.update(
                driver="GTiff",
                count=1,
                height=mosaic.shape[1],
                width=mosaic.shape[2],
                transform=transform,
                compress="lzw",
                BIGTIFF="IF_SAFER",
            )
            data = mosaic[0]
            utils.write_raster(
                data,
                out_path,
                profile,
                nodata=profile.get("nodata"),
                compress="lzw",
            )
        finally:
            for src in sources:
                src.close()

    _record_raster(stats, "dem_clip", cfg.DEM_RAW, data, profile)


def _clip_vector(
    key: str,
    source: Path,
    buffered: gpd.GeoDataFrame,
    stats: dict,
    *,
    layer: str | None = None,
    out_layer: str | None = None,
    drop_duplicates: bool = False,
) -> gpd.GeoDataFrame:
    gdf = utils.read_vector(source, layer=layer) if layer else utils.read_vector(source)
    if gdf.crs is None:
        raise RuntimeError(f"Vector layer {source} has no CRS")

    source_count = len(gdf)
    clipped = utils.clip_vector_to_polygon(gdf, buffered)
    duplicates = 0
    if drop_duplicates:
        duplicates = int(clipped.duplicated().sum())
        if duplicates:
            clipped = clipped.drop_duplicates().reset_index(drop=True)

    clipped = utils.reproject_vector(clipped, cfg.CRS_WORKING)
    if clipped.empty:
        raise RuntimeError(f"Clipped vector layer {key} is empty")

    out_path = cfg.output_path(key)
    _remove_if_exists(out_path)
    layer_name = out_layer or key.replace("_clip", "")
    utils.write_vector(clipped, out_path, layer=layer_name)

    stats["vectors"][key] = {
        "source": str(source),
        "output": str(out_path),
        "source_features": source_count,
        "clipped_features": len(clipped),
        "duplicates_dropped": duplicates,
    }
    LOGGER.info(
        "%s: %s -> %s features (%s duplicates dropped)",
        key,
        f"{source_count:,}",
        f"{len(clipped):,}",
        f"{duplicates:,}",
    )
    return clipped


def _clip_buildings(
    bounds: tuple[float, float, float, float],
    buffered: gpd.GeoDataFrame,
    stats: dict,
) -> gpd.GeoDataFrame:
    LOGGER.info("Reading the buildings CSV in chunks (this can take a few minutes)")
    buffered_geographic = buffered.to_crs(cfg.CRS_GEOGRAPHIC)
    buffered_geom = buffered_geographic.geometry.union_all()
    counts: dict[str, int] = {}
    buildings, total_rows = utils.read_buildings_csv(
        cfg.BUILDINGS_RAW,
        bounds=bounds,
        chunksize=cfg.BUILDINGS_CHUNK_SIZE,
        clip_geometry=buffered_geom,
        counts=counts,
        logger=LOGGER,
    )
    LOGGER.info(
        "Buildings CSV: %s rows total, %s inside bbox, %s intersect the "
        "buffered district",
        f"{total_rows:,}",
        f"{counts.get('bbox_rows', 0):,}",
        f"{len(buildings):,}",
    )

    geometries = buildings.geometry.to_numpy()
    fully_inside = shapely.within(geometries, buffered_geom)
    clipped_geometries = geometries.copy()
    if not np.all(fully_inside):
        crossing = ~fully_inside
        clipped_geometries[crossing] = shapely.intersection(
            geometries[crossing],
            buffered_geom,
        )

    polygonal = np.isin(shapely.get_type_id(clipped_geometries), [3, 6])
    buildings = buildings.loc[polygonal].copy()
    buildings["geometry"] = clipped_geometries[polygonal]
    buildings = buildings.set_geometry("geometry")
    clipped = buildings.to_crs(cfg.CRS_WORKING)
    if clipped.empty:
        raise RuntimeError("Clipped buildings layer is empty")

    out_path = cfg.output_path("buildings_clip")
    _remove_if_exists(out_path)
    utils.write_vector(clipped, out_path, layer="buildings")

    stats["buildings"] = {
        "source": str(cfg.BUILDINGS_RAW),
        "output": str(out_path),
        "csv_rows": int(total_rows),
        "bbox_rows": int(counts.get("bbox_rows", 0)),
        "clip_candidates": int(counts.get("clip_rows", len(buildings))),
        "clipped_features": len(clipped),
    }
    LOGGER.info("buildings_clip: %s features kept", f"{len(clipped):,}")
    return clipped


def _facility_type(row: pd.Series) -> str:
    for column in (
        "amenity",
        "health_facility_level",
        "natural",
        "waterway",
        "power",
        "public_transport",
    ):
        value = row.get(column)
        if pd.notna(value) and str(value).strip():
            return str(value).strip()
    return "unknown"


def _clip_facilities(
    buffered: gpd.GeoDataFrame,
    stats: dict,
) -> gpd.GeoDataFrame:
    layers = utils.list_gpkg_layers(cfg.FACILITIES_RAW)
    LOGGER.info("Facilities layers: %s", layers)

    frames: list[gpd.GeoDataFrame] = []
    source_total = 0
    for layer_name, geometry_type in layers:
        gdf = utils.read_vector(cfg.FACILITIES_RAW, layer=layer_name)
        source_total += len(gdf)
        clipped = utils.clip_vector_to_polygon(gdf, buffered)
        if clipped.empty:
            LOGGER.warning("Facilities layer %s is empty after clipping", layer_name)
            continue

        clipped = utils.reproject_vector(clipped, cfg.CRS_WORKING)
        clipped["type"] = layer_name
        clipped["source_geometry_type"] = geometry_type
        clipped["facility_type"] = [
            _facility_type(row) for _, row in clipped.iterrows()
        ]
        frames.append(clipped)

    if not frames:
        raise RuntimeError("No facilities remain after clipping")

    merged = gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True, sort=False),
        geometry="geometry",
        crs=cfg.CRS_WORKING,
    )

    out_path = cfg.output_path("facilities_clip")
    _remove_if_exists(out_path)
    utils.write_vector(
        merged,
        out_path,
        layer="facilities",
        geometry_type="Unknown",
    )

    stats["vectors"]["facilities_clip"] = {
        "source": str(cfg.FACILITIES_RAW),
        "output": str(out_path),
        "source_features": source_total,
        "clipped_features": len(merged),
        "duplicates_dropped": 0,
        "layers": [name for name, _ in layers],
    }
    LOGGER.info(
        "facilities_clip: %s -> %s features",
        f"{source_total:,}",
        f"{len(merged):,}",
    )
    return merged


def _clip_villages(
    district: gpd.GeoDataFrame,
    stats: dict,
) -> gpd.GeoDataFrame:
    villages = utils.read_vector(cfg.VILLAGES_RAW)
    if villages.crs is None:
        raise RuntimeError("Village layer has no CRS")

    district_geographic = district.to_crs(cfg.CRS_GEOGRAPHIC)
    district_geom = district_geographic.geometry.union_all()
    villages_geographic = (
        villages
        if villages.crs == cfg.CRS_GEOGRAPHIC
        else villages.to_crs(cfg.CRS_GEOGRAPHIC)
    )
    selected = villages_geographic[
        villages_geographic.intersects(district_geom)
    ].copy()
    selected = utils.reproject_vector(selected, cfg.CRS_WORKING)
    if selected.empty:
        raise RuntimeError("No villages touch the district")

    out_path = cfg.output_path("villages_clip")
    _remove_if_exists(out_path)
    utils.write_vector(selected, out_path, layer="villages")

    stats["vectors"]["villages_clip"] = {
        "source": str(cfg.VILLAGES_RAW),
        "output": str(out_path),
        "source_features": len(villages),
        "clipped_features": len(selected),
        "duplicates_dropped": 0,
    }
    LOGGER.info(
        "villages_clip: %s -> %s whole villages",
        f"{len(villages):,}",
        f"{len(selected):,}",
    )
    return selected


def main() -> None:
    cfg.check_input_paths()
    LOGGER.info("Starting Task 2 clipping")

    district, buffered = _save_district()
    bounds = _buffered_bbox_wgs84(buffered)

    stats: dict = {
        "district": {
            "name": cfg.DISTRICT_NAME,
            "area_km2": district.geometry.union_all().area / 1e6,
            "buffered_area_km2": buffered.geometry.union_all().area / 1e6,
            "buffer_m": cfg.DISTRICT_BUFFER_M,
            "buffered_bounds_wgs84": list(bounds),
        },
        "rasters": {},
        "vectors": {},
        "buildings": {},
    }

    _clip_dem(bounds, stats)
    _clip_raster("landcover_clip", cfg.LANDCOVER_RAW, bounds, stats)
    _clip_raster("ksdma_100yr_clip", cfg.KSDMA_SCENARIO_RAW, bounds, stats)
    _clip_raster("flood2018_clip", cfg.FLOOD_2018_RAW, bounds, stats)
    _clip_raster("pop2018_clip", cfg.POPULATION_2018_RAW, bounds, stats)

    _clip_vector(
        "roads_clip",
        cfg.ROADS_RAW,
        buffered,
        stats,
        layer="Alappuzha_roads",
        out_layer="roads",
        drop_duplicates=True,
    )
    _clip_vector(
        "waterways_clip",
        cfg.WATERWAYS_RAW,
        buffered,
        stats,
        layer="Alappuzha_waterways",
        out_layer="waterways",
        drop_duplicates=True,
    )
    _clip_buildings(bounds, buffered, stats)
    _clip_facilities(buffered, stats)
    _clip_villages(district, stats)

    stats_path = cfg.output_path("clip_stats")
    utils.make_folder(stats_path.parent)
    stats_path.write_text(
        json.dumps(stats, indent=2, default=str),
        encoding="utf-8",
    )

    missing = [key for key in TASK_OUTPUTS if not cfg.output_path(key).exists()]
    if missing:
        raise RuntimeError(f"Missing Task 2 outputs: {missing}")

    LOGGER.info("Task 2 clipping complete; stats written to %s", stats_path)


if __name__ == "__main__":
    main()
