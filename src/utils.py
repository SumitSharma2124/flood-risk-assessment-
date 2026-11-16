"""Shared helpers for the Alappuzha flood-risk assessment.

The functions here keep raster/vector handling consistent across all task
scripts: same CRS, same 90 m grid, same NoData convention and the same output
paths (defined in :mod:`src.config`).

Typical use::

    from src import config as cfg
    from src import utils

    district = utils.load_district()
    data, profile, nodata = utils.read_raster(cfg.DEM_RAW)
"""

from __future__ import annotations

import logging
import random
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

try:  # allows both "python -m src.script" and "python src/script.py"
    from . import config as cfg
except ImportError:  # pragma: no cover - direct execution fallback
    import config as cfg  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------


def make_folder(path: str | Path) -> Path:
    """Create a folder (and parents) if it does not exist and return it."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_logger(name: str = "flood_risk", level: int = logging.INFO) -> logging.Logger:
    """Return a console logger without adding duplicate handlers."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def set_seed(seed: int = cfg.RANDOM_SEED) -> int:
    """Seed Python, NumPy and (if installed) PyTorch."""
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return seed

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


# ---------------------------------------------------------------------------
# Reading and writing
# ---------------------------------------------------------------------------


def read_raster(
    path: str | Path, band: int = 1
) -> tuple[np.ndarray, dict[str, Any], float | int | None]:
    """Read one raster band and return ``(data, profile, nodata)``."""
    import rasterio

    with rasterio.open(path) as src:
        data = src.read(band)
        profile = src.profile.copy()
        nodata = src.nodata
    return data, profile, nodata


def write_raster(
    data: np.ndarray,
    out_path: str | Path,
    template: str | Path | Mapping[str, Any],
    *,
    nodata: float | int | None = None,
    dtype: str | None = None,
    compress: str | None = "lzw",
    **profile_updates: Any,
) -> Path:
    """Write ``data`` to ``out_path`` using another raster's profile.

    ``template`` may be a raster path or a rasterio profile dict.  The output
    is always a single-band GeoTIFF.
    """
    import rasterio

    out_path = Path(out_path)
    make_folder(out_path.parent)

    if isinstance(template, (str, Path)):
        with rasterio.open(template) as src:
            profile = src.profile.copy()
    else:
        profile = dict(template)

    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError(f"Expected a 2-D array, got shape {data.shape}")

    profile.update(
        driver="GTiff",
        count=1,
        height=data.shape[0],
        width=data.shape[1],
    )
    if dtype is not None:
        profile["dtype"] = dtype
    if nodata is not None:
        profile["nodata"] = nodata
    if compress is not None:
        profile["compress"] = compress
    profile.update(profile_updates)

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data.astype(profile["dtype"], copy=False), 1)

    return out_path


def write_multiband_raster(
    data: np.ndarray,
    out_path: str | Path,
    template: str | Path | Mapping[str, Any],
    *,
    band_names: Sequence[str] | None = None,
    nodata: float | int | None = cfg.NODATA,
    dtype: str = "float32",
    compress: str | None = "lzw",
    **profile_updates: Any,
) -> Path:
    """Write a multi-band raster and optionally name its bands."""
    import rasterio

    array = np.asarray(data)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3-D array, got shape {array.shape}")
    if band_names is not None and len(band_names) != array.shape[0]:
        raise ValueError(
            f"Got {len(band_names)} band names for {array.shape[0]} bands"
        )

    grid = _coerce_grid(template)
    out_path = Path(out_path)
    make_folder(out_path.parent)

    profile = grid["profile"].copy()
    profile.update(
        driver="GTiff",
        count=int(array.shape[0]),
        height=int(array.shape[1]),
        width=int(array.shape[2]),
        dtype=np.dtype(dtype).name,
        nodata=nodata,
        compress=compress,
        BIGTIFF="IF_SAFER",
    )
    if compress is None:
        profile.pop("compress", None)
    profile.update(profile_updates)

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(array.astype(dtype, copy=False))
        if band_names is not None:
            for index, name in enumerate(band_names, start=1):
                dst.set_band_description(index, name)

    return out_path


def read_vector(path: str | Path, **kwargs: Any):
    """Read a vector layer as a GeoDataFrame."""
    import geopandas as gpd

    return gpd.read_file(path, **kwargs)


def list_gpkg_layers(path: str | Path) -> list[tuple[str, str]]:
    """Return ``(layer_name, geometry_type)`` pairs from a GeoPackage."""
    import pyogrio

    layers = pyogrio.list_layers(path)
    return [(str(name), str(geometry_type)) for name, geometry_type in layers]


def filter_bbox(
    gdf: Any,
    bounds: Sequence[float],
    *,
    crs: Any | None = None,
):
    """Keep features whose geometry intersects a ``(minx, miny, maxx, maxy)`` box."""
    if crs is not None and gdf.crs is not None and crs != gdf.crs:
        from rasterio.warp import transform_bounds

        bounds = transform_bounds(crs, gdf.crs, *bounds, densify_pts=21)

    minx, miny, maxx, maxy = (float(value) for value in bounds)
    return gdf.cx[minx:maxx, miny:maxy]


def iter_buildings_csv(
    path: str | Path = cfg.BUILDINGS_RAW,
    *,
    bounds: Sequence[float] | None = None,
    chunksize: int = cfg.BUILDINGS_CHUNK_SIZE,
    confidence_threshold: float | None = None,
    usecols: Sequence[str] | None = None,
    lat_col: str = cfg.BUILDINGS_LAT_COLUMN,
    lon_col: str = cfg.BUILDINGS_LON_COLUMN,
    **read_csv_kwargs: Any,
):
    """Yield chunks of the buildings CSV, optionally filtered by bounding box.

    The CSV is never loaded in full.  ``bounds`` must be in EPSG:4326 and is
    applied to the latitude/longitude columns before geometry is parsed.
    """
    import pandas as pd

    if usecols is None:
        usecols = [
            lat_col,
            lon_col,
            "area_in_meters",
            "confidence",
            cfg.BUILDINGS_GEOMETRY_COLUMN,
            "full_plus_code",
        ]

    for chunk in pd.read_csv(
        path,
        chunksize=chunksize,
        usecols=list(usecols),
        **read_csv_kwargs,
    ):
        if bounds is not None:
            minx, miny, maxx, maxy = (float(value) for value in bounds)
            chunk = chunk[
                (chunk[lon_col] >= minx)
                & (chunk[lon_col] <= maxx)
                & (chunk[lat_col] >= miny)
                & (chunk[lat_col] <= maxy)
            ]

        if (
            confidence_threshold is not None
            and "confidence" in chunk.columns
        ):
            chunk = chunk[chunk["confidence"] >= confidence_threshold]

        if not chunk.empty:
            yield chunk


def read_buildings_csv(
    path: str | Path = cfg.BUILDINGS_RAW,
    *,
    bounds: Sequence[float] | None = None,
    chunksize: int = cfg.BUILDINGS_CHUNK_SIZE,
    confidence_threshold: float | None = None,
    usecols: Sequence[str] | None = None,
    lat_col: str = cfg.BUILDINGS_LAT_COLUMN,
    lon_col: str = cfg.BUILDINGS_LON_COLUMN,
    geometry_col: str = cfg.BUILDINGS_GEOMETRY_COLUMN,
    crs: Any = cfg.CRS_GEOGRAPHIC,
    clip_geometry: Any | None = None,
    counts: dict[str, int] | None = None,
    progress_every: int = 10,
    logger: logging.Logger | None = None,
    **read_csv_kwargs: Any,
) -> tuple[Any, int]:
    """Read buildings from the CSV in chunks and return ``(gdf, total_rows)``.

    Only rows inside ``bounds`` are converted from WKT to polygons, which keeps
    memory use low for the 4 GB, Kerala-wide source file.  If ``clip_geometry``
    is given (in the same CRS as the CSV coordinates), only buildings that
    intersect it are kept before the WKT is parsed.
    """
    import geopandas as gpd
    import pandas as pd
    import shapely

    if usecols is None:
        usecols = [
            lat_col,
            lon_col,
            "area_in_meters",
            "confidence",
            geometry_col,
            "full_plus_code",
        ]

    frames: list[Any] = []
    total_rows = 0
    bbox_rows = 0
    clip_rows = 0
    chunk_number = 0

    for chunk in pd.read_csv(
        path,
        chunksize=chunksize,
        usecols=list(usecols),
        **read_csv_kwargs,
    ):
        chunk_number += 1
        total_rows += len(chunk)

        if bounds is not None:
            minx, miny, maxx, maxy = (float(value) for value in bounds)
            chunk = chunk[
                (chunk[lon_col] >= minx)
                & (chunk[lon_col] <= maxx)
                & (chunk[lat_col] >= miny)
                & (chunk[lat_col] <= maxy)
            ]

        if (
            confidence_threshold is not None
            and "confidence" in chunk.columns
        ):
            chunk = chunk[chunk["confidence"] >= confidence_threshold]

        bbox_rows += len(chunk)

        if not chunk.empty:
            geometries = shapely.from_wkt(chunk[geometry_col].to_numpy())
            if clip_geometry is not None:
                keep = shapely.intersects(geometries, clip_geometry)
                clip_rows += int(keep.sum())
                if not np.all(keep):
                    chunk = chunk.loc[keep].reset_index(drop=True)
                    geometries = geometries[keep]
            else:
                clip_rows += len(chunk)

            if not chunk.empty:
                frame = chunk.drop(columns=[geometry_col], errors="ignore").copy()
                frame["geometry"] = geometries
                frames.append(frame)

        if logger is not None and progress_every and chunk_number % progress_every == 0:
            logger.info(
                "Buildings CSV: %s rows read, %s in bbox, %s kept so far",
                f"{total_rows:,}",
                f"{bbox_rows:,}",
                f"{sum(len(frame) for frame in frames):,}",
            )

    if counts is not None:
        counts.update(
            {
                "csv_rows": total_rows,
                "bbox_rows": bbox_rows,
                "clip_rows": clip_rows,
            }
        )

    if not frames:
        empty = gpd.GeoDataFrame(
            {
                lat_col: pd.Series(dtype="float64"),
                lon_col: pd.Series(dtype="float64"),
                "geometry": gpd.GeoSeries([], dtype="geometry"),
            },
            geometry="geometry",
            crs=crs,
        )
        return empty, total_rows

    combined = pd.concat(frames, ignore_index=True)
    geometries = combined["geometry"].to_numpy()
    valid = ~shapely.is_missing(geometries) & ~shapely.is_empty(geometries)
    if not np.all(valid):
        combined = combined.loc[valid].reset_index(drop=True)
        geometries = geometries[valid]

    combined = combined.drop(columns=["geometry"], errors="ignore")
    gdf = gpd.GeoDataFrame(combined, geometry=geometries, crs=crs)
    return gdf, total_rows


def write_vector(
    gdf: Any,
    out_path: str | Path,
    driver: str | None = None,
    **kwargs: Any,
) -> Path:
    """Write a GeoDataFrame, inferring the driver from the file extension."""
    out_path = Path(out_path)
    make_folder(out_path.parent)

    if driver is None:
        suffix = out_path.suffix.lower()
        driver = {
            ".gpkg": "GPKG",
            ".geojson": "GeoJSON",
            ".json": "GeoJSON",
            ".shp": "ESRI Shapefile",
        }.get(suffix)

    if driver is None:
        gdf.to_file(out_path, **kwargs)
    else:
        gdf.to_file(out_path, driver=driver, **kwargs)
    return out_path


# ---------------------------------------------------------------------------
# CRS and district
# ---------------------------------------------------------------------------


def reproject_vector(gdf: Any, crs: Any):
    """Reproject a GeoDataFrame to ``crs`` (no-op if it is already there)."""
    if gdf.crs is None:
        return gdf.set_crs(crs)
    return gdf.to_crs(crs)


def load_district(
    name: str = cfg.DISTRICT_NAME,
    crs: Any = cfg.CRS_WORKING,
    *,
    path: str | Path = cfg.DISTRICTS_RAW,
    name_column: str = cfg.DISTRICT_NAME_COLUMN,
    as_geometry: bool = False,
):
    """Load one district polygon by name in the requested CRS.

    Returns a one-row GeoDataFrame, or a single shapely geometry when
    ``as_geometry=True``.
    """
    gdf = read_vector(path)
    if name_column not in gdf.columns:
        raise KeyError(
            f"Column '{name_column}' not found in {path}. "
            f"Available columns: {list(gdf.columns)}"
        )

    matches = gdf[
        gdf[name_column].astype(str).str.strip().str.casefold() == name.strip().casefold()
    ]
    if matches.empty:
        available = sorted(gdf[name_column].dropna().astype(str).unique())
        close = [item for item in available if name.casefold() in item.casefold()]
        raise ValueError(
            f"District '{name}' not found in column '{name_column}'. "
            f"Possible matches: {close or available[:10]}"
        )

    district = reproject_vector(matches.copy(), crs)
    if as_geometry:
        return district.geometry.union_all()
    return district


def _as_geometry_and_crs(
    obj: Any,
) -> tuple[Any, Any]:
    """Coerce a geometry, GeoDataFrame, GeoSeries or path to (geometry, crs)."""
    from shapely.geometry import shape

    if isinstance(obj, (str, Path)):
        gdf = read_vector(obj)
        return gdf.geometry.union_all(), gdf.crs

    if isinstance(obj, Mapping) and "type" in obj:
        return shape(obj), None

    if hasattr(obj, "geometry") and hasattr(obj, "crs"):
        return obj.geometry.union_all(), obj.crs

    if hasattr(obj, "union_all"):
        return obj.union_all(), getattr(obj, "crs", None)

    if hasattr(obj, "geom_type"):
        return obj, None

    raise TypeError(f"Cannot interpret {type(obj)!r} as a geometry layer")


def _buffer_geometry(geom: Any, crs: Any, buffer_m: float) -> tuple[Any, Any]:
    """Buffer a geometry by metres, projecting to the working CRS if needed."""
    from pyproj import CRS, Transformer
    from shapely.ops import transform as shapely_transform

    if not buffer_m:
        return geom, crs

    if crs is None:
        crs = cfg.CRS_WORKING

    parsed = CRS.from_user_input(crs)
    if parsed.is_projected:
        return geom.buffer(buffer_m), crs

    transformer = Transformer.from_crs(crs, cfg.CRS_WORKING, always_xy=True)
    geom_working = shapely_transform(transformer.transform, geom)
    return geom_working.buffer(buffer_m), cfg.CRS_WORKING


def buffer_geometry(
    geometry: Any,
    buffer_m: float = cfg.DISTRICT_BUFFER_M,
    *,
    crs: Any = cfg.CRS_WORKING,
) -> Any:
    """Buffer a geometry by metres and return it in the working CRS.

    Accepts a shapely geometry, GeoDataFrame, GeoSeries or vector path.
    """
    from pyproj import CRS, Transformer
    from shapely.ops import transform as shapely_transform

    geom, geom_crs = _as_geometry_and_crs(geometry)
    if geom_crs is None:
        geom_crs = crs

    if CRS.from_user_input(geom_crs) != CRS.from_user_input(crs):
        transformer = Transformer.from_crs(geom_crs, crs, always_xy=True)
        geom = shapely_transform(transformer.transform, geom)

    return geom.buffer(buffer_m)


# ---------------------------------------------------------------------------
# Clipping
# ---------------------------------------------------------------------------


def clip_raster_to_polygon(
    src_path: str | Path,
    polygon: Any,
    out_path: str | Path | None = None,
    *,
    all_touched: bool = False,
    nodata: float | int | None = None,
    buffer_m: float = 0.0,
    band: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Clip a raster to a polygon using a windowed read.

    Only the window covering the polygon is read, so very large tiles (for
    example the 36,000 x 36,000 WorldCover tile) are never loaded in full.
    Cells outside the polygon are set to ``nodata`` (default: the source
    raster's NoData, else :data:`src.config.NODATA`).
    """
    import rasterio
    from rasterio.features import geometry_mask
    from rasterio.warp import transform_geom
    from rasterio.windows import Window, from_bounds
    from shapely.geometry import mapping, shape

    geom, geom_crs = _as_geometry_and_crs(polygon)
    geom, geom_crs = _buffer_geometry(geom, geom_crs, buffer_m)

    with rasterio.open(src_path) as src:
        if geom_crs is not None and src.crs is not None and geom_crs != src.crs:
            geom = shape(
                transform_geom(
                    geom_crs,
                    src.crs,
                    mapping(geom),
                )
            )

        window = from_bounds(*geom.bounds, transform=src.transform)
        window = window.round_offsets().round_lengths()
        window = window.intersection(Window(0, 0, src.width, src.height))
        if window.width <= 0 or window.height <= 0:
            raise ValueError(
                f"Polygon does not overlap raster {src_path} "
                f"(bounds {tuple(src.bounds)})"
            )

        data = src.read(band, window=window)
        transform = src.window_transform(window)
        profile = src.profile.copy()

        requested_nodata = (
            nodata if nodata is not None else src.nodata if src.nodata is not None else cfg.NODATA
        )
        out_nodata = _safe_nodata(data.dtype, requested_nodata)

        inside = geometry_mask(
            [geom],
            out_shape=data.shape,
            transform=transform,
            invert=True,
            all_touched=all_touched,
        )
        clipped = data.copy()
        clipped[~inside] = out_nodata

        profile.update(
            driver="GTiff",
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            transform=transform,
            nodata=out_nodata,
            compress="lzw",
        )

    if out_path is not None:
        write_raster(
            clipped,
            out_path,
            profile,
            nodata=out_nodata,
            compress="lzw",
        )

    return clipped, profile


def clip_vector_to_polygon(
    vector: Any,
    polygon: Any,
    out_path: str | Path | None = None,
    *,
    buffer_m: float = 0.0,
):
    """Clip a vector layer to a polygon and optionally write the result."""
    import geopandas as gpd
    from rasterio.warp import transform_geom
    from shapely.geometry import mapping, shape

    gdf = read_vector(vector) if isinstance(vector, (str, Path)) else vector.copy()
    geom, geom_crs = _as_geometry_and_crs(polygon)
    geom, geom_crs = _buffer_geometry(geom, geom_crs, buffer_m)

    if (
        geom_crs is not None
        and gdf.crs is not None
        and geom_crs != gdf.crs
    ):
        geom = shape(
            transform_geom(
                geom_crs,
                gdf.crs,
                mapping(geom),
            )
        )

    clipped = gpd.clip(gdf, geom)

    if out_path is not None:
        write_vector(clipped, out_path)

    return clipped


def clip_raster_to_bounds(
    src_path: str | Path,
    bounds: Sequence[float],
    out_path: str | Path | None = None,
    *,
    crs: Any | None = None,
    band: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Crop a raster to a bounding box without masking by a polygon.

    Only the window covering ``bounds`` is read.  The source data type and
    NoData value are preserved.  ``bounds`` are assumed to be in the source
    CRS unless ``crs`` is given, in which case they are transformed first.
    """
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import Window, from_bounds

    with rasterio.open(src_path) as src:
        if crs is not None and src.crs is not None and crs != src.crs:
            bounds = transform_bounds(crs, src.crs, *bounds, densify_pts=21)

        window = from_bounds(*bounds, transform=src.transform)
        window = window.round_offsets().round_lengths()
        window = window.intersection(Window(0, 0, src.width, src.height))
        if window.width <= 0 or window.height <= 0:
            raise ValueError(
                f"Bounding box does not overlap raster {src_path} "
                f"(bounds {tuple(src.bounds)})"
            )

        data = src.read(band, window=window)
        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            count=1,
            height=data.shape[0],
            width=data.shape[1],
            transform=src.window_transform(window),
            compress="lzw",
            BIGTIFF="IF_SAFER",
        )

    if out_path is not None:
        write_raster(
            data,
            out_path,
            profile,
            nodata=profile.get("nodata"),
            compress="lzw",
        )

    return data, profile


# ---------------------------------------------------------------------------
# Grid alignment
# ---------------------------------------------------------------------------


def _coerce_grid(template: Any) -> dict[str, Any]:
    """Return a grid dict from a profile, grid dict or raster path."""
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import array_bounds

    if isinstance(template, (str, Path)):
        with rasterio.open(template) as src:
            profile = src.profile.copy()
            bounds = src.bounds
    elif isinstance(template, Mapping):
        profile = dict(template.get("profile", template))
        if "transform" not in profile or "width" not in profile:
            raise ValueError("Grid mapping must contain transform and width/height")
        bounds = None
    else:
        raise TypeError(f"Cannot interpret {type(template)!r} as a grid")

    profile.setdefault("count", 1)
    transform = profile["transform"]
    width = int(profile["width"])
    height = int(profile["height"])

    if bounds is None:
        left, bottom, right, top = array_bounds(height, width, transform)
    else:
        left, bottom, right, top = bounds

    return {
        "crs": CRS.from_user_input(profile["crs"]),
        "transform": transform,
        "width": width,
        "height": height,
        "profile": profile,
        "bounds": (left, bottom, right, top),
    }


def load_grid(template: Any = cfg.GRID_PATH) -> dict[str, Any]:
    """Return the grid definition stored in a raster or mapping."""
    return _coerce_grid(template)


def build_grid(
    bounds: Sequence[float] | None = None,
    *,
    district: Any | None = None,
    resolution: float = cfg.GRID_RESOLUTION,
    crs: Any = cfg.CRS_WORKING,
    buffer_m: float = cfg.DISTRICT_BUFFER_M,
    dtype: str = "float32",
) -> dict[str, Any]:
    """Build the common analysis grid from an extent, resolution and CRS.

    Provide either ``bounds`` (minx, miny, maxx, maxy) or a ``district``
    (geometry, GeoDataFrame or path).  The grid is snapped outwards to whole
    resolution steps so that every script gets identical cell edges.
    """
    from math import ceil, floor

    import geopandas as gpd
    from rasterio.crs import CRS
    from rasterio.transform import from_origin

    crs = CRS.from_user_input(crs)

    if district is not None:
        geom, geom_crs = _as_geometry_and_crs(district)
        if geom_crs is not None and geom_crs != crs:
            geometry = gpd.GeoSeries([geom], crs=geom_crs).to_crs(crs).iloc[0]
        else:
            geometry = geom
        bounds = geometry.bounds
    elif bounds is None:
        bounds = load_district(crs=crs, as_geometry=True).bounds

    minx, miny, maxx, maxy = (float(value) for value in bounds)
    minx -= buffer_m
    miny -= buffer_m
    maxx += buffer_m
    maxy += buffer_m

    left = floor(minx / resolution) * resolution
    bottom = floor(miny / resolution) * resolution
    right = ceil(maxx / resolution) * resolution
    top = ceil(maxy / resolution) * resolution

    width = int(round((right - left) / resolution))
    height = int(round((top - bottom) / resolution))
    transform = from_origin(left, top, resolution, resolution)

    profile = {
        "driver": "GTiff",
        "dtype": dtype,
        "count": 1,
        "crs": crs,
        "transform": transform,
        "width": width,
        "height": height,
        "nodata": cfg.NODATA,
        "compress": "lzw",
        "BIGTIFF": "IF_SAFER",
    }

    return {
        "crs": crs,
        "transform": transform,
        "width": width,
        "height": height,
        "resolution": resolution,
        "bounds": (left, bottom, right, top),
        "profile": profile,
    }


def save_grid(grid: Mapping[str, Any], out_path: str | Path = cfg.GRID_PATH) -> Path:
    """Save an all-zero raster that records the common grid definition."""
    data = np.zeros((int(grid["height"]), int(grid["width"])), dtype="float32")
    return write_raster(
        data,
        out_path,
        grid["profile"],
        nodata=cfg.NODATA,
        dtype="float32",
    )


def _grid_source_window(src: Any, grid: Mapping[str, Any]):
    """Return the source window that fully covers the target grid."""
    from rasterio.warp import transform_bounds
    from rasterio.windows import Window, from_bounds

    src_bounds = transform_bounds(
        grid["crs"],
        src.crs,
        *grid["bounds"],
        densify_pts=21,
    )
    window = from_bounds(*src_bounds, transform=src.transform)

    col_off = max(0, int(np.floor(window.col_off)))
    row_off = max(0, int(np.floor(window.row_off)))
    col_end = min(src.width, int(np.ceil(window.col_off + window.width)))
    row_end = min(src.height, int(np.ceil(window.row_off + window.height)))
    window = Window(
        col_off,
        row_off,
        col_end - col_off,
        row_end - row_off,
    )
    if window.width <= 0 or window.height <= 0:
        raise ValueError(f"Raster {src.name} does not overlap the grid")
    return window


def read_raster_window(
    src_path: str | Path,
    template: Any = cfg.GRID_PATH,
    *,
    band: int = 1,
) -> tuple[np.ndarray, Any, Any, float | int | None]:
    """Read the source window that covers the target grid."""
    import rasterio

    grid = _coerce_grid(template)
    with rasterio.open(src_path) as src:
        if src.crs is None:
            raise ValueError(f"Raster {src_path} has no CRS")
        window = _grid_source_window(src, grid)
        data = src.read(band, window=window)
        transform = src.window_transform(window)
        crs = src.crs
        nodata = src.nodata
    return data, transform, crs, nodata


def align_raster(
    src_path: str | Path,
    template: Any = cfg.GRID_PATH,
    out_path: str | Path | None = None,
    *,
    method: str | None = None,
    resampling: Any | None = None,
    dtype: str | None = None,
    nodata: float | int | None = None,
    preserve_sum: bool = False,
    band: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Resample a raster onto the common grid.

    The resampling method defaults to nearest for integer rasters (classes)
    and bilinear for continuous float rasters.  Pass ``preserve_sum=True``
    (or ``method="sum"``) for population counts.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    grid = _coerce_grid(template)

    with rasterio.open(src_path) as src:
        if src.crs is None:
            raise ValueError(f"Raster {src_path} has no CRS")
        window = _grid_source_window(src, grid)
        source = src.read(band, window=window)
        src_transform = src.window_transform(window)
        src_crs = src.crs
        src_nodata = src.nodata

    if resampling is not None:
        method_enum = resampling
    elif method is not None:
        method_enum = Resampling[method.lower()]
    elif preserve_sum:
        method_enum = Resampling.sum
    elif np.issubdtype(source.dtype, np.integer):
        method_enum = Resampling.nearest
    else:
        method_enum = Resampling.bilinear

    dst_dtype = np.dtype(dtype) if dtype is not None else source.dtype

    is_sum = method_enum == Resampling.sum
    if is_sum:
        # Sum resampling needs a zero background so overlapping source cells
        # add up correctly. Valid-source cells are tracked separately below.
        fill = 0
    else:
        fill = nodata if nodata is not None else cfg.NODATA
    dst_nodata = _safe_nodata(dst_dtype, fill)

    destination = np.full(
        (grid["height"], grid["width"]),
        fill,
        dtype=dst_dtype,
    )

    reproject(
        source=source,
        destination=destination,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=src_nodata,
        dst_transform=grid["transform"],
        dst_crs=grid["crs"],
        dst_nodata=dst_nodata,
        resampling=method_enum,
    )

    if is_sum:
        # Keep genuine zeros (for example unpopulated cells) but replace areas
        # with no source coverage by the standard processed NoData value.
        if src_nodata is not None:
            source_valid = source != src_nodata
            if np.issubdtype(source.dtype, np.floating):
                source_valid &= np.isfinite(source)
        elif np.issubdtype(source.dtype, np.floating):
            source_valid = np.isfinite(source)
        else:
            source_valid = np.ones(source.shape, dtype=bool)

        valid_destination = np.zeros(
            (grid["height"], grid["width"]),
            dtype="uint8",
        )
        reproject(
            source=source_valid.astype("uint8"),
            destination=valid_destination,
            src_transform=src_transform,
            src_crs=src_crs,
            src_nodata=None,
            dst_transform=grid["transform"],
            dst_crs=grid["crs"],
            dst_nodata=0,
            resampling=Resampling.nearest,
        )

        dst_nodata = _safe_nodata(dst_dtype, cfg.NODATA)
        destination[valid_destination == 0] = dst_nodata

    profile = grid["profile"].copy()
    profile.update(
        dtype=dst_dtype.name,
        count=1,
        nodata=dst_nodata,
    )

    if out_path is not None:
        write_raster(
            destination,
            out_path,
            profile,
            nodata=dst_nodata,
            dtype=dst_dtype.name,
        )

    return destination, profile


def reproject_array(
    source: np.ndarray,
    src_transform: Any,
    src_crs: Any,
    template: Any = cfg.GRID_PATH,
    out_path: str | Path | None = None,
    *,
    method: str | Any = "nearest",
    src_nodata: float | int | None = None,
    dst_nodata: float | int | None = cfg.NODATA,
    dtype: str | None = None,
    fill: float | int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reproject an in-memory array onto a grid defined by ``template``."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    grid = _coerce_grid(template)
    array = np.asarray(source)
    dst_dtype = np.dtype(dtype or array.dtype)
    resampling = (
        Resampling[method.lower()] if isinstance(method, str) else method
    )

    if fill is None:
        fill = dst_nodata if dst_nodata is not None else 0
    destination = np.full(
        (grid["height"], grid["width"]),
        fill,
        dtype=dst_dtype,
    )

    reproject(
        source=array,
        destination=destination,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=src_nodata,
        dst_transform=grid["transform"],
        dst_crs=grid["crs"],
        dst_nodata=dst_nodata,
        resampling=resampling,
    )

    profile = grid["profile"].copy()
    profile.update(
        dtype=dst_dtype.name,
        count=1,
        nodata=dst_nodata,
    )

    if out_path is not None:
        write_raster(
            destination,
            out_path,
            profile,
            nodata=dst_nodata,
            dtype=dst_dtype.name,
        )

    return destination, profile


def population_to_grid(
    counts: np.ndarray,
    src_transform: Any,
    src_crs: Any,
    template: Any = cfg.GRID_PATH,
    out_path: str | Path | None = None,
    *,
    source_nodata: float | int | None = None,
    district_geometry: Any | None = None,
    district_mask: np.ndarray | None = None,
    source_cell_area_m2: float = cfg.POPULATION_SOURCE_CELL_AREA_M2,
    target_cell_area_m2: float = cfg.POPULATION_TARGET_CELL_AREA_M2,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Convert count cells to density, resample, and restore the district total.

    ``district_geometry`` must be in the source CRS.  ``district_mask`` is a
    boolean mask on the target grid.
    """
    from rasterio.features import geometry_mask
    from shapely.geometry import mapping

    array = np.asarray(counts)
    valid = valid_mask(array, source_nodata)
    density = np.where(valid, array / source_cell_area_m2, np.nan).astype(
        "float32"
    )

    density_grid, _ = reproject_array(
        density,
        src_transform,
        src_crs,
        template,
        method="average",
        src_nodata=np.nan,
        dst_nodata=cfg.NODATA,
        dtype="float32",
    )
    target = np.where(
        density_grid == cfg.NODATA,
        cfg.NODATA,
        density_grid * target_cell_area_m2,
    ).astype("float32")

    if district_geometry is not None:
        source_inside = geometry_mask(
            [mapping(district_geometry)],
            out_shape=array.shape,
            transform=src_transform,
            invert=True,
            all_touched=False,
        )
        source_selection = valid & source_inside
    else:
        source_selection = valid
    source_total = float(array[source_selection].sum())

    target_valid = target != cfg.NODATA
    if district_mask is not None:
        target_selection = target_valid & np.asarray(district_mask, dtype=bool)
    else:
        target_selection = target_valid
    target_before = float(target[target_selection].sum())

    scale = source_total / target_before if target_before > 0 else 0.0
    if scale > 0:
        target = np.where(target_valid, target * scale, target).astype("float32")

    target_after = float(target[target_selection].sum())
    relative = (
        (target_after - source_total) / source_total if source_total else 0.0
    )
    stats = {
        "source_total": source_total,
        "target_before_rescale": target_before,
        "target_after_rescale": target_after,
        "scale": scale,
        "relative_error": relative,
    }

    if out_path is not None:
        grid = _coerce_grid(template)
        profile = grid["profile"].copy()
        profile.update(dtype="float32", count=1, nodata=cfg.NODATA)
        write_raster(
            target,
            out_path,
            profile,
            nodata=cfg.NODATA,
            dtype="float32",
        )

    return target, stats


def rasterize_vector(
    vector: Any,
    template: Any = cfg.GRID_PATH,
    out_path: str | Path | None = None,
    *,
    attribute: str | None = None,
    burn_value: float | int = 1,
    fill: float | int = 0,
    all_touched: bool = False,
    dtype: str = "uint8",
    nodata: float | int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Rasterise a vector layer onto the common grid."""
    import rasterio
    from rasterio.features import rasterize

    gdf = read_vector(vector) if isinstance(vector, (str, Path)) else vector.copy()
    grid = _coerce_grid(template)

    if gdf.crs is not None and gdf.crs != grid["crs"]:
        gdf = gdf.to_crs(grid["crs"])

    if attribute is not None:
        if attribute not in gdf.columns:
            raise KeyError(f"Column '{attribute}' not found in vector layer")
        shapes = list(zip(gdf.geometry, gdf[attribute]))
    else:
        shapes = [(geom, burn_value) for geom in gdf.geometry if geom is not None]

    result = rasterize(
        shapes=shapes,
        out_shape=(grid["height"], grid["width"]),
        transform=grid["transform"],
        fill=fill,
        all_touched=all_touched,
        dtype=dtype,
    )

    out_nodata = _safe_nodata(result.dtype, nodata) if nodata is not None else None
    profile = grid["profile"].copy()
    profile.update(dtype=np.dtype(dtype).name, count=1, nodata=out_nodata)

    if out_path is not None:
        write_raster(result, out_path, profile, nodata=out_nodata, dtype=dtype)

    return result, profile


# ---------------------------------------------------------------------------
# NoData and checks
# ---------------------------------------------------------------------------


def _safe_nodata(dtype: Any, value: float | int | None) -> float | int | None:
    """Return a NoData value that is representable by ``dtype``."""
    dtype = np.dtype(dtype)
    if value is None:
        return None
    if np.issubdtype(dtype, np.floating):
        return float(value)
    if isinstance(value, float) and np.isnan(value):
        return int(np.iinfo(dtype).min)
    value = int(round(float(value)))
    info = np.iinfo(dtype)
    if info.min <= value <= info.max:
        return value
    return int(info.max) if np.issubdtype(dtype, np.unsignedinteger) else int(info.min)


def valid_area_m2(
    data: np.ndarray,
    transform: Any,
    crs: Any,
    *,
    valid: np.ndarray | None = None,
) -> float:
    """Return the ground area of valid cells in square metres."""
    from pyproj import CRS, Geod

    array = np.asarray(data)
    if valid is None:
        valid = valid_mask(array)
    else:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != array.shape:
            raise ValueError(
                f"Mask shape {valid.shape} does not match data shape {array.shape}"
            )

    crs = CRS.from_user_input(crs)
    if crs.is_geographic:
        geod = Geod(ellps="WGS84")
        total = 0.0
        for row in range(array.shape[0]):
            count = int(valid[row].sum())
            if count == 0:
                continue
            x0 = transform.c
            y0 = transform.f + row * transform.e
            x1 = x0 + transform.a
            y1 = y0 + transform.e
            area, _ = geod.polygon_area_perimeter(
                [x0, x1, x1, x0],
                [y0, y0, y1, y1],
            )
            total += abs(area) * count
        return total

    pixel_area = abs(transform.a * transform.e - transform.b * transform.d)
    return float(valid.sum()) * pixel_area


def nodata_to_nan(
    data: np.ndarray,
    nodata_values: float | int | Iterable[float | int] | None = None,
) -> np.ndarray:
    """Convert one or more NoData values to NaN and return a float array."""
    array = np.asarray(data)
    out_dtype = np.float64 if array.dtype == np.float64 else np.float32
    result = np.array(array, dtype=out_dtype, copy=True)

    if nodata_values is None:
        nodata_values = (cfg.NODATA,)
    elif np.isscalar(nodata_values):
        nodata_values = (nodata_values,)

    for value in nodata_values:
        if value is None:
            continue
        if isinstance(value, (float, np.floating)) and np.isnan(value):
            continue
        result[result == value] = np.nan

    return result


def valid_mask(
    data: np.ndarray,
    nodata_values: float | int | Iterable[float | int] | None = None,
) -> np.ndarray:
    """Return a boolean mask of cells that contain valid data."""
    array = np.asarray(data)
    if np.issubdtype(array.dtype, np.floating):
        mask = np.isfinite(array)
    else:
        mask = np.ones(array.shape, dtype=bool)

    if nodata_values is not None:
        if np.isscalar(nodata_values):
            nodata_values = (nodata_values,)
        for value in nodata_values:
            if value is None:
                continue
            if isinstance(value, (float, np.floating)) and np.isnan(value):
                continue
            mask &= array != value

    return mask


def raster_summary(
    raster: str | Path | np.ndarray,
    *,
    nodata_values: float | int | Iterable[float | int] | None = None,
    band: int = 1,
    name: str | None = None,
) -> dict[str, Any]:
    """Print and return a compact raster summary."""
    crs = None
    nodata = None

    if isinstance(raster, (str, Path)):
        data, profile, nodata = read_raster(raster, band=band)
        crs = profile.get("crs")
        label = name or Path(raster).name
    else:
        data = np.asarray(raster)
        label = name or "array"

    if nodata_values is None and nodata is not None:
        nodata_values = (nodata,)

    array = nodata_to_nan(data, nodata_values)
    finite = array[np.isfinite(array)]

    stats = {
        "name": label,
        "shape": tuple(array.shape),
        "crs": str(crs) if crs is not None else None,
        "min": float(finite.min()) if finite.size else float("nan"),
        "max": float(finite.max()) if finite.size else float("nan"),
        "percent_nan": float(np.isnan(array).mean() * 100.0),
        "nodata": nodata,
    }

    print(
        f"{stats['name']}: shape={stats['shape']}, crs={stats['crs']}, "
        f"min={stats['min']:.4g}, max={stats['max']:.4g}, "
        f"NaN={stats['percent_nan']:.2f}%"
    )
    return stats


def assert_same_grid(
    raster_a: str | Path | Mapping[str, Any],
    raster_b: str | Path | Mapping[str, Any],
    *,
    atol: float = 1e-6,
) -> bool:
    """Raise ``ValueError`` unless two rasters share CRS, transform and shape."""
    grid_a = _coerce_grid(raster_a)
    grid_b = _coerce_grid(raster_b)

    problems: list[str] = []
    if grid_a["crs"] != grid_b["crs"]:
        problems.append(f"CRS differs ({grid_a['crs']} != {grid_b['crs']})")

    transform_a = tuple(grid_a["transform"])[:6]
    transform_b = tuple(grid_b["transform"])[:6]
    if not np.allclose(transform_a, transform_b, atol=atol, rtol=0):
        problems.append(f"transform differs ({transform_a} != {transform_b})")

    shape_a = (grid_a["height"], grid_a["width"])
    shape_b = (grid_b["height"], grid_b["width"])
    if shape_a != shape_b:
        problems.append(f"shape differs ({shape_a} != {shape_b})")

    if problems:
        raise ValueError("Rasters are not on the same grid: " + "; ".join(problems))

    return True


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_raster(
    data: np.ndarray,
    out_path: str | Path | None = None,
    *,
    transform: Any | None = None,
    crs: Any | None = None,
    district: Any | None = None,
    title: str | None = None,
    cmap: str = "viridis",
    nodata_values: float | int | Iterable[float | int] | None = None,
    colorbar: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
    alpha: float = 1.0,
    figsize: tuple[float, float] = (10, 8),
    dpi: int = 150,
):
    """Plot a raster with an optional district outline and save it to disk."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pyproj import CRS
    from rasterio.transform import array_bounds

    array = np.asarray(data)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D raster, got shape {array.shape}")

    array = nodata_to_nan(array, nodata_values)

    extent = None
    if transform is not None:
        extent = array_bounds(array.shape[0], array.shape[1], transform)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    image = ax.imshow(
        array,
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=alpha,
    )

    if district is not None:
        gdf = read_vector(district) if isinstance(district, (str, Path)) else district
        if (
            crs is not None
            and getattr(gdf, "crs", None) is not None
            and gdf.crs != crs
        ):
            gdf = gdf.to_crs(crs)
        gdf.boundary.plot(ax=ax, color="black", linewidth=0.8)

    if title:
        ax.set_title(title)

    if crs is not None:
        try:
            projected = CRS.from_user_input(crs).is_projected
        except Exception:
            projected = False
        ax.set_xlabel("Easting (m)" if projected else "Longitude")
        ax.set_ylabel("Northing (m)" if projected else "Latitude")

    if colorbar:
        fig.colorbar(image, ax=ax, shrink=0.75, label="Value")

    ax.set_aspect("equal")

    if out_path is not None:
        out_path = Path(out_path)
        make_folder(out_path.parent)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return out_path

    return fig, ax
