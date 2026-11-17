"""Task 7b: give every camp a location.

Order: Excel X/Y -> OSM fuzzy name match -> Nominatim -> village centroid.

    python src/06_locate_camps.py              # Nominatim only for camps still unlocated
    python src/06_locate_camps.py --no-geocode # fully offline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import geopandas as gpd
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from shapely.geometry import Point

from src import config as cfg
from src import utils

LOGGER = utils.get_logger("locate_camps")


def _load_cache() -> dict:
    path = cfg.output_path("camps_geocode_cache")
    return json.loads(path.read_text()) if path.exists() else {}


def _make_geocoder():
    from geopy.extra.rate_limiter import RateLimiter
    from geopy.geocoders import Nominatim

    geo = Nominatim(user_agent="alappuzha-flood-student-project", timeout=10)
    return RateLimiter(geo.geocode, min_delay_seconds=1.1, max_retries=2,
                       error_wait_seconds=5, swallow_exceptions=True)


def _facility_pool(villages: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    fac = utils.read_vector(cfg.output_path("facilities_clip"))
    fac = utils.reproject_vector(fac, cfg.CRS_WORKING)
    name_col = next((c for c in ("name", "name_en", "name:en") if c in fac.columns), None)
    if name_col is None:
        LOGGER.warning("No name column in facilities; OSM matching disabled")
        return fac.iloc[0:0].assign(name_key="", village_idx=pd.NA, osm_name="")
    fac = fac[fac[name_col].notna()].copy()
    fac["osm_name"] = fac[name_col].astype(str)
    fac["name_key"] = fac["osm_name"].map(utils.normalize_place_name)
    fac = fac[fac["name_key"] != ""].copy()
    fac["geometry"] = fac.geometry.representative_point()
    joined = gpd.sjoin(fac, villages[["geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]
    fac["village_idx"] = joined["index_right"]
    return fac


def _source_xy(camps: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Excel X/Y (lon/lat) -> UTM; swapped or out-of-range values become NaN."""
    sx = pd.Series(np.nan, index=camps.index)
    sy = sx.copy()
    if not {"src_lon", "src_lat"} <= set(camps.columns):
        return sx, sy
    lon, lat = camps["src_lon"], camps["src_lat"]
    swap = ~lon.between(74, 78) & lat.between(74, 78)
    lon, lat = lon.where(~swap, camps["src_lat"]), lat.where(~swap, camps["src_lon"])
    ok = lon.between(74, 78) & lat.between(8, 11)
    if ok.any():
        pts = gpd.GeoSeries(gpd.points_from_xy(lon[ok], lat[ok]), index=camps.index[ok],
                            crs=cfg.CRS_GEOGRAPHIC).to_crs(cfg.CRS_WORKING)
        sx.loc[ok], sy.loc[ok] = pts.x, pts.y
    return sx, sy


def main(use_geocode: bool) -> None:
    camps = pd.read_csv(cfg.output_path("camps_clean"))
    for col in ("camp", "village", "taluk", "name_key"):
        camps[col] = camps[col].astype(object).fillna("").astype(str)
    district = utils.load_district(as_geometry=True)
    district_tol = district.buffer(500)  # coastal points can fall just outside

    villages = gpd.read_file(cfg.output_path("villages_clip"))
    villages = villages[villages["DISTRICT"].astype(str).str.casefold() == cfg.DISTRICT_NAME.casefold()]
    villages = utils.reproject_vector(villages.reset_index(drop=True), cfg.CRS_WORKING)
    village_keys = villages["NAME"].map(utils.normalize_place_name).tolist()
    village_points = villages.geometry.representative_point()
    sub_points = villages.dissolve(by="SUB_DIST").geometry.representative_point()
    sub_lookup = {str(k).casefold(): v for k, v in sub_points.items()}

    fac = _facility_pool(villages)
    sx, sy = _source_xy(camps)
    cache = _load_cache()
    geocode = _make_geocoder() if use_geocode else None

    rows = []
    for camp in camps.itertuples():
        vkey = utils.normalize_place_name(camp.village)
        v_idx, vscore = None, 0.0
        if vkey:
            hit = process.extractOne(vkey, village_keys, scorer=fuzz.WRatio, score_cutoff=80)
            if hit:
                v_idx, vscore = hit[2], hit[1]
        v_geom = villages.geometry.iloc[v_idx] if v_idx is not None else None

        method, score, matched, point = "none", 0.0, "", None

        # 0) coordinates given in the Excel file
        if pd.notna(sx[camp.Index]):
            cand = Point(sx[camp.Index], sy[camp.Index])
            if district_tol.contains(cand):
                method, score, matched, point = "source", 100.0, "Excel X/Y", cand

        # 1) fuzzy match to OSM facilities
        if point is None and len(fac):
            pools = []
            if v_idx is not None:
                pools.append((fac[fac["village_idx"] == v_idx], cfg.CAMPS_FUZZY_THRESHOLD))
            pools.append((fac, cfg.CAMPS_FUZZY_THRESHOLD_WIDE))
            for pool, cutoff in pools:
                if pool.empty:
                    continue
                hit = process.extractOne(camp.name_key, pool["name_key"].tolist(),
                                         scorer=fuzz.token_sort_ratio, score_cutoff=cutoff)
                if hit:
                    row = pool.iloc[hit[2]]
                    method, score, matched, point = "osm", hit[1], row["osm_name"], row.geometry
                    break

        # 2) Nominatim (cached)
        if point is None and geocode is not None:
            query = f"{camp.camp}, {camp.village}, Alappuzha, Kerala, India"
            if query not in cache:
                loc = geocode(query)
                cache[query] = [loc.latitude, loc.longitude] if loc else None
                cfg.output_path("camps_geocode_cache").write_text(json.dumps(cache, indent=1))
            if cache[query]:
                lat, lon = cache[query]
                cand = gpd.GeoSeries([Point(lon, lat)], crs=cfg.CRS_GEOGRAPHIC).to_crs(cfg.CRS_WORKING).iloc[0]
                ok = district_tol.contains(cand)
                if ok and v_geom is not None:
                    ok = cand.distance(v_geom) <= cfg.CAMPS_MAX_VILLAGE_DIST_M
                if ok:
                    method, score, matched, point = "nominatim", 100.0, query, cand

        # 3) village, then sub-district, centroid
        if point is None and v_idx is not None:
            method, score, matched, point = "centroid", vscore, villages["NAME"].iloc[v_idx], village_points.iloc[v_idx]
        if point is None and camp.taluk.casefold() in sub_lookup:
            method, score, matched, point = "centroid", 0.0, f"taluk {camp.taluk}", sub_lookup[camp.taluk.casefold()]

        rows.append({"camp_id": camp.camp_id, "method": method, "match_score": round(float(score), 1),
                     "matched_name": matched,
                     "x": point.x if point is not None else None,
                     "y": point.y if point is not None else None})

    located = camps.merge(pd.DataFrame(rows), on="camp_id")
    has = located["x"].notna()
    located["lon"], located["lat"] = np.nan, np.nan
    if has.any():
        pts = gpd.GeoSeries(gpd.points_from_xy(located.loc[has, "x"], located.loc[has, "y"]),
                            index=located.index[has], crs=cfg.CRS_WORKING).to_crs(cfg.CRS_GEOGRAPHIC)
        located.loc[has, "lon"], located.loc[has, "lat"] = pts.x, pts.y
    located.to_csv(cfg.output_path("camps_located"), index=False)
    LOGGER.info("Method counts:\n%s", located["method"].value_counts().to_string())
    LOGGER.info("Saved -> %s", cfg.output_path("camps_located"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-geocode", action="store_true")
    main(use_geocode=not parser.parse_args().no_geocode)