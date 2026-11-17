"""One-off diagnostic: why are camp coordinates being rejected?

    python src/06_diagnose_camps.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import geopandas as gpd
import pandas as pd

from src import config as cfg
from src import utils

pd.set_option("display.width", 200, "display.max_columns", 20, "display.max_colwidth", 40)

camps = pd.read_csv(cfg.output_path("camps_located"))
district = utils.load_district()
geom = district.geometry.union_all()

print("\n== source X/Y quality ==")
lon, lat = camps["src_lon"], camps["src_lat"]
print("missing:", int((lon.isna() | lat.isna()).sum()))
print("lon describe:\n", lon.describe().round(4).to_string())
print("lat describe:\n", lat.describe().round(4).to_string())
print("lon in 74-78 and lat in 8-11:", int((lon.between(74, 78) & lat.between(8, 11)).sum()))
print("swapped (lat in 74-78):", int(lat.between(74, 78).sum()))
print("zero values:", int(((lon == 0) | (lat == 0)).sum()))

ok = lon.between(74, 78) & lat.between(8, 11)
pts = gpd.GeoSeries(gpd.points_from_xy(lon[ok], lat[ok]), index=camps.index[ok],
                    crs=cfg.CRS_GEOGRAPHIC).to_crs(cfg.CRS_WORKING)
dist = pts.apply(lambda p: 0.0 if geom.contains(p) else p.distance(geom))
print("\n== distance outside the district (m), for camps with valid-range X/Y ==")
print("inside:", int((dist == 0).sum()), "| outside:", int((dist > 0).sum()))
print(dist[dist > 0].describe(percentiles=[.1, .25, .5, .75, .9]).round(0).to_string())

cols = [c for c in ("camp", "village", "lsg", "src_lon", "src_lat") if c in camps.columns]
far = dist[dist > 500].sort_values(ascending=False).index
print("\n== 12 camps whose X/Y fall >500 m outside the district ==")
print(camps.loc[far[:12], cols].to_string())

bad = camps.index[~ok]
print("\n== 12 camps with unusable X/Y ==")
print(camps.loc[bad[:12], cols].to_string())

none = camps[camps["method"] == "none"]
print("\n== 15 camps with method == none ==")
print(none[cols + ["method"]].head(15).to_string())

print("\n== 15 village names in the Excel (random) ==")
print(camps["village"].sample(15, random_state=1).tolist())