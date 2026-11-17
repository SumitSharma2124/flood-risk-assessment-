"""Task 7c: validate camp locations and flag camps inside flood zones.

    python src/06_validate_camps.py
"""

from __future__ import annotations

import json
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

from src import config as cfg
from src import utils

LOGGER = utils.get_logger("validate_camps")
COLORS = {"source": "tab:purple", "osm": "tab:green", "nominatim": "tab:blue",
          "centroid": "tab:orange", "none": "tab:red"}
GOOD_METHODS = ["source", "osm", "nominatim"]


def _sample(path, xs, ys) -> np.ndarray:
    with rasterio.open(path) as src:
        vals = np.array([v[0] for v in src.sample(zip(xs, ys))], dtype="float64")
        nodata = src.nodata
    if nodata is not None:
        vals[vals == nodata] = np.nan
    vals[~np.isfinite(vals)] = np.nan
    return vals


def main() -> None:
    camps = pd.read_csv(cfg.output_path("camps_located"))
    district = utils.load_district()
    geom = district.geometry.union_all()
    geom_tol = geom.buffer(500)

    located = camps[camps["method"] != "none"].copy()
    gdf = gpd.GeoDataFrame(located, geometry=gpd.points_from_xy(located["x"], located["y"]),
                           crs=cfg.CRS_WORKING)
    xs, ys = gdf.geometry.x.values, gdf.geometry.y.values

    flood = _sample(cfg.output_path("flood2018_grid"), xs, ys)
    ksdma = _sample(cfg.output_path("ksdma_grid"), xs, ys)
    gdf["in_flood_2018"] = flood == 1
    gdf["in_ksdma_100yr"] = ksdma > 0  # NoData is NOT treated as safe or unsafe
    gdf["ksdma_modelled_here"] = ~np.isnan(ksdma)
    gdf["unsafe"] = gdf["in_flood_2018"] | gdf["in_ksdma_100yr"]
    gdf["inside_district"] = gdf.geometry.within(geom_tol)
    gdf["dup_coords"] = gdf.duplicated(subset=["x", "y"], keep=False)

    utils.write_vector(gdf, cfg.output_path("camps"))
    n = len(camps)
    stats = {
        "camps_total": n,
        "camps_located": len(gdf),
        "camps_unlocated": n - len(gdf),
        "method_share": (camps["method"].value_counts() / n).round(3).to_dict(),
        "inside_district_500m_tolerance": int(gdf["inside_district"].sum()),
        "in_flood_2018": int(gdf["in_flood_2018"].sum()),
        "in_ksdma_100yr": int(gdf["in_ksdma_100yr"].sum()),
        "unsafe_total": int(gdf["unsafe"].sum()),
        "ksdma_modelled_at_camp": int(gdf["ksdma_modelled_here"].sum()),
        "camps_sharing_coordinates": int(gdf["dup_coords"].sum()),
        "checks": {
            "precise_methods_at_least_60pct": bool(camps["method"].isin(GOOD_METHODS).mean() >= 0.6),
            "all_inside_district": bool(gdf["inside_district"].all()),
        },
    }
    cfg.output_path("camps_validation").write_text(json.dumps(stats, indent=2))
    gdf.drop(columns="geometry").to_csv(cfg.output_path("camps_summary"), index=False)
    LOGGER.info("%s", json.dumps(stats, indent=2))

    with rasterio.open(cfg.output_path("flood2018_grid")) as src:
        img = src.read(1, out_shape=(src.height // 2, src.width // 2)).astype(float)
        img[img != 1] = np.nan
        ext = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)
    fig, ax = plt.subplots(figsize=(7, 10))
    district.boundary.plot(ax=ax, color="black", linewidth=0.8)
    ax.imshow(img, extent=ext, cmap="Blues", alpha=0.6)
    for method, sub in gdf.groupby("method"):
        ax.scatter(sub.geometry.x, sub.geometry.y, s=14, c=COLORS.get(method, "gray"), label=method)
    bad = gdf[gdf["unsafe"]]
    ax.scatter(bad.geometry.x, bad.geometry.y, s=60, facecolors="none", edgecolors="red", label="unsafe")
    ax.legend()
    ax.set_title("Relief camps (blue = 2018 flood)")
    fig.savefig(cfg.output_path("camps_overview"), dpi=150, bbox_inches="tight")
    LOGGER.info("Saved %s", cfg.output_path("camps_overview"))


if __name__ == "__main__":
    main()