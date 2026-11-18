"""Task 8c: snap relief camps and village origins to road-graph nodes.

    python src/07_camps_origins_to_nodes.py
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
from rasterio.features import rasterize
from scipy.spatial import cKDTree

from src import config as cfg
from src import utils

LOGGER = utils.get_logger("snap_nodes")


def _snap(tree, node_ids, xs, ys):
    dist, i = tree.query(np.column_stack([xs, ys]))
    return node_ids[i], dist


def _dist_stats(dist: np.ndarray) -> dict:
    return {"count": int(len(dist)), "median_m": round(float(np.median(dist)), 1),
            "p95_m": round(float(np.percentile(dist, 95)), 1), "max_m": round(float(dist.max()), 1),
            "over_warn_threshold": int((dist > cfg.SNAP_WARN_M).sum())}


def _add_access_flags(df: pd.DataFrame) -> pd.DataFrame:
    """snap_warn / poorly_connected: farther than SNAP_WARN_M from a road node.
    access_cost_m: straight-line gap to the node. Task 9 must add this to the
    route length at the start (origin) or end (camp), so a far snap is never free."""
    df["snap_warn"] = df["snap_dist_m"] > cfg.SNAP_WARN_M
    df["poorly_connected"] = df["snap_warn"]
    df["access_cost_m"] = df["snap_dist_m"].round(1)
    return df


def main() -> None:
    nodes = gpd.read_file(cfg.output_path("road_nodes"))
    edges = gpd.read_file(cfg.output_path("road_edges"))
    node_ids = nodes["node_id"].to_numpy()
    tree = cKDTree(np.column_stack([nodes.geometry.x, nodes.geometry.y]))

    # ---- camps ----
    camps = gpd.read_file(cfg.output_path("camps"))
    camps["node_id"], camps["snap_dist_m"] = _snap(tree, node_ids, camps.geometry.x, camps.geometry.y)
    camps = _add_access_flags(camps)
    keep = [c for c in ("camp_id", "camp", "village", "capacity", "method", "unsafe",
                        "in_flood_2018", "in_ksdma_100yr", "node_id", "snap_dist_m",
                        "snap_warn", "poorly_connected", "access_cost_m")
            if c in camps.columns]
    camps["x"], camps["y"] = camps.geometry.x, camps.geometry.y
    camps[keep + ["x", "y"]].to_csv(cfg.output_path("camps_nodes"), index=False)

    # ---- village origins ----
    villages = gpd.read_file(cfg.output_path("villages_clip"))
    villages = villages[villages["DISTRICT"].astype(str).str.casefold() == cfg.DISTRICT_NAME.casefold()]
    villages = utils.reproject_vector(villages.reset_index(drop=True), cfg.CRS_WORKING)
    with rasterio.open(cfg.output_path("population_grid")) as src:
        pop = src.read(1).astype("float64")
        if src.nodata is not None:
            pop[pop == src.nodata] = np.nan
        pop = np.nan_to_num(pop, nan=0.0)
        ids = rasterize([(g, i + 1) for i, g in enumerate(villages.geometry)],
                        out_shape=pop.shape, transform=src.transform, fill=0, dtype="int32")
    sums = np.bincount(ids.ravel(), weights=pop.ravel(), minlength=len(villages) + 1)
    villages["population_2018"] = sums[1:]

    pts = villages.geometry.representative_point()
    origins = pd.DataFrame({"origin_id": range(1, len(villages) + 1),
                            "village": villages["NAME"].astype(str).values,
                            "population_2018": villages["population_2018"].round(0).values,
                            "x": pts.x.values, "y": pts.y.values})
    origins["node_id"], origins["snap_dist_m"] = _snap(tree, node_ids, origins["x"], origins["y"])
    origins = _add_access_flags(origins)
    origins.to_csv(cfg.output_path("origins_nodes"), index=False)

    far = origins[origins["poorly_connected"]].sort_values("snap_dist_m", ascending=False)
    for r in far.itertuples():
        LOGGER.warning("Poorly connected origin: %s | snap %.0f m | population %.0f",
                       r.village, r.snap_dist_m, r.population_2018)

    # ---- stats ----
    stats_path = cfg.output_path("graph_stats")
    stats = json.loads(stats_path.read_text())
    pop_total = float(origins["population_2018"].sum())
    pop_far = float(origins.loc[origins["poorly_connected"], "population_2018"].sum())
    stats["snapping"] = {
        "camps": _dist_stats(camps["snap_dist_m"].to_numpy()),
        "origins": _dist_stats(origins["snap_dist_m"].to_numpy()),
        "origins_population_total": pop_total,
        "origins_population_beyond_warn": pop_far,
        "camps_unsafe": int(camps["unsafe"].sum()) if "unsafe" in camps else None,
        "camps_snap_share_within_threshold": round(float(1 - camps["snap_warn"].mean()), 4),
        "origins_snap_share_within_threshold": round(float(1 - origins["snap_warn"].mean()), 4),
        "origins_poorly_connected": {
            "count": int(len(far)),
            "population_share": round(pop_far / pop_total, 4) if pop_total else None,
            "villages": far["village"].tolist(),
            "note": ("Snap gap is straight-line, not road distance. Routing adds "
                     "access_cost_m at the start; access for these origins is low-confidence "
                     "(OSM has almost no roads in the central Kuttanad gap). Target of 95% "
                     "within 500 m is not met without ferry/jetty links."),
        },
    }
    stats_path.write_text(json.dumps(stats, indent=2))
    LOGGER.info("%s", json.dumps(stats["snapping"], indent=2))

    # ---- overview map ----
    fig, ax = plt.subplots(figsize=(7, 10))
    edges.plot(ax=ax, color="lightgray", linewidth=0.3)
    flooded = edges[edges["flooded_2018_share"].fillna(0) > 0]
    flooded.plot(ax=ax, color="tab:blue", linewidth=0.8)
    safe = camps[~camps["unsafe"].astype(bool)] if "unsafe" in camps else camps
    unsafe = camps[camps["unsafe"].astype(bool)] if "unsafe" in camps else camps.iloc[0:0]
    ax.scatter(safe.geometry.x, safe.geometry.y, s=10, c="tab:green", label="camp (not flagged)")
    ax.scatter(unsafe.geometry.x, unsafe.geometry.y, s=14, c="red", label="camp (unsafe)")
    good = origins[~origins["poorly_connected"]]
    ax.scatter(good["x"], good["y"], s=30, c="black", marker="^", label="village origin")
    if len(far):
        ax.scatter(far["x"], far["y"], s=40, c="orange", edgecolors="black", marker="^",
                   label=f"origin > {int(cfg.SNAP_WARN_M)} m from road")
        nxy = nodes.set_index("node_id").loc[far["node_id"]]
        for (ox, oy), (nx_, ny_) in zip(far[["x", "y"]].to_numpy(),
                                        zip(nxy.geometry.x, nxy.geometry.y)):
            ax.plot([ox, nx_], [oy, ny_], color="orange", linewidth=0.8, linestyle="--")
    ax.legend(loc="upper right")
    ax.set_title("Road graph (blue = roads in 2018 flood)")
    fig.savefig(cfg.output_path("graph_overview"), dpi=150, bbox_inches="tight")
    LOGGER.info("Saved %s", cfg.output_path("graph_overview"))


if __name__ == "__main__":
    main()