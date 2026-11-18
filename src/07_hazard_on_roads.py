"""Task 8b: attach flood hazard to every road edge (road-flood-impact proxy).

    python src/07_hazard_on_roads.py
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.transform
import shapely

from src import config as cfg
from src import utils

LOGGER = utils.get_logger("hazard_on_roads")


def _values_at(path, xs, ys) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float64")
        nodata, transform = src.nodata, src.transform
    if nodata is not None:
        arr[arr == nodata] = np.nan
    arr[~np.isfinite(arr)] = np.nan
    rows, cols = rasterio.transform.rowcol(transform, xs, ys)
    rows, cols = np.asarray(rows), np.asarray(cols)
    ok = (rows >= 0) & (rows < arr.shape[0]) & (cols >= 0) & (cols < arr.shape[1])
    out = np.full(len(xs), np.nan)
    out[ok] = arr[rows[ok], cols[ok]]
    return out


def main() -> None:
    edges = gpd.read_file(cfg.output_path("road_edges")).reset_index(drop=True)
    lengths = edges["length_m"].to_numpy()
    counts = np.maximum(2, np.ceil(lengths / cfg.ROAD_SAMPLE_STEP_M).astype(int) + 1)
    idx = np.repeat(np.arange(len(edges)), counts)
    start = np.repeat(np.cumsum(counts) - counts, counts)
    frac = (np.arange(counts.sum()) - start) / (counts[idx] - 1)
    geoms = edges.geometry.to_numpy()
    pts = shapely.line_interpolate_point(geoms[idx], frac * lengths[idx])
    xs, ys = shapely.get_x(pts), shapely.get_y(pts)
    LOGGER.info("Sampling %d points on %d edges", len(xs), len(edges))

    ksdma = _values_at(cfg.output_path("ksdma_grid"), xs, ys)
    flood = _values_at(cfg.output_path("flood2018_grid"), xs, ys)
    hand = _values_at(cfg.output_path("hand_grid"), xs, ys)

    df = pd.DataFrame({"i": idx, "k": ksdma, "f": (flood == 1).astype(float), "h": hand})
    df["m"] = df["k"].notna().astype(float)  # NaN = not modelled, never "safe"
    agg = df.groupby("i").agg(ksdma_max=("k", "max"), ksdma_modelled_share=("m", "mean"),
                              flooded_2018_share=("f", "mean"), hand_min=("h", "min"))
    for col in agg.columns:
        edges[col] = agg[col].reindex(range(len(edges))).to_numpy()

    utils.write_vector(edges, cfg.output_path("road_edges"))

    with open(cfg.output_path("road_graph"), "rb") as fh:
        graph = pickle.load(fh)
    lookup = edges.set_index("edge_id")[list(agg.columns)].to_dict("index")
    for _, _, data in graph.edges(data=True):
        data.update(lookup[data["edge_id"]])
    with open(cfg.output_path("road_graph"), "wb") as fh:
        pickle.dump(graph, fh)

    edges["flooded_km"] = edges["length_m"] * edges["flooded_2018_share"] / 1000
    edges["ksdma_km"] = edges["length_m"] * (edges["ksdma_max"] > 0) / 1000
    edges["length_km"] = edges["length_m"] / 1000
    table = edges.groupby("highway")[["length_km", "flooded_km", "ksdma_km"]].sum()
    table.loc["ALL"] = table.sum()
    table["flooded_2018_share"] = table["flooded_km"] / table["length_km"]
    table["ksdma_100yr_share"] = table["ksdma_km"] / table["length_km"]
    table.round(3).to_csv(cfg.output_path("road_flood_by_class"))
    LOGGER.info("\n%s", table.round(3).to_string())

    stats_path = cfg.output_path("graph_stats")
    stats = json.loads(stats_path.read_text())
    stats["road_flood_impact_proxy"] = {
        "definition": "road length inside the Sentinel-1 2018 observed flood mask",
        "flooded_2018_km": round(float(table.loc["ALL", "flooded_km"]), 1),
        "flooded_2018_share": round(float(table.loc["ALL", "flooded_2018_share"]), 4),
        "ksdma_100yr_km": round(float(table.loc["ALL", "ksdma_km"]), 1),
        "mean_ksdma_modelled_share": round(float(edges["ksdma_modelled_share"].mean()), 4),
    }
    stats_path.write_text(json.dumps(stats, indent=2))
    LOGGER.info("%s", json.dumps(stats["road_flood_impact_proxy"], indent=2))


if __name__ == "__main__":
    main()