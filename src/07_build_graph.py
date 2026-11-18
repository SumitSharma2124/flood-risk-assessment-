"""Task 8a: build a routable road graph from the clipped OSM road lines.

    python src/07_build_graph.py
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
import networkx as nx
import numpy as np
import pandas as pd
import shapely
from scipy.spatial import cKDTree
from shapely.ops import unary_union
from shapely.strtree import STRtree

from src import config as cfg
from src import utils

LOGGER = utils.get_logger("build_graph")
ATTRS = ["highway", "name", "oneway"]


def _pick(frame, names):
    return next((c for c in names if c in frame.columns), None)


def _flag(series: pd.Series) -> pd.Series:
    """True only for real tag values; missing values (NaN/None/<NA>) are False."""
    s = series.astype("object").where(series.notna(), "").astype(str).str.strip().str.lower()
    return ~s.isin({"", "no", "nan", "none", "<na>", "0", "false"})


def _node_lines(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Split lines at every intersection and carry the attributes over."""
    if frame.empty:
        return gpd.GeoDataFrame(columns=ATTRS + ["geometry"], geometry="geometry", crs=cfg.CRS_WORKING)
    merged = unary_union(frame.geometry.values)
    pieces = [g for g in getattr(merged, "geoms", [merged]) if g.length > 0]
    arr = np.array(pieces, dtype=object)
    mids = shapely.line_interpolate_point(arr, 0.5, normalized=True)
    idx = STRtree(frame.geometry.values).nearest(mids)
    attrs = frame.iloc[idx][ATTRS].reset_index(drop=True)
    return gpd.GeoDataFrame(attrs, geometry=pieces, crs=cfg.CRS_WORKING)


def main() -> None:
    roads = utils.read_vector(cfg.output_path("roads_clip"))
    roads = utils.reproject_vector(roads, cfg.CRS_WORKING)
    LOGGER.info("Road columns: %s", list(roads.columns))
    hw_col = _pick(roads, ("highway", "fclass", "road_type", "type"))
    if hw_col is None:
        raise KeyError(f"No highway-type column found. Columns: {list(roads.columns)}")
    raw_features = len(roads)

    roads["highway"] = roads[hw_col].astype(str).str.lower().str.strip()
    LOGGER.info("Highway values: %s", roads["highway"].value_counts().to_dict())
    roads = roads[roads["highway"].isin(cfg.ROAD_HIGHWAY_TYPES)].copy()
    if roads.empty:
        raise ValueError("No roads left after the highway-type filter; check ROAD_HIGHWAY_TYPES")

    name_col = _pick(roads, ("name", "name_en"))
    roads["name"] = roads[name_col].fillna("").astype(str) if name_col else ""
    one_col = _pick(roads, ("oneway",))
    roads["oneway"] = roads[one_col].fillna("").astype(str) if one_col else ""

    roads = roads.explode(index_parts=False)
    roads = roads[(roads.geom_type == "LineString") & ~roads.geometry.is_empty].reset_index(drop=True)
    kept_features = len(roads)

    grade = pd.Series(False, index=roads.index)  # bridges / tunnels / layered roads
    for col in ("bridge", "tunnel"):
        if col in roads.columns:
            grade |= _flag(roads[col])
    if "layer" in roads.columns:
        grade |= pd.to_numeric(roads["layer"], errors="coerce").fillna(0) != 0
    LOGGER.info("Grade-separated features: %d of %d", int(grade.sum()), kept_features)
    if grade.mean() > 0.5:
        raise ValueError("More than half the roads are flagged bridge/tunnel/layer; check the flag logic")

    ground = _node_lines(roads[~grade])
    elevated = roads.loc[grade, ATTRS + ["geometry"]].reset_index(drop=True)

    # Node ids from rounded coordinates (0.5 m); ground junctions share coordinates exactly.
    key_to_id: dict[tuple[int, int], int] = {}
    coords: list[tuple[float, float]] = []

    def node_id(x: float, y: float) -> int:
        key = (round(x * 2), round(y * 2))
        if key not in key_to_id:
            key_to_id[key] = len(coords)
            coords.append((x, y))
        return key_to_id[key]

    edges = []
    for geom, attrs in zip(ground.geometry, ground[ATTRS].itertuples(index=False)):
        a, b = geom.coords[0], geom.coords[-1]
        edges.append([node_id(*a[:2]), node_id(*b[:2]), geom, attrs])
    ground_coords = np.array(coords) if coords else np.empty((0, 2))
    tree = cKDTree(ground_coords) if len(ground_coords) else None

    def snap(x: float, y: float) -> int:
        if tree is not None:
            dist, i = tree.query([x, y])
            if dist <= cfg.ROAD_NODING_TOL_M:
                return int(i)
        return node_id(x, y)

    for geom, attrs in zip(elevated.geometry, elevated[ATTRS].itertuples(index=False)):
        a, b = geom.coords[0], geom.coords[-1]
        edges.append([snap(*a[:2]), snap(*b[:2]), geom, attrs])

    graph = nx.Graph()
    graph.add_nodes_from((i, {"x": x, "y": y}) for i, (x, y) in enumerate(coords))
    loops = parallel = 0
    for u, v, geom, attrs in edges:
        if u == v:
            loops += 1
            continue
        length = float(geom.length)
        if graph.has_edge(u, v):
            parallel += 1
            if graph[u][v]["length_m"] <= length:
                continue
        graph.add_edge(u, v, length_m=length, highway=attrs.highway, name=attrs.name,
                       oneway=attrs.oneway, geometry=geom)

    total_km = sum(d["length_m"] for *_, d in graph.edges(data=True)) / 1000
    comps = sorted(nx.connected_components(graph), key=len, reverse=True)
    lcc = comps[0]
    kept = graph.subgraph(lcc).copy()
    kept_km = sum(d["length_m"] for *_, d in kept.edges(data=True)) / 1000
    for eid, (u, v) in enumerate(kept.edges):
        kept[u][v]["edge_id"] = eid

    with open(cfg.output_path("road_graph"), "wb") as fh:
        pickle.dump(kept, fh)
    edge_rows = [{"edge_id": d["edge_id"], "u": u, "v": v, "length_m": d["length_m"],
                  "highway": d["highway"], "name": d["name"], "oneway": d["oneway"],
                  "geometry": d["geometry"]} for u, v, d in kept.edges(data=True)]
    edges_gdf = gpd.GeoDataFrame(edge_rows, geometry="geometry", crs=cfg.CRS_WORKING)
    nodes_gdf = gpd.GeoDataFrame(
        {"node_id": list(kept.nodes)},
        geometry=gpd.points_from_xy([kept.nodes[n]["x"] for n in kept.nodes],
                                    [kept.nodes[n]["y"] for n in kept.nodes]),
        crs=cfg.CRS_WORKING)
    utils.write_vector(edges_gdf, cfg.output_path("road_edges"))
    utils.write_vector(nodes_gdf, cfg.output_path("road_nodes"))

    stats = {
        "raw_features": raw_features,
        "drivable_features": kept_features,
        "grade_separated_features": int(grade.sum()),
        "self_loops_dropped": loops,
        "parallel_edges_seen": parallel,
        "graph_nodes_all": graph.number_of_nodes(),
        "graph_edges_all": graph.number_of_edges(),
        "graph_km_all": round(total_km, 1),
        "components": len(comps),
        "top5_component_nodes": [len(c) for c in comps[:5]],
        "largest_component_nodes": len(lcc),
        "nodes": kept.number_of_nodes(),
        "edges": kept.number_of_edges(),
        "km_kept": round(kept_km, 1),
        "km_dropped_disconnected": round(total_km - kept_km, 1),
        "share_km_kept": round(kept_km / total_km, 4) if total_km else None,
        "km_by_highway": (edges_gdf.groupby("highway")["length_m"].sum() / 1000).round(1).to_dict(),
    }
    cfg.output_path("graph_stats").write_text(json.dumps(stats, indent=2))
    LOGGER.info("%s", json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()