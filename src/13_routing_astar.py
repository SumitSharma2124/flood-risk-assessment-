"""
Task 9a - Risk-aware routing (v2).
Run from the src directory: python 08_routing_astar.py

Hazard per edge (0..1) = max( KSDMA 100-yr depth / KSDMA_DEPTH_CAP_CM , flooded_2018_share )
  - KSDMA depth is in cm; NaN means dry land (depth 0), per the KSDMA NoData audit.
  - The 2018 flood mask is a proxy, so it acts as a hazard floor, not a hard closure.
Cost = length_m * (1 + lambda * hazard).
"""

import json
import logging
import math
import warnings
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString

import config as cfg

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
LOGGER = logging.getLogger(__name__)

LAMBDA_DEFAULT = 1.0
HIGH_HAZARD_THRESHOLD = 0.5      # hazard >= 0.5  ->  depth >= 150 cm (or >= 50% in 2018 flood)
KSDMA_DEPTH_CAP_CM = 300.0       # depth that counts as hazard = 1 (median flooded depth ~250 cm)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def output_file(folder, filename):
    target = OUTPUT_DIR / folder
    target.mkdir(parents=True, exist_ok=True)
    return target / filename


def _number(value, default=0.0):
    """Convert nullable CSV/GPKG values to finite floats."""
    try:
        value = float(value)
        return value if np.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def edge_hazard(depth_cm, flooded_2018_share) -> float:
    """0..1 hazard. NaN depth = dry = 0. The 2018 mask is a floor (proxy, not a closure)."""
    depth = max(0.0, _number(depth_cm, 0.0))
    flooded = min(1.0, max(0.0, _number(flooded_2018_share, 0.0)))
    return max(min(depth / KSDMA_DEPTH_CAP_CM, 1.0), flooded)


def build_nx_graph(edges: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame | None = None) -> nx.Graph:
    """Build the routing graph. Fails loudly if the hazard columns are missing."""
    required = {"u", "v", "ksdma_max", "flooded_2018_share"}
    missing = required - set(edges.columns)
    if missing:
        raise ValueError(
            f"road_edges is missing columns: {sorted(missing)}. "
            "Run 07_hazard_on_roads.py first."
        )

    G = nx.Graph()
    for row in edges.itertuples(index=False):
        u, v = int(row.u), int(row.v)
        geometry = getattr(row, "geometry", None)
        length = _number(getattr(row, "length_m", None), default=np.nan)
        if not np.isfinite(length) or length <= 0:
            length = float(geometry.length) if geometry is not None else 0.0
        if length <= 0:
            LOGGER.warning("Skipping invalid edge %s-%s (length=%s)", u, v, length)
            continue

        flooded = min(1.0, max(0.0, _number(row.flooded_2018_share)))
        G.add_edge(
            u, v, length_m=length,
            hazard=edge_hazard(row.ksdma_max, row.flooded_2018_share),
            flooded_2018=flooded, geometry=geometry,
        )
    if G.number_of_edges() == 0:
        raise ValueError("The road graph contains no valid edges.")

    if nodes is not None:   # node coordinates for the A* heuristic
        for n, geom in zip(nodes["node_id"], nodes.geometry):
            if int(n) in G:
                G.nodes[int(n)]["x"], G.nodes[int(n)]["y"] = geom.x, geom.y

    hz = np.array([d["hazard"] for *_, d in G.edges(data=True)])
    LOGGER.info("Edge hazard: %.1f%% of edges > 0 | %.1f%% >= %.1f | mean %.3f",
                100 * (hz > 0).mean(), 100 * (hz >= HIGH_HAZARD_THRESHOLD).mean(),
                HIGH_HAZARD_THRESHOLD, hz.mean())
    if not (hz > 0).any():
        raise ValueError("All edge hazards are 0; check ksdma_max / flooded_2018_share.")
    return G


def risk_weight(u, v, data, lam: float) -> float:
    return data["length_m"] * (1.0 + lam * data["hazard"])


def euclid_heuristic(G: nx.Graph):
    """Straight-line distance: admissible because cost >= length_m >= Euclidean distance."""
    def h(a, b):
        na, nb = G.nodes[a], G.nodes[b]
        if "x" not in na or "x" not in nb:
            return 0.0
        return math.hypot(na["x"] - nb["x"], na["y"] - nb["y"])
    return h


def route_metrics(G: nx.Graph, path: list) -> dict:
    """Distance-weighted hazard and flood exposure of a route."""
    if len(path) < 2:
        return dict(length_m=0.0, mean_hazard=0.0, max_hazard=0.0,
                    high_hazard_share=0.0, flooded_2018_share=0.0, edge_count=0)

    lengths, hazards, flooded = [], [], []
    for a, b in zip(path[:-1], path[1:]):
        data = G[a][b]
        lengths.append(data["length_m"])
        hazards.append(data["hazard"])
        flooded.append(data["flooded_2018"])

    lengths_a = np.array(lengths)
    total = float(lengths_a.sum())
    haz_a = np.array(hazards)
    return dict(
        length_m=round(total, 1),
        mean_hazard=round(float((haz_a * lengths_a).sum() / total), 4),
        max_hazard=round(float(haz_a.max()), 4),
        high_hazard_share=round(float(lengths_a[haz_a >= HIGH_HAZARD_THRESHOLD].sum() / total), 4),
        flooded_2018_share=round(float((np.array(flooded) * lengths_a).sum() / total), 4),
        edge_count=len(path) - 1,
    )


def path_to_linestring(G: nx.Graph, path: list) -> LineString | None:
    """Join edge geometries while respecting the direction of travel."""
    coords = []
    for a, b in zip(path[:-1], path[1:]):
        geom = G[a][b].get("geometry")
        if geom is None or geom.is_empty:
            continue
        edge_coords = list(geom.coords)
        if coords:
            gap_first = (coords[-1][0] - edge_coords[0][0]) ** 2 + (coords[-1][1] - edge_coords[0][1]) ** 2
            gap_last = (coords[-1][0] - edge_coords[-1][0]) ** 2 + (coords[-1][1] - edge_coords[-1][1]) ** 2
            if gap_last < gap_first:
                edge_coords.reverse()
            coords.extend(edge_coords[1:])
        else:
            # First edge orientation: look ahead so it ends at the shared node.
            nxt = G[path[1]][path[2]].get("geometry") if len(path) > 2 else None
            if nxt is not None and not nxt.is_empty:
                ref = list(nxt.coords)
                ends = [ref[0], ref[-1]]
                d_first = min((edge_coords[0][0] - e[0]) ** 2 + (edge_coords[0][1] - e[1]) ** 2 for e in ends)
                d_last = min((edge_coords[-1][0] - e[0]) ** 2 + (edge_coords[-1][1] - e[1]) ** 2 for e in ends)
                if d_first < d_last:
                    edge_coords.reverse()
            coords.extend(edge_coords)
    return LineString(coords) if len(coords) >= 2 else None


def nearest_safe_camp(camps: pd.DataFrame, origin_node, G, component_nodes):
    """Nearest safe camp by road length. Returns (node_id, camp access cost) or None.
    Unsafe camps (KSDMA 100-yr zone or 2018 flood mask) are never used as destinations."""
    safe = camps[(~camps["unsafe"].fillna(True).astype(bool)) & camps["node_id"].isin(component_nodes)]
    if safe.empty:
        return None
    try:
        dist = nx.single_source_dijkstra_path_length(G, origin_node, weight="length_m")
    except nx.NodeNotFound:
        return None

    best = None
    for row in safe.itertuples():
        node = int(row.node_id)
        if node in dist:
            access = _number(getattr(row, "access_cost_m", 0.0))
            key = (dist[node], access, node)
            if best is None or key < best:
                best = key
    return (best[2], best[1]) if best else None


def main():
    LOGGER.info("Loading road graph and snapped locations ...")
    edges = gpd.read_file(cfg.output_path("road_edges"))
    nodes = gpd.read_file(cfg.output_path("road_nodes"))
    camps = pd.read_csv(cfg.output_path("camps_nodes"))
    origins = pd.read_csv(cfg.output_path("origins_nodes"))

    if origins.empty or camps.empty:
        raise ValueError("origins_nodes.csv or camps_nodes.csv is empty.")
    for column in ("node_id", "unsafe"):
        if column not in camps.columns:
            raise ValueError(f"camps_nodes is missing required column: {column}")
    for column in ("node_id", "village", "population_2018", "access_cost_m", "poorly_connected"):
        if column not in origins.columns:
            raise ValueError(f"origins_nodes is missing required column: {column}")

    G = build_nx_graph(edges, nodes)
    largest_cc = max(nx.connected_components(G), key=len)
    G_main = G.subgraph(largest_cc).copy()
    heuristic = euclid_heuristic(G_main)
    LOGGER.info("Largest component: %d nodes, %d edges", G_main.number_of_nodes(), G_main.number_of_edges())

    weight = lambda u, v, data: risk_weight(u, v, data, LAMBDA_DEFAULT)

    records, geom_rows, skipped = [], [], []
    for orig in origins.itertuples(index=False):
        o_node, village = int(orig.node_id), str(orig.village)

        if o_node not in G_main:
            LOGGER.warning("Origin %s is outside the largest component; skipped", village)
            skipped.append(village)
            continue
        found = nearest_safe_camp(camps, o_node, G_main, largest_cc)
        if found is None:
            LOGGER.warning("No reachable safe camp for %s; skipped", village)
            skipped.append(village)
            continue
        camp_node, camp_access = found

        try:
            sp_path = nx.shortest_path(G_main, o_node, camp_node, weight="length_m")
            ra_path = nx.astar_path(G_main, o_node, camp_node, weight=weight, heuristic=heuristic)
        except (nx.NetworkXNoPath, nx.NodeNotFound) as exc:
            LOGGER.warning("Routing failed for %s: %s", village, exc)
            skipped.append(village)
            continue

        sp_metrics, ra_metrics = route_metrics(G_main, sp_path), route_metrics(G_main, ra_path)
        origin_access = _number(orig.access_cost_m)
        for m in (sp_metrics, ra_metrics):
            m["total_length_with_access_m"] = round(m["length_m"] + origin_access + camp_access, 1)
            m["same_route_as_shortest"] = bool(ra_path == sp_path)

        base = dict(
            village=village, population_2018=_number(orig.population_2018),
            origin_node=o_node, camp_node=camp_node,
            access_cost_m=origin_access, camp_access_cost_m=round(camp_access, 1),
            poorly_connected=bool(orig.poorly_connected), lambda_used=LAMBDA_DEFAULT,
        )
        records.extend([{**base, "route_type": "shortest_path", **sp_metrics},
                        {**base, "route_type": "risk_aware_astar", **ra_metrics}])
        for rtype, lam, path in (("shortest_path", 0.0, sp_path), ("risk_aware_astar", LAMBDA_DEFAULT, ra_path)):
            geom = path_to_linestring(G_main, path)
            if geom is not None:
                geom_rows.append({"village": village, "route_type": rtype, "lambda": lam, "geometry": geom})

    if not records:
        raise RuntimeError("No origins were routed. Check node IDs, components and safe camps.")

    results = pd.DataFrame(records)
    out_csv = output_file("tables", "08_routing_results.csv")
    results.to_csv(out_csv, index=False)
    LOGGER.info("Saved %s (%d rows)", out_csv, len(results))

    if geom_rows:
        routes_gdf = gpd.GeoDataFrame(geom_rows, geometry="geometry", crs=edges.crs or cfg.CRS_WORKING)
        routes_gdf.to_file(output_file("maps", "08_routing_routes.gpkg"), driver="GPKG")

    sp = results[results.route_type == "shortest_path"].set_index("village")
    ra = results[results.route_type == "risk_aware_astar"].set_index("village")
    changed = int((~ra["same_route_as_shortest"]).sum())
    stats = {
        "lambda": LAMBDA_DEFAULT,
        "hazard_definition": f"max(KSDMA 100-yr depth cm / {KSDMA_DEPTH_CAP_CM:g} capped at 1, flooded_2018_share); "
                             "NaN depth = dry = 0; 2018 mask is a proxy floor, not a closure",
        "high_hazard_threshold": HIGH_HAZARD_THRESHOLD,
        "origins_total": int(len(origins)),
        "origins_routed": int(len(sp)),
        "origins_skipped": skipped,
        "routes_changed_by_risk_awareness": changed,
        "mean_hazard_shortest": round(float(sp["mean_hazard"].mean()), 4),
        "mean_hazard_risk_aware": round(float(ra["mean_hazard"].mean()), 4),
        "high_hazard_share_shortest": round(float(sp["high_hazard_share"].mean()), 4),
        "high_hazard_share_risk_aware": round(float(ra["high_hazard_share"].mean()), 4),
        "median_extra_length_m": round(float((ra["length_m"] - sp["length_m"]).median()), 1),
        "poorly_connected_routed": int(sp["poorly_connected"].sum()),
        "note": "Distances exclude straight-line access gaps unless total_length_with_access_m is used. "
                "Poorly connected origins have low-confidence access.",
    }
    output_file("tables", "08_routing_stats.json").write_text(json.dumps(stats, indent=2))
    LOGGER.info("%s", json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()