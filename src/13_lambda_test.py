"""
Task 9b – Lambda sensitivity test.
Run from the src directory: python 08_lambda_test.py
"""

import importlib.util
import logging
from pathlib import Path
import warnings

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd

import config as cfg

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
LOGGER = logging.getLogger(__name__)

LAMBDAS = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def output_file(folder, filename):
    target = OUTPUT_DIR / folder
    target.mkdir(parents=True, exist_ok=True)
    return target / filename
def _load_routing_helpers():
    """Load Task 9a helpers despite numbered script filenames."""
    here = Path(__file__).resolve().parent
    candidates = (
        here / "routing_astar.py",
        here / "08_routing_astar.py",
        here / "13_routing_astar.py",
    )
    module_path = next((p for p in candidates if p.exists()), None)
    if module_path is None:
        raise FileNotFoundError(
            "Could not find routing_astar.py, 08_routing_astar.py, "
            "or 13_routing_astar.py beside this script."
        )
    spec = importlib.util.spec_from_file_location("_routing_astar_helpers", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_nx_graph, module.route_metrics


build_nx_graph, route_metrics = _load_routing_helpers()


def main():
    edges = gpd.read_file(cfg.output_path("road_edges"))
    camps = pd.read_csv(cfg.output_path("camps_nodes"))
    origins = pd.read_csv(cfg.output_path("origins_nodes"))

    if origins.empty or camps.empty:
        raise ValueError("Origins or camps file is empty.")

    G = build_nx_graph(edges, gpd.read_file(cfg.output_path("road_nodes")))
    largest_cc = max(nx.connected_components(G), key=len)
    G_main = G.subgraph(largest_cc).copy()

    safe_camps = camps[
        (~camps["unsafe"].fillna(True).astype(bool))
        & camps["node_id"].isin(largest_cc)
    ]
    if safe_camps.empty:
        raise ValueError("No safe camps are snapped to the largest road component.")

    LOGGER.info("Pre-computing nearest safe camp for each origin …")
    origin_to_camp = {}
    for origin in origins.itertuples(index=False):
        origin_node = int(origin.node_id)
        if origin_node not in G_main:
            continue
        distances = nx.single_source_dijkstra_path_length(
            G_main, origin_node, weight="length_m"
        )
        reachable = [
            (distances[int(camp.node_id)], int(camp.node_id))
            for camp in safe_camps.itertuples()
            if int(camp.node_id) in distances
        ]
        if reachable:
            origin_to_camp[origin_node] = min(reachable)[1]

    if not origin_to_camp:
        raise RuntimeError("No origins can reach a safe camp in the largest component.")

    records = []
    for lam in LAMBDAS:
        LOGGER.info("Running lambda = %.1f …", lam)
        lengths, hazards, high_shares, flood_shares = [], [], [], []

        for origin in origins.itertuples(index=False):
            origin_node = int(origin.node_id)
            camp_node = origin_to_camp.get(origin_node)
            if camp_node is None:
                continue

            weight = lambda u, v, data, value=lam: (
                data["length_m"] * (1.0 + value * data["hazard"])
            )
            try:
                # h(n)=0 is admissible and guarantees optimal weighted cost,
                # although it makes this equivalent to Dijkstra.
                path = nx.astar_path(
                    G_main, origin_node, camp_node,
                    heuristic=lambda *_: 0, weight=weight,
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

            metrics = route_metrics(G_main, path)
            total_distance = metrics["length_m"] + float(
                getattr(origin, "access_cost_m", 0.0)
            )
            lengths.append(total_distance)
            hazards.append(metrics["mean_hazard"])
            high_shares.append(metrics["high_hazard_share"])
            flood_shares.append(metrics["flooded_2018_share"])

        if not lengths:
            LOGGER.warning("No routes found for lambda=%.1f", lam)
            records.append({
                "lambda_value": lam, "origins_routed": 0,
                "median_total_length_m": np.nan,
                "mean_route_hazard": np.nan,
                "mean_high_hazard_share": np.nan,
                "mean_flooded_2018_share": np.nan,
            })
            continue

        records.append({
            "lambda_value": lam,
            "origins_routed": len(lengths),
            "median_total_length_m": round(float(np.median(lengths)), 1),
            "mean_route_hazard": round(float(np.nanmean(hazards)), 4),
            "mean_high_hazard_share": round(float(np.nanmean(high_shares)), 4),
            "mean_flooded_2018_share": round(float(np.nanmean(flood_shares)), 4),
        })
        LOGGER.info(
            "lambda=%.1f | routes=%d | median length=%.0f m | "
            "mean hazard=%.3f | high-hazard share=%.3f",
            lam, len(lengths), np.median(lengths),
            np.nanmean(hazards), np.nanmean(high_shares),
        )

    out = output_file("stats", "08_routing_lambda_summary.csv")
    pd.DataFrame(records).to_csv(out, index=False)
    LOGGER.info("Saved %s", out)


if __name__ == "__main__":
    main()
