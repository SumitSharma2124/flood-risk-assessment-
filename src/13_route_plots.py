
"""
Task 9c – Route plots.
Run from the src directory: python 08_route_plots.py

Outputs:
    outputs/figures/08_routing_sample_map.png
    outputs/figures/08_routing_lambda_curve.png
"""

import logging
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

import config as cfg

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
LOGGER = logging.getLogger(__name__)

SAMPLE_VILLAGES = 6

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def output_file(folder, filename):
    """Create output directory if necessary and return file path."""
    target = OUTPUT_DIR / folder
    target.mkdir(parents=True, exist_ok=True)
    return target / filename


def _require_columns(frame, columns, filename):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(
            f"{filename} is missing columns: {sorted(missing)}"
        )


def plot_sample_map(routes_gdf, edges_gdf, camps_df, results_df):

    _require_columns(
        routes_gdf, {"village", "route_type", "geometry"}, "routing_routes"
    )
    _require_columns(edges_gdf, {"geometry"}, "road_edges")
    _require_columns(camps_df, {"unsafe", "x", "y"}, "camps_nodes")
    _require_columns(
        results_df,
        {"poorly_connected", "route_type", "village"},
        "routing_results",
    )

    # Select 3 poorly connected and 3 well-connected villages
    shortest = results_df[
        results_df["route_type"] == "shortest_path"
    ].drop_duplicates("village")

    poor = shortest[
        shortest["poorly_connected"].fillna(False).astype(bool)
    ]["village"].tolist()

    good = shortest[
        ~shortest["poorly_connected"].fillna(False).astype(bool)
    ]["village"].tolist()

    sample = (poor[:3] + good[:3])[:SAMPLE_VILLAGES]

    if not sample:
        raise ValueError("No successfully routed origins are available to plot.")

    fig, ax = plt.subplots(figsize=(8, 11))

    try:
        # Background road network
        edges_gdf.plot(
            ax=ax,
            color="lightgray",
            linewidth=0.3,
            zorder=1,
        )

        colors = {
            "shortest_path": "steelblue",
            "risk_aware_astar": "darkorange",
        }

        labels = {
            "shortest_path": "Shortest path",
            "risk_aware_astar": "Risk-aware route",
        }

        # Plot selected routes
        selected = routes_gdf[
            routes_gdf["village"].isin(sample)
        ]

        for route_type, color in colors.items():
            subset = selected[
                selected["route_type"] == route_type
            ]

            if not subset.empty:
                subset.plot(
                    ax=ax,
                    color=color,
                    linewidth=1.8,
                    zorder=3,
                )

        # Safe camp markers
        safe = camps_df[
            ~camps_df["unsafe"].fillna(True).astype(bool)
        ]

        if not safe.empty:
            ax.scatter(
                safe["x"],
                safe["y"],
                s=12,
                c="green",
                zorder=4,
            )

        # Legend
        handles = [
            mpatches.Patch(color=color, label=labels[key])
            for key, color in colors.items()
        ]

        handles.append(
            mpatches.Patch(color="green", label="Safe camp")
        )

        ax.legend(
            handles=handles,
            loc="upper right",
            fontsize=8,
        )

        ax.set_title(
            f"Route comparison — {len(sample)} sample origins\n"
            "(orange = risk-aware route, blue = shortest path)"
        )

        ax.set_xlabel("Easting [m]")
        ax.set_ylabel("Northing [m]")

        fig.tight_layout()

        # Save figure
        out = output_file(
            "figures",
            "08_routing_sample_map.png",
        )

        fig.savefig(out, dpi=150, bbox_inches="tight")
        LOGGER.info("Saved %s", out)

    finally:
        plt.close(fig)


def plot_lambda_curve(summary_df):

    _require_columns(
        summary_df,
        {
            "lambda_value",
            "median_total_length_m",
            "mean_high_hazard_share",
            "mean_route_hazard",
        },
        "routing_lambda_summary",
    )

    if summary_df.empty:
        raise ValueError("routing_lambda_summary contains no rows.")

    summary_df = summary_df.sort_values("lambda_value")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    try:
        ax1, ax2 = axes

        # Route length vs lambda
        ax1.plot(
            summary_df["lambda_value"],
            summary_df["median_total_length_m"] / 1000,
            marker="o",
            color="steelblue",
        )

        ax1.set_xlabel("λ (risk-aversion weight)")
        ax1.set_ylabel("Median total route length (km)")
        ax1.set_title("Route length vs λ")
        ax1.grid(True, linestyle="--", alpha=0.5)

        # Hazard exposure vs lambda
        ax2.plot(
            summary_df["lambda_value"],
            summary_df["mean_high_hazard_share"],
            marker="o",
            color="darkorange",
            label="High-hazard share",
        )

        ax2.plot(
            summary_df["lambda_value"],
            summary_df["mean_route_hazard"],
            marker="s",
            color="firebrick",
            linestyle="--",
            label="Mean hazard",
        )

        ax2.set_xlabel("λ (risk-aversion weight)")
        ax2.set_ylabel("Hazard share / index")
        ax2.set_title("Hazard exposure vs λ")
        ax2.legend(fontsize=8)
        ax2.grid(True, linestyle="--", alpha=0.5)

        fig.suptitle(
            "Distance–hazard trade-off across λ values",
            fontsize=11,
        )

        fig.tight_layout()

        # Save figure
        out = output_file(
            "figures",
            "08_routing_lambda_curve.png",
        )

        fig.savefig(out, dpi=150, bbox_inches="tight")
        LOGGER.info("Saved %s", out)

    finally:
        plt.close(fig)


def main():

    LOGGER.info("Loading routing data...")

    # Read generated Task 9 files from root/outputs
    routes = gpd.read_file(
        output_file("maps", "08_routing_routes.gpkg")
    )

    edges = gpd.read_file(
        cfg.output_path("road_edges")
    )

    camps = pd.read_csv(
        cfg.output_path("camps_nodes")
    )

    results = pd.read_csv(
        output_file("tables", "08_routing_results.csv")
    )

    summary = pd.read_csv(
        output_file("stats", "08_routing_lambda_summary.csv")
    )

    LOGGER.info("Generating route comparison map...")
    plot_sample_map(routes, edges, camps, results)

    LOGGER.info("Generating lambda sensitivity curve...")
    plot_lambda_curve(summary)

    LOGGER.info("All routing plots generated successfully.")


if __name__ == "__main__":
    main()
