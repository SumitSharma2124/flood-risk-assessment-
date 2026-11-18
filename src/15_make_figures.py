
"""
Task 16B - Final Project Figure Generation

Run from project root:
    python src/15_make_figures.py

Reads existing project outputs only.
Does not retrain models or modify analysis results.

Output:
    outputs/figures/15_01_fuzzy_ahp_weights.png
    outputs/figures/15_02_baseline_validation.png
    outputs/figures/15_03_test_model_comparison.png
    outputs/figures/15_04_hazard_validation.png
    outputs/figures/15_05_integrated_risk_validation.png
    outputs/figures/15_06_routing_comparison.png
    outputs/figures/15_07_agent_advisory_priorities.png
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "outputs"
TABLES_DIR = OUTPUT_DIR / "tables"
STATS_DIR = OUTPUT_DIR / "stats"
FIGURES_DIR = OUTPUT_DIR / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)

LOGGER = logging.getLogger("task16_figures")


# ============================================================
# 2. FILE READING HELPERS
# ============================================================

def read_csv(filename: str) -> pd.DataFrame | None:
    """Read a CSV from outputs/tables."""

    path = TABLES_DIR / filename

    if not path.exists():
        LOGGER.warning("Missing CSV: %s", filename)
        return None

    try:
        df = pd.read_csv(path)
        LOGGER.info("Read CSV: %s (%d rows)", filename, len(df))
        return df

    except Exception as exc:
        LOGGER.warning("Could not read %s: %s", filename, exc)
        return None


def read_json(filename: str) -> dict | list | None:
    """
    Read a JSON file.

    Search outputs/tables first because most project metrics
    are stored there, then outputs/stats.
    """

    candidates = [
        TABLES_DIR / filename,
        STATS_DIR / filename,
    ]

    for path in candidates:

        if not path.exists():
            continue

        try:
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)

            LOGGER.info("Read JSON: %s", path.name)
            return data

        except Exception as exc:
            LOGGER.warning("Could not read %s: %s", path, exc)

    LOGGER.warning("Missing JSON: %s", filename)
    return None


def find_column(
    df: pd.DataFrame,
    candidates: list[str]
) -> str | None:
    """Find a column using case-insensitive matching."""

    lookup = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


def save_figure(filename: str) -> None:
    """Save the current Matplotlib figure."""

    path = FIGURES_DIR / filename

    plt.tight_layout()
    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close()

    LOGGER.info("Saved figure: %s", path)


# ============================================================
# 3. FIGURE 1 - FUZZY-AHP WEIGHTS
# ============================================================

def figure_fuzzy_weights():

    df = read_csv("08_fuzzy_ahp_weights.csv")

    if df is None or df.empty:
        return

    name_col = find_column(
        df,
        ["criterion", "factor", "parameter", "name", "feature"]
    )

    weight_col = find_column(
        df,
        [
            "weight",
            "final_weight",
            "crisp_weight",
            "normalised_weight"
        ]
    )

    if name_col is None or weight_col is None:
        LOGGER.warning("Fuzzy-AHP columns not recognised.")
        return

    values = pd.to_numeric(
        df[weight_col],
        errors="coerce"
    )

    valid = values.notna()

    if not valid.any():
        return

    plt.figure(figsize=(9, 5))

    plt.barh(
        df.loc[valid, name_col].astype(str),
        values[valid]
    )

    plt.xlabel("Criterion Weight")
    plt.ylabel("Criterion")
    plt.title("Fuzzy-AHP Criterion Weights")

    plt.gca().invert_yaxis()

    save_figure("15_01_fuzzy_ahp_weights.png")


# ============================================================
# 4. FIGURE 2 - RANDOM FOREST VS XGBOOST
# ============================================================

def figure_baseline_validation():

    df = read_csv("09_baseline_validation_metrics.csv")

    if df is None or df.empty:
        return

    model_col = find_column(
        df,
        ["model", "model_name", "estimator"]
    )

    if model_col is None:
        LOGGER.warning("Baseline model column not found.")
        return

    metric_candidates = {
        "MAE": ["mae", "mean_absolute_error"],
        "RMSE": ["rmse", "root_mean_squared_error"],
        "R2": ["r2", "r_squared", "r2_score"],
    }

    available = []

    for label, candidates in metric_candidates.items():

        col = find_column(df, candidates)

        if col is not None:
            available.append((label, col))

    if not available:
        LOGGER.warning("No baseline metrics found.")
        return

    fig, axes = plt.subplots(
        1,
        len(available),
        figsize=(5 * len(available), 4)
    )

    if len(available) == 1:
        axes = [axes]

    for ax, (label, column) in zip(axes, available):

        values = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        valid = values.notna()

        bars = ax.bar(
            df.loc[valid, model_col].astype(str),
            values[valid]
        )

        ax.set_title(label)
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=20)

        for bar, value in zip(bars, values[valid]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8
            )

    fig.suptitle(
        "Random Forest vs XGBoost - Validation Performance"
    )

    save_figure("15_02_baseline_validation.png")


# ============================================================
# 5. FIGURE 3 - RF VS XGBOOST VS U-NET TEST PERFORMANCE
# ============================================================

def figure_test_performance():

    data = read_json("10_all_models_test_metrics.json")

    if not isinstance(data, dict):
        LOGGER.warning("Test metrics JSON unavailable.")
        return

    metrics = data.get("metrics")

    if not isinstance(metrics, dict):
        LOGGER.warning("Test metrics structure not recognised.")
        return

    model_labels = {
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
        "unet": "U-Net",
    }

    metric_names = ["MAE", "RMSE", "R2"]

    metric_titles = {
        "MAE": "Mean Absolute Error",
        "RMSE": "Root Mean Squared Error",
        "R2": "R² Score",
    }

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    for ax, metric in zip(axes, metric_names):

        names = []
        values = []

        for key, label in model_labels.items():

            model_metrics = metrics.get(key)

            if not isinstance(model_metrics, dict):
                continue

            value = model_metrics.get(metric)

            if value is None:
                continue

            names.append(label)
            values.append(float(value))

        if not values:
            ax.set_visible(False)
            continue

        bars = ax.bar(names, values)

        ax.set_title(metric_titles[metric])
        ax.set_ylabel(metric)
        ax.tick_params(
            axis="x",
            rotation=20
        )

        for bar, value in zip(bars, values):

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

    test_cells = data.get("test_cells", "N/A")

    fig.suptitle(
        "RF vs XGBoost vs U-Net - "
        f"Held-out Test Performance (n={test_cells})"
    )

    save_figure("15_03_test_model_comparison.png")


# ============================================================
# 6. FIGURE 4 - 2018 FLOOD HAZARD VALIDATION
# ============================================================

def figure_hazard_validation():

    df = read_csv("13_hazard_validation_metrics.csv")

    if df is None or df.empty:
        return

    model_col = find_column(
        df,
        ["model", "model_name"]
    )

    metric_col = find_column(
        df,
        ["metric", "metric_name"]
    )

    value_col = find_column(
        df,
        ["value", "score", "metric_value"]
    )

    if model_col and metric_col and value_col:

        selected = df[
            df[metric_col]
            .astype(str)
            .str.lower()
            .isin([
                "auc",
                "roc_auc",
                "average_precision",
                "ap"
            ])
        ].copy()

        if selected.empty:
            LOGGER.warning(
                "No AUC/AP rows found in hazard validation."
            )
            return

        selected[value_col] = pd.to_numeric(
            selected[value_col],
            errors="coerce"
        )

        selected = selected.dropna(
            subset=[value_col]
        )

        if selected.empty:
            return

        pivot = selected.pivot_table(
            index=model_col,
            columns=metric_col,
            values=value_col,
            aggfunc="mean"
        )

        ax = pivot.plot(
            kind="bar",
            figsize=(9, 5)
        )

        ax.set_ylabel("Score")
        ax.set_title(
            "2018 Sentinel-1 Flood Extent - Hazard Validation"
        )

        ax.tick_params(
            axis="x",
            rotation=20
        )

        save_figure("15_04_hazard_validation.png")
        return

    # Support wide-format tables.

    metric_columns = [
        col for col in df.columns
        if any(
            term in str(col).lower()
            for term in ["auc", "precision", "capture"]
        )
    ]

    if not metric_columns:
        LOGGER.warning(
            "Hazard validation table format not recognised."
        )
        return

    ax = df.set_index(
        df.columns[0]
    )[metric_columns].plot(
        kind="bar",
        figsize=(10, 5)
    )

    ax.set_title(
        "2018 Sentinel-1 Flood Extent - Hazard Validation"
    )

    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=20)

    save_figure("15_04_hazard_validation.png")


# ============================================================
# 7. FIGURE 5 - INTEGRATED RISK VALIDATION
# ============================================================

def figure_risk_validation():

    df = read_csv("13_risk_validation_metrics.csv")

    if df is None or df.empty:
        return

    model_col = find_column(
        df,
        ["model", "model_name"]
    )

    metric_col = find_column(
        df,
        ["metric", "metric_name"]
    )

    value_col = find_column(
        df,
        ["value", "score", "metric_value"]
    )

    if model_col and metric_col and value_col:

        selected = df[
            df[metric_col]
            .astype(str)
            .str.lower()
            .isin([
                "auc",
                "roc_auc",
                "average_precision",
                "ap"
            ])
        ].copy()

        if selected.empty:
            LOGGER.warning(
                "No AUC/AP rows found in risk validation."
            )
            return

        selected[value_col] = pd.to_numeric(
            selected[value_col],
            errors="coerce"
        )

        selected = selected.dropna(
            subset=[value_col]
        )

        if selected.empty:
            return

        pivot = selected.pivot_table(
            index=model_col,
            columns=metric_col,
            values=value_col,
            aggfunc="mean"
        )

        ax = pivot.plot(
            kind="bar",
            figsize=(9, 5)
        )

        ax.set_ylabel("Score")
        ax.set_title(
            "Integrated Risk - 2018 Flood Validation"
        )

        ax.tick_params(
            axis="x",
            rotation=20
        )

        save_figure("15_05_integrated_risk_validation.png")
        return

    metric_columns = [
        col for col in df.columns
        if any(
            term in str(col).lower()
            for term in ["auc", "precision", "capture"]
        )
    ]

    if not metric_columns:
        LOGGER.warning(
            "Risk validation table format not recognised."
        )
        return

    ax = df.set_index(
        df.columns[0]
    )[metric_columns].plot(
        kind="bar",
        figsize=(10, 5)
    )

    ax.set_title(
        "Integrated Risk - 2018 Flood Validation"
    )

    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=20)

    save_figure("15_05_integrated_risk_validation.png")


# ============================================================
# 8. FIGURE 6 - ROUTING COMPARISON
# ============================================================

def figure_routing_comparison():

    df = read_csv("08_routing_results.csv")

    if df is None or df.empty:
        return

    route_col = find_column(
        df,
        ["route_type"]
    )

    hazard_col = find_column(
        df,
        ["mean_hazard"]
    )

    length_col = find_column(
        df,
        ["total_length_with_access_m"]
    )

    high_col = find_column(
        df,
        ["high_hazard_share"]
    )

    if not route_col or not hazard_col or not length_col:
        LOGGER.warning(
            "Routing result columns not recognised."
        )
        return

    df[hazard_col] = pd.to_numeric(
        df[hazard_col],
        errors="coerce"
    )

    df[length_col] = pd.to_numeric(
        df[length_col],
        errors="coerce"
    )

    if high_col:
        df[high_col] = pd.to_numeric(
            df[high_col],
            errors="coerce"
        )

    grouped = df.groupby(route_col)

    metrics_to_plot = {
        "Mean Route Hazard":
            grouped[hazard_col].mean(),

        "Mean Total Journey Length (m)":
            grouped[length_col].mean(),
    }

    if high_col:
        metrics_to_plot["Mean High-Hazard Share"] = (
            grouped[high_col].mean()
        )

    fig, axes = plt.subplots(
        1,
        len(metrics_to_plot),
        figsize=(5 * len(metrics_to_plot), 4)
    )

    if len(metrics_to_plot) == 1:
        axes = [axes]

    for ax, (title, values) in zip(
        axes,
        metrics_to_plot.items()
    ):

        values.plot(
            kind="bar",
            ax=ax
        )

        ax.set_title(title)
        ax.set_ylabel(title)
        ax.tick_params(
            axis="x",
            rotation=20
        )

    fig.suptitle(
        "Shortest Path vs Risk-Aware A* Routing"
    )

    save_figure("15_06_routing_comparison.png")


# ============================================================
# 9. FIGURE 7 - MULTI-AGENT ADVISORY PRIORITIES
# ============================================================

def figure_agent_priorities():

    df = read_csv(
        "14_agent_village_advisories.csv"
    )

    if df is None or df.empty:
        return

    priority_col = find_column(
        df,
        ["priority"]
    )

    if priority_col is None:
        LOGGER.warning(
            "Advisory priority column not found."
        )
        return

    counts = (
        df[priority_col]
        .fillna("Unknown")
        .value_counts()
    )

    plt.figure(figsize=(9, 5))

    plt.bar(
        counts.index.astype(str),
        counts.values
    )

    plt.xlabel("Advisory Priority")
    plt.ylabel("Number of Villages")

    plt.title(
        "Multi-Agent System - Village Advisory Priorities"
    )

    plt.xticks(
        rotation=20,
        ha="right"
    )

    save_figure(
        "15_07_agent_advisory_priorities.png"
    )


# ============================================================
# 10. MAIN
# ============================================================

def main():

    LOGGER.info("=" * 60)
    LOGGER.info("TASK 16B - FINAL FIGURE GENERATION")
    LOGGER.info("=" * 60)

    functions = [
        figure_fuzzy_weights,
        figure_baseline_validation,
        figure_test_performance,
        figure_hazard_validation,
        figure_risk_validation,
        figure_routing_comparison,
        figure_agent_priorities,
    ]

    generated = 0
    failed = 0

    for function in functions:

        before = set(
            FIGURES_DIR.glob("15_*.png")
        )

        try:
            function()

            after = set(
                FIGURES_DIR.glob("15_*.png")
            )

            if after - before:
                generated += 1

        except Exception:

            failed += 1

            LOGGER.exception(
                "Failed: %s",
                function.__name__
            )

    LOGGER.info("=" * 60)
    LOGGER.info("FIGURE GENERATION COMPLETED")
    LOGGER.info("New figures generated: %d", generated)
    LOGGER.info("Failed figure functions: %d", failed)
    LOGGER.info("Output directory: %s", FIGURES_DIR)
    LOGGER.info("=" * 60)


if __name__ == "__main__":
    main()
