
"""
Task 16A - Consolidate existing project results.

Run from project root:
    python src/15_collect_results.py

Reads existing CSV and JSON outputs.
Does not retrain models or rerun analysis.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"

TABLES_DIR = OUTPUT_DIR / "tables"
STATS_DIR = OUTPUT_DIR / "stats"

CONSOLIDATED_CSV = TABLES_DIR / "15_consolidated_metrics.csv"
CONSOLIDATED_JSON = STATS_DIR / "15_consolidated_results.json"
AUDIT_JSON = STATS_DIR / "15_collection_audit.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)

LOGGER = logging.getLogger("result_collector")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def is_finite_number(value):
    """Return True if a value can be converted to a finite number."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def flatten_json(data, prefix=""):
    """
    Flatten nested JSON dictionaries into key-value pairs.
    Lists and complex objects are retained as JSON strings.
    """
    flattened = {}

    if isinstance(data, dict):
        for key, value in data.items():
            name = f"{prefix}.{key}" if prefix else str(key)

            if isinstance(value, dict):
                flattened.update(flatten_json(value, name))
            elif isinstance(value, list):
                flattened[name] = json.dumps(value)
            else:
                flattened[name] = value

    else:
        flattened[prefix or "value"] = data

    return flattened


def add_metric(records, source, metric, value, metric_type="value"):
    """Append a single metric if it is meaningful."""
    if value is None:
        return

    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return
        value = float(value)

    elif isinstance(value, (int, np.integer)):
        value = int(value)

    elif isinstance(value, (bool, np.bool_)):
        value = bool(value)

    records.append({
        "source_file": str(source.relative_to(OUTPUT_DIR)),
        "metric": str(metric),
        "value": value,
        "metric_type": metric_type,
    })


# ---------------------------------------------------------------------
# JSON collection
# ---------------------------------------------------------------------

def collect_json_file(path, records, audit):
    """Extract scalar metrics from one JSON file."""
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        flattened = flatten_json(data)

        for key, value in flattened.items():
            add_metric(
                records,
                path,
                key,
                value,
                "json_value"
            )

        audit["json_files_read"] += 1

        LOGGER.info("Read JSON: %s", path.name)

    except Exception as exc:
        audit["errors"].append({
            "file": str(path.relative_to(OUTPUT_DIR)),
            "error": str(exc),
        })

        LOGGER.warning("Could not read JSON %s: %s", path.name, exc)


# ---------------------------------------------------------------------
# CSV collection
# ---------------------------------------------------------------------

def collect_csv_file(path, records, audit):
    """
    Extract row counts, numeric summaries and categorical counts
    from one CSV.

    Original CSV files remain unchanged.
    """
    try:
        df = pd.read_csv(path, low_memory=False)

        relative_path = str(path.relative_to(OUTPUT_DIR))

        add_metric(
            records, path, "row_count", len(df), "file_structure"
        )

        add_metric(
            records, path, "column_count", len(df.columns),
            "file_structure"
        )

        for column in df.columns:
            series = df[column]

            # Numeric columns: collect descriptive summaries.
            numeric = pd.to_numeric(series, errors="coerce")
            valid = numeric.dropna()

            if len(valid) > 0:
                add_metric(
                    records, path, f"{column}.mean",
                    valid.mean(), "numeric_summary"
                )
                add_metric(
                    records, path, f"{column}.median",
                    valid.median(), "numeric_summary"
                )
                add_metric(
                    records, path, f"{column}.min",
                    valid.min(), "numeric_summary"
                )
                add_metric(
                    records, path, f"{column}.max",
                    valid.max(), "numeric_summary"
                )

            # Missing-value count for every column.
            missing = int(series.isna().sum())

            add_metric(
                records, path, f"{column}.missing_count",
                missing, "data_quality"
            )

            # Categorical columns: unique-value count.
            if not pd.api.types.is_numeric_dtype(series):
                unique_count = int(series.nunique(dropna=True))

                add_metric(
                    records, path, f"{column}.unique_count",
                    unique_count, "categorical_summary"
                )

        audit["csv_files_read"] += 1

        audit["csv_inventory"].append({
            "file": relative_path,
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
        })

        LOGGER.info(
            "Read CSV: %s (%d rows, %d columns)",
            path.name, len(df), len(df.columns)
        )

    except Exception as exc:
        audit["errors"].append({
            "file": str(path.relative_to(OUTPUT_DIR)),
            "error": str(exc),
        })

        LOGGER.warning("Could not read CSV %s: %s", path.name, exc)


# ---------------------------------------------------------------------
# Main collector
# ---------------------------------------------------------------------

def main():

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    STATS_DIR.mkdir(parents=True, exist_ok=True)

    audit = {
        "project": "Alappuzha Flood Risk Assessment",
        "collection_status": "started",
        "csv_files_found": 0,
        "csv_files_read": 0,
        "json_files_found": 0,
        "json_files_read": 0,
        "csv_inventory": [],
        "missing_expected_files": [],
        "errors": [],
    }

    records = []

    LOGGER.info("Starting results collection...")
    LOGGER.info("Output directory: %s", OUTPUT_DIR)

    # Exclude this collector's own outputs.
    excluded = {
        CONSOLIDATED_CSV.resolve(),
        CONSOLIDATED_JSON.resolve(),
        AUDIT_JSON.resolve(),
    }

    csv_files = sorted(
        p for p in OUTPUT_DIR.rglob("*.csv")
        if p.resolve() not in excluded
    )

    json_files = sorted(
        p for p in OUTPUT_DIR.rglob("*.json")
        if p.resolve() not in excluded
    )

    audit["csv_files_found"] = len(csv_files)
    audit["json_files_found"] = len(json_files)

    LOGGER.info(
        "Found %d CSV files and %d JSON files.",
        len(csv_files), len(json_files)
    )

    # Read all existing CSV result tables.
    for path in csv_files:
        collect_csv_file(path, records, audit)

    # Read all existing JSON statistics and reports.
    for path in json_files:
        collect_json_file(path, records, audit)

    # Important outputs expected from the completed workflow.
    expected_files = [
        OUTPUT_DIR / "tables" / "08_routing_results.csv",
        OUTPUT_DIR / "tables" / "08_routing_stats.json",
        OUTPUT_DIR / "tables" / "14_agent_route_recommendations.csv",
        OUTPUT_DIR / "tables" / "14_agent_village_advisories.csv",
        OUTPUT_DIR / "stats" / "14_agent_situation_report.json",
    ]

    for path in expected_files:
        if not path.exists():
            audit["missing_expected_files"].append(
                str(path.relative_to(OUTPUT_DIR))
            )

    # Build consolidated metrics table.
    metrics_df = pd.DataFrame(
        records,
        columns=[
            "source_file",
            "metric",
            "value",
            "metric_type",
        ],
    )

    metrics_df.to_csv(CONSOLIDATED_CSV, index=False)

    # Build machine-readable consolidated result.
    consolidated = {
        "project": "AI-Based Flood Risk Assessment",
        "study_area": "Alappuzha, Kerala",
        "scenario": "100-year return period",
        "collection_summary": {
            "csv_files_read": audit["csv_files_read"],
            "json_files_read": audit["json_files_read"],
            "total_metrics_collected": len(metrics_df),
        },
        "metrics": records,
    }

    with CONSOLIDATED_JSON.open("w", encoding="utf-8") as f:
        json.dump(consolidated, f, indent=2, default=str)

    # Final audit.
    audit["total_metrics_collected"] = len(metrics_df)
    audit["collection_status"] = (
        "completed_with_warnings"
        if audit["errors"] or audit["missing_expected_files"]
        else "completed"
    )

    with AUDIT_JSON.open("w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, default=str)

    # Final summary.
    LOGGER.info("=" * 60)
    LOGGER.info("RESULT COLLECTION COMPLETED")
    LOGGER.info("=" * 60)

    LOGGER.info("CSV files read: %d", audit["csv_files_read"])
    LOGGER.info("JSON files read: %d", audit["json_files_read"])
    LOGGER.info("Metrics collected: %d", len(metrics_df))
    LOGGER.info("Errors: %d", len(audit["errors"]))
    LOGGER.info(
        "Missing expected files: %d",
        len(audit["missing_expected_files"])
    )

    LOGGER.info("Saved: %s", CONSOLIDATED_CSV)
    LOGGER.info("Saved: %s", CONSOLIDATED_JSON)
    LOGGER.info("Saved: %s", AUDIT_JSON)


if __name__ == "__main__":
    main()
