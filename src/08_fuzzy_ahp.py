
"""Task 8: Fuzzy-AHP flood-susceptibility mapping for Alappuzha.

Run from the project root:
    python src/09_fuzzy_ahp.py

Uses the existing Task 4 90 m feature stack. Pairwise judgments and land-cover
scores are initial expert assumptions; review and sensitivity-test them before
using the results in the final report.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

from src import config as cfg
from src import utils

LOGGER = utils.get_logger("fuzzy_ahp")


# ============================================================
# 1. CRITERIA
# ============================================================

CRITERIA = (
    "elevation",
    "slope",
    "twi",
    "distance_waterway",
    "landcover_susceptibility",
)

LANDCOVER_BANDS = (
    "trees",
    "cropland",
    "builtup",
    "water",
    "wetland",
    "mangrove",
)

LANDCOVER_SCORES = {
    "trees": 0.20,
    "cropland": 0.75,
    "builtup": 0.85,
    "water": 1.00,
    "wetland": 0.95,
    "mangrove": 0.80,
    "other": 0.45,
}


# ============================================================
# 2. PAIRWISE COMPARISON MATRIX
# ============================================================

# Initial Saaty 1-9 pairwise judgments.
# Row criterion / column criterion.
# These are expert assumptions, not measured parameters.

PAIRWISE = np.array([
    [1,   3,   1/2, 2,   3],
    [1/3, 1,   1/3, 1/2, 2],
    [2,   3,   1,   2,   2],
    [1/2, 2,   1/2, 1,   2],
    [1/3, 1/2, 1/2, 1/2, 1],
], dtype="float64")


# ============================================================
# 3. TRIANGULAR FUZZY NUMBERS
# ============================================================

TFN = {
    1: (1, 1, 1),
    2: (1, 2, 3),
    3: (2, 3, 4),
    4: (3, 4, 5),
    5: (4, 5, 6),
    6: (5, 6, 7),
    7: (6, 7, 8),
    8: (7, 8, 9),
    9: (8, 9, 9),
}

RI = {
    1: 0,
    2: 0,
    3: 0.58,
    4: 0.90,
    5: 1.12,
    6: 1.24,
}

CR_LIMIT = 0.10


# ============================================================
# 4. FUZZY IF-THEN RULES
# ============================================================

# Rule activations are explanatory outputs.
# They are not added again to the weighted Fuzzy-AHP score.

RULES = [
    (
        "R1",
        "Low elevation AND high TWI",
        "Very high",
        "Low-lying, wet terrain",
        "elevation",
        "twi",
        False,
        False,
    ),
    (
        "R2",
        "Near waterways AND high TWI",
        "Very high",
        "Near-water, wet terrain",
        "distance_waterway",
        "twi",
        False,
        False,
    ),
    (
        "R3",
        "Low slope AND high land-cover susceptibility",
        "High",
        "Gentle terrain with susceptible land cover",
        "slope",
        "landcover_susceptibility",
        False,
        False,
    ),
    (
        "R4",
        "Low elevation AND high land-cover susceptibility",
        "High",
        "Low-lying land with susceptible cover",
        "elevation",
        "landcover_susceptibility",
        False,
        False,
    ),
    (
        "R5",
        "High elevation AND far from waterways",
        "Low",
        "Elevated terrain distant from waterways",
        "elevation",
        "distance_waterway",
        True,
        True,
    ),
]

CLASS_NAMES = {
    1: "Very low",
    2: "Low",
    3: "Moderate",
    4: "High",
    5: "Very high",
}


# ============================================================
# 5. CONVERT JUDGMENTS TO FUZZY NUMBERS
# ============================================================

def _to_tfn(value: float) -> tuple[float, float, float]:

    if value >= 1:

        key = int(round(value))

        if not np.isclose(value, key) or key not in TFN:
            raise ValueError(f"Invalid Saaty judgment: {value}")

        return tuple(float(x) for x in TFN[key])

    reciprocal = 1.0 / value
    key = int(round(reciprocal))

    if value <= 0 or not np.isclose(reciprocal, key) or key not in TFN:
        raise ValueError(f"Invalid reciprocal Saaty judgment: {value}")

    low, middle, high = TFN[key]

    return (
        1.0 / high,
        1.0 / middle,
        1.0 / low,
    )


# ============================================================
# 6. FUZZY-AHP WEIGHTS AND CONSISTENCY
# ============================================================

def _calculate_weights(matrix: np.ndarray):

    n = matrix.shape[0]

    if matrix.shape != (n, n) or not np.allclose(
        matrix * matrix.T, 1.0
    ):
        raise ValueError("Pairwise matrix must be square and reciprocal")

    if not np.allclose(np.diag(matrix), 1.0):
        raise ValueError("Pairwise matrix diagonal must equal 1")

    fuzzy = np.empty((n, n, 3), dtype="float64")

    for i in range(n):
        for j in range(n):
            fuzzy[i, j] = _to_tfn(float(matrix[i, j]))

    # Buckley's fuzzy geometric mean method.

    gm = np.empty((n, 3), dtype="float64")

    for k in range(3):
        gm[:, k] = np.prod(
            fuzzy[:, :, k],
            axis=1,
        ) ** (1 / n)

    totals = gm.sum(axis=0)

    fuzzy_weights = np.column_stack((
        gm[:, 0] / totals[2],
        gm[:, 1] / totals[1],
        gm[:, 2] / totals[0],
    ))

    # Defuzzification and normalization.

    crisp = fuzzy_weights.mean(axis=1)
    weights = crisp / crisp.sum()

    # Classical AHP consistency check.

    lambda_max = float(
        np.max(np.linalg.eigvals(matrix).real)
    )

    ci = (lambda_max - n) / (n - 1)
    cr = ci / RI[n] if RI[n] else 0.0

    return fuzzy_weights, weights, lambda_max, float(cr)


# ============================================================
# 7. READ EXISTING FEATURE STACK
# ============================================================

def _read_inputs():

    required_keys = (
        "features_stack",
        "features_stack_bands",
        "features_valid_mask",
        "district_mask",
    )

    missing = [
        str(cfg.output_path(key))
        for key in required_keys
        if not Path(cfg.output_path(key)).exists()
    ]

    if not Path(cfg.GRID_PATH).exists():
        missing.append(str(cfg.GRID_PATH))

    if missing:
        raise FileNotFoundError(
            "Missing Task 4 inputs:\n"
            + "\n".join(f" - {p}" for p in missing)
        )

    stack_path = cfg.output_path("features_stack")

    utils.assert_same_grid(
        stack_path,
        cfg.output_path("features_valid_mask"),
    )

    utils.assert_same_grid(
        stack_path,
        cfg.output_path("district_mask"),
    )

    utils.assert_same_grid(
        stack_path,
        cfg.GRID_PATH,
    )

    metadata = json.loads(
        Path(
            cfg.output_path("features_stack_bands")
        ).read_text(encoding="utf-8")
    )

    band_index = {
        str(item["name"]): int(item["index"]) - 1
        for item in metadata
    }

    required_bands = set(CRITERIA[:4]) | set(LANDCOVER_BANDS)

    missing_bands = required_bands - set(band_index)

    if missing_bands:
        raise ValueError(
            "Feature stack missing band(s): "
            + ", ".join(sorted(missing_bands))
        )

    with rasterio.open(stack_path) as src:
        stack = src.read().astype("float32")
        nodata = src.nodata

    valid_raw, _, _ = utils.read_raster(
        cfg.output_path("features_valid_mask")
    )

    district_raw, _, _ = utils.read_raster(
        cfg.output_path("district_mask")
    )

    valid = (
        (valid_raw == 1)
        & (district_raw == 1)
    )

    if nodata is not None:
        stack[stack == nodata] = np.nan

    stack[~np.isfinite(stack)] = np.nan

    for name in required_bands:

        index = band_index[name]

        valid &= (
            np.isfinite(stack[index])
            & (stack[index] != cfg.NODATA)
        )

    if not valid.any():
        raise ValueError(
            "No valid Alappuzha cells found in the feature stack"
        )

    return stack, valid, band_index


# ============================================================
# 8. FUZZY MEMBERSHIP FUNCTIONS
# ============================================================

def _membership(
    values: np.ndarray,
    valid: np.ndarray,
    *,
    increasing: bool,
):

    sample = values[valid]

    if not np.isfinite(sample).all():
        raise ValueError(
            "Non-finite criterion values found inside valid mask"
        )

    p10, p90 = np.percentile(sample, [10, 90])

    if p90 <= p10:

        result = np.full(
            values.shape,
            0.5,
            dtype="float32",
        )

    else:

        result = np.clip(
            (values - p10) / (p90 - p10),
            0,
            1,
        ).astype("float32")

    if not increasing:
        result = 1.0 - result

    result[~valid] = np.nan

    return result, {
        "p10": float(p10),
        "p90": float(p90),
        "direction": (
            "increasing" if increasing else "decreasing"
        ),
    }


# ============================================================
# 9. LAND-COVER SUSCEPTIBILITY
# ============================================================

def _landcover_score(
    stack: np.ndarray,
    band_index: dict[str, int],
    valid: np.ndarray,
):

    fractions = {
        name: np.clip(
            stack[band_index[name]],
            0,
            1,
        ).astype("float32")
        for name in LANDCOVER_BANDS
    }

    total = np.sum(
        np.stack(list(fractions.values())),
        axis=0,
    )

    # Normalize cells with minor fraction overruns.

    over = total > 1.0

    if over.any():

        for name in fractions:
            fractions[name][over] /= total[over]

        total = np.sum(
            np.stack(list(fractions.values())),
            axis=0,
        )

    # Residual fraction for remaining land-cover classes.

    other = np.clip(
        1.0 - total,
        0,
        1,
    )

    score = sum(
        fractions[name] * LANDCOVER_SCORES[name]
        for name in LANDCOVER_BANDS
    )

    score = score + other * LANDCOVER_SCORES["other"]

    score[~valid] = np.nan

    details = {
        "class_scores": LANDCOVER_SCORES,
        "mean_other_fraction": float(other[valid].mean()),
        "min_fraction_sum_after_residual": float(
            (total[valid] + other[valid]).min()
        ),
        "max_fraction_sum_after_residual": float(
            (total[valid] + other[valid]).max()
        ),
        "cells_with_fraction_overrun_normalized": int(
            (over & valid).sum()
        ),
    }

    return score.astype("float32"), details


# ============================================================
# 10. FUZZY RULE ACTIVATIONS
# ============================================================

def _build_rule_activations(
    memberships: dict[str, np.ndarray],
    valid: np.ndarray,
):

    rasters = []
    rows = []

    for rule in RULES:

        (
            rule_id,
            antecedents,
            consequent,
            meaning,
            left_name,
            right_name,
            inv_left,
            inv_right,
        ) = rule

        left = memberships[left_name]
        right = memberships[right_name]

        if inv_left:
            left = 1.0 - left

        if inv_right:
            right = 1.0 - right

        # Mamdani minimum operator for AND.

        activation = np.minimum(
            left,
            right,
        ).astype("float32")

        activation[~valid] = np.nan

        values = activation[valid]

        rasters.append(activation)

        rows.append({
            "rule": rule_id,
            "antecedents": antecedents,
            "consequent": consequent,
            "interpretation": meaning,
            "mean_activation": float(values.mean()),
            "share_activation_ge_0_5": float(
                (values >= 0.5).mean()
            ),
            "cells_activation_ge_0_5": int(
                (values >= 0.5).sum()
            ),
        })

    return np.stack(rasters), rows


# ============================================================
# 11. OVERVIEW FIGURE
# ============================================================

def _plot_overview(
    memberships,
    score,
    valid,
    out_path: Path,
):

    panels = [
        ("elevation", "Low-elevation membership"),
        ("slope", "Low-slope membership"),
        ("twi", "High-TWI membership"),
        ("distance_waterway", "Near-waterway membership"),
        (
            "landcover_susceptibility",
            "Land-cover membership",
        ),
    ]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 9),
        constrained_layout=True,
    )

    for ax, (key, title) in zip(axes.flat[:5], panels):

        arr = np.ma.masked_where(
            ~valid,
            memberships[key],
        )

        im = ax.imshow(
            arr,
            vmin=0,
            vmax=1,
            cmap="YlOrRd",
        )

        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

        fig.colorbar(
            im,
            ax=ax,
            fraction=0.046,
            pad=0.02,
        )

    arr = np.ma.masked_where(
        ~valid,
        score,
    )

    im = axes.flat[5].imshow(
        arr,
        vmin=0,
        vmax=1,
        cmap="YlOrRd",
    )

    axes.flat[5].set_title(
        "Fuzzy-AHP susceptibility score"
    )

    axes.flat[5].set_xticks([])
    axes.flat[5].set_yticks([])

    fig.colorbar(
        im,
        ax=axes.flat[5],
        fraction=0.046,
        pad=0.02,
    )

    fig.suptitle(
        "Alappuzha: Fuzzy-AHP flood susceptibility",
        fontsize=14,
    )

    utils.make_folder(out_path.parent)

    fig.savefig(
        out_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# 12. MAIN
# ============================================================

def main():

    stack, valid, band_index = _read_inputs()

    LOGGER.info(
        "Valid Alappuzha cells: %s",
        f"{int(valid.sum()):,}",
    )

    memberships = {}
    bounds = {}

    # Higher membership always means higher susceptibility.

    memberships["elevation"], bounds["elevation"] = _membership(
        stack[band_index["elevation"]],
        valid,
        increasing=False,
    )

    memberships["slope"], bounds["slope"] = _membership(
        stack[band_index["slope"]],
        valid,
        increasing=False,
    )

    memberships["twi"], bounds["twi"] = _membership(
        stack[band_index["twi"]],
        valid,
        increasing=True,
    )

    memberships["distance_waterway"], bounds["distance_waterway"] = _membership(
        stack[band_index["distance_waterway"]],
        valid,
        increasing=False,
    )

    (
        memberships["landcover_susceptibility"],
        lc_stats,
    ) = _landcover_score(
        stack,
        band_index,
        valid,
    )

    bounds["landcover_susceptibility"] = {
        "method": "fraction-weighted ESA WorldCover scores",
        "direction": "higher is more susceptible",
    }

    # Calculate Fuzzy-AHP weights.

    fuzzy_weights, weights, lambda_max, cr = (
        _calculate_weights(PAIRWISE)
    )

    if cr > CR_LIMIT:
        raise ValueError(
            f"AHP consistency ratio {cr:.4f} exceeds "
            f"{CR_LIMIT:.2f}; review pairwise judgments"
        )

    # Calculate final weighted susceptibility score.

    score = np.full(
        valid.shape,
        cfg.NODATA,
        dtype="float32",
    )

    weighted = np.zeros(
        valid.shape,
        dtype="float32",
    )

    for name, weight in zip(CRITERIA, weights):

        weighted[valid] += (
            memberships[name][valid] * float(weight)
        )

    score[valid] = np.clip(
        weighted[valid],
        0,
        1,
    )

    # Classify into five susceptibility levels.
    # Thresholds are initial equal-interval breaks.

    classes = np.full(
        valid.shape,
        cfg.NODATA,
        dtype="float32",
    )

    classes[valid] = (
        np.digitize(
            score[valid],
            bins=[0.2, 0.4, 0.6, 0.8],
        ).astype("float32") + 1
    )

    # Rule and membership outputs.

    rule_stack, rule_rows = _build_rule_activations(
        memberships,
        valid,
    )

    membership_stack = np.stack(
        [memberships[name] for name in CRITERIA]
    )

    membership_stack[:, ~valid] = cfg.NODATA
    rule_stack[:, ~valid] = cfg.NODATA

    # Save GeoTIFF outputs on the existing grid.

    utils.write_raster(
        score,
        cfg.output_path("fuzzy_ahp_score"),
        cfg.GRID_PATH,
        nodata=cfg.NODATA,
        dtype="float32",
    )

    utils.write_raster(
        classes,
        cfg.output_path("fuzzy_ahp_classes"),
        cfg.GRID_PATH,
        nodata=cfg.NODATA,
        dtype="float32",
    )

    utils.write_multiband_raster(
        membership_stack,
        cfg.output_path("fuzzy_ahp_memberships"),
        cfg.GRID_PATH,
        band_names=list(CRITERIA),
        nodata=cfg.NODATA,
    )

    utils.write_multiband_raster(
        rule_stack,
        cfg.output_path("fuzzy_ahp_rules"),
        cfg.GRID_PATH,
        band_names=[r[0] for r in RULES],
        nodata=cfg.NODATA,
    )

    # Save pairwise matrix.

    pairwise_df = pd.DataFrame(
        PAIRWISE,
        index=CRITERIA,
        columns=CRITERIA,
    )

    pairwise_df.index.name = "criterion"

    pairwise_df.to_csv(
        cfg.output_path("fuzzy_ahp_pairwise"),
        float_format="%.6f",
    )

    # Save weights.

    weight_rows = []

    for i, name in enumerate(CRITERIA):

        weight_rows.append({
            "criterion": name,
            "fuzzy_weight_lower": float(fuzzy_weights[i, 0]),
            "fuzzy_weight_middle": float(fuzzy_weights[i, 1]),
            "fuzzy_weight_upper": float(fuzzy_weights[i, 2]),
            "final_weight": float(weights[i]),
        })

    pd.DataFrame(weight_rows).sort_values(
        "final_weight",
        ascending=False,
    ).to_csv(
        cfg.output_path("fuzzy_ahp_weights"),
        index=False,
        float_format="%.6f",
    )

    # Save rule summary.

    pd.DataFrame(rule_rows).to_csv(
        cfg.output_path("fuzzy_ahp_rule_summary"),
        index=False,
        float_format="%.6f",
    )

    # Calculate class area and share.

    cell_area_km2 = (
        cfg.GRID_RESOLUTION ** 2
    ) / 1_000_000

    counts = np.bincount(
        classes[valid].astype("uint8"),
        minlength=6,
    )

    class_stats = {
        CLASS_NAMES[k]: {
            "cells": int(counts[k]),
            "area_km2": round(
                float(counts[k] * cell_area_km2),
                3,
            ),
            "share": round(
                float(counts[k] / valid.sum()),
                4,
            ),
        }
        for k in CLASS_NAMES
    }

    # Save statistics.

    stats = {
        "district": cfg.DISTRICT_NAME,
        "scenario": cfg.SCENARIO_NAME,
        "method": (
            "Buckley Fuzzy-AHP weighted overlay "
            "with Mamdani AND-rule activation diagnostics"
        ),
        "valid_district_cells": int(valid.sum()),
        "grid_resolution_m": int(cfg.GRID_RESOLUTION),
        "score_min": float(score[valid].min()),
        "score_max": float(score[valid].max()),
        "score_mean": float(score[valid].mean()),
        "score_median": float(np.median(score[valid])),
        "criteria": list(CRITERIA),
        "membership_bounds": bounds,
        "landcover": lc_stats,
        "weights": {
            r["criterion"]: r["final_weight"]
            for r in weight_rows
        },
        "weights_sum": float(weights.sum()),
        "lambda_max": lambda_max,
        "consistency_ratio": cr,
        "consistency_ratio_limit": CR_LIMIT,
        "consistency_passed": bool(cr <= CR_LIMIT),
        "class_distribution": class_stats,
        "fuzzy_rules": rule_rows,
        "interpretation": (
            "Relative flood susceptibility, not flood probability, "
            "depth or a declaration of safety. Pairwise judgments "
            "and land-cover scores are initial expert assumptions "
            "requiring review and sensitivity analysis."
        ),
    }

    Path(
        cfg.output_path("fuzzy_ahp_stats")
    ).write_text(
        json.dumps(stats, indent=2),
        encoding="utf-8",
    )

    # Save figure.

    _plot_overview(
        memberships,
        score,
        valid,
        Path(cfg.output_path("fuzzy_ahp_overview")),
    )

    LOGGER.info("AHP weights: %s", stats["weights"])

    LOGGER.info(
        "Weights sum: %.6f | lambda_max: %.4f | CR: %.4f",
        weights.sum(),
        lambda_max,
        cr,
    )

    LOGGER.info(
        "Class distribution: %s",
        class_stats,
    )

    LOGGER.info(
        "Fuzzy-AHP complete. Outputs saved under data/processed and outputs."
    )


if __name__ == "__main__":
    main()
    