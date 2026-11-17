"""Task 6: clean Census 2011 data and build vulnerability indicators.

Run after Task 2 (village boundaries are not needed until the join step)::

    python src/05_clean_census.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

from src import config as cfg
from src import utils

LOGGER = utils.get_logger("clean_census")

PCA_COUNT_COLUMNS = (
    "State",
    "District",
    "Subdistt",
    "Town/Village",
    "Ward",
    "TOT_P",
    "P_06",
    "P_SC",
    "P_ST",
    "MARGWORK_P",
    "NON_WORK_P",
)

INDICATORS = tuple(cfg.VULNERABILITY_INDICATORS)


def _require_inputs() -> None:
    required = {name: path for name, path in cfg.CENSUS_FILES.items()}
    missing = [
        f"{name}: {path}"
        for name, path in required.items()
        if not Path(path).exists()
    ]
    if missing:
        formatted = "\n".join(f"  - {item}" for item in missing)
        raise FileNotFoundError(f"Missing Census 2011 input(s):\n{formatted}")


def _normalise_level(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.upper()


def _find_hlpca() -> tuple[Path, list[dict]]:
    """Verify each HLPCA candidate and return the Alappuzha file."""
    checked: list[dict] = []
    for path in cfg.CENSUS_HLPCA_CANDIDATES:
        if not Path(path).exists():
            checked.append({"path": str(path), "exists": False})
            continue
        raw = pd.read_excel(path, header=None, nrows=12)
        row = cfg.CENSUS_HLPCA_DATA_START_ROW
        district_code = pd.to_numeric(raw.iloc[row, 2], errors="coerce")
        district_name = str(raw.iloc[row, 3]).strip()
        checked.append(
            {
                "path": str(path),
                "exists": True,
                "district_code": (
                    int(district_code) if np.isfinite(district_code) else None
                ),
                "district_name": district_name,
            }
        )
        is_target = district_name.casefold() == cfg.DISTRICT_NAME.casefold()
        if np.isfinite(district_code):
            is_target = is_target or int(district_code) == cfg.CENSUS_DISTRICT_CODE
        if is_target:
            return Path(path), checked
    raise FileNotFoundError(
        "Could not find an HLPCA file for Alappuzha. Checked: "
        + ", ".join(item["path"] for item in checked)
    )


def _read_pca() -> pd.DataFrame:
    pca = pd.read_excel(cfg.CENSUS_FILES["pca"])
    pca["Level"] = _normalise_level(pca["Level"])
    for column in PCA_COUNT_COLUMNS:
        pca[column] = pd.to_numeric(pca[column], errors="coerce")
    pca["Name"] = pca["Name"].astype(str).str.strip()
    pca["TRU"] = pca["TRU"].astype(str).str.strip()

    selected = pca.loc[
        (pca["State"] == cfg.CENSUS_STATE_CODE)
        & (pca["District"] == cfg.CENSUS_DISTRICT_CODE)
        & (pca["Level"].isin(cfg.CENSUS_LOCAL_LEVELS))
    ].copy()
    if selected.empty:
        raise ValueError("No Alappuzha village/town/ward rows found in PCA file")

    selected = selected.dropna(subset=["Town/Village", "TOT_P"])
    area = selected.loc[selected["Level"].isin(cfg.CENSUS_AREA_LEVELS)].copy()
    area = area.sort_values("TOT_P", ascending=False).drop_duplicates(
        "Town/Village", keep="first"
    )
    wards = selected.loc[selected["Level"] == "WARD"].copy()
    wards = wards.drop_duplicates(["Town/Village", "Ward"], keep="first")

    combined = pd.concat([area, wards], ignore_index=True)
    combined = combined.sort_values(
        ["Level", "Subdistt", "Town/Village", "Ward"],
        kind="stable",
    ).reset_index(drop=True)
    return combined


def _read_hlpca(path: Path) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_excel(path, header=None)
    data = raw.iloc[cfg.CENSUS_HLPCA_DATA_START_ROW :].reset_index(drop=True)

    no_latrine_col = cfg.CENSUS_HLPCA_COLUMNS["no_latrine_pct"]
    temporary_col = cfg.CENSUS_HLPCA_COLUMNS["temporary_structure_pct"]
    result = pd.DataFrame(
        {
            "state_code": pd.to_numeric(data[0], errors="coerce"),
            "district_code": pd.to_numeric(data[2], errors="coerce"),
            "area_code": pd.to_numeric(data[6], errors="coerce"),
            "ward_code": pd.to_numeric(data[7], errors="coerce"),
            "no_latrine_pct": pd.to_numeric(
                data[no_latrine_col], errors="coerce"
            ),
            "temporary_structure_pct": pd.to_numeric(
                data[temporary_col], errors="coerce"
            ),
        }
    )
    result = result.loc[
        (result["state_code"] == cfg.CENSUS_STATE_CODE)
        & (result["district_code"] == cfg.CENSUS_DISTRICT_CODE)
        & (result["area_code"] > 0)
        & result["ward_code"].notna()
    ].copy()
    result["housing_deprivation"] = (
        result["no_latrine_pct"] + result["temporary_structure_pct"]
    ) / 200.0

    keep = [
        "area_code",
        "ward_code",
        "no_latrine_pct",
        "temporary_structure_pct",
        "housing_deprivation",
    ]
    result = result[keep].drop_duplicates(["area_code", "ward_code"])
    info = {
        "path": str(path),
        "rows": int(len(result)),
        "area_rows": int((result["ward_code"] == 0).sum()),
        "ward_rows": int((result["ward_code"] > 0).sum()),
    }
    return result, info


def _read_scst_totals(path: Path) -> dict:
    frame = pd.read_excel(path)
    frame["Level"] = _normalise_level(frame["Level"])
    frame["TRU"] = frame["TRU"].astype(str).str.strip()
    frame["State"] = pd.to_numeric(frame["State"], errors="coerce")
    frame["District"] = pd.to_numeric(frame["District"], errors="coerce")
    frame["TOT_P"] = pd.to_numeric(frame["TOT_P"], errors="coerce")
    frame = frame.loc[
        (frame["State"] == cfg.CENSUS_STATE_CODE)
        & (frame["District"] == cfg.CENSUS_DISTRICT_CODE)
    ]
    district_total = frame.loc[
        (frame["Level"] == "DISTRICT")
        & (frame["TRU"].str.casefold() == "total"),
        "TOT_P",
    ]
    town_total = frame.loc[frame["Level"] == "TOWN", "TOT_P"]
    return {
        "district_total": float(district_total.sum()),
        "town_total": float(town_total.sum()),
        "town_rows": int(len(town_total)),
    }


def _build_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    population = pd.to_numeric(frame["TOT_P"], errors="coerce").fillna(0.0)
    population = population.clip(lower=0.0)

    child_pop = pd.to_numeric(frame["P_06"], errors="coerce").fillna(0.0)
    sc_pop = pd.to_numeric(frame["P_SC"], errors="coerce").fillna(0.0)
    st_pop = pd.to_numeric(frame["P_ST"], errors="coerce").fillna(0.0)
    marginal = pd.to_numeric(frame["MARGWORK_P"], errors="coerce").fillna(0.0)
    non_workers = pd.to_numeric(frame["NON_WORK_P"], errors="coerce").fillna(0.0)

    positive = population > 0
    frame["population"] = population.astype("int64")
    frame["households"] = (
        pd.to_numeric(frame["households_raw"], errors="coerce")
        .fillna(0)
        .astype("int64")
    )
    frame["child_pop"] = child_pop.astype("int64")
    frame["sc_pop"] = sc_pop.astype("int64")
    frame["st_pop"] = st_pop.astype("int64")
    frame["marginal_workers"] = marginal.astype("int64")
    frame["non_workers"] = non_workers.astype("int64")
    frame["zero_population"] = ~positive

    frame["child_share"] = np.where(positive, child_pop / population, np.nan)
    frame["scst_share"] = np.where(
        positive,
        (sc_pop + st_pop) / population,
        np.nan,
    )
    frame["worker_fragility"] = np.where(
        positive,
        (marginal + non_workers) / population,
        np.nan,
    )
    return frame


def _fill_indicators(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    frame = frame.copy()
    raw_missing = frame[list(INDICATORS)].isna()
    frame["missing_indicators"] = raw_missing.apply(
        lambda row: ",".join(row.index[row]),
        axis=1,
    )

    fill_info: dict[str, dict] = {}
    for indicator in INDICATORS:
        original = frame[indicator].copy()
        group_median = frame.groupby("subdistrict_code")[indicator].transform(
            "median"
        )
        district_median = frame[indicator].median()
        frame[indicator] = frame[indicator].fillna(group_median)
        frame[indicator] = frame[indicator].fillna(district_median)
        filled = original.isna() & frame[indicator].notna()
        frame[f"{indicator}_filled"] = filled
        fill_info[indicator] = {
            "filled_rows": int(filled.sum()),
            "subdistrict_fill_rows": int((original.isna() & group_median.notna()).sum()),
            "district_median_fallback_rows": int(
                (original.isna() & group_median.isna() & frame[indicator].notna()).sum()
            ),
            "district_median": (
                float(district_median) if np.isfinite(district_median) else None
            ),
            "filled_values": {
                str(key): int(value)
                for key, value in frame.loc[filled, indicator].value_counts().items()
            },
        }
    return frame, fill_info


def main() -> None:
    cfg.check_input_paths()
    _require_inputs()

    pca = _read_pca()
    hlpca_path, hlpca_candidates = _find_hlpca()
    hlpca, hlpca_info = _read_hlpca(hlpca_path)

    frame = pca.rename(
        columns={
            "State": "state_code",
            "District": "district_code",
            "Subdistt": "subdistrict_code",
            "Town/Village": "area_code",
            "Ward": "ward_code",
            "Level": "level",
            "Name": "area_name",
            "TRU": "tru",
            "No_HH": "households_raw",
        }
    )
    frame["area_name"] = frame["area_name"].astype(str).str.strip()
    frame["is_area_total"] = frame["level"].isin(cfg.CENSUS_AREA_LEVELS)
    frame = _build_indicators(frame)

    frame = frame.merge(
        hlpca,
        on=["area_code", "ward_code"],
        how="left",
        validate="one_to_one",
        indicator="housing_join",
    )
    frame, fill_info = _fill_indicators(frame)

    output_columns = [
        "state_code",
        "district_code",
        "subdistrict_code",
        "area_code",
        "ward_code",
        "area_name",
        "level",
        "tru",
        "is_area_total",
        "population",
        "households",
        "child_pop",
        "sc_pop",
        "st_pop",
        "marginal_workers",
        "non_workers",
        "zero_population",
        "child_share",
        "scst_share",
        "worker_fragility",
        "no_latrine_pct",
        "temporary_structure_pct",
        "housing_deprivation",
        "missing_indicators",
        "child_share_filled",
        "scst_share_filled",
        "worker_fragility_filled",
        "housing_deprivation_filled",
    ]
    frame = frame[output_columns].sort_values(
        ["is_area_total", "level", "subdistrict_code", "area_code", "ward_code"],
        ascending=[False, True, True, True, True],
        kind="stable",
    )

    output_path = cfg.output_path("census_clean")
    utils.make_folder(output_path.parent)
    frame.to_csv(output_path, index=False, float_format="%.6f")
    LOGGER.info("Wrote %s rows to %s", len(frame), output_path)

    area = frame.loc[frame["is_area_total"]].copy()
    pca_population = float(area["population"].sum())
    pca_sc = float(area["sc_pop"].sum())
    pca_st = float(area["st_pop"].sum())
    town = area.loc[area["level"] == "TOWN"]
    sc_totals = _read_scst_totals(cfg.CENSUS_FILES["pca_sc"])
    st_totals = _read_scst_totals(cfg.CENSUS_FILES["pca_st"])

    checks = {
        "area_rows": int(len(area)),
        "village_rows": int((area["level"] == "VILLAGE").sum()),
        "town_rows": int(len(town)),
        "ward_rows": int((frame["level"] == "WARD").sum()),
        "census_reference_population": cfg.ALAPPUZHA_POPULATION_2011,
        "pca_area_population": pca_population,
        "population_matches_reference": bool(
            abs(pca_population - cfg.ALAPPUZHA_POPULATION_2011) < 1
        ),
        "pca_sc_population": pca_sc,
        "sc_file_district_total": sc_totals["district_total"],
        "sc_total_matches_pca": bool(abs(pca_sc - sc_totals["district_total"]) < 1),
        "pca_st_population": pca_st,
        "st_file_district_total": st_totals["district_total"],
        "st_total_matches_pca": bool(abs(pca_st - st_totals["district_total"]) < 1),
        "pca_town_sc_population": float(town["sc_pop"].sum()),
        "sc_file_town_total": sc_totals["town_total"],
        "town_sc_matches_file": bool(
            abs(float(town["sc_pop"].sum()) - sc_totals["town_total"]) < 1
        ),
        "pca_town_st_population": float(town["st_pop"].sum()),
        "st_file_town_total": st_totals["town_total"],
        "town_st_matches_file": bool(
            abs(float(town["st_pop"].sum()) - st_totals["town_total"]) < 1
        ),
        "housing_area_rows": int(
            area["housing_deprivation"].notna().sum()
        ),
        "housing_ward_rows": int(
            frame.loc[frame["level"] == "WARD", "housing_deprivation"]
            .notna()
            .sum()
        ),
        "zero_population_rows": int(frame["zero_population"].sum()),
        "filled_rows": int(frame["missing_indicators"].ne("").sum()),
    }

    stats = {
        "inputs": {name: str(path) for name, path in cfg.CENSUS_FILES.items()},
        "hlpca_selected": str(hlpca_path),
        "hlpca_candidates": hlpca_candidates,
        "hlpca": hlpca_info,
        "indicators": cfg.VULNERABILITY_INDICATORS,
        "cleaning": {
            "levels_kept": list(cfg.CENSUS_LOCAL_LEVELS),
            "area_levels": list(cfg.CENSUS_AREA_LEVELS),
            "duplicate_area_rule": "keep largest population per area code",
        },
        "fill": {
            "method": cfg.VULNERABILITY_FILL_METHOD,
            "details": fill_info,
        },
        "checks": checks,
    }
    stats_path = cfg.output_path("census_stats")
    utils.make_folder(stats_path.parent)
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    LOGGER.info(
        "Census areas: %d (%d villages, %d towns); population %.0f",
        checks["area_rows"],
        checks["village_rows"],
        checks["town_rows"],
        pca_population,
    )
    LOGGER.info(
        "SC %.0f / %.0f; ST %.0f / %.0f (PCA / auxiliary district totals)",
        pca_sc,
        sc_totals["district_total"],
        pca_st,
        st_totals["district_total"],
    )
    if not checks["population_matches_reference"]:
        raise RuntimeError("Cleaned census population does not match 2011 reference")
    if not checks["sc_total_matches_pca"] or not checks["st_total_matches_pca"]:
        raise RuntimeError("PCA SC/ST totals do not match auxiliary Census files")
    if not checks["town_sc_matches_file"] or not checks["town_st_matches_file"]:
        raise RuntimeError("Urban SC/ST totals do not match auxiliary Census files")
    LOGGER.info("Census cleaning complete")


if __name__ == "__main__":
    main()
