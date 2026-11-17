"""Task 6: join Census 2011 areas to Alappuzha village polygons.

Run after ``src/05_clean_census.py``::

    python src/05_join_boundaries.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import geopandas as gpd
import numpy as np
import pandas as pd

from src import config as cfg
from src import utils

LOGGER = utils.get_logger("join_boundaries")

INDICATORS = tuple(cfg.VULNERABILITY_INDICATORS)
COUNT_COLUMNS = (
    "population",
    "child_pop",
    "sc_pop",
    "st_pop",
    "marginal_workers",
    "non_workers",
)


def _require_inputs() -> None:
    required = {
        "census_clean": cfg.output_path("census_clean"),
        "villages_clip": cfg.output_path("villages_clip"),
        "mapping": cfg.VILLAGE_CODE_MAPPING_RAW,
    }
    missing = [
        f"{name}: {path}"
        for name, path in required.items()
        if not Path(path).exists()
    ]
    if missing:
        formatted = "\n".join(f"  - {item}" for item in missing)
        raise FileNotFoundError(f"Missing Task 6 input(s):\n{formatted}")


def _name_key(value: object) -> str:
    """Normalise names and drop parenthetical suffixes such as (Part)/(CT)."""
    text = str(value).casefold().replace("&", " and ")
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _read_census() -> pd.DataFrame:
    frame = pd.read_csv(cfg.output_path("census_clean"))
    frame = frame.loc[frame["is_area_total"]].copy()
    frame["area_code"] = pd.to_numeric(frame["area_code"], errors="raise").astype(int)
    frame["subdistrict_code"] = pd.to_numeric(
        frame["subdistrict_code"], errors="raise"
    ).astype(int)
    frame["name_key"] = frame["area_name"].map(_name_key)
    frame = frame.set_index("area_code", drop=False)
    return frame.sort_index()


def _read_villages() -> gpd.GeoDataFrame:
    villages = gpd.read_file(cfg.output_path("villages_clip"))
    if "DISTRICT" not in villages.columns:
        raise KeyError("village layer has no DISTRICT column")
    villages = villages.loc[
        villages["DISTRICT"].astype(str).str.casefold()
        == cfg.DISTRICT_NAME.casefold()
    ].copy()
    if villages.empty:
        raise ValueError("No Alappuzha polygons found in villages_clip")

    villages = villages.loc[
        villages.geometry.notna() & ~villages.geometry.is_empty
    ].copy()
    villages["polygon_id"] = villages.index
    villages["polygon_name"] = villages["NAME"].astype(str).str.strip()
    villages["polygon_type"] = villages["TYPE"].astype(str).str.strip()
    villages["subdistrict"] = villages["SUB_DIST"].astype(str).str.strip()
    villages["code_2001"] = (
        villages["CEN_2001"].astype(str).str[-8:].astype(int)
    )
    villages["name_key"] = villages["polygon_name"].map(_name_key)
    return villages


def _read_mapping() -> pd.DataFrame:
    mapping = pd.read_csv(
        cfg.VILLAGE_CODE_MAPPING_RAW,
        sep=";",
        dtype=str,
    )
    mapping = mapping.loc[
        mapping["district_code_2011"] == str(cfg.CENSUS_DISTRICT_CODE)
    ].copy()
    mapping["code_2011"] = pd.to_numeric(
        mapping["village_code_2011"], errors="raise"
    ).astype(int)
    mapping["code_2001"] = pd.to_numeric(
        mapping["village_code_2001"], errors="coerce"
    )
    mapping = mapping.dropna(subset=["code_2001"]).copy()
    mapping["code_2001"] = mapping["code_2001"].astype(int)
    mapping["key_2001"] = mapping["village_name_2001"].map(_name_key)
    mapping["key_2011"] = mapping["village_name_2011"].map(_name_key)
    return mapping


def _match_polygon(
    polygon: pd.Series,
    mapping: pd.DataFrame,
    census: pd.DataFrame,
    used_codes: set[int],
) -> tuple[int | None, str, str]:
    candidate_rows = mapping.loc[mapping["code_2001"] == polygon["code_2001"]]
    for key in ("key_2001", "key_2011"):
        candidates = candidate_rows.loc[candidate_rows[key] == polygon["name_key"]]
        if len(candidates) == 1:
            code = int(candidates.iloc[0]["code_2011"])
            if code in census.index and code not in used_codes:
                return code, f"mapping_{key}", "matched"

    candidates = census.loc[census["name_key"] == polygon["name_key"]]
    if candidates.empty:
        return None, "unmatched", "no_census_name"

    by_type = candidates.loc[
        candidates["level"].astype(str).str.casefold()
        == polygon["polygon_type"].casefold()
    ]
    if len(by_type) == 1:
        candidates = by_type
        method = "name_type"
    elif len(candidates) == 1:
        method = "name"
    else:
        return None, "ambiguous_name", f"{len(candidates)} candidates"

    code = int(candidates.iloc[0]["area_code"])
    if code in used_codes:
        return None, "duplicate_match", f"census code {code} already used"
    return code, method, "matched"


def _subdistrict_codes(matched: pd.DataFrame) -> dict[str, int]:
    lookup: dict[str, int] = {}
    joined = matched.loc[matched["census_code"].notna()].copy()
    if joined.empty:
        return lookup
    for name, group in joined.groupby("subdistrict"):
        values = group["subdistrict_code"].dropna()
        if not values.empty:
            lookup[name] = int(values.mode().iloc[0])
    return lookup


def _fill_values(
    census: pd.DataFrame,
    matched_codes: list[int],
    subdistrict_code: int | None,
) -> tuple[dict[str, float], str]:
    matched = census.loc[census.index.isin(matched_codes)]
    fill: dict[str, float] = {}
    for indicator in INDICATORS:
        if subdistrict_code is not None:
            group = matched.loc[
                matched["subdistrict_code"] == subdistrict_code, indicator
            ]
            value = group.median()
        else:
            value = np.nan
        if not np.isfinite(value):
            value = matched[indicator].median()
        if not np.isfinite(value):
            value = census[indicator].median()
        fill[indicator] = float(value)
    return fill, "subdistrict_median" if subdistrict_code is not None else "district_median"


def main() -> None:
    cfg.check_input_paths()
    _require_inputs()

    census = _read_census()
    villages = _read_villages()
    mapping = _read_mapping()

    records: list[dict] = []
    used_codes: set[int] = set()
    for _, polygon in villages.iterrows():
        code, method, status = _match_polygon(
            polygon,
            mapping,
            census,
            used_codes,
        )
        record = {
            "record_type": "polygon",
            "polygon_id": int(polygon["polygon_id"]),
            "polygon_name": polygon["polygon_name"],
            "polygon_type": polygon["polygon_type"],
            "subdistrict": polygon["subdistrict"],
            "code_2001": int(polygon["code_2001"]),
            "census_code": code,
            "census_code_2011": str(code) if code is not None else "",
            "census_name": "",
            "census_level": "",
            "subdistrict_code": np.nan,
            "match_method": method,
            "status": status,
            "geometry_filled": False,
            "fill_source": "",
            "population": 0,
            "zero_population": False,
            "missing_indicators": "",
            "housing_source": "",
        }
        if code is not None:
            used_codes.add(code)
            row = census.loc[code]
            record.update(
                {
                    "census_name": row["area_name"],
                    "census_level": row["level"],
                    "subdistrict_code": int(row["subdistrict_code"]),
                    "population": int(row["population"]),
                    "zero_population": bool(row["zero_population"]),
                    "missing_indicators": row["missing_indicators"],
                    "housing_source": (
                        "filled" if row["housing_deprivation_filled"] else "hlpca"
                    ),
                }
            )
            for indicator in INDICATORS:
                record[indicator] = float(row[indicator])
            for column in (
                "no_latrine_pct",
                "temporary_structure_pct",
            ):
                record[column] = float(row[column])
        records.append(record)

    matched = pd.DataFrame(records)
    subdistrict_codes = _subdistrict_codes(matched)

    for index, row in matched.loc[matched["census_code"].isna()].iterrows():
        subdistrict_code = subdistrict_codes.get(row["subdistrict"])
        fill, source = _fill_values(
            census,
            sorted(used_codes),
            subdistrict_code,
        )
        for indicator in INDICATORS:
            matched.at[index, indicator] = fill[indicator]
        matched.at[index, "census_name"] = (
            f"{row['subdistrict']} sub-district median"
        )
        matched.at[index, "census_level"] = "FILL"
        matched.at[index, "geometry_filled"] = True
        matched.at[index, "fill_source"] = source
        matched.at[index, "housing_source"] = source

    matched_codes = [
        int(code) for code in matched["census_code"].dropna().unique()
    ]
    matched_population = float(
        census.loc[census.index.isin(matched_codes), "population"].sum()
    )
    total_population = float(census["population"].sum())
    matched_polygons = int(matched["census_code"].notna().sum())
    total_polygons = int(len(matched))
    matched_village_share = matched_polygons / total_polygons
    matched_population_share = (
        matched_population / total_population if total_population else 0.0
    )

    unmatched_census = census.loc[~census.index.isin(matched_codes)].copy()
    unmatched_rows: list[dict] = []
    for code, row in unmatched_census.iterrows():
        unmatched_rows.append(
            {
                "record_type": "census_area_without_polygon",
                "polygon_id": np.nan,
                "polygon_name": "",
                "polygon_type": "",
                "subdistrict": "",
                "code_2001": np.nan,
                "census_code": int(code),
                "census_code_2011": str(code),
                "census_name": row["area_name"],
                "census_level": row["level"],
                "subdistrict_code": int(row["subdistrict_code"]),
                "match_method": "",
                "status": "no_polygon",
                "geometry_filled": False,
                "fill_source": "",
                "population": int(row["population"]),
                "zero_population": bool(row["zero_population"]),
                "missing_indicators": row["missing_indicators"],
                "housing_source": "",
            }
        )

    report = pd.concat(
        [matched, pd.DataFrame(unmatched_rows)],
        ignore_index=True,
    )
    report["matched_villages_share"] = matched_village_share
    report["matched_population_share"] = matched_population_share
    report["note"] = ""
    missing_polygons = report.loc[report["status"] != "matched"]
    for index, row in missing_polygons.iterrows():
        if row["record_type"] == "polygon":
            report.at[index, "note"] = (
                "No census area matched; sub-district median used"
            )
        else:
            report.at[index, "note"] = (
                "Census area has no separate polygon in the 2001 layer"
            )

    report_path = cfg.output_path("join_report")
    utils.make_folder(report_path.parent)
    report.to_csv(report_path, index=False)

    output_columns = [
        "polygon_id",
        "polygon_name",
        "polygon_type",
        "subdistrict",
        "code_2001",
        "census_code_2011",
        "census_name",
        "census_level",
        "match_method",
        "status",
        "geometry_filled",
        "fill_source",
        "population",
        "zero_population",
        "missing_indicators",
        "housing_source",
        "child_share",
        "scst_share",
        "worker_fragility",
        "housing_deprivation",
        "no_latrine_pct",
        "temporary_structure_pct",
        "geometry",
    ]
    report_polygons = report.loc[
        report["record_type"] == "polygon",
        [
            "polygon_id",
            "census_code_2011",
            "census_name",
            "census_level",
            "match_method",
            "status",
            "geometry_filled",
            "fill_source",
            "population",
            "zero_population",
            "missing_indicators",
            "housing_source",
            *INDICATORS,
            "no_latrine_pct",
            "temporary_structure_pct",
        ],
    ]
    output = villages.drop(columns=["name_key"]).merge(
        report_polygons,
        on="polygon_id",
        how="left",
        validate="one_to_one",
    )
    output = gpd.GeoDataFrame(
        output[output_columns],
        geometry="geometry",
        crs=villages.crs,
    )
    output_path = cfg.output_path("villages_vuln")
    utils.make_folder(output_path.parent)
    if output_path.exists():
        output_path.unlink()
    output.to_file(output_path, driver="GPKG")

    method_counts = (
        report.loc[report["record_type"] == "polygon", "match_method"]
        .value_counts()
        .to_dict()
    )
    checks = {
        "total_polygons": total_polygons,
        "matched_polygons": matched_polygons,
        "matched_villages_share": matched_village_share,
        "matched_population": matched_population,
        "total_population": total_population,
        "matched_population_share": matched_population_share,
        "unmatched_polygons": int(total_polygons - matched_polygons),
        "unmatched_census_areas": int(len(unmatched_census)),
        "filled_polygons": int(matched["geometry_filled"].sum()),
        "meets_village_threshold": bool(
            matched_village_share >= cfg.VULNERABILITY_MIN_MATCHED_VILLAGE_SHARE
        ),
        "meets_population_threshold": bool(
            matched_population_share
            >= cfg.VULNERABILITY_MIN_MATCHED_POPULATION_SHARE
        ),
    }
    stats = {
        "inputs": {
            "census_clean": str(cfg.output_path("census_clean")),
            "villages_clip": str(cfg.output_path("villages_clip")),
            "mapping": str(cfg.VILLAGE_CODE_MAPPING_RAW),
        },
        "match_methods": method_counts,
        "checks": checks,
        "unmatched_census_areas": unmatched_census[
            ["area_name", "level", "population"]
        ].to_dict(orient="records"),
        "unmatched_polygons": report.loc[
            (report["record_type"] == "polygon")
            & (report["status"] != "matched"),
            ["polygon_name", "polygon_type", "status", "match_method"],
        ].to_dict(orient="records"),
    }
    stats_path = cfg.output_path("join_stats")
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    utils.make_folder(output_path.parent)
    LOGGER.info(
        "Matched %d/%d polygons (%.2f%%); population %.0f/%.0f (%.2f%%)",
        matched_polygons,
        total_polygons,
        matched_village_share * 100.0,
        matched_population,
        total_population,
        matched_population_share * 100.0,
    )
    LOGGER.info("Match methods: %s", method_counts)
    if not checks["meets_village_threshold"]:
        raise RuntimeError("Matched village share is below the 90% threshold")
    if not checks["meets_population_threshold"]:
        raise RuntimeError("Matched population share is below the 90% threshold")
    LOGGER.info("Village/census join complete")


if __name__ == "__main__":
    main()
