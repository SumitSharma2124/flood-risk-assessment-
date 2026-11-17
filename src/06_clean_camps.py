"""Task 7a: clean the Kerala monsoon-camp Excel file down to Alappuzha camps.

    python src/06_clean_camps.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from src import config as cfg
from src import utils

LOGGER = utils.get_logger("clean_camps")

DISTRICT_LABELS = {"alappuzha", "alapuzha", "alleppey"}  # exact match, not substring
HINTS = {
    "district": ["district"],
    "taluk": ["taluk", "tehsil", "sub district", "subdistrict"],
    "village": ["village"],
    "camp": ["camp name", "name of camp", "name of the camp", "camp",
             "institution", "building", "name"],
    "capacity": ["capacity"],
    "inmates": ["inmate", "persons", "individual", "people", "member"],
    "families": ["famil"],
}
# Columns matched by exact (normalised) name, so "x" never matches another column.
EXACT = {
    "src_lon": ["x", "longitude", "lon"],
    "src_lat": ["y", "latitude", "lat"],
    "lsg": ["lsg"],
    "camp_type": ["type"],
    "ownership": ["ownership"],
}
HEADER_WORDS = ("district", "taluk", "village", "camp", "name")


def _norm_col(col) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(col).lower()).strip()


def _read_best_sheet() -> pd.DataFrame:
    sheets = pd.read_excel(cfg.CAMPS_RAW, sheet_name=None, header=None, dtype=object)
    name, raw = max(sheets.items(), key=lambda kv: len(kv[1]))
    LOGGER.info("Sheets: %s; using '%s' (%d rows)",
                {k: len(v) for k, v in sheets.items()}, name, len(raw))

    scores = []
    for r in range(min(15, len(raw))):
        cells = [str(c).lower() for c in raw.iloc[r].tolist() if pd.notna(c)]
        scores.append(sum(any(w in c for w in HEADER_WORDS) for c in cells))
    header_row = max(range(len(scores)), key=scores.__getitem__)
    frame = raw.iloc[header_row + 1:].copy()
    frame.columns = [
        str(c).strip() if pd.notna(c) else f"unnamed_{i}"
        for i, c in enumerate(raw.iloc[header_row].tolist())
    ]
    return frame.reset_index(drop=True)


def _detect(columns) -> dict[str, str]:
    cols = {c: _norm_col(c) for c in columns}
    found, used = {}, set()
    for key, hints in HINTS.items():
        for hint in hints:
            hit = next((c for c, n in cols.items() if c not in used and hint in n), None)
            if hit:
                found[key] = hit
                used.add(hit)
                break
    exact = {n: c for c, n in cols.items()}
    for key, names in EXACT.items():
        for nm in names:
            if nm in exact and exact[nm] not in used:
                found[key] = exact[nm]
                used.add(exact[nm])
                break
    return found


def main() -> None:
    frame = _read_best_sheet()
    LOGGER.info("Columns: %s", list(frame.columns))
    found = _detect(frame.columns)
    LOGGER.info("Detected columns: %s", found)
    for required in ("district", "camp"):
        if required not in found:
            raise KeyError(f"Could not detect '{required}'. Columns: {list(frame.columns)}")

    out = pd.DataFrame({key: frame[col] for key, col in found.items()})
    for key in ("district", "taluk", "village"):  # merged cells leave blanks
        if key in out:
            out[key] = out[key].ffill()
    rows_read = len(out)

    out = out.dropna(subset=["camp"])
    out["camp"] = out["camp"].astype(str).str.strip()
    out = out[out["camp"] != ""]

    district = out["district"].astype(str).str.strip().str.casefold()
    LOGGER.info("District labels containing 'alap': %s",
                sorted(district[district.str.contains("alap")].unique()))
    out = out[district.isin(DISTRICT_LABELS)].copy()
    rows_alappuzha = len(out)

    for key in ("taluk", "village"):
        if key not in out:
            out[key] = ""
        out[key] = out[key].fillna("").astype(str).str.strip()
    for key in ("capacity", "inmates", "families", "src_lon", "src_lat"):
        if key in out:
            out[key] = pd.to_numeric(out[key], errors="coerce")

    out["name_key"] = out["camp"].map(utils.normalize_place_name)
    out["village_key"] = out["village"].map(utils.normalize_place_name)
    out = out[out["name_key"] != ""]
    out = out.drop_duplicates(subset=["name_key", "village_key"]).reset_index(drop=True)
    out.insert(0, "camp_id", range(1, len(out) + 1))

    out.to_csv(cfg.output_path("camps_clean"), index=False)
    has_xy = (out[["src_lon", "src_lat"]].notna().all(axis=1).sum()
              if {"src_lon", "src_lat"} <= set(out.columns) else 0)
    stats = {
        "detected_columns": found,
        "rows_read": rows_read,
        "rows_alappuzha": rows_alappuzha,
        "rows_after_dedup": len(out),
        "villages": int(out["village"].nunique()),
        "with_village": int((out["village"] != "").sum()),
        "with_source_xy": int(has_xy),
    }
    cfg.output_path("camps_clean_stats").write_text(json.dumps(stats, indent=2))
    LOGGER.info("Saved %d camps -> %s", len(out), cfg.output_path("camps_clean"))
    LOGGER.info("Stats: %s", stats)


if __name__ == "__main__":
    main()