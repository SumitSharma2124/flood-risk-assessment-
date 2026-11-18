# TASK 7b: Audit of KSDMA NoData Handling in Tasks 1–6

**Date:** 2026-11-17  
**Scope:** Read-only audit of scripts (`src/`), configuration (`src/config.py`), utilities (`src/utils.py`), documentation (`README.md`, `docs/`), and generated outputs (`data/`, `outputs/`).

---

## 1. Executive Summary

- **Physical Reality:** The KSDMA 100-year raster provides inundation depth in **centimetres (cm)** (valid range: 0.81 cm to 1,126 cm in Alappuzha; median 248 cm / ~2.5 m). Cells with NoData represent **unflooded dry land (depth = 0 cm)**. These dry cells have a median elevation of ~9.1–12.3 m and median HAND of ~1.6–3.0 m, compared to 2.0–2.6 m elevation and 0.0 m HAND for flooded cells.
- **The Core Issue:** Earlier tasks interpreted NoData as "unmodelled coverage gap / missing data". In `src/02_align_rasters.py`, dry cells inside Alappuzha were reprojected as `NODATA` (`-9999.0`) and isolated as `ksdma_modelled_mask.tif`.
- **Key Finding on Past Tasks (Tasks 1–6):**
  - **Task 4 (Feature Stack):** `features_stack.tif` and `features_valid_mask.tif` were built purely from terrain and land cover; **they did NOT mask out KSDMA NoData cells**. The feature stack covers the full district.
  - **Task 5 (Exposure):** The combined exposure score (`exposure_score.tif`) was computed on **all district cells**; it was **NOT** masked by KSDMA NoData.
  - **Task 6 (Vulnerability):** Built on census indicators and village boundaries; **unaffected by KSDMA**.
  - **Task 7 (Relief Camps):** In `camps_clean.gpkg`, 290 of 379 camps (76.5%) sit in NoData cells. They were kept, but flagged as `in_ksdma_coverage = False` and given `flood_depth_at_camp = NaN` instead of `0.0 cm`.
- **Critical Risk for Downstream Tasks (Tasks 8+):** If `ksdma_100yr_90m.tif` retains `-9999.0` for dry land or if `ksdma_modelled_mask.tif` is used as an exclusion mask for training the U-Net, baseline ML models, or road network risk scoring, **59.4% of the district (all safe, dry high ground) would be discarded**, leaving models with 100% flooded cells.

---

## 2. Code Audit Table

| File | Line(s) | What it does with NoData | Impact |
| :--- | :--- | :--- | :--- |
| `src/config.py` | 263–274 | Defines `KSDMA_NODATA = -3.4e38` and `KSDMA_NODATA_500 = 0.0`. | **None** (accurately defines raw raster nodata values). |
| `src/config.py` | 276–279 | Defines `KSDMA_MASK_NAME = "ksdma_modelled_mask.tif"` with docstring stating it masks "cells where KSDMA actually modelled flood probability (as opposed to background NoData)". | **Must fix** (conceptually misidentifies the flood footprint as a data coverage boundary). |
| `src/config.py` | 435–436, 450, 453 | Defines aliases `ksdma_grid` and `ksdma_mask`. | **Minor** (clarify aliases so downstream tasks don't treat `ksdma_mask` as a domain exclusion mask). |
| `src/01_clip_layers.py` | 482 | Clips raw KSDMA tile to district buffer window; passes raw NoData through. | **None** (raw interim clipping is neutral). |
| `src/01_inspect_layers.py` | 325–365 | `_ksdma_check`: computes `values = data[inside & valid]`; logs valid share (40.67%) as "valid inside the district"; writes range note on valid cells only. | **Minor** (diagnostic report mislabels flood extent as "valid coverage"). |
| `src/02_align_rasters.py` | 247–260 | `_align_ksdma`: sets non-valid cells to `np.nan` and writes `dst_nodata=cfg.NODATA` (`-9999.0`) to `ksdma_100yr_90m.tif`. | **Must fix** (dry cells inside `district_mask == 1` are assigned `-9999.0` instead of `0.0 cm`). |
| `src/02_align_rasters.py` | 261–268 | Creates `ksdma_modelled_mask.tif` as `(ksdma != NODATA).astype(float32)` and saves it. | **Must fix** (this raster is actually `ksdma_100yr_flood_footprint.tif`; treating it as "modelled vs unmodelled" causes downstream misuse). |
| `src/02_align_rasters.py` | 270–290 | Compares `clipped_area` (1,074.8 km²) with `target_area` (1,072.0 km²). Checks relative change (-0.26%). | **Minor** (stat is mathematically correct for the flood footprint, but labeled as "modelled area"). |
| `src/02_align_rasters.py` | 292–302 | Computes min (0.81 cm) and max (1,126 cm) on `modelled == 1` only; logs range. | **Minor** (diagnostic log). |
| `src/02_align_rasters.py` | 566–567 | Plots `ksdma_100yr_90m` and `ksdma_modelled_mask` in `02_aligned_overview.png`. | **Minor** (diagnostic visualization). |
| `src/03_stack_features.py` | 47, 114–165 | `_ksdma_link_check`: reads `ksdma_modelled_mask` to compare HAND and TWI between flooded and unflooded cells. | **None / Beneficial** (diagnostic check; correctly proved terrain differences). |
| `src/03_stack_features.py` | 252–258 | `feature_valid`: builds `features_valid_mask.tif` from terrain + landcover + district. **Does NOT include KSDMA.** | **None** (feature stack correctly kept 100% of district cells). |
| `src/04_exposure_combine.py`| 35, 197–198 | Reads `ksdma_modelled_mask` to produce breakdown summary row in `04_exposure_summary.csv`. | **Minor** (exposure score itself is unaffected; only breakdown table row is labeled "ksdma_modelled"). |
| `src/06_clean_camp.py` | 240–249 | Sets `gdf["in_ksdma_coverage"] = v` and `gdf["flood_depth_at_camp"] = NaN` for dry cells. | **Must fix** (dry camps should have `in_ksdma_flood_zone = False` and `flood_depth_at_camp = 0.0 cm`; fix deferred per user instruction). |
| `src/06_clean_camp.py` | 259, 264, 287 | Logs and records `"camps in KSDMA NoData (masked) area: 290"`. | **Minor** (diagnostic logging). |

---

## 3. Direct Raster Verification (`ksdma_100yr_90m.tif`)

A direct inspection of [ksdma_100yr_90m.tif](file:///C:/Users/sumit/documents/code/projects/AI-course/flood_risk/data/processed/ksdma_100yr_90m.tif) against [district_mask.tif](file:///C:/Users/sumit/documents/code/projects/AI-course/flood_risk/data/processed/district_mask.tif) reveals:

- **Configured NoData value:** `-9999.0`
- **Total district cells (`district_mask == 1`):** `176,745` (1,431.63 km²)
- **Valid (flooded) cells inside district:** `71,799` (**40.62%** of district area; 581.57 km²)
- **NoData (dry) cells inside district:** `104,946` (**59.38%** of district area; 850.06 km²)
- **Total valid cells in full bounding box (including 1 km buffer):** `132,342` (1,071.97 km² ~ 1,072 km²)
- **Total raster cells in bounding box:** `523,406` (533 × 982)

### Task 3 Verification: "Flood area changed by under 0.5%"
- In `src/02_align_rasters.py`:
  - `clipped_area` = `1,074.79 km²` (computed only on `clipped_valid`, cells with depth > 0)
  - `target_area` = `1,071.97 km²` (computed as `modelled.sum() * 8100 / 1e6`, cells with depth > 0)
  - `relative_change` = `-0.002628` (**-0.26%**, safely below the 5% error threshold and under 0.5%).
- **Finding:** This calculation was performed **strictly on valid (flooded) cells only**.

---

## 4. Affected Outputs & Status

### Outputs Requiring Regeneration / Fixes
1. `data/processed/ksdma_100yr_90m.tif`:
   - *Current state:* Inside the district, unflooded cells hold `-9999.0`.
   - *Required change:* Inside `district_mask == 1`, change `-9999.0` to `0.0` (depth = 0 cm). Values outside the district boundary remain `-9999.0`.
2. `data/processed/ksdma_modelled_mask.tif`:
   - *Current state:* Binary raster where 1 = depth > 0, 0 = NoData.
   - *Required change:* Re-label/alias conceptually as `ksdma_100yr_flood_footprint.tif` or `flood_footprint_100yr_bin.tif`. Ensure downstream tasks understand 0 means dry land, not an invalid cell.
3. `data/interim/camps_clean.gpkg` (and `camps_stats.json` / `camps_summary.csv`):
   - *Current state:* 290 camps have `in_ksdma_coverage = False` and `flood_depth_at_camp = NaN`.
   - *Required change:* Rename column to `in_ksdma_flood_zone` and fill unflooded camp depth with `0.0`. *(Hold off rerunning until approved).*

### Outputs Already Correct (Do NOT need regeneration)
- `data/processed/features_stack.tif`: Built from terrain and land cover only; does not mask KSDMA.
- `data/processed/features_valid_mask.tif`: Covers 100% of district cells (`valid.sum() == 176,745`).
- `data/processed/exposure_score.tif`: Computed across 100% of district cells (`exposure = _mask_domain(exposure, district)`).
- `data/processed/vulnerability_score.tif`: Independent of KSDMA; covers 100% of district cells.

---

## 5. Text & Documentation Stating NoData is "Masked / Missing"

1. **`src/config.py` (Lines 276–277):**
   `# Mask of cells where KSDMA actually modelled flood probability (as opposed to the background NoData surrounding the modelled area).`
2. **`README.md` (Line 252 & Project Notes):**
   Notes describing "the 25% NoData must stay masked in the models" or raw NoData descriptions implying coverage gaps.
3. **`outputs/tables/01_layer_summary.csv` (Row 4):**
   Notes column: `inside_district_valid=40.67%`.
4. **`outputs/tables/02_alignment_stats.json` (Lines 45–51):**
   Uses keys `clipped_modelled_km2` and `target_modelled_km2`.
5. **`outputs/tables/04_exposure_summary.csv` (Row 3):**
   Row labeled `ksdma_modelled` instead of `ksdma_100yr_flood_zone`.
6. **`outputs/stats/camps_stats.json` (Line 50):**
   Key `"in_ksdma_masked_area": 290`.

---

## 6. Proposed Fixes for "Must Fix" Items

> [!IMPORTANT]
> **No changes have been made yet.** Execution will wait for user approval.

### Proposed Fix 1: Update `src/02_align_rasters.py` (`_align_ksdma`)
- When reprojecting KSDMA to `data/processed/ksdma_100yr_90m.tif`:
  ```python
  # Set cells inside district mask that are unflooded to 0.0 cm depth
  ksdma_depth = np.where(district_mask == 1, np.nan_to_num(ksdma_reprojected, nan=0.0), cfg.NODATA)
  ```
- Cells inside the district mask will have depth = `0.0 cm` where dry, and depth > 0 where flooded.
- Cells outside the district mask will retain `NODATA` (`-9999.0`).
- Update `ksdma_modelled_mask.tif` docstring / metadata to specify it is the **binary flood footprint (1 = flooded, 0 = dry)**.

### Proposed Fix 2: Update `src/config.py`
- Clarify comments for `KSDMA_MASK_NAME` and `KSDMA_MASK_PATH` to indicate it is a **flood hazard footprint / binary inundation mask**, not a spatial data validity mask.
- Define explicit constants for flood depth units: `KSDMA_DEPTH_UNITS = "cm"`.

### Proposed Fix 3: Update `src/06_clean_camp.py` (When ready for Task 7 refinement)
- Rename `in_ksdma_coverage` -> `in_ksdma_flood_zone`.
- Set `flood_depth_at_camp = 0.0` where camp is outside the flood footprint (`~v`).
- Keep camp records intact (all 379 camps are verified valid shelters on dry ground).
