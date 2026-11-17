# Alappuzha Flood-Risk Assessment

Machine-learning flood-risk assessment for Alappuzha district, Kerala. The
project combines terrain, land cover, rainfall/return-period flood probability,
population, buildings, roads, waterways, critical facilities and census
indicators on a common 90 m grid in UTM zone 43N (EPSG:32643).

Code and raw data are versioned in Git. Large raw files use Git LFS; the only
file kept out of Git is the reconstructed 4 GB building CSV. All paths, CRS,
grid settings, thresholds and output file names live in `src/config.py`, so
every script uses the same inputs and writes to the same outputs.

## Repository layout

```
flood_risk/
|-- src/
|   |-- __init__.py
|   |-- config.py          # paths, CRS, grid, NoData, thresholds, output names
|   |-- utils.py           # shared raster/vector/grid/plot helpers
|   |-- 01_clip_layers.py  # Task 2: clip raw layers to Alappuzha + 1 km
|   |-- 01_inspect_layers.py
|   |-- 02_build_grid.py   # Task 3: common 90 m grid and district mask
|   |-- 02_align_rasters.py
|   |-- 03_prepare_regional_dem.py  # Task 4: regional DEM and waterways
|   |-- 03_terrain_features.py
|   |-- 03_waterway_distance.py
|   |-- 03_stack_features.py
|   |-- 04_exposure_population.py   # Task 5: population exposure
|   |-- 04_exposure_buildings.py
|   |-- 04_exposure_facilities.py
|   |-- 04_exposure_combine.py
|   |-- 05_clean_census.py          # Task 6: Census vulnerability indicators
|   |-- 05_join_boundaries.py
|   `-- 05_vulnerability_score.py
|-- data/
|   |-- raw_data/          # inputs, tracked with Git LFS
|   |-- interim/           # clipped layers (ignored by Git)
|   `-- processed/         # aligned 90 m grid and feature stack (ignored by Git)
|-- outputs/
|   |-- maps/
|   |-- figures/
|   `-- tables/
|-- models/                # trained models (ignored by Git)
|-- notebooks/             # quick exploration only
|-- dashboard/             # Task 17 dashboard
|-- docs/                  # SRS and plan of action
|-- requirements.txt
|-- requirement-cpu.txt
|-- requirement-gpu.txt
`-- README.md
```

`data/interim/`, `data/processed/`, `models/`, `outputs/`, `__pycache__/`,
`.ipynb_checkpoints/` and `.env` are ignored by Git. `data/raw_data/` is tracked
with Git LFS. Keep the raw folder names exactly as delivered (some contain
spaces and dots); `config.py` holds the exact paths.

## Getting the data

Raw data is stored in this repository with Git LFS. Install Git LFS once, then
clone and pull the data:

```powershell
git lfs install
git clone https://github.com/SumitSharma2124/flood-risk-assessment-.git
cd flood-risk-assessment-
git lfs pull
```

If you already cloned the repository, run:

```powershell
git lfs install
git lfs pull
```

The raw files should end up in `data/raw_data/`. Expected contents:

| Folder | Contents |
| --- | --- |
| `Building_footprints/` | `3b1_buildings.csv` (Google Open Buildings, Kerala) |
| `census_2011_vulnerability_indicators/` | Census 2011 DDW/PCA tables (file names use `3211`; Alappuzha is district code `598` inside the files) |
| `critical_facilities/` | Facilities GeoPackage and the original clipping boundary |
| `district_boundaries/` | `geoBoundaries-IND-ADM2.geojson` |
| `elevation/` | Copernicus 30 m DEM tiles: `N09_00_E076_00`, `N09_00_E077_00`, `N10_00_E076_00`, `N10_00_E077_00` |
| `flood_extent/` | `kerala_flood_2018_s1.tif` and the 2018 flood reports |
| `flood_probability/` | KSDMA return-period rasters (10–500 years) |
| `land_cover/` | ESA WorldCover 10 m 2021 (`N09E075` tile is needed) |
| `population_count_100m/` | Annual 100 m population counts (2018 is used) |
| `rainfall/` | CHIRPS monthly rainfall (not used in the first pass) |
| `roads/` | `Alappuzha_roads.gpkg` |
| `shelters/` | Kerala monsoon camp Excel workbook |
| `village_boundaries/` | `kl.geojson` and the 2001/2011 village code mapping |
| `waterways/` | `Alappuzha`, `Kottayam` and `Pathanamthitta` waterways GeoPackages |

## Rebuilding the building footprints CSV

GitHub LFS rejects any single file larger than 2 GB. The complete file

```
data/raw_data/Building_footprints/3b1_buildings.csv   (4,082,787,227 bytes)
```

is therefore stored as three byte-exact parts that must be concatenated **in
order**:

| Part | Size (bytes) |
| --- | --- |
| `3b1_buildings.csv.part00` | 1,992,294,400 |
| `3b1_buildings.csv.part01` | 1,992,294,400 |
| `3b1_buildings.csv.part02` | 98,198,427 |

### Windows (PowerShell or cmd)

```powershell
cmd /c copy /b data\raw_data\Building_footprints\3b1_buildings.csv.part00+data\raw_data\Building_footprints\3b1_buildings.csv.part01+data\raw_data\Building_footprints\3b1_buildings.csv.part02 data\raw_data\Building_footprints\3b1_buildings.csv
```

### Linux / macOS

```bash
cat data/raw_data/Building_footprints/3b1_buildings.csv.part00 \
    data/raw_data/Building_footprints/3b1_buildings.csv.part01 \
    data/raw_data/Building_footprints/3b1_buildings.csv.part02 \
    > data/raw_data/Building_footprints/3b1_buildings.csv
```

### Verify the rebuilt file

The SHA-256 checksum of the complete file must be:

```
372463300AACC982FCEC99A244D9EBCEEDB27088CEB2FF7FF72E8E748C6E8302
```

```powershell
Get-FileHash data\raw_data\Building_footprints\3b1_buildings.csv -Algorithm SHA256
```

The combined `3b1_buildings.csv` is listed in `.gitignore`, so only the three
parts are committed.

Check that every configured input exists:

```powershell
python -c "from src import config as cfg; cfg.check_input_paths(); print('all inputs present')"
```

## Installation

The pinned versions were tested with **Python 3.14, 64-bit, on Windows**. Use a
virtual environment so the whole team gets the same results.

### CPU-only machines (most teammates)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirement-cpu.txt --index-url https://download.pytorch.org/whl/cpu
```

### GPU machines (GTX 1650, CUDA 13.2)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirement-gpu.txt --index-url https://download.pytorch.org/whl/cu132
```

`requirements.txt` is a pinned freeze of the development environment.
`requirement-cpu.txt` and `requirement-gpu.txt` only add the matching PyTorch
build.

## Running the pipeline

Always run scripts **from the project root** so package imports work. The
digit-prefixed files cannot be imported as modules, so run them directly:

```powershell
.venv\Scripts\python.exe src\01_clip_layers.py
```

Scripts are added task by task. Current run order:

1. Clip raw layers to the district and inspect them:

   ```powershell
   .venv\Scripts\python.exe src\01_clip_layers.py
   .venv\Scripts\python.exe src\01_inspect_layers.py
   ```

2. Build the common 90 m grid and align every raster layer:

   ```powershell
   .venv\Scripts\python.exe src\02_build_grid.py
   .venv\Scripts\python.exe src\02_align_rasters.py
   ```

3. Prepare the regional DEM/waterways, derive terrain features and distances,
   then stack them:

   ```powershell
   .venv\Scripts\python.exe src\03_prepare_regional_dem.py
   .venv\Scripts\python.exe src\03_terrain_features.py
   .venv\Scripts\python.exe src\03_waterway_distance.py
   .venv\Scripts\python.exe src\03_stack_features.py
   ```

4. Build exposure layers and the combined 0-1 exposure score:

   ```powershell
   .venv\Scripts\python.exe src\04_exposure_population.py
   .venv\Scripts\python.exe src\04_exposure_buildings.py
   .venv\Scripts\python.exe src\04_exposure_facilities.py
   .venv\Scripts\python.exe src\04_exposure_combine.py
   ```

5. Clean the Census 2011 indicators, join them to village polygons and build
   the relative vulnerability score:

   ```powershell
   .venv\Scripts\python.exe src\05_clean_census.py
   .venv\Scripts\python.exe src\05_join_boundaries.py
   .venv\Scripts\python.exe src\05_vulnerability_score.py
   ```

6. Build flood labels (KSDMA 100-year scenario).
7. Create the spatial-block train/validation/test split.
8. Train and tune the model.
9. Evaluate, interpret and export risk maps/tables.
10. Build the dashboard.

## Configuration

`src/config.py` is the single source of truth:

- `PROJECT_ROOT` is derived from the file location, so the project runs on any
  machine.
- Folders under `data/`, `outputs/`, `models/`, `docs/`, `notebooks/` and
  `dashboard/` are created automatically on import.
- Study settings: `DISTRICT_NAME = "Alappuzha"`, `SCENARIO_NAME = "100yr"`,
  `RETURN_PERIODS = (10, 25, 50, 100, 200, 500)`.
- CRS: `EPSG:4326` for geographic data and `EPSG:32643` (UTM 43N) for analysis.
- Grid: 90 m cells, 1 km buffer around the district boundary, saved at
  `data/processed/grid_template.tif`. The regional routing grid is saved at
  `data/processed/regional_grid_template.tif`.
- NoData: `-9999` for processed rasters; raw KSDMA NoData is `-3.4e38` (and
  `0` for the 500-year raster).
- Terrain: four regional Copernicus DEM tiles, waterway burn depth `5.0 m`,
  stream accumulation threshold `1,000` cells, HAND drainage threshold `250`
  cells, TWI minimum slope `0.1°`, and flat resolution epsilon `1e-7`.
- Model settings: building confidence `0.7`, flood-mask threshold `0.5`,
  random seed `42`, patch size `64`, 5 km spatial blocks, 70/15/15 split.
- Exposure settings: minimum building footprint `10 m2`, facility dedup radius
  `50 m`, facility smoothing radius `500 m`, exposure cap at the 99th
  percentile, and weights population `0.5` / buildings `0.3` / facilities
  `0.2`.
- Vulnerability settings: four equal-weighted indicators (child share,
  combined SC/ST share, marginal/non-worker share and housing deprivation),
  percentile-rank scaling, sub-district median filling, and an alternative
  housing-heavy weight set for rank-sensitivity checks.
- `OUTPUT_FILES` defines every interim/processed output path. Use
  `cfg.output_path("dem_grid")` rather than hard-coding file names in scripts.

`src/utils.py` provides the shared helpers: logger and seed setup, raster/vector
read-write, district loading, buffered clipping with windowed reads, grid
construction/alignment/rasterisation, point-to-grid counting and
deduplication, percentile scaling and smoothing, NoData handling, grid checks,
raster summaries and map plotting.

## Data checks performed during setup

- `Building_footprints/3b1_buildings.csv` is complete. Its SHA-256 is
  `372463300AACC982FCEC99A244D9EBCEEDB27088CEB2FF7FF72E8E748C6E8302` and its
  extent covers Alappuzha. The three LFS parts are kept so the file can be
  reconstructed after cloning.
- `geoBoundaries-IND-ADM2.geojson` uses the column **`shapeName`**; the
  `Alappuzha` row is present.
- The DEM tile `Copernicus_DSM_COG_30_N09_00_E076_00_DEM.tif` covers the whole
  district (district bounds: 76.277–76.693°E, 9.110–9.892°N). The four
  `N09/N10` × `E076/E077` tiles are used for regional flow routing, so upstream
  catchments in Kottayam and Pathanamthitta are included.
- Waterways are available for Alappuzha, Kottayam and Pathanamthitta. Task 4
  merges and de-duplicates them, then burns flow lines into the regional DEM.
- The critical facilities were **not** clipped to Alappuzha. They follow
  `clipping_boundary.geojson`, a custom polygon of about 3,679 km² (the district
  is about 1,432 km²), and only 1,195 of 3,675 facilities fall inside
  Alappuzha. Task 2 re-clips them to the district boundary.
- The village file contains 88 Alappuzha villages, and the district name column
  is `DISTRICT`.
- Census file names use `3211`, while the rows inside use state/district code
  `32`/`598` for Alappuzha. The shortlisted PCA and HLPCA files are configured
  in `CENSUS_FILES`; `HLPCA-32598` is the Alappuzha housing-census file
  (`32597` is Kottayam and `32599` is Pathanamthitta).
- The 2018 population grid totals 2,703,786 inside the district, 27.1% above
  the 2011 census (2,127,789). WorldPop R2025A is used as-is; the difference
  should be treated as a documented source caveat.
- The building layer has 1,111,252 features. After confidence `>=0.7`, minimum
  footprint `10 m2` and district filtering, 791,918 buildings remain
  (68.1 km2 footprint, about 1.58x the 500,000-household reference). Google
  Open Buildings uses newer imagery than 2018.
- Facilities start at 1,285 features; 14 water/dam/reservoir records are
  excluded, and 50 m same-category deduplication leaves 1,031 facilities
  inside Alappuzha.

## Task 6: Census vulnerability

`05_clean_census.py` keeps village, town and ward rows, removes the
sub-district/district summary rows and keeps the larger `Alappuzha (M + OG)`
record instead of the duplicate `Alappuzha (M)` row. The cleaned area totals
are 90 areas (52 villages, 38 towns) and 207 wards; the area population is
exactly the 2011 census total, `2,127,789`.

The PCA tables supply village-level SC/ST counts. The separate SC/ST PCA tables
include district totals plus town rows (not village rows) and are used as
validation: their district totals match the base PCA totals (`201,211` SC and
`6,574` ST), and their town totals match the base urban totals (`89,280` SC and
`3,613` ST).

`05_join_boundaries.py` links the 88 Alappuzha polygons to 2011 areas through
the 2011→2001 mapping, with name and type fallback for municipal towns. All
88/88 polygons match; 83 matches use the code/name mapping and 5 use name/type
fallback. The matched population share is `2,071,460 / 2,127,789 = 97.35%`.
`Vayalar (CT)` and `Thanneermukkam (CT)` have no separate 2001 polygons and are
listed in the join report. `Aryad South (Part)` has zero population and is
filled from its sub-district median and flagged.

The vulnerability score uses four equal-weighted indicators, each converted to
a percentile rank across areas:

- children aged 0-6 share (`P_06 / TOT_P`);
- combined SC/ST share (`(P_SC + P_ST) / TOT_P`);
- marginal or non-worker share (`(MARGWORK_P + NON_WORK_P) / TOT_P`);
- housing deprivation, the mean of the no-latrine and temporary-structure
  percentages from HLPCA.

The score ranges from `0.112` to `0.810` (mean `0.500`). The raster has
`176,745` district cells, with `3,062` cells without a village filled by the
median area score (`0.514`) and NoData outside the district. Validation results:
neighbour correlation `0.994`, equal-vs-housing-heavy Spearman `0.951`, and
top-10 overlap `80%`. The score is relative within Alappuzha, not an absolute
national scale.

Outputs:

- `data/interim/census_clean.csv`
- `data/processed/villages_vuln.gpkg`
- `data/processed/vulnerability_score.tif`
- `outputs/tables/05_join_report.csv`
- `outputs/tables/05_census_stats.json`
- `outputs/tables/05_join_stats.json`
- `outputs/tables/05_vulnerability_summary.csv`
- `outputs/tables/05_vulnerability_stats.json`
- `outputs/figures/05_vulnerability_overview.png`

Limitations: the census is from 2011 and predates both the 2018 flood and the
2018 WorldPop layer; the 2001 polygons do not represent every 2011 census town;
towns receive one area-level score while the 207 ward rows are not scored
separately; disability and elderly shares were not used, with child share
serving as the age proxy.
