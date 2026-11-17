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
|   `-- 03_stack_features.py
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
| `census_2011_vulnerability_indicators/` | Census 2011 DDW/PCA tables (Alappuzha is district code 11; look for `3211`) |
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

4. Build flood labels (KSDMA 100-year scenario).
5. Create the spatial-block train/validation/test split.
6. Train and tune the model.
7. Evaluate, interpret and export risk maps/tables.
8. Build the dashboard.

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
- `OUTPUT_FILES` defines every interim/processed output path. Use
  `cfg.output_path("dem_grid")` rather than hard-coding file names in scripts.

`src/utils.py` provides the shared helpers: logger and seed setup, raster/vector
read-write, district loading, buffered clipping with windowed reads, grid
construction/alignment/rasterisation, NoData handling, grid checks, raster
summaries and map plotting.

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
- Census files use state/district code `32`/`11`, so the Alappuzha files start
  with `3211`. The shortlist is added to `CENSUS_FILES` in `config.py` in
  Task 6.
