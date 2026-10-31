# Flood Risk Assessment — Raw Data

Raw geospatial and socio-economic datasets used for the Kerala flood-risk
assessment (Alappuzha, Kottayam, and Pathanamthitta districts).

Large files are stored with [Git LFS](https://git-lfs.com/). Install it before
cloning, otherwise the files will be downloaded as small pointer files.

```bash
git lfs install
git clone https://github.com/SumitSharma2124/flood-risk-assessment-.git
```

If you already cloned without LFS:

```bash
git lfs pull
```

## Rebuilding the building footprints CSV

GitHub LFS rejects any single file larger than 2 GB. The original file

```
raw_data/Building_footprints/3b1_buildings.csv   (4,082,787,227 bytes)
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
cmd /c copy /b raw_data\Building_footprints\3b1_buildings.csv.part00+raw_data\Building_footprints\3b1_buildings.csv.part01+raw_data\Building_footprints\3b1_buildings.csv.part02 raw_data\Building_footprints\3b1_buildings.csv
```

### Linux / macOS

```bash
cat raw_data/Building_footprints/3b1_buildings.csv.part00 \
    raw_data/Building_footprints/3b1_buildings.csv.part01 \
    raw_data/Building_footprints/3b1_buildings.csv.part02 \
    > raw_data/Building_footprints/3b1_buildings.csv
```

### Verify the rebuilt file

The SHA-256 checksum of the complete file must be:

```
372463300AACC982FCEC99A244D9EBCEEDB27088CEB2FF7FF72E8E748C6E8302
```

```powershell
# Windows
Get-FileHash raw_data\Building_footprints\3b1_buildings.csv -Algorithm SHA256
```

```bash
# Linux / macOS
sha256sum raw_data/Building_footprints/3b1_buildings.csv
```

The rebuilt `3b1_buildings.csv` is listed in `.gitignore`, so it will not be
committed again.

## Directory layout

| Directory | Contents |
| --- | --- |
| `Building_footprints/` | Building footprint polygons for Kerala (Google Open Buildings export, split into parts) |
| `census_2011_vulnerability_indicators/` | Census 2011 tables (DDW, PCA, SC/ST) for district code 3200 |
| `critical_facilities/` | Critical facilities vector layer and clipping boundary |
| `district_boundaries/` | India ADM2 district boundaries (geoBoundaries) |
| `elevation/` | Copernicus 30 m DEM tiles |
| `flood_extent/` | 2018 Kerala flood extent, duration, and inundation reports |
| `flood_probability/` | Return-period flood probability rasters (10–500 years) |
| `land_cover/` | ESA WorldCover 10 m 2021 land cover tiles |
| `population_count_100m/` | Annual 100 m population count rasters (2015–2025) |
| `rainfall/` | CHIRPS v3.0 monthly rainfall rasters (June–September, 2018–2020) |
| `roads/` | Road networks for Alappuzha, Kottayam, and Pathanamthitta |
| `shelters/` | Kerala monsoon camp and shelter lists |
| `village_boundaries/` | Kerala village boundaries and 2001/2011 code mapping |
| `waterways/` | Waterway networks for Alappuzha, Kottayam, and Pathanamthitta |
