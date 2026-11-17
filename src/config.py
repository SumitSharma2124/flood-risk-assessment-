"""Central configuration for the Alappuzha flood-risk assessment.

Every script imports paths, CRS and modelling settings from here so that all
teammates work on the same projection, grid and output files.

Run scripts from the project root, for example::

    python src/01_clip_layers.py
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Project root and folders
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw_data"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MAPS_DIR = OUTPUTS_DIR / "maps"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"

MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
DOCS_DIR = PROJECT_ROOT / "docs"

_FOLDERS = (
    DATA_DIR,
    RAW_DIR,
    INTERIM_DIR,
    PROCESSED_DIR,
    OUTPUTS_DIR,
    MAPS_DIR,
    FIGURES_DIR,
    TABLES_DIR,
    MODELS_DIR,
    NOTEBOOKS_DIR,
    DASHBOARD_DIR,
    DOCS_DIR,
)

for _folder in _FOLDERS:
    _folder.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Raw input paths (exact folder names, including spaces and dots)
# ---------------------------------------------------------------------------

DEM_RAW = (
    RAW_DIR
    / "elevation"
    / "Copernicus_DSM_COG_30_N09_00_E076_00_DEM.tif"
)

# Four Copernicus tiles cover the three target districts and their upstream
# catchments (76-78 E, 9-11 N).
DEM_REGIONAL_TILES = (
    RAW_DIR / "elevation" / "Copernicus_DSM_COG_30_N09_00_E076_00_DEM.tif",
    RAW_DIR / "elevation" / "Copernicus_DSM_COG_30_N09_00_E077_00_DEM.tif",
    RAW_DIR / "elevation" / "Copernicus_DSM_COG_30_N10_00_E076_00_DEM.tif",
    RAW_DIR / "elevation" / "Copernicus_DSM_COG_30_N10_00_E077_00_DEM.tif",
)

LANDCOVER_RAW = (
    RAW_DIR
    / "land_cover"
    / "ESA_WorldCover_10m_2021_v200_N09E075_Map.tif"
)

KSDMA_DIR = RAW_DIR / "flood_probability"

RETURN_PERIOD_FILES = {
    10: KSDMA_DIR
    / "1. Flood Return Probability History (10, 25, 50 years)"
    / "Kerala_Flood_10_yr_Historical.tif",
    25: KSDMA_DIR
    / "1. Flood Return Probability History (10, 25, 50 years)"
    / "Kerala_Flood_25_yr_Historical.tif",
    50: KSDMA_DIR
    / "1. Flood Return Probability History (10, 25, 50 years)"
    / "Kerala_Flood_50_yr_Historical.tif",
    100: KSDMA_DIR
    / "2.Flood Return Probability History (100, 200, 500 years)"
    / "Kerala_Flood_100_yr_Historical.tif",
    200: KSDMA_DIR
    / "2.Flood Return Probability History (100, 200, 500 years)"
    / "Kerala_Flood_200_yr_Historical.tif",
    500: KSDMA_DIR
    / "2.Flood Return Probability History (100, 200, 500 years)"
    / "Kerala_Flood_500_yr_Historical.tif",
}

FLOOD_2018_RAW = RAW_DIR / "flood_extent" / "kerala_flood_2018_s1.tif"

ROADS_RAW = RAW_DIR / "roads" / "Alappuzha_roads.gpkg"
WATERWAYS_RAW = RAW_DIR / "waterways" / "Alappuzha_waterways.gpkg"
WATERWAYS_BY_DISTRICT = {
    "Alappuzha": WATERWAYS_RAW,
    "Kottayam": RAW_DIR / "waterways" / "Kottayam_waterways.gpkg",
    "Pathanamthitta": RAW_DIR / "waterways" / "Pathanamthitta_waterways.gpkg",
}

POPULATION_2018_RAW = (
    RAW_DIR
    / "population_count_100m"
    / "ind_pop_2018_CN_100m_R2025A_v1.tif"
)

BUILDINGS_RAW = (
    RAW_DIR / "Building_footprints" / "3b1_buildings.csv"
)

FACILITIES_RAW = (
    RAW_DIR / "critical_facilities" / "critical_facilities.gpkg"
)
# Original QA polygon supplied with the facilities (~3,679 km2). It is much
# larger than Alappuzha (~1,432 km2), so only about one third of the facilities
# fall inside the district. Re-clip facilities to the real district boundary.
FACILITIES_CLIP_BOUNDARY_RAW = (
    RAW_DIR / "critical_facilities" / "clipping_boundary.geojson"
)

DISTRICTS_RAW = (
    RAW_DIR / "district_boundaries" / "geoBoundaries-IND-ADM2.geojson"
)

VILLAGES_RAW = RAW_DIR / "village_boundaries" / "kl.geojson"
VILLAGE_CODE_MAPPING_RAW = (
    RAW_DIR / "village_boundaries" / "kl_village_2011_2001_code_mapping.txt"
)

CAMPS_RAW = (
    RAW_DIR / "shelters" / "Monsoon-Camp-Details-ALL-Kerala.xlsx"
)

CENSUS_DIR = RAW_DIR / "census_2011_vulnerability_indicators"

# Filled in Task 6. Alappuzha is district code 11, so look for the "3211"
# files first. Example: {"population": CENSUS_DIR / "PCA_CDB-3211-F-Census.xlsx"}
CENSUS_FILES: dict[str, Path] = {}


# ---------------------------------------------------------------------------
# Study settings
# ---------------------------------------------------------------------------

DISTRICT_NAME = "Alappuzha"
REGIONAL_DISTRICT_NAMES = ("Alappuzha", "Kottayam", "Pathanamthitta")

# Column in the geoBoundaries ADM2 file that holds district names.
DISTRICT_NAME_COLUMN = "shapeName"

SCENARIO_NAME = "100yr"
SCENARIO_RETURN_PERIOD = 100
RETURN_PERIODS = (10, 25, 50, 100, 200, 500)

KSDMA_SCENARIO_RAW = RETURN_PERIOD_FILES[SCENARIO_RETURN_PERIOD]

CRS_GEOGRAPHIC = "EPSG:4326"
CRS_WORKING = "EPSG:32643"  # UTM zone 43N


# ---------------------------------------------------------------------------
# Grid settings
# ---------------------------------------------------------------------------

GRID_RESOLUTION = 90  # metres
DEM_30M_RESOLUTION = 30  # optional full-detail DEM
DISTRICT_BUFFER_M = 1000  # metres around the district boundary
GRID_PATH = PROCESSED_DIR / "grid_template.tif"
REGIONAL_GRID_PATH = PROCESSED_DIR / "regional_grid_template.tif"


# ---------------------------------------------------------------------------
# NoData rules
# ---------------------------------------------------------------------------

NODATA = -9999.0

# Raw KSDMA rasters use -3.4e38; the 500-year raster records NoData as 0.
KSDMA_NODATA = -3.4028230607370965e38
KSDMA_NODATA_500 = 0.0
KSDMA_NODATA_BY_RETURN_PERIOD = {
    10: (KSDMA_NODATA,),
    25: (KSDMA_NODATA,),
    50: (KSDMA_NODATA,),
    100: (KSDMA_NODATA,),
    200: (KSDMA_NODATA,),
    500: (KSDMA_NODATA_500,),
}
KSDMA_NODATA_VALUES = KSDMA_NODATA_BY_RETURN_PERIOD[SCENARIO_RETURN_PERIOD]

# Mask of cells where KSDMA actually modelled flood probability (as opposed to
# the background NoData surrounding the modelled area).
KSDMA_MASK_NAME = "ksdma_modelled_mask.tif"
KSDMA_MASK_PATH = PROCESSED_DIR / KSDMA_MASK_NAME


# ---------------------------------------------------------------------------
# Thresholds and model settings
# ---------------------------------------------------------------------------

BUILDING_CONFIDENCE_THRESHOLD = 0.7
FLOOD_MASK_THRESHOLD = 0.5

# Terrain and drainage settings (Task 4).
WATERWAY_FLOW_TYPES = (
    "river",
    "stream",
    "canal",
    "ditch",
    "drain",
    "tidal_channel",
    "brook",
)
WATERWAY_BURN_DEPTH_M = 5.0
STREAM_ACCUMULATION_THRESHOLD = 1000  # upstream cells at 90 m
HAND_ACCUMULATION_THRESHOLD = 250  # upstream cells for HAND drainage cells
TWI_MIN_SLOPE_DEG = 0.1
RESOLVE_FLATS_EPS = 1e-7

# Columns in the Google Open Buildings CSV.
BUILDINGS_LAT_COLUMN = "latitude"
BUILDINGS_LON_COLUMN = "longitude"
BUILDINGS_GEOMETRY_COLUMN = "geometry"
BUILDINGS_CHUNK_SIZE = 250_000

# 2011 Census population of Alappuzha district, used as the sanity-check
# reference for the 2018 gridded population layer.
ALAPPUZHA_POPULATION_2011 = 2_127_789

# Population counts are stored per 100 m source cell; the common grid is 90 m.
POPULATION_SOURCE_CELL_AREA_M2 = 10_000.0
POPULATION_TARGET_CELL_AREA_M2 = float(GRID_RESOLUTION**2)

RANDOM_SEED = 42

PATCH_SIZE = 64
BLOCK_SIZE_M = 5000.0  # spatial block size for the geographic train/test split

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# Weights (lambda values) for the routing cost surface.
ROUTING_LAMBDAS = {
    "slope": 1.0,
    "land_cover": 1.0,
    "roads": 0.5,
    "waterways": 0.5,
}


# ---------------------------------------------------------------------------
# ESA WorldCover classes (v200, 2021)
# ---------------------------------------------------------------------------

ESA_WORLDCOVER_CLASSES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}

# Key fractions saved separately for modelling; the remaining classes are
# stored as an "other" fraction so that every valid cell sums to one.
LANDCOVER_KEY_CLASSES = {
    "trees": 10,
    "cropland": 40,
    "builtup": 50,
    "water": 80,
    "wetland": 90,
    "mangrove": 95,
}
LANDCOVER_CLASS_VALUES = tuple(sorted(ESA_WORLDCOVER_CLASSES))


# ---------------------------------------------------------------------------
# Output file names: every script must use these
# ---------------------------------------------------------------------------

OUTPUT_FILES = {
    # Interim clipped layers (Task 2)
    "district": INTERIM_DIR / "district.gpkg",
    "dem_clip": INTERIM_DIR / "dem_clip.tif",
    "landcover_clip": INTERIM_DIR / "landcover_clip.tif",
    "ksdma_100yr_clip": INTERIM_DIR / "ksdma_100yr_clip.tif",
    "flood2018_clip": INTERIM_DIR / "flood2018_clip.tif",
    "pop2018_clip": INTERIM_DIR / "pop2018_clip.tif",
    "roads_clip": INTERIM_DIR / "roads_clip.gpkg",
    "waterways_clip": INTERIM_DIR / "waterways_clip.gpkg",
    "buildings_clip": INTERIM_DIR / "buildings_clip.gpkg",
    "facilities_clip": INTERIM_DIR / "facilities_clip.gpkg",
    "villages_clip": INTERIM_DIR / "villages_clip.gpkg",
    # Task 2 inspection outputs
    "layer_summary": TABLES_DIR / "01_layer_summary.csv",
    "layers_overview": FIGURES_DIR / "01_layers_overview.png",
    "clip_stats": TABLES_DIR / "01_clip_stats.json",
    # Aligned 90 m grid
    "grid": GRID_PATH,
    "district_mask": PROCESSED_DIR / "district_mask.tif",
    "dem_grid": PROCESSED_DIR / "dem_90m.tif",
    "landcover_mode": PROCESSED_DIR / "landcover_mode.tif",
    "landcover_frac": PROCESSED_DIR / "landcover_frac.tif",
    "ksdma_100yr_90m": PROCESSED_DIR / "ksdma_100yr_90m.tif",
    "ksdma_modelled_mask": KSDMA_MASK_PATH,
    "flood2018_frac": PROCESSED_DIR / "flood2018_frac.tif",
    "flood2018_bin": PROCESSED_DIR / "flood2018_bin.tif",
    "pop2018_90m": PROCESSED_DIR / "pop2018_90m.tif",
    "dem_30m": PROCESSED_DIR / "dem_30m.tif",
    # Regional terrain inputs (Task 4)
    "regional_grid": REGIONAL_GRID_PATH,
    "dem_regional_raw_90m": PROCESSED_DIR / "dem_regional_raw_90m.tif",
    "dem_regional_90m": PROCESSED_DIR / "dem_regional_90m.tif",
    "waterways_regional": INTERIM_DIR / "waterways_regional.gpkg",
    "waterways_regional_grid": PROCESSED_DIR / "waterways_regional_90m.tif",
    "regional_dem_stats": TABLES_DIR / "03_regional_dem_stats.json",
    # Aliases retained for later tasks
    "landcover_grid": PROCESSED_DIR / "landcover_mode.tif",
    "ksdma_grid": PROCESSED_DIR / "ksdma_100yr_90m.tif",
    "flood2018_grid": PROCESSED_DIR / "flood2018_bin.tif",
    "population_grid": PROCESSED_DIR / "pop2018_90m.tif",
    "ksdma_mask": KSDMA_MASK_PATH,
    "builtup_grid": PROCESSED_DIR / "builtup_90m.tif",
    "roads_grid": PROCESSED_DIR / "roads_90m.tif",
    "waterways_grid": PROCESSED_DIR / "waterways_90m.tif",
    "distance_roads_grid": PROCESSED_DIR / "distance_to_roads_90m.tif",
    "distance_waterways_grid": PROCESSED_DIR / "distance_to_waterways_90m.tif",
    # Task 3 inspection outputs
    "aligned_overview": FIGURES_DIR / "02_aligned_overview.png",
    "alignment_stats": TABLES_DIR / "02_alignment_stats.json",
    # Terrain features (Task 4)
    "flow_direction": PROCESSED_DIR / "flow_direction_90m.tif",
    "flow_accumulation": PROCESSED_DIR / "flow_accumulation_90m.tif",
    "log_flow_accumulation": PROCESSED_DIR / "log_flow_accumulation_90m.tif",
    "streams_grid": PROCESSED_DIR / "streams_90m.tif",
    "hand_grid": PROCESSED_DIR / "hand_90m.tif",
    "slope_grid": PROCESSED_DIR / "slope_90m.tif",
    "twi_grid": PROCESSED_DIR / "twi_90m.tif",
    "distance_permanent_water_grid": (
        PROCESSED_DIR / "distance_to_permanent_water_90m.tif"
    ),
    "features_stack": PROCESSED_DIR / "features_stack.tif",
    "features_stack_bands": PROCESSED_DIR / "features_stack_bands.json",
    "features_valid_mask": PROCESSED_DIR / "features_valid_mask.tif",
    "terrain_stats": TABLES_DIR / "03_terrain_stats.json",
    "waterway_distance_stats": TABLES_DIR / "03_waterway_distance_stats.json",
    "feature_summary": TABLES_DIR / "03_feature_summary.csv",
    "features_overview": FIGURES_DIR / "03_features_overview.png",
    "feature_stats": TABLES_DIR / "03_feature_stats.json",
    # Later modelling outputs
    "flood_labels": PROCESSED_DIR / "flood_labels_90m.tif",
    "model": MODELS_DIR / "flood_risk_model.pt",
}


def output_path(name: str) -> Path:
    """Return a configured output path by logical name."""
    try:
        return OUTPUT_FILES[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown output '{name}'. Available: {sorted(OUTPUT_FILES)}"
        ) from exc


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

RAW_INPUTS = {
    "dem": DEM_RAW,
    "landcover": LANDCOVER_RAW,
    **{f"ksdma_{rp}yr": path for rp, path in RETURN_PERIOD_FILES.items()},
    "flood_2018": FLOOD_2018_RAW,
    "roads": ROADS_RAW,
    "waterways": WATERWAYS_RAW,
    "population_2018": POPULATION_2018_RAW,
    "buildings": BUILDINGS_RAW,
    "facilities": FACILITIES_RAW,
    "facilities_clip_boundary": FACILITIES_CLIP_BOUNDARY_RAW,
    "districts": DISTRICTS_RAW,
    "villages": VILLAGES_RAW,
    "village_code_mapping": VILLAGE_CODE_MAPPING_RAW,
    "camps": CAMPS_RAW,
}


def missing_input_paths(paths: dict[str, Path] | None = None) -> list[Path]:
    """Return raw input paths that do not exist (empty list means all good)."""
    paths = paths or RAW_INPUTS
    return [Path(path) for path in paths.values() if not Path(path).exists()]


def check_input_paths(paths: dict[str, Path] | None = None) -> None:
    """Raise FileNotFoundError if any configured raw input is missing."""
    missing = missing_input_paths(paths)
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing {len(missing)} input file(s):\n{formatted}")
