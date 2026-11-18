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

# Task 6 Census 2011 inputs. Alappuzha is district code 598 in these files
# (state 32); the "3211" in the file names is a legacy file code.
CENSUS_FILES: dict[str, Path] = {
    "pca": CENSUS_DIR / "DDW_PCA3211_2011_MDDS with UI.xlsx",
    "pca_sc": CENSUS_DIR / "PCA_SC3211_2011_MDDS_DDW.xlsx",
    "pca_st": CENSUS_DIR / "PCA_ST3211_2011_MDDS_DDW.xlsx",
    "hlpca": CENSUS_DIR / "HLPCA-32598-2011_H14_census.xlsx",
}

# The HLPCA files are split by district: 32597=Kottayam, 32598=Alappuzha,
# 32599=Pathanamthitta. Verify the district name inside before using one.
CENSUS_HLPCA_CANDIDATES = (
    CENSUS_DIR / "HLPCA-32597-2011_H14_census.xlsx",
    CENSUS_DIR / "HLPCA-32598-2011_H14_census.xlsx",
    CENSUS_DIR / "HLPCA-32599-2011_H14_census.xlsx",
)
CENSUS_STATE_CODE = 32
CENSUS_DISTRICT_CODE = 598
CENSUS_HLPCA_DATA_START_ROW = 7
CENSUS_LOCAL_LEVELS = ("VILLAGE", "TOWN", "WARD")
CENSUS_AREA_LEVELS = ("VILLAGE", "TOWN")
CENSUS_HLPCA_COLUMNS = {
    "no_latrine_pct": 99,
    "temporary_structure_pct": 141,
}

# Task 6 vulnerability indicators. All directions are +1 because higher values
# mean greater vulnerability. The score is relative within Alappuzha.
VULNERABILITY_INDICATORS = {
    "child_share": {
        "label": "Children aged 0-6 share",
        "definition": "P_06 / TOT_P",
        "direction": 1,
    },
    "scst_share": {
        "label": "Scheduled caste and scheduled tribe share",
        "definition": "(P_SC + P_ST) / TOT_P",
        "direction": 1,
    },
    "worker_fragility": {
        "label": "Marginal or non-worker share",
        "definition": "(MARGWORK_P + NON_WORK_P) / TOT_P",
        "direction": 1,
    },
    "housing_deprivation": {
        "label": "Housing deprivation",
        "definition": (
            "mean(no latrine within premises, temporary structure), HLPCA"
        ),
        "direction": 1,
    },
}
VULNERABILITY_WEIGHTS = {
    "child_share": 0.25,
    "scst_share": 0.25,
    "worker_fragility": 0.25,
    "housing_deprivation": 0.25,
}
VULNERABILITY_WEIGHT_SETS = {
    "equal": VULNERABILITY_WEIGHTS,
    "housing_heavy": {
        "child_share": 0.20,
        "scst_share": 0.20,
        "worker_fragility": 0.20,
        "housing_deprivation": 0.40,
    },
}
VULNERABILITY_FILL_METHOD = "subdistrict_median"
VULNERABILITY_MIN_MATCHED_VILLAGE_SHARE = 0.90
VULNERABILITY_MIN_MATCHED_POPULATION_SHARE = 0.90
VULNERABILITY_MAX_MISSING_SHARE = 0.10
VULNERABILITY_MIN_WEIGHT_SPEARMAN = 0.80
VULNERABILITY_MIN_NEIGHBOUR_CORRELATION = 0.30


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

# Exposure settings (Task 5).
MIN_BUILDING_FOOTPRINT_M2 = 10.0
BUILDING_COUNT_REFERENCE = 500_000  # order-of-magnitude 2011 household check
EXPOSURE_CAP_PERCENTILE = 99.0
EXPOSURE_SCALE_LOG = False
EXPOSURE_BUILDING_METRIC = "count"  # "count" or "area"
FACILITY_DEDUP_RADIUS_M = 50.0
FACILITY_SMOOTH_RADIUS_M = 500.0
FACILITY_CATEGORIES = (
    "hospital",
    "fire_station",
    "police",
    "substation",
    "power_plant",
    "school",
    "college",
    "clinic",
    "water_tower",
    "pump_house",
    "water_point",
)
FACILITY_CATEGORY_WEIGHTS = {
    "hospital": 1.0,
    "fire_station": 0.9,
    "police": 0.8,
    "substation": 0.8,
    "power_plant": 0.8,
    "school": 0.5,
    "college": 0.5,
    "clinic": 0.4,
    "water_tower": 0.3,
    "pump_house": 0.3,
    "water_point": 0.2,
}
EXPOSURE_WEIGHTS = {
    "population": 0.5,
    "buildings": 0.3,
    "facilities": 0.2,
}

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
    # Exposure layers (Task 5)
    "exposure_population_count": PROCESSED_DIR / "exposure_population_count.tif",
    "exposure_pop": PROCESSED_DIR / "exposure_pop.tif",
    "exposure_buildings_count": PROCESSED_DIR / "exposure_buildings_count.tif",
    "exposure_buildings_area": PROCESSED_DIR / "exposure_buildings_area.tif",
    "exposure_facilities_counts": PROCESSED_DIR / "exposure_facilities_counts.tif",
    "exposure_facilities_categories": (
        PROCESSED_DIR / "exposure_facilities_categories.json"
    ),
    "exposure_facilities_score": PROCESSED_DIR / "exposure_facilities_score.tif",
    "exposure_facilities_score_raw": (
        PROCESSED_DIR / "exposure_facilities_score_raw.tif"
    ),
    "exposure_score": PROCESSED_DIR / "exposure_score.tif",
    "exposure_population_stats": TABLES_DIR / "04_exposure_population_stats.json",
    "exposure_buildings_stats": TABLES_DIR / "04_exposure_buildings_stats.json",
    "exposure_facilities_stats": TABLES_DIR / "04_exposure_facilities_stats.json",
    "exposure_combine_stats": TABLES_DIR / "04_exposure_combine_stats.json",
    "exposure_summary": TABLES_DIR / "04_exposure_summary.csv",
    "exposure_overview": FIGURES_DIR / "04_exposure_overview.png",
    # Vulnerability layers (Task 6)
    "census_clean": INTERIM_DIR / "census_clean.csv",
    "villages_vuln": PROCESSED_DIR / "villages_vuln.gpkg",
    "vulnerability_score": PROCESSED_DIR / "vulnerability_score.tif",
    "join_report": TABLES_DIR / "05_join_report.csv",
    "join_stats": TABLES_DIR / "05_join_stats.json",
    "census_stats": TABLES_DIR / "05_census_stats.json",
    "vulnerability_summary": TABLES_DIR / "05_vulnerability_summary.csv",
    "vulnerability_stats": TABLES_DIR / "05_vulnerability_stats.json",
    "vulnerability_overview": FIGURES_DIR / "05_vulnerability_overview.png",
    # Later modelling outputs
    "flood_labels": PROCESSED_DIR / "flood_labels_90m.tif",
    "model": MODELS_DIR / "flood_risk_model.pt",

    "routing_results":        "routing_results.csv",
    "routing_routes":         "routing_routes.gpkg",
    "routing_lambda_summary": "routing_lambda_summary.csv",
    "routing_sample_map":     "routing_sample_map.png",
    "routing_lambda_curve":   "routing_lambda_curve.png",
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
    "census_pca": CENSUS_FILES["pca"],
    "census_pca_sc": CENSUS_FILES["pca_sc"],
    "census_pca_st": CENSUS_FILES["pca_st"],
    "census_hlpca": CENSUS_FILES["hlpca"],
}

# Task 7: relief camps
OUTPUT_FILES.update({
    "camps_clean": INTERIM_DIR / "camps_clean.csv",
    "camps_located": INTERIM_DIR / "camps_located.csv",
    "camps_geocode_cache": INTERIM_DIR / "camps_geocode_cache.json",
    "camps": PROCESSED_DIR / "camps.gpkg",
    "camps_clean_stats": TABLES_DIR / "06_camps_clean_stats.json",
    "camps_validation": TABLES_DIR / "06_camps_validation.json",
    "camps_summary": TABLES_DIR / "06_camps_summary.csv",
    "camps_overview": FIGURES_DIR / "06_camps_overview.png",
})
CAMPS_FUZZY_THRESHOLD = 85       # name match score (0-100) inside the same village
CAMPS_FUZZY_THRESHOLD_WIDE = 92  # stricter score when searching the whole district
CAMPS_MAX_VILLAGE_DIST_M = 5000  # geocoded point must be near its village

# Task 8: Fuzzy-AHP susceptibility mapping
OUTPUT_FILES.update({
    "fuzzy_ahp_score": PROCESSED_DIR / "09_fuzzy_ahp_susceptibility.tif",
    "fuzzy_ahp_classes": PROCESSED_DIR / "09_fuzzy_ahp_classes.tif",
    "fuzzy_ahp_memberships": PROCESSED_DIR / "09_fuzzy_ahp_memberships.tif",
    "fuzzy_ahp_rules": PROCESSED_DIR / "09_fuzzy_ahp_rule_activations.tif",

    "fuzzy_ahp_pairwise": TABLES_DIR / "09_fuzzy_ahp_pairwise.csv",
    "fuzzy_ahp_weights": TABLES_DIR / "09_fuzzy_ahp_weights.csv",
    "fuzzy_ahp_rule_summary": TABLES_DIR / "09_fuzzy_ahp_rule_summary.csv",
    "fuzzy_ahp_stats": TABLES_DIR / "09_fuzzy_ahp_stats.json",

    "fuzzy_ahp_overview": FIGURES_DIR / "09_fuzzy_ahp_overview.png",
})

# Task 8: Fuzzy-AHP susceptibility mapping
OUTPUT_FILES.update({
    "fuzzy_ahp_score": PROCESSED_DIR / "08_fuzzy_ahp_susceptibility.tif",
    "fuzzy_ahp_classes": PROCESSED_DIR / "08_fuzzy_ahp_classes.tif",
    "fuzzy_ahp_memberships": PROCESSED_DIR / "08_fuzzy_ahp_memberships.tif",
    "fuzzy_ahp_rules": PROCESSED_DIR / "08_fuzzy_ahp_rule_activations.tif",

    "fuzzy_ahp_pairwise": TABLES_DIR / "08_fuzzy_ahp_pairwise.csv",
    "fuzzy_ahp_weights": TABLES_DIR / "08_fuzzy_ahp_weights.csv",
    "fuzzy_ahp_rule_summary": TABLES_DIR / "08_fuzzy_ahp_rule_summary.csv",
    "fuzzy_ahp_stats": TABLES_DIR / "08_fuzzy_ahp_stats.json",

    "fuzzy_ahp_overview": FIGURES_DIR / "08_fuzzy_ahp_overview.png",
})


# Task 9 — baseline model outputs
OUTPUT_FILES.update({
    "rf_model": MODELS_DIR / "task9_random_forest.joblib",
    "xgb_model": MODELS_DIR / "task9_xgboost.json",
    "baseline_metrics": TABLES_DIR / "09_baseline_validation_metrics.csv",
    "baseline_metadata": TABLES_DIR / "09_baseline_metadata.json",
})

# Task 13: road graph
ROAD_HIGHWAY_TYPES = (
    "motorway", "trunk", "primary", "secondary", "tertiary", "unclassified",
    "residential", "service", "living_street",
    "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link",
)
ROAD_SAMPLE_STEP_M = 30     # hazard is sampled along each road every 30 m
ROAD_NODING_TOL_M = 5       # bridge/tunnel ends snap to a road node within 5 m
SNAP_WARN_M = 500           # warn when a camp/origin is farther than this from a road node

OUTPUT_FILES.update({
    "road_graph": PROCESSED_DIR / "road_graph.pkl",
    "road_edges": PROCESSED_DIR / "road_edges.gpkg",
    "road_nodes": PROCESSED_DIR / "road_nodes.gpkg",
    "camps_nodes": PROCESSED_DIR / "camps_nodes.csv",
    "origins_nodes": PROCESSED_DIR / "origins_nodes.csv",
    "graph_stats": TABLES_DIR / "07_graph_stats.json",
    "road_flood_by_class": TABLES_DIR / "07_road_flood_by_class.csv",
    "graph_overview": FIGURES_DIR / "07_graph_overview.png",
})

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
