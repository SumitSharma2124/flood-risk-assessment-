
from pathlib import Path
import sys
import json

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio
import folium
import matplotlib.pyplot as plt

from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from streamlit_folium import st_folium
from folium import LayerControl

# =========================================================
# 1. PROJECT PATHS
# =========================================================

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

sys.path.insert(0, str(SRC))

try:
    import config as cfg

    PROCESSED = cfg.PROCESSED_DIR
    INTERIM = cfg.INTERIM_DIR
    OUTPUTS = cfg.OUTPUTS_DIR
    FIGURES = cfg.FIGURES_DIR
    TABLES = cfg.TABLES_DIR
    MAPS = cfg.MAPS_DIR
    STATS = OUTPUTS / "stats"

except Exception:
    PROCESSED = ROOT / "data/processed"
    INTERIM = ROOT / "data/interim"
    OUTPUTS = ROOT / "outputs"
    FIGURES = OUTPUTS / "figures"
    TABLES = OUTPUTS / "tables"
    MAPS = OUTPUTS / "maps"
    STATS = OUTPUTS / "stats"

RASTERS = OUTPUTS / "rasters"


# =========================================================
# 2. STREAMLIT CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Alappuzha Flood Risk Dashboard",
    page_icon="🌊",
    layout="wide"
)

st.title("Alappuzha Flood Risk Assessment")

st.caption(
    "AI-based flood susceptibility, hazard modelling, "
    "integrated risk and evacuation decision support"
)

st.warning(
    "Research prototype: This dashboard presents a static "
    "100-year scenario analysis. It does not use live rainfall, "
    "river levels, verified road closures or real-time emergency "
    "information. It is not a safety-certified evacuation system."
)


# =========================================================
# 3. DATA LOADING HELPERS
# =========================================================

def existing(*paths):
    for path in paths:
        if path and Path(path).exists():
            return Path(path)
    return None


@st.cache_data
def load_csv(filename):

    path = existing(
        TABLES / filename,
        STATS / filename
    )

    if path is None:
        return None

    try:
        return pd.read_csv(path)
    except Exception:
        return None


@st.cache_data
def load_json(filename):

    path = existing(
        TABLES / filename,
        STATS / filename
    )

    if path is None:
        return None

    try:
        return json.loads(
            path.read_text(encoding="utf-8")
        )
    except Exception:
        return None


def show_table(data):

    if data is None or data.empty:
        st.info("Output table not found.")
    else:
        st.dataframe(
            data,
            use_container_width=True,
            hide_index=True
        )


def show_figure(filename, caption=None):

    path = existing(FIGURES / filename)

    if path:
        st.image(
            str(path),
            caption=caption,
            use_container_width=True
        )
    else:
        st.info(f"{filename} not found.")


# =========================================================
# 4. RASTER DISPLAY
# =========================================================

@st.cache_data
def load_raster_for_map(path):

    with rasterio.open(path) as src:

        scale = min(
            1.0,
            700 / max(src.width, src.height)
        )

        height = max(1, int(src.height * scale))
        width = max(1, int(src.width * scale))

        data = src.read(
            1,
            out_shape=(height, width),
            masked=True,
            resampling=Resampling.nearest
        )

        bounds = transform_bounds(
            src.crs,
            "EPSG:4326",
            *src.bounds,
            densify_pts=15
        )

    values = np.asarray(
        data.filled(np.nan),
        dtype=np.float32
    )

    valid = np.isfinite(values)

    if not valid.any():
        return None, bounds

    low, high = np.nanpercentile(
        values[valid],
        [2, 98]
    )

    if high <= low:
        high = low + 0.000001

    normalized = np.clip(
        (values - low) / (high - low),
        0,
        1
    )

    rgba = (
        plt.get_cmap("turbo")(
            np.nan_to_num(normalized)
        ) * 255
    ).astype(np.uint8)

    rgba[:, :, 3] = np.where(
        valid,
        205,
        0
    ).astype(np.uint8)

    return rgba, bounds


# =========================================================
# 5. INTERACTIVE MAP
# =========================================================

def get_map_center():

    boundary = existing(
        INTERIM / "district.gpkg"
    )

    if boundary:

        try:
            gdf = gpd.read_file(boundary)
            gdf = gdf.to_crs("EPSG:4326")

            center = gdf.geometry.union_all().centroid

            return [center.y, center.x]

        except Exception:
            pass

    return [9.50, 76.35]


def create_map(layers):

    m = folium.Map(
        location=get_map_center(),
        zoom_start=9,
        tiles="CartoDB positron",
        control_scale=True
    )

    for name, path in layers.items():

        path = Path(path)

        if not path.exists():
            continue

        if path.suffix.lower() in [".tif", ".tiff"]:

            try:

                image, bounds = load_raster_for_map(
                    str(path)
                )

                if image is not None:

                    folium.raster_layers.ImageOverlay(
                        image=image,
                        bounds=[
                            [bounds[1], bounds[0]],
                            [bounds[3], bounds[2]]
                        ],
                        name=name,
                        opacity=0.72,
                        interactive=True
                    ).add_to(m)

            except Exception as error:
                st.warning(
                    f"Could not display {name}: {error}"
                )

        elif path.suffix.lower() == ".gpkg":

            try:

                gdf = gpd.read_file(path)

                if gdf.empty or gdf.crs is None:
                    continue

                if len(gdf) > 6000:
                    gdf = gdf.sample(
                        6000,
                        random_state=42
                    )

                gdf = gdf.to_crs("EPSG:4326")

                color = (
                    "#c0392b"
                    if "route" in name.lower()
                    else "#222222"
                )

                folium.GeoJson(
                    json.loads(gdf.to_json()),
                    name=name,
                    style_function=lambda feature, c=color: {
                        "color": c,
                        "weight": 2,
                        "fillOpacity": 0.04
                    }
                ).add_to(m)

            except Exception as error:
                st.warning(
                    f"Could not display {name}: {error}"
                )

    LayerControl(collapsed=False).add_to(m)

    return m


def display_map(layers, height=570):

    available = {
        name: path
        for name, path in layers.items()
        if Path(path).exists()
    }

    if not available:
        st.warning("No requested map layers were found.")
        return

    map_object = create_map(available)

    st_folium(
        map_object,
        use_container_width=True,
        height=height,
        returned_objects=[]
    )


# =========================================================
# 6. SIDEBAR NAVIGATION
# =========================================================

pages = [
    "Overview",
    "Interactive Map",
    "Fuzzy-AHP",
    "Model Comparison",
    "Exposure and Vulnerability",
    "Flood Validation and Risk",
    "Evacuation Routing",
    "Multi-Agent Advisories",
    "Output Audit"
]

with st.sidebar:

    st.title("Navigation")

    page = st.radio(
        "Dashboard Sections",
        pages
    )

    st.divider()

    st.markdown("### Study Settings")
    st.write("District: Alappuzha")
    st.write("Scenario: 100-year")
    st.write("Resolution: 90 metres")
    st.write("CRS: EPSG:32643")


# =========================================================
# 7. LOAD TABULAR OUTPUTS
# =========================================================

ahp_weights = load_csv(
    "08_fuzzy_ahp_weights.csv"
)

baseline_metrics = load_csv(
    "09_baseline_validation_metrics.csv"
)

hazard_validation = load_csv(
    "13_hazard_validation_metrics.csv"
)

risk_validation = load_csv(
    "13_risk_validation_metrics.csv"
)

routing_results = load_csv(
    "08_routing_results.csv"
)

advisories = load_csv(
    "14_agent_village_advisories.csv"
)

recommendations = load_csv(
    "14_agent_route_recommendations.csv"
)

test_metrics = load_json(
    "10_all_models_test_metrics.json"
)

agent_report = load_json(
    "14_agent_situation_report.json"
)


# =========================================================
# 8. OVERVIEW
# =========================================================

if page == "Overview":

    st.header("Project Overview")

    a, b, c, d = st.columns(4)

    a.metric("District", "Alappuzha")
    b.metric("Valid Grid Cells", "176,745")
    c.metric("Grid Resolution", "90 m")
    d.metric("Village Origins", "88")

    st.subheader("Methodology")

    col1, col2, col3 = st.columns(3)

    col1.markdown("""
    **1. Susceptibility Mapping**

    - Fuzzy-AHP
    - Terrain and drainage indicators
    - Land-cover susceptibility
    """)

    col2.markdown("""
    **2. Hazard and Risk**

    - Random Forest
    - XGBoost
    - U-Net
    - Exposure and vulnerability
    """)

    col3.markdown("""
    **3. Evacuation Support**

    - A* route planning
    - Risk-aware routing
    - Three-agent architecture
    """)

    st.subheader("Project Results")

    left, right = st.columns(2)

    with left:

        show_figure(
            "15_03_test_model_comparison.png",
            "Test-set model comparison"
        )

        show_figure(
            "15_04_hazard_validation.png",
            "Hazard validation"
        )

    with right:

        show_figure(
            "15_06_routing_comparison.png",
            "Routing comparison"
        )

        show_figure(
            "15_07_agent_advisory_priorities.png",
            "Agent advisory priorities"
        )

    st.info(
        "U-Net performed best on the reported held-out test "
        "regression metrics. XGBoost performed best in the "
        "separate 2018 flood-extent hazard validation."
    )


# =========================================================
# 9. INTERACTIVE MAP
# =========================================================

elif page == "Interactive Map":

    st.header("Interactive Spatial Analysis")

    layers = {

        "Fuzzy-AHP Susceptibility":
            PROCESSED / "08_fuzzy_ahp_susceptibility.tif",

        "Fuzzy-AHP Classes":
            PROCESSED / "08_fuzzy_ahp_classes.tif",

        "XGBoost Hazard":
            RASTERS / "09_xgb_full_district_prediction.tif",

        "U-Net Hazard":
            RASTERS / "10_unet_full_district_prediction.tif",

        "XGBoost Integrated Risk":
            RASTERS / "11_xgb_integrated_risk.tif",

        "U-Net Integrated Risk":
            RASTERS / "11_unet_integrated_risk.tif",

        "Exposure":
            PROCESSED / "exposure_score.tif",

        "Vulnerability":
            PROCESSED / "vulnerability_score.tif",

        "2018 Flood Extent":
            PROCESSED / "flood2018_bin.tif",

        "KSDMA 100-year Reference":
            PROCESSED / "ksdma_100yr_90m.tif",

        "District Boundary":
            INTERIM / "district.gpkg",

        "Village Boundaries":
            PROCESSED / "villages_vuln.gpkg",

        "Evacuation Routes":
            MAPS / "08_routing_routes.gpkg"
    }

    available = {
        name: path
        for name, path in layers.items()
        if path.exists()
    }

    selected = st.multiselect(
        "Select map layers",
        list(available.keys()),
        default=[
            name for name in [
                "Fuzzy-AHP Susceptibility",
                "District Boundary"
            ]
            if name in available
        ]
    )

    display_map({
        name: available[name]
        for name in selected
    })

    st.caption(
        "Raster layers are downsampled for browser display. "
        "Use original GeoTIFFs for quantitative analysis."
    )


# =========================================================
# 10. FUZZY-AHP
# =========================================================

elif page == "Fuzzy-AHP":

    st.header("Fuzzy-AHP Susceptibility")

    left, right = st.columns(2)

    with left:

        show_figure(
            "15_01_fuzzy_ahp_weights.png",
            "Fuzzy-AHP criterion weights"
        )

    with right:

        st.metric(
            "Consistency Ratio",
            "0.0464"
        )

        st.caption(
            "The reported consistency ratio is below 0.10."
        )

        st.markdown("### Susceptibility Distribution")

        st.dataframe(
            pd.DataFrame({
                "Class": [
                    "Very Low",
                    "Low",
                    "Moderate",
                    "High",
                    "Very High"
                ],
                "District Share (%)": [
                    7.73,
                    25.58,
                    28.10,
                    23.57,
                    15.02
                ]
            }),
            hide_index=True,
            use_container_width=True
        )

    st.subheader("Criterion Weights")

    show_table(ahp_weights)

    st.subheader("Spatial Distribution")

    display_map({
        "Susceptibility":
            PROCESSED / "08_fuzzy_ahp_susceptibility.tif",

        "Susceptibility Classes":
            PROCESSED / "08_fuzzy_ahp_classes.tif",

        "District Boundary":
            INTERIM / "district.gpkg"
    })


# =========================================================
# 11. MODEL COMPARISON
# =========================================================

elif page == "Model Comparison":

    st.header("Model Performance")

    show_figure(
        "15_03_test_model_comparison.png",
        "Held-out test-set comparison"
    )

    if test_metrics is not None:
        st.json(test_metrics, expanded=False)
    else:
        st.info(
            "10_all_models_test_metrics.json not found."
        )

    st.subheader("Baseline Validation")

    show_table(baseline_metrics)

    st.caption(
        "The held-out test set is for final reporting, "
        "not model tuning. Validation and test metrics "
        "are separate evaluations."
    )


# =========================================================
# 12. EXPOSURE AND VULNERABILITY
# =========================================================

elif page == "Exposure and Vulnerability":

    st.header("Exposure and Social Vulnerability")

    a, b, c = st.columns(3)

    a.metric("Population (2018)", "2,703,786")
    b.metric("Buildings", "791,918")
    c.metric("Village Polygons", "88")

    display_map({

        "Exposure":
            PROCESSED / "exposure_score.tif",

        "Population":
            PROCESSED / "exposure_pop.tif",

        "Vulnerability":
            PROCESSED / "vulnerability_score.tif",

        "District Boundary":
            INTERIM / "district.gpkg",

        "Villages":
            PROCESSED / "villages_vuln.gpkg"
    })

    left, right = st.columns(2)

    with left:
        show_figure("04_exposure_overview.png")

    with right:
        show_figure("05_vulnerability_overview.png")

    st.subheader("Exposure Summary")

    show_table(
        load_csv("04_exposure_summary.csv")
    )

    st.subheader("Vulnerability Summary")

    show_table(
        load_csv("05_vulnerability_summary.csv")
    )

    st.caption(
        "Exposure represents people and assets, not flood "
        "probability. Vulnerability is a relative score "
        "based on Census 2011 indicators."
    )


# =========================================================
# 13. FLOOD VALIDATION AND INTEGRATED RISK
# =========================================================

elif page == "Flood Validation and Risk":

    st.header("2018 Flood Validation")

    left, right = st.columns(2)

    with left:
        show_figure("15_04_hazard_validation.png")

    with right:
        show_figure("15_05_integrated_risk_validation.png")

    st.subheader("Hazard Validation Metrics")

    show_table(hazard_validation)

    st.subheader("Integrated Risk Validation Metrics")

    show_table(risk_validation)

    st.subheader("Integrated Risk Maps")

    display_map({

        "XGBoost Risk":
            RASTERS / "11_xgb_integrated_risk.tif",

        "U-Net Risk":
            RASTERS / "11_unet_integrated_risk.tif",

        "KSDMA Reference Risk":
            RASTERS / "11_ksdma_reference_risk.tif",

        "District Boundary":
            INTERIM / "district.gpkg"
    })

    st.caption(
        "Integrated risk is a relative index, not flood "
        "probability or expected loss. Hazard validation "
        "and integrated-risk validation measure different things."
    )


# =========================================================
# 14. EVACUATION ROUTING
# =========================================================

elif page == "Evacuation Routing":

    st.header("Risk-Aware Evacuation Routing")

    a, b, c = st.columns(3)

    a.metric("Village Origins Routed", "88")

    b.metric(
        "Shortest Route Mean Hazard",
        "0.2399"
    )

    c.metric(
        "Risk-Aware Mean Hazard",
        "0.2290"
    )

    show_figure(
        "15_06_routing_comparison.png",
        "Shortest vs risk-aware routing"
    )

    st.subheader("Route Map")

    display_map({

        "Evacuation Routes":
            MAPS / "08_routing_routes.gpkg",

        "District Boundary":
            INTERIM / "district.gpkg",

        "Villages":
            PROCESSED / "villages_vuln.gpkg"
    })

    st.subheader("Routing Results")

    show_table(routing_results)

    st.warning(
        "Routing is scenario-based, not live navigation. "
        "Nine origins were flagged as poorly connected. "
        "Seven routes had zero road-network length but "
        "non-zero total journey length because access costs "
        "were included."
    )


# =========================================================
# 15. MULTI-AGENT ADVISORIES
# =========================================================

elif page == "Multi-Agent Advisories":

    st.header("Multi-Agent Decision Support")

    if agent_report is not None:
        st.json(agent_report, expanded=False)

    left, right = st.columns(2)

    with left:

        st.subheader("Village Advisories")

        show_table(advisories)

    with right:

        st.subheader("Route Recommendations")

        show_table(recommendations)

    st.caption(
        "These advisories are generated from static scenario "
        "outputs. They are not live public warnings or "
        "operational evacuation instructions."
    )


# =========================================================
# 16. OUTPUT AUDIT
# =========================================================

elif page == "Output Audit":

    st.header("Project Output Audit")

    expected = {

        "Fuzzy-AHP Susceptibility":
            PROCESSED / "08_fuzzy_ahp_susceptibility.tif",

        "Fuzzy-AHP Classes":
            PROCESSED / "08_fuzzy_ahp_classes.tif",

        "Exposure":
            PROCESSED / "exposure_score.tif",

        "Vulnerability":
            PROCESSED / "vulnerability_score.tif",

        "XGBoost Prediction":
            RASTERS / "09_xgb_full_district_prediction.tif",

        "U-Net Prediction":
            RASTERS / "10_unet_full_district_prediction.tif",

        "XGBoost Integrated Risk":
            RASTERS / "11_xgb_integrated_risk.tif",

        "U-Net Integrated Risk":
            RASTERS / "11_unet_integrated_risk.tif",

        "Routing GeoPackage":
            MAPS / "08_routing_routes.gpkg",

        "Agent Advisories":
            TABLES / "14_agent_village_advisories.csv",

        "Agent Recommendations":
            TABLES / "14_agent_route_recommendations.csv",

        "Consolidated Metrics":
            TABLES / "15_consolidated_metrics.csv",

        "Collection Audit":
            STATS / "15_collection_audit.json"
    }

    audit = pd.DataFrame([

        {
            "Output": name,
            "Status": (
                "Available"
                if path.exists()
                else "Missing"
            ),
            "Path": str(path.relative_to(ROOT))
        }

        for name, path in expected.items()
    ])

    st.dataframe(
        audit,
        use_container_width=True,
        hide_index=True
    )

    available_count = (
        audit["Status"] == "Available"
    ).sum()

    st.metric(
        "Available Outputs",
        f"{available_count}/{len(audit)}"
    )

    st.subheader("Consolidated Metrics")

    show_table(
        load_csv("15_consolidated_metrics.csv")
    )

    collection_audit = load_json(
        "15_collection_audit.json"
    )

    if collection_audit is not None:

        st.subheader("Collection Audit Details")

        st.json(
            collection_audit,
            expanded=False
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Alappuzha Flood Risk Assessment | Static research "
    "prototype | Existing project outputs only"
)
