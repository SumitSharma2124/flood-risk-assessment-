# AI-Based Flood Risk Assessment and Risk-Aware Evacuation Route Planning

**A district-level study of Alappuzha, Kerala, India**

**Scenario:** 100-year return-period reference layer  
**Observed-event validation:** 2018 Sentinel-1 flood extent  
**Spatial resolution:** 90 m  
**Coordinate reference system:** EPSG:32643

---

## Abstract

This project develops an integrated, district-scale flood-risk assessment and risk-aware evacuation-route planning prototype for Alappuzha, Kerala. The workflow combines geospatial data preparation, Fuzzy Analytic Hierarchy Process (Fuzzy-AHP) susceptibility mapping, supervised hazard modelling using Random Forest, XGBoost and U-Net, exposure and vulnerability assessment, comparison with the 2018 Sentinel-1 flood extent, road-network routing, and a three-agent decision-support prototype.

The analysis uses a common 90 m grid in UTM zone 43N (EPSG:32643). Fuzzy-AHP covers 176,745 valid district cells. On the held-out test split of 10,767 common cells, U-Net achieved MAE 86.33, RMSE 112.05 and R² 0.188. Against the 2018 flood extent on the common KSDMA modelled domain, XGBoost achieved the strongest hazard ROC-AUC (0.896) and average precision (0.795), followed by U-Net (0.875 and 0.769). Integrated-risk surfaces did not show positive spatial discrimination for the 2018 flood mask.

Risk-aware routing for 88 village origins reduced mean route hazard from 0.2399 to 0.2290 while increasing mean total journey length from 2,659.63 m to 2,672.99 m. A controller coordinated situation-awareness, route-optimisation and communication functions to generate village-level advisories.

The system is an academic, scenario-based prototype, not a live or safety-certified emergency service.

## 1. Introduction

Flood risk is shaped by physical hazard, the distribution of people and assets, social vulnerability and the accessibility of evacuation routes. This project connects these dimensions in a single GIS- and AI-based workflow for Alappuzha district.

### 1.1 Aim

To develop an AI- and GIS-based prototype for flood hazard and integrated-risk assessment and compare shortest-path routes with hazard-aware evacuation routes.

### 1.2 Objectives

1. Prepare and align terrain, rainfall, land-cover, hydrological, population, building, facility, census, road and observed-flood datasets.
2. Generate Fuzzy-AHP susceptibility using five terrain and land-cover criteria.
3. Train and compare Random Forest, XGBoost and U-Net against the 100-year KSDMA reference using geographic blocks.
4. Estimate exposure and village-level vulnerability and combine them into relative risk surfaces.
5. Evaluate hazard and risk surfaces against the 2018 Sentinel-1 flood extent.
6. Build shortest-path and risk-aware A* routes and generate village advisories through three agent functions.

### 1.3 Scope and assumptions

- Geographic scope: Alappuzha district only.
- Scenario: 100-year reference layer.
- Common grid: 90 m; CRS: EPSG:32643.
- KSDMA is a modelling reference target; its pixel unit is not independently confirmed.
- The 2018 Sentinel-1 extent is an observed-event comparison, not an equivalent 100-year event.
- The prototype uses prepared/static data and has no live weather, traffic, road-closure or emergency feeds.

## 2. Data and Spatial Framework

The workflow integrates Copernicus DEM GLO-90, CHIRPS rainfall, ESA WorldCover 2021, OpenStreetMap, WorldPop 2018, KSDMA reference records, census indicators and a Sentinel-1-derived 2018 flood extent.

| Data group | Role | Qualification |
|---|---|---|
| DEM and terrain | Elevation, slope and terrain derivatives | Analysis uses a 90 m grid |
| CHIRPS | Historical rainfall context | Not a live forecast |
| ESA WorldCover 2021 | Land-cover fractions and susceptibility | Class scores are expert assumptions |
| KSDMA 100-year layer | Supervised target/reference hazard | Pixel unit unconfirmed; empty cells are not zero |
| WorldPop 2018 | Population exposure | Source year differs from some other layers |
| OSM and facilities | Buildings, facilities and road graph | Completeness and currency are not guaranteed |
| Census 2011 | Village vulnerability indicators | Older census geography and missing indicators |
| Sentinel-1 2018 | Observed-event validation | Paddy-field inundation may affect interpretation |

## 3. Methodology

### 3.1 Data preparation

Inputs were clipped to Alappuzha, reprojected to EPSG:32643 and aligned to a common 90 m grid. NoData was kept distinct from low or zero hazard. Terrain derivatives included slope, topographic wetness index (TWI) and distance to waterways. Camp locations, village boundaries, census indicators and road-network data were prepared for analysis.

### 3.2 Fuzzy-AHP susceptibility

Buckley Fuzzy-AHP weighted overlay used elevation, slope, TWI, distance to waterways and land-cover susceptibility.

| Criterion | Weight |
|---|---:|
| Elevation | 0.2727 |
| Slope | 0.1172 |
| TWI | 0.3205 |
| Distance to waterways | 0.1859 |
| Land-cover susceptibility | 0.1038 |

Weights sum to 1.0000 and the consistency ratio is 0.0464, below the configured 0.10 threshold. Pairwise judgments and land-cover scores remain expert assumptions requiring review and sensitivity analysis.

Mamdani AND-rule activation diagnostics explain combinations of conditions and were not added a second time to the weighted overlay.

### 3.3 Supervised hazard modelling

Random Forest and XGBoost were used as tabular baselines; U-Net was trained on spatial patches. Features included elevation, slope, TWI, waterway distance and land-cover indicators.

Geographic block separation was used to reduce neighbouring-cell leakage. The KSDMA 100-year raster was the supervised target. Cells without target values were excluded rather than labelled zero.

Model fit therefore measures agreement with KSDMA, not independent real-world flood-depth accuracy.

### 3.4 Exposure, vulnerability and integrated risk

Exposure combined:
- Population: 0.50
- Buildings: 0.30
- Critical facilities: 0.20

Building and facility counts were robust-scaled using the 99th percentile.

Village vulnerability combined child share, SC/ST share, worker fragility and housing deprivation, each weighted 0.25.

Integrated risk combined normalised hazard, exposure and vulnerability. These are relative indices, not flood probabilities or expected monetary losses.

### 3.5 2018 validation

The Sentinel-1 flood extent was reprojected and aligned to the 90 m grid. Comparisons used the common KSDMA modelled domain of 71,799 cells.

The primary observed-flood fraction threshold was 0.5, with sensitivity thresholds 0.1, 0.25 and 0.75. ROC-AUC, average precision and top-prevalence capture were reported.

Top-prevalence capture selects the same number of highest-scoring cells as the observed flooded-cell count; it is not a calibrated probability threshold.

### 3.6 Routing and multi-agent prototype

A road graph was built and hazard values sampled along road segments. Risk-aware A* edge cost was:

**Cost = length × (1 + λ × hazard)**

This was compared with plain shortest path. The selected comparison used λ = 1.0. The 2018 flood mask acts as a hazard proxy, not a confirmed road-closure layer.

Three functions—situation awareness, route optimisation and communication—are coordinated by a controller to produce village-level advisories.

## 4. Results

### 4.1 Fuzzy-AHP susceptibility

The analysis covered 176,745 valid cells. TWI had the largest weight (0.3205), followed by elevation (0.2727), distance to waterways (0.1859), slope (0.1172) and land-cover susceptibility (0.1038).

| Class | Cells | Area (km²) | Share |
|---|---:|---:|---:|
| Very low | 13,669 | 110.719 | 7.73% |
| Low | 45,203 | 366.144 | 25.58% |
| Moderate | 49,667 | 402.303 | 28.10% |
| High | 41,653 | 337.389 | 23.57% |
| Very high | 26,553 | 215.079 | 15.02% |

High and very-high classes account for 38.59% of valid cells. This is relative susceptibility, not flood probability.

### 4.2 Model performance

Validation metrics:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random Forest | 100.305 | 132.989 | -0.0006 |
| XGBoost | 98.339 | 129.854 | 0.0460 |

XGBoost was modestly better on validation, but both fits were weak.

Held-out test metrics:

| Model | Test MAE ↓ | Test RMSE ↓ | Test R² ↑ |
|---|---:|---:|---:|
| Random Forest | 89.689 | 114.860 | 0.1467 |
| XGBoost | 90.859 | 114.142 | 0.1573 |
| U-Net | **86.333** | **112.046** | **0.1880** |

All three models were evaluated on the same 10,767 held-out cells. The test split was not used for training or tuning.

U-Net performed best on all three test metrics, although R² = 0.188 indicates modest explanatory performance. Full-district predictions outside the KSDMA modelled domain are extrapolations.

XGBoost mean absolute SHAP contributions ranked elevation (31.66), trees (10.41), distance to waterways (8.55), slope (5.08) and cropland (4.92). These are model attributions, not causal effects. RF SHAP did not complete; its impurity importance is not directly comparable to XGBoost SHAP.

### 4.3 Exposure and vulnerability

The exposure surface covered 176,745 cells (approximately 1,431.63 km²), with rasterised population 2,703,786 and 791,918 buildings.

Within the KSDMA modelled domain, population was 617,010 and buildings 165,537. The district-wide 2018 flood-mask preparation summary recorded 29,929 cells, population 67,807 and 3,292 buildings intersecting the mask.

The vulnerability layer contained 88 village/polygon units. Population-share matching was 97.35%; two census areas remained unmatched and 207 town wards were not scored separately. Disability and elderly indicators were not included.

Some missing vulnerability values were filled using the district median, although configuration referred to a subdistrict median. Exposure reconstruction matched its source reconstruction exactly (MAE 0, maximum difference 0, Pearson r = 1); this verifies computation, not source completeness.

### 4.4 Hazard validation against 2018 flood extent

The common validation domain contained 71,799 cells, of which 28,400 were flooded at the primary 0.5 threshold.

| Hazard surface | ROC-AUC | Average precision | Top-prevalence capture |
|---|---:|---:|---:|
| KSDMA | 0.7391 | 0.5588 | 59.77% |
| XGBoost | **0.8962** | **0.7948** | **77.91%** |
| U-Net | 0.8751 | 0.7690 | 74.39% |

Across flood-fraction thresholds 0.1 / 0.25 / 0.5 / 0.75, capture was:

- XGBoost: 84.57% / 82.70% / 77.91% / 70.87%
- U-Net: 83.83% / 80.44% / 74.39% / 66.86%
- KSDMA: 70.86% / 66.71% / 59.77% / 52.08%

This indicates stronger spatial ranking agreement with the 2018 event, not general forecasting ability.

### 4.5 Integrated-risk validation

| Risk surface | ROC-AUC | Average precision | Top-prevalence capture |
|---|---:|---:|---:|
| KSDMA integrated risk | 0.1968 | 0.2771 | 13.33% |
| XGBoost integrated risk | 0.1506 | 0.2686 | 8.12% |
| U-Net integrated risk | 0.1615 | 0.2703 | 9.64% |

Integrated-risk surfaces did not discriminate 2018 flooded cells positively in this comparison.

This differs from hazard-only validation because exposure describes the distribution of people and assets, not flood likelihood. The result should be reported transparently; exposure weights should not be tuned solely to maximise flood-mask AUC.

### 4.6 Risk-aware routing

| Measure | Shortest path | Risk-aware A* |
|---|---:|---:|
| Mean route hazard | 0.2399 | 0.2290 |
| Mean high-hazard share | 0.2422 | 0.2310 |
| Mean total journey length | 2,659.63 m | 2,672.99 m |
| Routes changed | — | 8 of 88 |

All 88 origins were routed and none skipped. Mean route hazard decreased by about 4.5%; mean total journey length increased by 13.36 m (about 0.50%).

Nine origins were poorly connected. Seven routes had zero road-network length but non-zero total journey length because access gaps were represented separately. These cases need manual scrutiny.

### 4.7 Multi-agent outputs

The controller generated 88 village advisories and 88 route recommendations. Both files contained 88 unique villages, with complete village matching.

Advisory priorities were checked against the communication-agent logic. The priority ranking is heuristic, not a validated emergency-priority score.

## 5. Discussion

The project integrates susceptibility mapping, supervised hazard modelling, exposure and vulnerability, observed-event comparison, routing and multi-agent reporting.

Two evaluations must be kept distinct:
- Held-out test metrics measure agreement with the KSDMA target.
- Sentinel-1 comparison measures spatial ranking agreement with a separate 2018 event.

U-Net leads the former test comparison; XGBoost leads the latter hazard-only comparison.

The weak integrated-risk validation is an important finding. A risk index combines hazard with the spatial distribution of exposed assets and vulnerability, so it is not expected to rank flood occurrence identically to a hazard-only surface.

Differences in source dates, coverage and rasterisation may also affect the relationship. The audit identifies zero-valued raster cells but cannot establish source completeness.

Routing shows a modest hazard reduction for a small distance increase, but route hazard is heuristic, the 2018 mask is a proxy rather than a verified closure layer, and some origin access links are low confidence.

The result is a comparative planning experiment, not operational evacuation guidance.

## 6. Limitations

1. KSDMA target units are unconfirmed; supervised models reproduce that reference rather than independent flood-depth observations.
2. Full-district predictions outside the KSDMA modelled domain are extrapolations.
3. The 2018 event is not equivalent to a 100-year event; paddy inundation can affect the flood mask.
4. Fuzzy-AHP judgments and land-cover scores are expert assumptions requiring sensitivity analysis.
5. Integrated risk is a relative index and performed poorly against the 2018 flood mask.
6. Population, buildings, facilities, land cover and census data differ in date, resolution and completeness.
7. Two census areas were unmatched; 207 town wards were not scored separately; elderly and disability indicators were absent.
8. District-median filling was used for some missing vulnerability values rather than the configured subdistrict median.
9. Nine origins were poorly connected; seven routes had zero road-network length with non-zero access distance.
10. No verified live road closures, traffic, bridge status or passability data were available.
11. Agent priority scores are heuristic and not validated with emergency-management stakeholders.
12. The project covers one district and one scenario and has no live operational validation.

## 7. Conclusion

The project produced a district-scale flood assessment and evacuation-route planning prototype for Alappuzha, integrating Fuzzy-AHP, Random Forest/XGBoost/U-Net modelling, exposure and vulnerability, 2018 flood comparison, A* routing and a three-agent advisory workflow.

Fuzzy-AHP covered 176,745 cells and passed its consistency screen. U-Net achieved the best held-out test metrics against KSDMA, while XGBoost achieved the strongest hazard-only spatial validation against the 2018 flood extent. Risk-aware routing reduced average route hazard modestly with a small increase in journey length.

Integrated-risk validation was poor and must remain explicit in the interpretation. The system is an academic scenario-based decision-support demonstration, not a calibrated flood-probability model, live evacuation platform or substitute for official warnings and emergency-management judgement.

## 8. Future Work

1. Verify KSDMA target units and metadata with the data provider.
2. Review Fuzzy-AHP pairwise judgments and land-cover scores with domain experts; conduct sensitivity analysis.
3. Improve observed-flood reference interpretation and distinguish paddy inundation from damaging floodwater.
4. Audit exposure-source dates, coverage and rasterisation; flag uncertain zero-valued cells.
5. Improve census matching and add reliable elderly/disability indicators.
6. Investigate the negative exposure/flood-mask relationship without tuning weights solely to maximise AUC.
7. Verify road connectivity and access gaps; add authoritative closure and bridge information.
8. Validate advisory priorities with emergency-management stakeholders before operational use.
9. Test transferability to other districts and return-period scenarios.

## 9. Reproducibility and Deliverables

The collector `src/15_collect_results.py` reported:
- 47 CSV files
- 46 JSON files
- 3,719 collected metrics
- Zero read errors
- Zero missing expected files

The figure generator `src/15_make_figures.py` generated the following figures from existing outputs; no model retraining was performed.

| Figure file | Content |
|---|---|
| 15_01_fuzzy_ahp_weights.png | Fuzzy-AHP criterion weights |
| 15_02_baseline_validation.png | Random Forest vs XGBoost validation |
| 15_03_test_model_comparison.png | RF vs XGBoost vs U-Net held-out test metrics |
| 15_04_hazard_validation.png | 2018 hazard validation |
| 15_05_integrated_risk_validation.png | Integrated-risk validation |
| 15_06_routing_comparison.png | Shortest path vs risk-aware A* |
| 15_07_agent_advisory_priorities.png | Village advisory priorities |

The figure files are stored separately in `outputs/figures/` and should be included alongside this report in the final submission package.

---

## Appendix A. Key Results at a Glance

| Component | Result |
|---|---|
| Fuzzy-AHP | 176,745 valid cells; CR 0.0464; high + very high 38.59% |
| Held-out test | 10,767 common cells; U-Net MAE 86.33, RMSE 112.05, R² 0.188 |
| 2018 hazard validation | XGBoost AUC 0.896; AP 0.795; capture 77.91% |
| 2018 risk validation | AUC below 0.5 for all three integrated-risk surfaces |
| Routing | 88/88 origins routed; mean hazard 0.2399 to 0.2290; 9 poorly connected |
| Multi-agent | 88 village advisories and 88 matching route recommendations |
| Collection audit | 47 CSV + 46 JSON; 3,719 metrics; zero errors; zero missing expected files |

## Appendix B. Interpretation Safeguards

- Hazard validation and integrated-risk validation answer different questions.
- U-Net test metrics describe agreement with KSDMA, not independent real-world flood accuracy.
- Top-prevalence capture is a ranking comparison, not a calibrated probability threshold.
- Risk scores are relative indices, not probabilities or expected losses.
- A risk-aware route is not guaranteed safe or passable; access gaps and missing live closure data remain.
- This prototype must not be used as an operational evacuation authority.