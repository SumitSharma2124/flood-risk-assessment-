# Task 4 Technical Audit — Terrain Features and Sensitivity Experiments

**Project:** AI-Based Flood Risk Assessment and Risk-Aware Evacuation Route Planning in Kerala, India  
**Implementation area:** Alappuzha district  
**Grid:** EPSG:32643, 90 m  
**District analysis cells:** 176,745  
**Evidence reviewed:** `sensitivity_task4.zip` (33 files; four burn-depth runs, baseline comparison, HAND fallback diagnostics, and TWI consistency outputs).

## 1. Executive conclusion

The Task 4 sensitivity experiments completed successfully. The experiment pipeline's 5 m control reproduced the existing production flow-accumulation, HAND, and TWI rasters exactly over all 176,745 Alappuzha cells. This establishes computational reproducibility between the experiment script and the production pipeline for the tested inputs and settings. It does **not** independently establish that a 5 m waterway burn is physically optimal.

The 5 m burn depth should therefore remain the project's documented baseline for now. It should be described as a modelling assumption, not as a calibrated or empirically validated parameter. The HAND Euclidean fallback affects only 124 district cells (0.0702%). The alternative TWI experiment shows a moderate distribution shift and strong, but not perfect, rank agreement.

No production data are changed by the archived experiments. All experimental rasters are stored under `outputs/sensitivity_task4/`.

## 2. Experimental setup and integrity checks

- The burn-depth experiment tested 0 m, 2 m, 5 m, and 10 m waterway burning.
- Each burn-depth run recalculated the conditioned DEM, D8 flow direction/accumulation, HAND, and TWI using the experiment's shared terrain-processing functions.
- The regional DEM rasters are 2,474 × 2,453 cells. District outputs are 982 × 533 cells.
- The district outputs in the archive use EPSG:32643, 90 m resolution, Float32 values, and -9999 NoData. The fallback mask uses UInt8 with 255 as NoData.
- All four burn-depth runs report 176,745 valid Alappuzha cells for each of flow accumulation, HAND, and TWI.
- The archived fallback mask contains 124 cells marked 1, 176,621 cells marked 0, and 346,661 outside-district NoData cells. This agrees with `hand_fallback/stats.json`.

## 3. Experiment A — Waterway burn-depth sensitivity

### 3.1 District-wide summary statistics

| Burn depth | Flow accumulation mean | Flow accumulation P90 | HAND mean (m) | HAND median (m) | HAND P90 (m) | TWI mean | TWI median | TWI P90 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 m | 1,942.37 | 46 | 3.273 | 1.085 | 8.909 | 10.349 | 9.656 | 13.524 |
| 2 m | 1,714.48 | 49 | 3.064 | 0.698 | 8.708 | 10.339 | 9.645 | 13.559 |
| 5 m | 1,449.67 | 52 | 2.942 | 0.498 | 8.526 | 10.339 | 9.649 | 13.587 |
| 10 m | 2,061.56 | 54 | 2.862 | 0.408 | 8.370 | 10.353 | 9.652 | 13.619 |

### 3.2 Comparison against the existing 5 m production baseline

| Layer | Burn | MAE | RMSE | Spearman correlation | Cells changed (%) |
|---|---:|---:|---:|---:|---:|
| Flow accumulation | 0 m | 2,618.645 | 25,755.267 | 0.619 | 43.12 |
| Flow accumulation | 2 m | 2,284.762 | 24,248.028 | 0.802 | 32.01 |
| Flow accumulation | 5 m | 0 | 0 | 1.000 | 0 |
| Flow accumulation | 10 m | 1,829.582 | 24,664.497 | 0.920 | 17.46 |
| HAND | 0 m | 0.727 m | 1.783 m | 0.863 | 41.69 |
| HAND | 2 m | 0.380 m | 1.232 m | 0.918 | 25.77 |
| HAND | 5 m | 0 | 0 | 1.000 | 0 |
| HAND | 10 m | 0.194 m | 1.010 m | 0.963 | 10.83 |
| TWI | 0 m | 0.921 | 2.006 | 0.796 | 43.12 |
| TWI | 2 m | 0.631 | 1.649 | 0.888 | 32.01 |
| TWI | 5 m | 0 | 0 | 1.000 | 0 |
| TWI | 10 m | 0.319 | 1.158 | 0.937 | 17.46 |

### 3.3 Interpretation

- Waterway burn depth changes the drainage network and derived HAND/TWI values. The flow-accumulation mean is non-monotonic across burn depths; it falls from 0 m to 5 m and rises at 10 m.
- HAND mean and median decrease as burn depth increases across the tested values.
- TWI summary means remain close, but cell-level differences remain measurable.
- The 10 m outputs have higher rank correlations with the 5 m baseline than the 0 m and 2 m outputs. This only describes similarity to the selected baseline; it does not make 10 m more physically correct.
- The 5 m control gives zero MAE/RMSE and 100% rank agreement for the three layers. This is a reproducibility/control result, not independent validation of the physical appropriateness of the 5 m parameter.

**Decision:** retain 5 m as the current baseline, document it as an assumption, and do not claim it is calibrated or optimal. Further selection would require independent hydrological evidence or a defensible calibration/sensitivity criterion.

## 4. Experiment B — HAND Euclidean fallback

| Measure | Result |
|---|---:|
| Alappuzha cells | 176,745 |
| Cells requiring fallback | 124 |
| Fallback share | 0.0702% |
| Cells using normal HAND calculation | 176,621 (99.9298%) |
| Regional fallback cells | 96,820 |

Final HAND statistics by group:

| Statistic | Fallback cells | Normal cells |
|---|---:|---:|
| Mean (m) | 0.907 | 2.943 |
| Median (m) | 0.023 | 0.500 |
| P90 (m) | 3.973 | 8.530 |
| Maximum (m) | 7.723 | 79.815 |

The fallback share inside Alappuzha is very small, despite the much larger regional fallback count. The fallback cells' lower HAND values are descriptive only; the two groups are spatially different and this comparison does not establish systematic bias.

**Decision:** retain the existing fallback for the current district-scale implementation, but document it as a limitation. The 124-cell mask is available at `hand_fallback/fallback_mask.tif`. A future improvement could inspect those cells individually or replace the Euclidean fallback with a hydrologically constrained method if the affected locations prove important.

## 5. Experiment C — TWI input consistency

The experiment compared:

- **Current method:** flow accumulation from the conditioned 90 m DEM and slope from the averaged original 30 m DEM.
- **Alternative method:** flow accumulation and slope from the conditioned 90 m DEM.

| Metric | Current method | Alternative method |
|---|---:|---:|
| Mean TWI | 10.339 | 10.030 |
| Median TWI | 9.649 | 9.345 |
| P90 TWI | 13.587 | 13.306 |
| Minimum | 5.601 | 5.602 |
| Maximum | 24.341 | 23.655 |

Cell-by-cell comparison:

- Valid cells: 176,745
- Mean absolute difference: 0.374
- RMSE: 0.786
- Spearman correlation: 0.903
- Cells with any numerical difference above 1e-6: 90.64%
- Cells differing by more than 1 TWI unit: 14.26%

The alternative method reduces the mean TWI by approximately 0.309 (about 3.0%). Rank agreement remains strong, although most cells differ numerically. The 90.64% figure uses a very small numerical threshold and should not be presented as a percentage of materially changed cells; the more interpretable >1-unit statistic is 14.26%.

**Decision:** retain the current slope source for now. It uses the averaged original 30 m DEM, and the alternative experiment does not establish that the conditioned 90 m slope is more accurate. Document the mixed terrain inputs as a methodological choice/limitation. If desired, a future comparison can validate both slope surfaces against independent elevation/slope reference data.

## 6. Limitations and interpretation cautions

1. These are sensitivity and reproducibility experiments, not ground-truth validation of hydrological correctness.
2. The 5 m control exactly reproduces production because it uses the same inputs and processing logic. It is not an independent replication.
3. The burn-depth comparison tests the configured waterway-burn mechanism; it does not test alternative channel datasets, burn geometry, stream thresholds, DEM conditioning parameters, or routing algorithms.
4. The HAND fallback group is small within Alappuzha, but its spatial importance was not assessed against flood-prone assets or validation labels.
5. The TWI alternative uses a different source surface for slope. The experiment compares input consistency, not which slope is closest to ground truth.
6. No downstream Fuzzy-AHP, RF/XGBoost, U-Net, flood-validation, or evacuation-routing outputs were recalculated with alternative terrain products. Therefore, downstream model sensitivity remains unmeasured.
7. The archive contains outputs and statistics, not the full experiment scripts or source inputs. This audit verifies the supplied outputs and their metadata; it does not perform a source-code audit of the scripts.

## 7. Final Task 4 decision log

| Component | Audit outcome | Decision |
|---|---|---|
| Regional DEM and grid alignment | Previously verified; archived experiment grids are aligned at 90 m | Retain |
| Waterway burn depth | Material sensitivity; 5 m control reproduces production | Retain 5 m as provisional baseline; document assumption |
| D8 flow accumulation | Changes with burn depth | Retain current production output; note parameter sensitivity |
| HAND fallback | 0.0702% of district cells affected | Retain; document small fallback limitation |
| Slope source for TWI | Alternative shifts TWI; Spearman 0.903 | Retain current 30 m-derived slope pending independent validation |
| TWI formula | Same production function used in both methods | Retain |

## 8. Recommended next actions

1. Preserve `outputs/sensitivity_task4/` as the experiment evidence folder.
2. Add the above assumptions and sensitivity findings to the methodology/limitations section of the project report.
3. Keep the current 5 m production pipeline unchanged for the next modelling stage unless new evidence justifies a change.
4. Proceed to the next SRS workstream: RF/XGBoost baseline reproduction of the KSDMA 100-year hazard raster, followed by the U-Net model, using the locked current feature stack and district mask.
5. When reporting results, distinguish **computational reproducibility**, **sensitivity**, and **independent validation**; they are different forms of evidence.
