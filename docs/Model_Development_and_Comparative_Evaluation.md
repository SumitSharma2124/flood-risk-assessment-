# Model Development and Comparative Evaluation

## 1. Objective

Three machine learning approaches were developed to reproduce the KSDMA 100-year flood hazard raster for Alappuzha district:

1. Random Forest (RF)
2. Extreme Gradient Boosting (XGBoost)
3. U-Net (Convolutional Neural Network)

All models use the same ten input features and the same geographically separated training, validation, and test partitions.

The target is the KSDMA modelled hazard raster. Its values are treated as a relative hazard index because the physical units have not been confirmed.

## 2. Dataset and experimental setup

- Study area: Alappuzha district, Kerala.
- Scenario: 100-year return period.
- Eligible supervised cells: 71,799.
- Training cells: 50,261.
- Validation cells: 10,771.
- Test cells: 10,767.
- Geographic partitioning: 5 km spatial blocks.
- Random seed: 42.

The ten input features were elevation, slope, TWI, distance to waterways, trees, cropland, built-up area, water, wetland, and mangrove.

The same spatial split was maintained across all three models to support a fair comparison and reduce spatial leakage.

## 3. Baseline model development

Random Forest and XGBoost were trained as tabular baseline models.

### Validation performance

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Training mean predictor | 105.90 | 137.72 | -0.0731 |
| Linear Regression | 101.73 | 134.93 | -0.0300 |
| Random Forest | 100.31 | 132.99 | -0.0006 |
| XGBoost | 98.34 | 129.85 | 0.0460 |

XGBoost performed marginally better than Random Forest on the validation set. However, both models showed limited predictive performance, indicating difficulty reproducing the spatial variation in the reference hazard raster.

### XGBoost feature interpretation

SHAP analysis was performed on 2,000 validation samples.

The most influential features by mean absolute SHAP value were:

| Feature | Mean absolute SHAP |
|---|---:|
| Elevation | 31.66 |
| Trees | 10.41 |
| Distance to waterways | 8.55 |
| Slope | 5.08 |
| Cropland | 4.92 |
| Water | 3.36 |
| TWI | 1.82 |
| Built-up area | 1.33 |
| Wetland | 1.25 |
| Mangrove | 0.03 |

Elevation was the dominant feature influencing XGBoost predictions. SHAP values describe the model's learned behaviour and should not be interpreted as proof of physical causation.

Random Forest SHAP computation was discontinued because of its high computational cost. Its built-in impurity-based feature importance was retained as supplementary analysis, not as a direct substitute for SHAP.

## 4. U-Net development

A U-Net architecture was trained using 16 × 16 spatial patches containing the ten selected input features.

The model used:
- Training-only feature normalisation.
- Training-only target normalisation.
- Masked Huber loss.
- AdamW optimisation.
- Spatial augmentation using flips and rotations.
- Early stopping based on validation RMSE.

The original model achieved its best validation performance at epoch 7.

### Original U-Net validation performance

| Metric | Result |
|---|---:|
| MAE | 98.84 |
| RMSE | 134.75 |
| R² | -0.0273 |

A weighted-loss experiment was also conducted to investigate whether increased emphasis on extreme hazard values improved performance.

The weighted model achieved MAE 99.03, RMSE 135.08, and R² -0.0324. It did not improve overall validation performance, so the original U-Net checkpoint was retained.

## 5. Final held-out test evaluation

After model selection, the original U-Net, Random Forest, and XGBoost models were evaluated on the same 10,767 held-out test cells.

The test set was not used for model training or tuning.

| Metric | Random Forest | XGBoost | U-Net |
|---|---:|---:|---:|
| MAE | 89.69 | 90.86 | **86.33** |
| RMSE | 114.86 | 114.14 | **112.05** |
| R² | 0.1467 | 0.1573 | **0.1880** |

U-Net achieved the best overall test performance across all three metrics.

Compared with Random Forest, U-Net reduced MAE by approximately 3.74% and RMSE by approximately 2.45%.

The improvement is modest, but it indicates that the spatial convolutional approach performed better overall on this particular held-out test partition.

## 6. Performance across hazard quantiles

Test errors were additionally analysed across five quantiles of actual hazard values.

| Quantile | Random Forest MAE | XGBoost MAE | U-Net MAE |
|---|---:|---:|---:|
| Q1 – Lowest | 134.55 | 161.84 | **124.38** |
| Q2 | 90.09 | 97.66 | **72.66** |
| Q3 | 68.67 | **53.64** | 60.29 |
| Q4 | 58.20 | **39.92** | 53.39 |
| Q5 – Highest | **96.66** | 100.73 | 120.92 |

The results show that model performance varies across the hazard distribution:

- U-Net performed best in Q1 and Q2.
- XGBoost performed best in Q3 and Q4.
- Random Forest performed best in Q5.

U-Net showed comparatively high error in the highest hazard quantile. Its predictions also exhibited a narrower distribution than the actual target values, indicating regression towards the mean.

Therefore, the best overall model does not necessarily provide the best predictions at every hazard level.

## 7. Key findings

1. XGBoost marginally outperformed Random Forest during validation.
2. The original U-Net achieved the best overall performance on the independent test partition.
3. U-Net's overall improvement over the baseline models was modest.
4. XGBoost performed particularly well in the moderate and high hazard quantiles.
5. Random Forest achieved the lowest error in the highest hazard quantile.
6. All three models showed limitations in reproducing the full range of hazard values.
7. Elevation was the dominant feature in the XGBoost SHAP analysis.

## 8. Limitations

- The models reproduce the KSDMA modelled hazard raster rather than directly predicting observed flood depths.
- The target raster's physical units remain unconfirmed and are treated as a relative hazard index.
- Performance is evaluated on one geographic block-based test partition.
- All models exhibit errors at the extremes of the hazard distribution.
- The test results should not be interpreted as independent real-world flood forecasting accuracy.

Independent validation against the 2018 Sentinel-1 flood extent is therefore required to assess how well the modelled hazard patterns correspond to observed flood-affected areas.

## 9. Conclusion

Among the three evaluated approaches, U-Net achieved the strongest overall held-out test performance, with MAE of 86.33, RMSE of 112.05, and R² of 0.1880.

However, XGBoost and Random Forest demonstrated advantages in specific hazard quantiles. These findings support retaining the three-model comparison rather than relying exclusively on a single overall metric.

The original U-Net checkpoint is retained as the best overall model candidate for subsequent project analysis. Its outputs will be used alongside the Fuzzy-AHP susceptibility results and other spatial information in the remaining flood-risk assessment workflow.