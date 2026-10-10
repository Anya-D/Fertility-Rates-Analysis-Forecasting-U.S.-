# Fertility Rates Analysis & Forecasting (U.S.)

> **Description:** An exploratory and predictive analysis of U.S. birth, pregnancy, abortion, and miscarriage data from 1973 to 2020, using XGBoost with SHAP explainability to identify what drives birth totals and Prophet to forecast California and Texas trends through 2030.

## Overview

This project investigates how reproductive-health indicators and state-level differences relate to birth totals in the United States. It combines three steps:

1. **Exploration:** trends in births, fertility, abortion, pregnancy, and miscarriage over time and by state.
2. **Prediction and interpretation:** an XGBoost regressor predicts total births, and SHAP explains which features matter most.
3. **Forecasting:** Prophet projects yearly totals for California and Texas through 2030 and is benchmarked against linear regression.

## Data Sources

| Dataset | Source |
|---|---|
| NCHS Births and General Fertility Rates, United States | [data.gov](https://catalog.data.gov/dataset/nchs-births-and-general-fertility-rates-united-states) |
| State fertility rates (`data-table.csv`) | [CDC NCHS](https://www.cdc.gov/nchs/pressroom/sosmap/fertility_rate/fertility_rates.htm) |
| National and State Pregnancy Data (`NationalAndStatePregnancy_PublicUse.csv`) | [OSF](https://osf.io/duj6a) |

Columns used from the pregnancy dataset: `state`, `year`, `birthratetotal`, `abortionratetotal`, `pregnancyratetotal`, `miscarriagestotal`, `birthstotal`, `abortionstotal`.

## Methods

**Exploratory analysis**
- Point plots of abortion rate, pregnancy rate, and miscarriage totals by year (1973–2020).
- Scatter plots of births and fertility rate by state with a trend line, plus per-state summary statistics.

**Predictive model**
- Target: `birthstotal`. Categorical features (state) are one-hot encoded.
- 80/20 train/test split (`random_state=42`).
- `XGBRegressor` with 1000 estimators, max depth 7, learning rate 0.1, subsample 0.7, column sample 0.8.
- Evaluation: repeated 10-fold cross-validation (3 repeats) on MAE, plus test MAE, MSE, and R².

**Explainability (SHAP)**
- `TreeExplainer` with a summary (mean |SHAP|) bar plot and dependence plots for every feature.

**Forecasting**
- Prophet fit per state on yearly totals of births, abortions, miscarriages, and pregnancy rate (California and Texas), projected to 2030.
- Backtest: train on data up to 2015, test on later years, comparing Prophet and linear regression using RMSE and MAPE.

## Key Findings

- **Most influential features:** state, miscarriage totals, and abortion totals, suggesting state-level policy and healthcare access shape outcomes.
- **Year effect:** SHAP values for year grow over time, consistent with changing data quality, healthcare, and social conditions.
- **Miscarriage totals** show a strong near-linear relationship with their SHAP values, so they dominate related predictions.
- **California forecast:** births are projected to decline overall through 2030, with some fluctuation.

| Year | Projected CA births |
|---|---|
| 2025 | 370,648 |
| 2027 | 313,865 (lowest point) |
| 2028 | 363,429 (temporary rise) |
| 2030 | 314,951 |

Widening confidence intervals show growing uncertainty further out.

## Getting Started

The script was exported from Google Colab.

```bash
pip install pandas numpy scikit-learn seaborn matplotlib xgboost shap prophet
```

1. Download the three datasets above and place the CSVs in the working directory (the script expects `/content/`).
2. Update the file paths if you are not running in Colab.
3. Remove or replace the `!pip install` lines if running as a plain `.py` file.

## Known Issues

- **Possible feature leakage:** the model predicts `birthstotal` while `birthratetotal` and `pregnancyratetotal` are included as features. These are closely tied to births, which may inflate performance. Consider removing them or re-framing the target.
- **Undefined helper functions:** a later section calls `prepare_data`, `train_xgboost_model`, `evaluate_model`, and `plot_shap_analysis`, which are not defined in the script.
- **Undefined variable:** the first Prophet evaluation block references `merged`, which is never created. The later version (`merged_prophet`) works.
- **Repeated code:** the Prophet blocks for each state and metric are copy-pasted and could be refactored into one function.
- **Deprecated frequency:** `freq='Y'` is deprecated in recent pandas versions; use `'YE'`.
- Forecasts rely on a single series per state with no external variables, so they should be treated as exploratory.

## Future Work

- Add external data (economic, healthcare access, immigration) to forecasts.
- Tune Prophet or try RNN and hybrid models.
- Extend forecasting to more states and compare policy environments.

## Tech Stack

Python, pandas, NumPy, scikit-learn, XGBoost, SHAP, Prophet, Seaborn, Matplotlib
