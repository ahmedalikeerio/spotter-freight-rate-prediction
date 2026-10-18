# Freight Rate Prediction — Machine Learning Assessment

## Project Overview

An end-to-end machine learning solution for predicting freight transportation rates using shipment characteristics, geographic information, and historical pricing data.

Developed as part of the **Spotter Machine Learning Engineer Assessment**.

The project emphasizes reproducible experimentation, leakage-safe preprocessing, chronological validation, and interpretable model evaluation.

## 1. Problem Statement

Predict the posted freight rate (`posted_rate`) for a shipment using available shipment features.

The assessment provides:

| Dataset | Records | Purpose |
|---|---:|---|
| Training/development | 48,000 | Exploratory analysis, model development, and validation |
| Final validation | 12,000 | Unseen shipments requiring predictions |
| December chart | 31 | Daily predictions for a fixed shipment configuration |

**Target:** `posted_rate` (USD)

**Task:** Supervised regression

## 2. Solution Approach

The modeling workflow consists of:

1. Exploratory data analysis and data-quality checks
2. Missing and invalid value handling
3. Temporal and geographic feature engineering
4. Chronological backtesting
5. Baseline and gradient-boosting model comparisons
6. Feature ablation and hyperparameter tuning
7. Error analysis and unseen-city stress testing
8. Final model training and prediction generation

## 3. Data Quality and Feature Engineering

The development dataset contains:

- 300 missing shipment weights
- 292 negative shipment weights
- 374 missing market-index observations

Invalid nonpositive weights are treated as missing, with indicators preserving information about the original data-quality issue.

Feature engineering includes:

- Calendar month, weekday, and weekend indicators
- Days elapsed since January 1, 2025
- Log-transformed shipment distance
- Missing/invalid weight indicators
- Optional geographic coordinates

Missing-value imputation and categorical encoding are fitted exclusively on the training portion of each backtest.

## 4. Validation Strategy

Because the final prediction period occurs after the labeled development period, chronological validation was selected instead of relying on a random train/test split.

Three expanding-window backtests were performed:

| Training period | Holdout period |
|---|---|
| January–July 2025 | August 2025 |
| January–August 2025 | September 2025 |
| January–September 2025 | October 2025 |

Evaluation metrics:

- MAE — Mean Absolute Error
- RMSE — Root Mean Squared Error
- WAPE — Weighted Absolute Percentage Error
- R² — Coefficient of Determination

Model selection primarily emphasizes MAE while also considering robustness, prediction errors, and feature availability.

## 5. Model Comparison

Three gradient-boosting algorithms were evaluated using common features.

| Algorithm | Average MAE | Average WAPE |
|---|---:|---:|
| LightGBM | $105.16 | 4.43% |
| CatBoost | $105.90 | 4.46% |
| XGBoost | $107.16 | 4.51% |

The experiment also included constant-median and equipment-based rate-per-mile baselines.

### Feature Ablation

Controlled experiments showed that adding geographic features improved predictive performance, while including market-index and quote signals worsened the tested backtest results.

### Selected Configurations

| Prediction task | Selected model | Average backtest MAE |
|---|---|---:|
| Final validation loads | LightGBM + Geographic Features, Higher Capacity | $103.12 |
| December chart | LightGBM + Common Features, Higher Capacity | $104.26 |

The reported scores are averages across the three historical holdout months, not performance on Spotter's private evaluation dataset.

## 6. Error Analysis and Generalization

The October diagnostic analysis revealed that the worst 1% of predictions account for approximately 48.9% of total absolute error.

Large errors are concentrated among unusually expensive shipments, indicating a limitation in predicting extreme rates.

An additional simulated unseen-city experiment compared:

| Experiment | MAE |
|---|---:|
| Normal training | $108.77 |
| Eight cities withheld from training | $111.21 |

The experiment suggests some robustness to unfamiliar locations, although it also reduces training-set size and therefore does not isolate the effect of unseen cities.

## 7. Project Structure

- `src/preprocessing.py` — Data cleaning and feature engineering
- `src/modeling.py` — Model definitions and preprocessing pipelines
- `src/evaluation.py` — Regression evaluation metrics
- `src/train.py` — Train and save final model pipelines
- `src/predict.py` — Generate required prediction files
- `notebooks/` — Exploratory analysis and experimental validation
- `reports/` — Experiment summaries and supporting documentation
- `requirements.txt` — Python dependencies

Employer-provided datasets, generated model artifacts, and submission CSVs are excluded from the repository.

## 8. Setup and Reproduction

### Prerequisites

Python 3.9+ with compatible dependency versions.

Clone the repository and create a virtual environment.

1. Run `python3 -m venv .venv`
2. Activate with `source .venv/bin/activate` on macOS/Linux
3. Install dependencies using `python -m pip install -r requirements.txt`

Place the original Spotter-provided files in the `data/` directory using these names:

- `train-test.csv`
- `validation.csv`
- `validation-predictions-template.csv`
- `december-chart-inputs.csv`

### Train the final models

Run `python -m src.train` from the repository root.

This saves the trained pipelines under `models/`.

### Generate predictions

Run `python -m src.predict`.

The script generates:

- `outputs/validation_predictions.csv`
- `outputs/december_chart_inputs.csv`

### Validate submission files

Place the original `score.py` in the project root and run:

`python score.py --predictions outputs/validation_predictions.csv --december-predictions outputs/december_chart_inputs.csv`

The scorer validates both prediction files and generates `scorer_results/candidate_december.png`.

**Note:** The supplied scorer validates submission structure and creates the required December chart. Final prediction metrics are computed independently by Spotter.

## 9. Key Engineering Decisions

- Used chronological backtesting to simulate future predictions.
- Prevented target leakage by explicitly selecting input features.
- Learned imputation and categorical transformations only from training data.
- Evaluated multiple algorithms rather than assuming one was best.
- Used feature ablation to measure the contribution of additional signals.
- Investigated extreme pricing errors and unseen-city generalization.
- Saved reusable preprocessing and model pipelines for consistent inference.

## 10. Limitations and Future Improvements

- Large prediction errors remain on a small set of extreme-rate shipments.
- Geographic generalization was tested through a simulated stress test rather than a fully independent unseen-city dataset.
- The exact business meaning and prediction-time availability of the optional market signals require further clarification.
- Additional time periods and independent evaluation data would strengthen confidence in model selection.

## Assessment Deliverables

The submission includes the required prediction CSV, technical report with the official December chart, accessible GitHub repository, and Loom walkthrough.

The repository documents the implementation and experiments; the required submission artifacts are provided separately.