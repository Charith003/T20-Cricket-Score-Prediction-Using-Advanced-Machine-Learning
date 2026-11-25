# T20 Cricket Match Score Prediction

End-to-end regression and ensemble-learning project that predicts the `Predicted Score` column of a public T20 cricket dataset from match-progress and match-condition inputs.

> **Important Note:** The metrics below describe performance **on this dataset**. The target-generation method has not been independently established, and 29.0% of records have estimated current runs above the final target score (see [Model Limitations](#model-limitations)). A high $R^2$ here is **not** evidence of real-world cricket forecasting ability.

---

## How It Works

The entire workflow lives in [T20_Cricket_Score_Prediction_Advanced.ipynb](T20_Cricket_Score_Prediction_Advanced.ipynb) with shared preprocessing in [t20_features.py](t20_features.py):

1. **Data Loading & Audit:** Loads `dataset/t20_cricket_match_score_prediction.csv` (1,500 rows × 9 columns, 0 missing values, 0 duplicates), validates ranges, and drops `Match ID` to prevent leakage.
2. **Leakage-Safe Splitting:** Splits data into **1,049 train / 226 validation / 225 untouched test** rows (`SEED = 42`). All preprocessing, feature selection, and hyperparameter tuning happen strictly on training/validation data.
3. **Feature Engineering (`t20_features.py`):** Stateless `CricketFeatureEngineer` transforms the 7 raw inputs into domain features. A 5-fold CV ablation compared `raw`, `basic`, and `full` feature sets and selected **`basic`**:
   - `current_est_runs` (`Overs Played * Run Rate`), `wickets_remaining`, `overs_remaining`, `wicket_pressure` (`Wickets Lost / max(Overs Played, 1)`), `remaining_innings_pct`
   - `scoring_phase` (`powerplay` / `middle` / `death`) and `run_rate_category` (`low` / `moderate` / `high` / `very_high`)
4. **Preprocessing Pipeline:** Median imputation for numeric features, most-frequent imputation + `OneHotEncoder(handle_unknown="ignore")` for categorical features, and `StandardScaler` for linear/ensemble models.
5. **Model Benchmarking & Tuning:** Trains and evaluates 11 individual regressors plus Voting and Stacking ensembles using 5-fold CV:
   - **Linear:** Linear Regression, Ridge (`GridSearchCV`), Lasso (`GridSearchCV`)
   - **Trees:** Decision Tree, Random Forest (`RandomizedSearchCV`), Extra Trees
   - **Boosting:** Gradient Boosting, HistGradientBoosting (`RandomizedSearchCV`), XGBoost (`RandomizedSearchCV`), LightGBM (`Optuna` TPE, 20 trials), CatBoost (`RandomizedSearchCV`)
   - **Ensembles:** Voting Regressor and **Stacking Regressor** (CatBoost, LightGBM, XGBoost, and HistGradientBoosting base learners with a Ridge meta-model)
   - **Baseline:** Dummy Regressor (mean predictor)
6. **Final Evaluation & Uncertainty:** Refits the selected **Stacking Regressor** on train + validation (1,275 rows), evaluates **once** on the untouched 225-row test set, and computes a **90% empirical prediction interval** from out-of-fold absolute residuals.
7. **Explainability & Export:** Generates permutation importance, native tree importance, and SHAP summary plots, then exports the fitted pipelines (`models/*.pkl`), metrics (`results/*`), and 14 figures (`visualizations/*`).

---

## Results

### Final Test Set Performance (Stacking Regressor, Untouched 225 Rows)

| Metric | Result |
|---|---|
| **MAE** | **1.35 runs** |
| **RMSE** | **1.88 runs** |
| **$R^2$** | **0.9993** |
| **Within ±5 runs** *(custom tolerance metric)* | **96.0%** |
| **Within ±10 / ±15 / ±20 runs** | **100.0%** |
| **5-Fold CV RMSE (train + val)** | **2.06 ± 0.15 runs** |
| **90% Empirical Prediction Interval** | **±2.97 runs** *(observed test coverage: 91.1%)* |
| **Mean-Predictor Baseline Test RMSE** | **70.90 runs** |

### Model Comparison (Validation Set, 226 Rows)

| Model | Stage | Validation MAE | Validation RMSE | Validation $R^2$ | ±5 Runs (%) | 5-Fold CV RMSE |
|---|---|---:|---:|---:|---:|---:|
| **Stacking Regressor** | Ensemble | **1.30** | **1.84** | **0.9993** | **97.8%** | **2.15 ± 0.12** |
| Voting Regressor | Ensemble | 1.31 | 1.93 | 0.9992 | 98.7% | 2.25 ± 0.19 |
| CatBoost | Baseline | 1.69 | 2.35 | 0.9988 | 96.0% | 2.91 ± 0.22 |
| CatBoost (tuned) | Tuned | 1.75 | 2.39 | 0.9988 | 93.8% | 2.61 ± 0.18 |
| XGBoost (tuned) | Tuned | 1.87 | 2.64 | 0.9985 | 95.1% | 3.20 ± 0.16 |
| LightGBM (tuned) | Tuned | 1.97 | 2.74 | 0.9984 | 93.8% | 3.16 ± 0.26 |
| HistGradientBoosting (tuned) | Tuned | 2.25 | 3.34 | 0.9976 | 92.5% | 3.82 ± 0.37 |
| Gradient Boosting | Baseline | 2.72 | 3.57 | 0.9973 | 85.8% | 4.30 ± 0.29 |
| Extra Trees | Baseline | 3.76 | 5.13 | 0.9943 | 76.1% | 5.92 ± 0.37 |
| Random Forest (tuned) | Tuned | 4.30 | 6.02 | 0.9922 | 65.9% | 7.32 ± 0.26 |
| Decision Tree | Baseline | 6.36 | 8.75 | 0.9835 | 52.7% | 10.21 ± 0.67 |
| Lasso Regression | Baseline | 8.10 | 11.61 | 0.9710 | 47.8% | 12.07 ± 0.64 |
| Linear Regression | Baseline | 8.38 | 11.75 | 0.9703 | 46.5% | 12.05 ± 0.64 |
| Ridge Regression (tuned) | Tuned | 8.39 | 11.78 | 0.9701 | 46.5% | 12.04 ± 0.66 |
| Dummy (predict mean) | Baseline | 54.16 | 68.27 | -0.0032 | 5.8% | 72.87 ± 1.54 |

### Top Input Features (Permutation Importance on Validation Set)

| Rank | Raw Input | Permutation Importance | Std |
|---|---|---:|---:|
| 1 | `Overs Played` | 81.62 | 3.06 |
| 2 | `Run Rate` | 48.15 | 2.30 |
| 3 | `Pitch Condition` | 26.98 | 1.36 |
| 4 | `Weather` | 9.76 | 0.46 |
| 5 | `Wickets Lost` | 4.42 | 0.20 |
| 6 | `Opponent Strength` | 2.44 | 0.16 |
| 7 | `Home/Away` | ~0.00 | 0.00 |

### Visual Outputs

All 14 diagnostic, EDA, and explainability plots are saved in `visualizations/`:
- **EDA:** `01_univariate_numeric.png` – `08_outlier_boxplots.png`
- **Model Diagnostics:** `09_validation_diagnostics.png`, `10_learning_curve.png`, `11_final_test_results.png`
- **Explainability:** `12_permutation_importance.png`, `13_native_importance.png`, `14_shap_summary.png`

---

## Quick Start & Prediction Usage

### Installation

```bash
git clone https://github.com/Charith003/T20-Cricket-Score-Prediction-Using-Advanced-Machine-Learning.git
cd T20-Cricket-Score-Prediction-Using-Advanced-Machine-Learning
python -m venv .venv
# Windows: .venv\Scripts\activate  |  Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt jupyter
```

### Run the Notebook

```bash
jupyter notebook T20_Cricket_Score_Prediction_Advanced.ipynb
```

Or execute non-interactively:

```bash
jupyter nbconvert --to notebook --execute --inplace T20_Cricket_Score_Prediction_Advanced.ipynb
```

### Run Inference in Python

```python
import joblib
import pandas as pd
import t20_features  # required in path to unpickle CricketFeatureEngineer

pipe = joblib.load("models/t20_prediction_pipeline.pkl")
row = pd.DataFrame([{
    "Overs Played": 10,
    "Wickets Lost": 3,
    "Run Rate": 8.0,
    "Home/Away": "Home",
    "Opponent Strength": 5,
    "Pitch Condition": "Balanced",
    "Weather": "Sunny",
}])
print(f"Predicted Score: {pipe.predict(row)[0]:.2f}")  # ~100.30 runs
```

---

## Project Structure

```text
T20-Cricket-Score-Prediction/
├── T20_Cricket_Score_Prediction_Advanced.ipynb   # Full executed analysis, training & evaluation notebook
├── t20_features.py                               # Stateless feature engineering & pipeline builder
├── requirements.txt                              # Exact Python 3.12 package versions
├── environment.yml                               # Conda environment specification
├── runtime_versions.json                         # Recorded runtime versions
├── dataset/
│   └── t20_cricket_match_score_prediction.csv    # 1,500-row dataset
├── models/
│   ├── best_t20_model.pkl                        # Fitted Stacking Regressor
│   ├── t20_preprocessing_pipeline.pkl            # Fitted feature engineering + encoder
│   └── t20_prediction_pipeline.pkl               # End-to-end raw DataFrame -> score pipeline
├── results/
│   ├── model_evaluation_results.csv              # All model CV & validation/test metrics
│   ├── actual_vs_predicted.csv                   # Test-set predictions vs actuals
│   ├── feature_importance.csv                    # Permutation & native importances
│   ├── model_metadata.json                       # Full run metadata & tuned hyperparameters
│   └── final_report.txt                          # Summary report
└── visualizations/                               # 14 EDA, diagnostic, and SHAP plots
```

---

## Model Limitations

- **Target provenance is unverified:** A simple linear model achieves $R^2 \approx 0.97$ and boosting achieves $R^2 \approx 0.999$, which is consistent with a formula-generated target rather than noisy real-world match outcomes.
- **Internal inconsistency:** 29.0% of rows have `current_est_runs > Predicted Score`. All rows are flagged and retained.
- **No chronology:** No match dates or sequential IDs are available for time-based validation; splits are random.
- **Single empirical interval:** The ±2.97-run 90% interval is a global band and does not model heteroscedasticity.
- **External validation required:** Evaluate on genuine historical ball-by-ball T20 data before making any real-world forecasting claims.
