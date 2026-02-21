# 🇱🇰 Sri Lanka Export Revenue Prediction

---

## 📌 Project Overview

Predict Sri Lanka's **monthly total export revenue (USD Millions)** using a
**Gradient Boosting Regressor** trained on economic indicators from 2014–2024.

**Problem Type:** Regression
**Algorithm:** Gradient Boosting Regressor (scikit-learn) / XGBoost
**Target Variable:** `total_export_revenue_usd_mn`

---

## 📁 Project Structure

```
export_revenue_prediction/
│
├── data/
│   ├── srilanka_exports.csv       ← Raw dataset
│   ├── X_preprocessed.csv         ← Feature matrix
│   ├── y_target.csv               ← Target variable
│   └── dates.csv                  ← Date index
│
├── models/
│   └── gbr_model.pkl              ← Trained model
│
├── outputs/
│   ├── 01_eda_timeseries.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_seasonal_pattern.png
│   ├── 04_actual_vs_predicted.png
│   ├── 05_residuals.png
│   ├── 06_metrics_comparison.png
│   ├── 07_feature_importance.png
│   ├── 08_permutation_importance.png
│   ├── 09_partial_dependence_plots.png
│   ├── 10_prediction_breakdown.png
│   ├── 11_error_over_time.png
│   └── evaluation_metrics.csv
│
├── app/
│   └── app.py                     ← Streamlit web app (Bonus)
│
├── 01_generate_data.py
├── 02_eda_preprocessing.py
├── 03_model_training.py
├── 04_explainability.py
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Pipeline (in order)

```bash
python 01_generate_data.py        # Generate/load dataset
python 02_eda_preprocessing.py    # EDA + preprocessing
python 03_model_training.py       # Train & evaluate model
python 04_explainability.py       # XAI analysis
```

### 3. Launch Web App (Bonus)

```bash
streamlit run app/app.py
```

---

## 📊 Features Used

| Feature                 | Description                       |
| ----------------------- | --------------------------------- |
| exchange_rate_lkr_usd   | USD/LKR monthly exchange rate     |
| tea_price_usd_per_kg    | Global tea commodity price        |
| tea_export_volume_mt    | Monthly tea export volume (MT)    |
| rubber_price_usd_per_kg | Global rubber price               |
| rubber_export_volume_mt | Monthly rubber export volume (MT) |
| apparel_revenue_usd_mn  | Garment sector revenue (USD Mn)   |
| tourism_earnings_usd_mn | Tourism foreign exchange earnings |
| inflation_rate_pct      | Monthly CPI inflation rate        |
| oil_import_cost_usd_mn  | Oil import costs (USD Mn)         |
| revenue_lag1            | Previous month's export revenue   |
| revenue_lag3            | Revenue 3 months ago              |
| revenue_rolling3        | 3-month rolling average revenue   |

---

## 🤖 Algorithm: Gradient Boosting Regressor

**About Algorithm:**

- Ensemble of weak learners built sequentially — each tree corrects its predecessor
- Uses gradient descent in function space (not covered in standard syllabus)
- Different from: Decision Trees, k-NN, Logistic Regression, SVM
- Used in top Kaggle solutions; industry standard for tabular data

**Hyperparameter Tuning:** GridSearchCV with TimeSeriesSplit (5-fold)

---

## 📈 Evaluation Metrics

- **RMSE** — Root Mean Squared Error
- **MAE** — Mean Absolute Error
- **R²** — Coefficient of Determination
- **MAPE** — Mean Absolute Percentage Error

---

## 🧠 Explainability Methods

1. **Built-in Feature Importance** — Mean decrease in impurity
2. **Permutation Importance** — Drop in R² per feature
3. **Partial Dependence Plots (PDP)** — Marginal effect of each feature
4. **Individual Prediction Breakdown** — Feature contributions to single predictions

---

## 📦 Data Sources

- [Central Bank of Sri Lanka](https://www.cbsl.gov.lk/en/statistics)
- [Export Development Board](https://www.srilankabusiness.com)
- [Sri Lanka Tourism Development Authority](https://www.sltda.gov.lk)
- [World Bank Commodity Prices](https://www.worldbank.org/en/research/commodity-markets)
