"""
============================================================
STEP 5: Streamlit Frontend App (Bonus)
Sri Lanka Export Revenue Predictor
============================================================
Run with: streamlit run app/app.py
============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Page Config ──────────────────────────────────────────
st.set_page_config(
    page_title="Sri Lanka Export Revenue Predictor",
    page_icon="🇱🇰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a6b3c, #2d8653);
        color: white; padding: 1.5rem 2rem;
        border-radius: 10px; margin-bottom: 1.5rem;
        text-align: center;
    }
    .metric-card {
        background: #f8f9fa; border-left: 4px solid #1a6b3c;
        padding: 1rem; border-radius: 8px; margin: 0.5rem 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #1a6b3c, #2d8653);
        color: white; padding: 2rem; border-radius: 12px;
        text-align: center; font-size: 1.2rem;
    }
    .warning-box {
        background: #fff3cd; border-left: 4px solid #ffc107;
        padding: 1rem; border-radius: 8px; margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Model ───────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "gbr_model.pkl")
    with open(model_path, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    base = os.path.join(os.path.dirname(__file__), "..")
    X = pd.read_csv(os.path.join(base, "data", "X_preprocessed.csv"))
    y = pd.read_csv(os.path.join(base, "data", "y_target.csv")).squeeze()
    d = pd.read_csv(os.path.join(base, "data", "dates.csv"), parse_dates=["date"]).squeeze()
    return X, y, d

try:
    model = load_model()
    X, y, dates = load_data()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Model not found. Please run 03_model_training.py first. Error: {e}")

# ── Header ───────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🇱🇰 Sri Lanka Export Revenue Predictor</h1>
    <p style="margin:0; opacity:0.9;">
        Dashboard | Gradient Boosting Regressor
    </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/1/11/Flag_of_Sri_Lanka.svg",
                 width=100)
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("", ["🔮 Predict Revenue", "📈 Historical Analysis", "🧠 Model Info"])

st.sidebar.markdown("---")
st.sidebar.markdown("""
**About This App**
Predicts Sri Lanka's monthly export revenue (USD Millions) using a Gradient Boosting model trained on:
- Tea & rubber commodities
- Apparel sector data
- Macroeconomic indicators
- Tourism earnings
""")

# ─────────────────────────────────────────────────────────
# PAGE 1: PREDICTION
# ─────────────────────────────────────────────────────────
if page == "🔮 Predict Revenue":
    st.header("🔮 Predict Monthly Export Revenue")
    st.markdown("Enter economic indicators below to get a prediction.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📅 Time Period")
        year  = st.slider("Year",  2024, 2030, 2025)
        month = st.selectbox("Month", range(1, 13),
                             format_func=lambda x: ["Jan","Feb","Mar","Apr","May","Jun",
                                                    "Jul","Aug","Sep","Oct","Nov","Dec"][x-1])

        st.subheader("🫖 Tea Sector")
        tea_price  = st.number_input("Tea Price (USD/kg)",  min_value=1.5, max_value=6.0, value=3.1, step=0.05)
        tea_volume = st.number_input("Tea Export Volume (MT)", min_value=8000, max_value=30000, value=18000, step=500)

    with col2:
        st.subheader("🌿 Rubber Sector")
        rubber_price  = st.number_input("Rubber Price (USD/kg)", min_value=0.5, max_value=4.0, value=1.7, step=0.05)
        rubber_volume = st.number_input("Rubber Export Volume (MT)", min_value=1000, max_value=7000, value=3500, step=200)

        st.subheader("👗 Apparel Sector")
        apparel_revenue = st.number_input("Apparel Revenue (USD Mn)", min_value=200.0, max_value=700.0, value=490.0, step=10.0)

    with col3:
        st.subheader("🏦 Macroeconomic Indicators")
        exchange_rate = st.number_input("Exchange Rate (LKR/USD)", min_value=100.0, max_value=500.0, value=315.0, step=5.0)
        inflation     = st.number_input("Inflation Rate (%)",       min_value=1.0,   max_value=80.0,  value=8.0,   step=0.5)
        oil_cost      = st.number_input("Oil Import Cost (USD Mn)", min_value=100.0, max_value=500.0, value=260.0, step=10.0)

        st.subheader("✈️ Tourism")
        tourism = st.number_input("Tourism Earnings (USD Mn)", min_value=0.0, max_value=350.0, value=185.0, step=10.0)

    # Recent revenue for lag features
    st.markdown("---")
    st.subheader("📉 Recent Revenue (for lag features)")
    lc1, lc2, lc3 = st.columns(3)
    with lc1:
        rev_lag1 = st.number_input("Last month revenue (USD Mn)", value=700.0, step=10.0)
    with lc2:
        rev_lag3 = st.number_input("3 months ago revenue (USD Mn)", value=695.0, step=10.0)
    with lc3:
        rev_roll3 = st.number_input("3-month rolling avg (USD Mn)", value=698.0, step=10.0)

    # ── Predict Button ────────────────────────────────────
    if st.button("🚀 Predict Export Revenue", type="primary", use_container_width=True):
        if model_loaded:
            input_data = pd.DataFrame([{
                "year": year, "month": month,
                "exchange_rate_lkr_usd": exchange_rate,
                "tea_price_usd_per_kg": tea_price,
                "tea_export_volume_mt": tea_volume,
                "rubber_price_usd_per_kg": rubber_price,
                "rubber_export_volume_mt": rubber_volume,
                "apparel_revenue_usd_mn": apparel_revenue,
                "tourism_earnings_usd_mn": tourism,
                "inflation_rate_pct": inflation,
                "oil_import_cost_usd_mn": oil_cost,
                "revenue_lag1": rev_lag1,
                "revenue_lag3": rev_lag3,
                "revenue_rolling3": rev_roll3,
            }])

            prediction = model.predict(input_data)[0]

            # Display prediction
            st.markdown(f"""
            <div class="prediction-box">
                <h2 style="margin:0;">🎯 Predicted Export Revenue</h2>
                <h1 style="font-size:3rem; margin:0.5rem 0;">USD {prediction:.1f} Million</h1>
                <p style="margin:0; opacity:0.85;">
                    Month: {"Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()[month-1]} {year}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Context cards
            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            hist_avg = y.mean() if model_loaded else 703
            c1.metric("vs Historical Avg", f"{prediction:.1f}", f"{prediction - hist_avg:+.1f} USD Mn")
            c2.metric("Tea Revenue",       f"{(tea_price * tea_volume / 1000):.1f} USD Mn")
            c3.metric("Rubber Revenue",    f"{(rubber_price * rubber_volume / 1000):.1f} USD Mn")
            c4.metric("Apparel Share",     f"{(apparel_revenue / prediction * 100):.1f}%")

            # Feature contribution chart
            contributions = []
            baseline_pred = prediction
            for feat in input_data.columns:
                X_perturbed = input_data.copy()
                X_perturbed[feat] = X[feat].mean()
                perturbed_pred = model.predict(X_perturbed)[0]
                contributions.append((feat, baseline_pred - perturbed_pred))

            contrib_df = pd.DataFrame(contributions, columns=["Feature", "Contribution"])
            contrib_df = contrib_df.sort_values("Contribution", key=abs, ascending=True).tail(8)

            fig, ax = plt.subplots(figsize=(9, 5))
            colors_bar = ["#2ecc71" if c > 0 else "#e74c3c" for c in contrib_df["Contribution"]]
            ax.barh(contrib_df["Feature"], contrib_df["Contribution"], color=colors_bar, edgecolor="white")
            ax.axvline(0, color="black", linewidth=1)
            ax.set_title("Feature Contributions to This Prediction", fontweight="bold")
            ax.set_xlabel("Contribution to Revenue (USD Mn)")
            ax.grid(True, axis="x", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

# ─────────────────────────────────────────────────────────
# PAGE 2: HISTORICAL ANALYSIS
# ─────────────────────────────────────────────────────────
elif page == "📈 Historical Analysis":
    st.header("📈 Historical Export Revenue Analysis")

    if model_loaded:
        all_preds = model.predict(X)
        n = len(X)
        val_end = int(n * 0.85)

        fig, axes = plt.subplots(2, 1, figsize=(13, 9))
        fig.suptitle("Sri Lanka Export Revenue (2014–2024)", fontsize=14, fontweight="bold")

        axes[0].plot(dates, y, label="Actual Revenue", color="#1a6b3c", linewidth=2)
        axes[0].plot(dates, all_preds, label="Model Prediction", color="#ff6b35",
                     linewidth=1.5, linestyle="--", alpha=0.85)
        axes[0].axvspan(dates.iloc[val_end], dates.iloc[-1], alpha=0.12, color="red", label="Test Period")
        axes[0].fill_between(dates, y, alpha=0.1, color="#1a6b3c")
        axes[0].set_ylabel("USD Millions")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title("Actual vs Predicted Revenue", fontweight="bold")

        X_raw = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "srilanka_exports.csv"),
                            parse_dates=["date"]).iloc[3:].reset_index(drop=True)
        axes[1].stackplot(
            dates,
            X_raw["apparel_revenue_usd_mn"],
            X_raw["tea_price_usd_per_kg"] * X_raw["tea_export_volume_mt"] / 1000,
            X_raw["rubber_price_usd_per_kg"] * X_raw["rubber_export_volume_mt"] / 1000,
            X_raw["tourism_earnings_usd_mn"] * 0.15,
            labels=["Apparel", "Tea", "Rubber", "Tourism (15%)"],
            colors=["#e65c00", "#4b7f2a", "#8b6914", "#0055a5"],
            alpha=0.8
        )
        axes[1].set_ylabel("USD Millions")
        axes[1].legend(loc="upper left", fontsize=9)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_title("Revenue Component Breakdown", fontweight="bold")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        from sklearn.metrics import mean_squared_error, r2_score
        test_preds = all_preds[val_end:]
        test_actual = y.iloc[val_end:]
        rmse = np.sqrt(mean_squared_error(test_actual, test_preds))
        r2   = r2_score(test_actual, test_preds)
        mape = np.mean(np.abs((test_actual - test_preds) / test_actual)) * 100
        c1.metric("Test RMSE",  f"{rmse:.2f} USD Mn")
        c2.metric("Test R²",    f"{r2:.4f}")
        c3.metric("Test MAPE",  f"{mape:.2f}%")

# ─────────────────────────────────────────────────────────
# PAGE 3: MODEL INFO
# ─────────────────────────────────────────────────────────
elif page == "🧠 Model Info":
    st.header("🧠 About the Model")

    st.markdown("""
    ### Algorithm: Gradient Boosting Regressor

    **Why Gradient Boosting?**
    - An ensemble method that builds trees sequentially — each tree corrects errors of the previous one
    - Not covered in standard ML lectures (satisfies assignment requirement)
    - Outperforms single Decision Trees, Linear Regression, and k-NN on tabular economic data
    - Handles non-linear relationships and feature interactions automatically
    - Supports partial dependence and permutation importance for explainability

    ---
    ### Dataset Features
    | Feature | Description | Source |
    |---|---|---|
    | exchange_rate_lkr_usd | USD/LKR exchange rate | CBSL |
    | tea_price_usd_per_kg | Global tea commodity price | World Bank |
    | tea_export_volume_mt | Monthly tea export volume | EDB |
    | rubber_price_usd_per_kg | Global rubber price | World Bank |
    | rubber_export_volume_mt | Monthly rubber export volume | EDB |
    | apparel_revenue_usd_mn | Garment sector export revenue | EDB |
    | tourism_earnings_usd_mn | Tourism foreign exchange earnings | SLTDA |
    | inflation_rate_pct | Monthly CPI inflation | CBSL |
    | oil_import_cost_usd_mn | Oil import expenditure | CBSL |
    | revenue_lag1/3 | Previous month/3-month revenue | Derived |
    | revenue_rolling3 | 3-month rolling average revenue | Derived |

    ---
    ### Evaluation Metrics
    - **RMSE** (Root Mean Square Error): Penalizes large errors
    - **MAE** (Mean Absolute Error): Average prediction error in USD Mn
    - **R²** (Coefficient of Determination): % of variance explained
    - **MAPE** (Mean Absolute Percentage Error): % error — intuitive for economics

    ---
    ### Key Findings
    1. **Apparel sector** is the dominant export driver (>60% of revenue)
    2. **Lag features** confirm strong month-to-month autocorrelation
    3. **Inflation** has a strong negative effect — consistent with economic theory
    4. **2022 economic crisis** caused the largest prediction errors due to unprecedented conditions
    5. **Tea price volatility** adds meaningful signal for seasonal patterns

    ---
    ### Data Sources
    - 🏦 [Central Bank of Sri Lanka](https://www.cbsl.gov.lk)
    - 📦 [Export Development Board](https://www.srilankabusiness.com)
    - ✈️ [Sri Lanka Tourism Development Authority](https://www.sltda.gov.lk)
    - 🌍 [World Bank Commodity Price Data](https://www.worldbank.org/en/research/commodity-markets)
    """)

    if model_loaded:
        st.markdown("---")
        st.subheader("Model Parameters")
        params = model.get_params()
        param_df = pd.DataFrame(list(params.items()), columns=["Parameter", "Value"])
        st.dataframe(param_df, use_container_width=True)
