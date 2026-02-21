"""
============================================================
STEP 2: Exploratory Data Analysis & Preprocessing
Sri Lanka Export Revenue Prediction
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os

os.makedirs("outputs", exist_ok=True)

# ── Load Data ────────────────────────────────────────────
df = pd.read_csv("data/srilanka_exports.csv", parse_dates=["date"])

print("=" * 55)
print("  EXPLORATORY DATA ANALYSIS")
print("=" * 55)
print(f"\nShape: {df.shape}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nStatistical Summary:")
print(df.describe().round(2))

# ── Plot 1: Target Variable Over Time ────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 12))
fig.suptitle("Sri Lanka Export Revenue Analysis (2014–2024)",
             fontsize=15, fontweight="bold", y=0.98)

axes[0].plot(df["date"], df["total_export_revenue_usd_mn"],
             color="#1a6b3c", linewidth=2, marker="o", markersize=2)
axes[0].fill_between(df["date"], df["total_export_revenue_usd_mn"],
                     alpha=0.15, color="#1a6b3c")
axes[0].set_title("Monthly Total Export Revenue (USD Millions)", fontweight="bold")
axes[0].set_ylabel("USD Millions")
axes[0].axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-06-01"),
                alpha=0.15, color="blue", label="COVID-19")
axes[0].axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2023-01-01"),
                alpha=0.15, color="red", label="Economic Crisis")
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)
axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# Key drivers
axes[1].plot(df["date"], df["apparel_revenue_usd_mn"],
             label="Apparel Revenue", color="#e65c00", linewidth=1.8)
axes[1].plot(df["date"], df["tea_price_usd_per_kg"] * df["tea_export_volume_mt"] / 1000,
             label="Tea Revenue", color="#4b7f2a", linewidth=1.8)
axes[1].plot(df["date"], df["tourism_earnings_usd_mn"],
             label="Tourism Earnings", color="#0055a5", linewidth=1.8)
axes[1].set_title("Key Revenue Components (USD Millions)", fontweight="bold")
axes[1].set_ylabel("USD Millions")
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# Exchange rate & inflation
ax2b = axes[2].twinx()
axes[2].plot(df["date"], df["exchange_rate_lkr_usd"],
             color="#8b0000", linewidth=2, label="Exchange Rate (LKR/USD)")
ax2b.plot(df["date"], df["inflation_rate_pct"],
          color="#ff7f0e", linewidth=2, linestyle="--", label="Inflation (%)")
axes[2].set_title("Macroeconomic Indicators", fontweight="bold")
axes[2].set_ylabel("LKR per USD", color="#8b0000")
ax2b.set_ylabel("Inflation Rate (%)", color="#ff7f0e")
axes[2].legend(loc="upper left", fontsize=9)
ax2b.legend(loc="upper right", fontsize=9)
axes[2].grid(True, alpha=0.3)
axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.tight_layout()
plt.savefig("outputs/01_eda_timeseries.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: outputs/01_eda_timeseries.png")

# ── Plot 2: Correlation Heatmap ──────────────────────────
features = [
    "exchange_rate_lkr_usd", "tea_price_usd_per_kg", "tea_export_volume_mt",
    "rubber_price_usd_per_kg", "rubber_export_volume_mt", "apparel_revenue_usd_mn",
    "tourism_earnings_usd_mn", "inflation_rate_pct", "oil_import_cost_usd_mn",
    "total_export_revenue_usd_mn"
]

fig, ax = plt.subplots(figsize=(11, 9))
corr_matrix = df[features].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, vmin=-1, vmax=1, ax=ax, square=True, linewidths=0.5,
            cbar_kws={"shrink": 0.8})
ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold", pad=15)
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig("outputs/02_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: outputs/02_correlation_heatmap.png")

# ── Plot 3: Seasonal Pattern ─────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
monthly_avg = df.groupby("month")["total_export_revenue_usd_mn"].mean()
months_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                 "Jul","Aug","Sep","Oct","Nov","Dec"]
bars = ax.bar(months_labels, monthly_avg.values, color="#1a6b3c", alpha=0.8, edgecolor="white")
ax.axhline(monthly_avg.mean(), color="red", linestyle="--", linewidth=1.5, label="Annual Average")
ax.set_title("Average Monthly Export Revenue — Seasonal Pattern", fontweight="bold")
ax.set_ylabel("USD Millions")
ax.legend()
ax.grid(True, axis="y", alpha=0.3)
for bar, val in zip(bars, monthly_avg.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{val:.0f}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig("outputs/03_seasonal_pattern.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: outputs/03_seasonal_pattern.png")

# ── Preprocessing ────────────────────────────────────────
print("\n" + "=" * 55)
print("  PREPROCESSING")
print("=" * 55)

# Feature matrix
FEATURE_COLS = [
    "year", "month",
    "exchange_rate_lkr_usd",
    "tea_price_usd_per_kg",
    "tea_export_volume_mt",
    "rubber_price_usd_per_kg",
    "rubber_export_volume_mt",
    "apparel_revenue_usd_mn",
    "tourism_earnings_usd_mn",
    "inflation_rate_pct",
    "oil_import_cost_usd_mn",
]
TARGET_COL = "total_export_revenue_usd_mn"

X = df[FEATURE_COLS].copy()
y = df[TARGET_COL].copy()

# Add lag features (previous month's revenue — useful for time series)
X["revenue_lag1"] = y.shift(1)
X["revenue_lag3"] = y.shift(3)
X["revenue_rolling3"] = y.shift(1).rolling(3).mean()

# Drop first 3 rows (NaN from lag)
X = X.iloc[3:].reset_index(drop=True)
y = y.iloc[3:].reset_index(drop=True)
dates_clean = df["date"].iloc[3:].reset_index(drop=True)

print(f"Feature matrix shape: {X.shape}")
print(f"Features used: {list(X.columns)}")
print(f"Missing values after lag: {X.isnull().sum().sum()}")

# Save preprocessed data
X.to_csv("data/X_preprocessed.csv", index=False)
y.to_csv("data/y_target.csv", index=False)
dates_clean.to_csv("data/dates.csv", index=False)

print("\nPreprocessed data saved.")
print("Ready for model training.")
