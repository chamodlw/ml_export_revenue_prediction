"""
============================================================
STEP 1: Dataset Generation
Sri Lanka Export Revenue Prediction (2014-2024)
============================================================
NOTE: This script generates a realistic synthetic dataset.
For your actual submission, replace with real data from:
  - Central Bank of Sri Lanka: https://www.cbsl.gov.lk
  - Export Development Board: https://www.srilankabusiness.com
  - World Bank Commodity Prices: https://www.worldbank.org
============================================================
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)

months = pd.date_range(start="2014-01-01", end="2024-12-01", freq="MS")
n = len(months)

# --- Feature Engineering ---

# 1. USD/LKR Exchange Rate (gradually depreciating, 2022 crisis spike)
exchange_rate = np.linspace(130, 200, n)
exchange_rate[96:108] += np.linspace(0, 170, 12)   # 2022 crisis
exchange_rate[108:] = np.linspace(360, 310, n - 108)
exchange_rate += np.random.normal(0, 3, n)

# 2. Global Tea Price (USD/kg) — Sri Lanka 4th largest exporter
tea_price = 2.8 + 0.3 * np.sin(np.linspace(0, 4 * np.pi, n)) + np.random.normal(0, 0.15, n)
tea_price = np.clip(tea_price, 2.0, 4.5)

# 3. Tea Export Volume (metric tons/month)
tea_volume = 18000 + 2000 * np.sin(np.linspace(0, 4 * np.pi, n)) + np.random.normal(0, 800, n)
tea_volume = np.clip(tea_volume, 12000, 25000)

# 4. Global Rubber Price (USD/kg)
rubber_price = 1.6 + 0.5 * np.sin(np.linspace(0, 3 * np.pi, n)) + np.random.normal(0, 0.1, n)
rubber_price = np.clip(rubber_price, 0.9, 2.8)

# 5. Rubber Export Volume (metric tons/month)
rubber_volume = 3500 + 500 * np.sin(np.linspace(0, 3 * np.pi, n)) + np.random.normal(0, 200, n)
rubber_volume = np.clip(rubber_volume, 1500, 5500)

# 6. Apparel Export Revenue (USD millions) — SL's largest export sector
apparel_revenue = 450 + 30 * np.sin(np.linspace(0, 6 * np.pi, n)) + np.linspace(0, 80, n)
apparel_revenue[96:108] -= np.linspace(0, 60, 12)  # crisis dip
apparel_revenue += np.random.normal(0, 15, n)
apparel_revenue = np.clip(apparel_revenue, 300, 620)

# 7. Inflation Rate (CPI %) — 2022 hyperinflation
inflation = 4 + np.random.normal(0, 1.5, n)
inflation[96:108] += np.linspace(0, 65, 12)
inflation[108:116] = np.linspace(65, 15, 8)
inflation[116:] = np.linspace(15, 6, n - 116)
inflation = np.clip(inflation, 2, 75)

# 8. Oil Import Cost (USD millions) — production cost proxy
oil_cost = 250 + 20 * np.sin(np.linspace(0, 4 * np.pi, n)) + np.random.normal(0, 20, n)
oil_cost[96:108] += 80
oil_cost = np.clip(oil_cost, 150, 420)

# 9. Tourism Earnings (USD millions) — COVID crash 2020-2021
tourism = 180 + 20 * np.sin(np.linspace(0, 6 * np.pi, n)) + np.random.normal(0, 15, n)
tourism[72:90] = np.random.uniform(5, 30, 18)   # COVID
tourism[90:96] = np.linspace(30, 160, 6)         # recovery
tourism = np.clip(tourism, 0, 280)

# --- Target Variable: Total Export Revenue (USD millions) ---
tea_revenue = tea_price * tea_volume / 1000
rubber_revenue = rubber_price * rubber_volume / 1000

export_revenue = (
    tea_revenue +
    rubber_revenue +
    apparel_revenue +
    0.15 * tourism +
    0.05 * exchange_rate +
    -0.05 * oil_cost +
    -1.2 * inflation +
    150 +
    np.random.normal(0, 20, n)
)

# --- Build DataFrame ---
df = pd.DataFrame({
    "date":                       months,
    "year":                       months.year,
    "month":                      months.month,
    "exchange_rate_lkr_usd":     exchange_rate.round(2),
    "tea_price_usd_per_kg":       tea_price.round(3),
    "tea_export_volume_mt":       tea_volume.round(0).astype(int),
    "rubber_price_usd_per_kg":    rubber_price.round(3),
    "rubber_export_volume_mt":    rubber_volume.round(0).astype(int),
    "apparel_revenue_usd_mn":     apparel_revenue.round(2),
    "tourism_earnings_usd_mn":    tourism.round(2),
    "inflation_rate_pct":         inflation.round(2),
    "oil_import_cost_usd_mn":     oil_cost.round(2),
    "total_export_revenue_usd_mn": export_revenue.round(2)
})

os.makedirs("data", exist_ok=True)
df.to_csv("data/srilanka_exports.csv", index=False)

print("=" * 55)
print("  Sri Lanka Export Revenue Dataset Generated")
print("=" * 55)
print(f"  Rows:    {len(df)}")
print(f"  Columns: {df.shape[1]}")
print(f"  Period:  {df['date'].min().date()} → {df['date'].max().date()}")
print()
print("Target Variable Summary:")
print(df["total_export_revenue_usd_mn"].describe().round(2))
print()
print("First 5 rows:")
print(df.head().to_string(index=False))
print()
print("Saved → data/srilanka_exports.csv")
