"""
Generate realistic Sri Lankan Export Revenue dataset (2014–2024)
Based on patterns from Central Bank of Sri Lanka statistical tables.
Replace this with real data from cbsl.gov.lk for actual submission.
"""

import pandas as pd
import numpy as np

np.random.seed(42)

months = pd.date_range(start="2014-01-01", end="2024-12-01", freq="MS")
n = len(months)

# USD/LKR Exchange Rate (gradually depreciating, crisis spike in 2022)
exchange_rate = np.linspace(130, 200, n)
exchange_rate[96:108] += np.linspace(0, 170, 12)   # 2022 crisis spike
exchange_rate[108:] = np.linspace(360, 310, n - 108)
exchange_rate += np.random.normal(0, 3, n)

# Global Tea Price (USD per kg) — Sri Lanka is world's 4th largest exporter
tea_price = 2.8 + 0.3 * np.sin(np.linspace(0, 4 * np.pi, n)) + np.random.normal(0, 0.15, n)
tea_price = np.clip(tea_price, 2.0, 4.5)

# Tea Export Volume (metric tons per month)
tea_volume = 18000 + 2000 * np.sin(np.linspace(0, 4 * np.pi, n)) + np.random.normal(0, 800, n)
tea_volume = np.clip(tea_volume, 12000, 25000)

# Rubber Price (USD per kg)
rubber_price = 1.6 + 0.5 * np.sin(np.linspace(0, 3 * np.pi, n)) + np.random.normal(0, 0.1, n)
rubber_price = np.clip(rubber_price, 0.9, 2.8)

# Rubber Export Volume (metric tons)
rubber_volume = 3500 + 500 * np.sin(np.linspace(0, 3 * np.pi, n)) + np.random.normal(0, 200, n)
rubber_volume = np.clip(rubber_volume, 1500, 5500)

# Apparel Export Revenue (USD millions) — largest export category
apparel_revenue = 450 + 30 * np.sin(np.linspace(0, 6 * np.pi, n)) + np.linspace(0, 80, n)
apparel_revenue[96:108] -= np.linspace(0, 60, 12)   # crisis dip
apparel_revenue += np.random.normal(0, 15, n)
apparel_revenue = np.clip(apparel_revenue, 300, 620)

# Inflation Rate (CPI %)
inflation = 4 + np.random.normal(0, 1.5, n)
inflation[96:108] += np.linspace(0, 65, 12)   # 2022 hyperinflation
inflation[108:116] = np.linspace(65, 15, 8)
inflation[116:] = np.linspace(15, 6, n - 116)
inflation = np.clip(inflation, 2, 75)

# Oil Import Cost (USD millions) — affects production costs
oil_cost = 250 + 20 * np.sin(np.linspace(0, 4 * np.pi, n)) + np.random.normal(0, 20, n)
oil_cost[96:108] += 80   # oil price surge 2022
oil_cost = np.clip(oil_cost, 150, 420)

# Tourism Earnings (USD millions) — COVID dip 2020-2021
tourism = 180 + 20 * np.sin(np.linspace(0, 6 * np.pi, n)) + np.random.normal(0, 15, n)
tourism[72:90] = np.random.uniform(5, 30, 18)    # COVID crash
tourism[90:96] = np.linspace(30, 160, 6)          # recovery
tourism = np.clip(tourism, 0, 280)

# Month (seasonality feature)
month_num = months.month

# TARGET: Total Export Revenue (USD millions)
# Constructed from components + noise to mimic real relationships
export_revenue = (
    (tea_price * tea_volume / 1000) +
    (rubber_price * rubber_volume / 1000) +
    apparel_revenue +
    0.15 * tourism +
    0.3 * exchange_rate +
    -0.2 * oil_cost +
    -0.8 * inflation +
    np.random.normal(0, 20, n)
)
export_revenue = np.clip(export_revenue, 700, 1400)

df = pd.DataFrame({
    "date": months,
    "year": months.year,
    "month": month_num,
    "exchange_rate_lkr_usd": exchange_rate.round(2),
    "tea_price_usd_per_kg": tea_price.round(3),
    "tea_export_volume_mt": tea_volume.round(0).astype(int),
    "rubber_price_usd_per_kg": rubber_price.round(3),
    "rubber_export_volume_mt": rubber_volume.round(0).astype(int),
    "apparel_revenue_usd_mn": apparel_revenue.round(2),
    "tourism_earnings_usd_mn": tourism.round(2),
    "inflation_rate_pct": inflation.round(2),
    "oil_import_cost_usd_mn": oil_cost.round(2),
    "total_export_revenue_usd_mn": export_revenue.round(2)
})

df.to_csv("/home/claude/export_revenue_prediction/data/srilanka_exports.csv", index=False)
print(f"Dataset created: {len(df)} rows, {df.shape[1]} columns")
print(df.head())
print("\nBasic stats:")
print(df["total_export_revenue_usd_mn"].describe())
