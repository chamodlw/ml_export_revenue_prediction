"""
============================================================
STEP 3: Model Training & Evaluation
Algorithm: Gradient Boosting Regressor
(Same family as XGBoost — not covered in standard lectures)
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import pickle
import os

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ── Load Preprocessed Data ───────────────────────────────
X = pd.read_csv("data/X_preprocessed.csv")
y = pd.read_csv("data/y_target.csv").squeeze()
dates = pd.read_csv("data/dates.csv", parse_dates=["date"]).squeeze()

print("=" * 55)
print("  MODEL TRAINING: Gradient Boosting Regressor")
print("=" * 55)
print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

# ── Train / Validation / Test Split (time-based) ─────────
# 70% train | 15% validation | 15% test — respects time order
n = len(X)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)

X_train = X.iloc[:train_end]
X_val   = X.iloc[train_end:val_end]
X_test  = X.iloc[val_end:]

y_train = y.iloc[:train_end]
y_val   = y.iloc[train_end:val_end]
y_test  = y.iloc[val_end:]

dates_train = dates.iloc[:train_end]
dates_val   = dates.iloc[train_end:val_end]
dates_test  = dates.iloc[val_end:]

print(f"\nTrain:      {len(X_train)} samples ({dates_train.iloc[0].date()} → {dates_train.iloc[-1].date()})")
print(f"Validation: {len(X_val)} samples  ({dates_val.iloc[0].date()} → {dates_val.iloc[-1].date()})")
print(f"Test:       {len(X_test)} samples  ({dates_test.iloc[0].date()} → {dates_test.iloc[-1].date()})")

# ── Hyperparameter Tuning (TimeSeriesSplit CV) ───────────
print("\nRunning hyperparameter search...")

param_grid = {
    "n_estimators":    [100, 200, 300],
    "max_depth":       [3, 4, 5],
    "learning_rate":   [0.05, 0.1, 0.2],
    "subsample":       [0.8, 1.0],
    "min_samples_leaf":[3, 5],
}

tscv = TimeSeriesSplit(n_splits=5)
gbr = GradientBoostingRegressor(random_state=42)

grid_search = GridSearchCV(
    gbr, param_grid,
    cv=tscv,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    verbose=0
)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print(f"\nBest Parameters Found:")
for k, v in best_params.items():
    print(f"  {k}: {v}")

# ── Train Final Model ────────────────────────────────────
model = GradientBoostingRegressor(**best_params, random_state=42)
model.fit(X_train, y_train)

# ── Evaluation Function ──────────────────────────────────
def evaluate(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"\n  {name}:")
    print(f"    RMSE : {rmse:.2f} USD mn")
    print(f"    MAE  : {mae:.2f} USD mn")
    print(f"    R²   : {r2:.4f}")
    print(f"    MAPE : {mape:.2f}%")
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}

print("\n" + "=" * 55)
print("  EVALUATION RESULTS")
print("=" * 55)

y_train_pred = model.predict(X_train)
y_val_pred   = model.predict(X_val)
y_test_pred  = model.predict(X_test)

metrics_train = evaluate("Training Set",    y_train, y_train_pred)
metrics_val   = evaluate("Validation Set",  y_val,   y_val_pred)
metrics_test  = evaluate("Test Set",        y_test,  y_test_pred)

# ── Results Table ────────────────────────────────────────
results_df = pd.DataFrame({
    "Split":  ["Train", "Validation", "Test"],
    "RMSE":   [metrics_train["RMSE"],  metrics_val["RMSE"],  metrics_test["RMSE"]],
    "MAE":    [metrics_train["MAE"],   metrics_val["MAE"],   metrics_test["MAE"]],
    "R²":     [metrics_train["R2"],    metrics_val["R2"],    metrics_test["R2"]],
    "MAPE %": [metrics_train["MAPE"],  metrics_val["MAPE"],  metrics_test["MAPE"]],
}).round(3)

print("\n  Summary Table:")
print(results_df.to_string(index=False))
results_df.to_csv("outputs/evaluation_metrics.csv", index=False)

# ── Plot 1: Actual vs Predicted ──────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle("Gradient Boosting Regressor — Actual vs Predicted\nSri Lanka Export Revenue (USD Millions)",
             fontsize=13, fontweight="bold")

# Full timeline
all_dates  = pd.concat([dates_train, dates_val, dates_test]).reset_index(drop=True)
all_actual = pd.concat([y_train, y_val, y_test]).reset_index(drop=True)
all_preds  = np.concatenate([y_train_pred, y_val_pred, y_test_pred])

axes[0].plot(all_dates, all_actual, label="Actual", color="#1a6b3c", linewidth=2)
axes[0].plot(dates_train, y_train_pred, label="Train Prediction", color="#4dabf7", linewidth=1.5, linestyle="--")
axes[0].plot(dates_val,   y_val_pred,   label="Val Prediction",   color="#ff922b", linewidth=1.5, linestyle="--")
axes[0].plot(dates_test,  y_test_pred,  label="Test Prediction",  color="#f03e3e", linewidth=2, linestyle="--")
axes[0].axvline(dates_val.iloc[0],  color="#999", linestyle=":", linewidth=1.5, label="Val/Test split")
axes[0].axvline(dates_test.iloc[0], color="#999", linestyle=":",  linewidth=1.5)
axes[0].set_title("Full Timeline — Actual vs Predicted", fontweight="bold")
axes[0].set_ylabel("USD Millions")
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

# Test set scatter
axes[1].scatter(y_test, y_test_pred, alpha=0.7, color="#1a6b3c", edgecolor="white", s=60)
min_val = min(y_test.min(), y_test_pred.min()) - 10
max_val = max(y_test.max(), y_test_pred.max()) + 10
axes[1].plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Prediction")
axes[1].set_title(f"Test Set — Actual vs Predicted Scatter (R²={metrics_test['R2']:.3f})", fontweight="bold")
axes[1].set_xlabel("Actual Export Revenue (USD Millions)")
axes[1].set_ylabel("Predicted Export Revenue (USD Millions)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/04_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: outputs/04_actual_vs_predicted.png")

# ── Plot 2: Residuals Analysis ───────────────────────────
residuals = y_test - y_test_pred

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Residual Analysis — Test Set", fontsize=13, fontweight="bold")

axes[0].scatter(y_test_pred, residuals, alpha=0.7, color="#5c6bc0", edgecolor="white", s=60)
axes[0].axhline(0, color="red", linestyle="--", linewidth=2)
axes[0].set_title("Residuals vs Predicted")
axes[0].set_xlabel("Predicted Values")
axes[0].set_ylabel("Residuals (Actual − Predicted)")
axes[0].grid(True, alpha=0.3)

axes[1].hist(residuals, bins=15, color="#5c6bc0", edgecolor="white", alpha=0.8)
axes[1].axvline(0, color="red", linestyle="--", linewidth=2)
axes[1].set_title("Residual Distribution")
axes[1].set_xlabel("Residual Value")
axes[1].set_ylabel("Frequency")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/05_residuals.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: outputs/05_residuals.png")

# ── Plot 3: Metrics Comparison Bar ──────────────────────
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
fig.suptitle("Model Performance Across Splits", fontsize=13, fontweight="bold")

splits  = ["Train", "Validation", "Test"]
colors  = ["#4dabf7", "#ff922b", "#f03e3e"]
metrics = {
    "RMSE (USD Mn)": [metrics_train["RMSE"], metrics_val["RMSE"], metrics_test["RMSE"]],
    "MAE (USD Mn)":  [metrics_train["MAE"],  metrics_val["MAE"],  metrics_test["MAE"]],
    "R² Score":      [metrics_train["R2"],   metrics_val["R2"],   metrics_test["R2"]],
}

for ax, (title, vals) in zip(axes, metrics.items()):
    bars = ax.bar(splits, vals, color=colors, edgecolor="white", width=0.5)
    ax.set_title(title, fontweight="bold")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002 * max(vals),
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(vals) * 1.2)

plt.tight_layout()
plt.savefig("outputs/06_metrics_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: outputs/06_metrics_comparison.png")

# ── Save Model ───────────────────────────────────────────
with open("models/gbr_model.pkl", "wb") as f:
    pickle.dump(model, f)

model_info = {
    "algorithm": "Gradient Boosting Regressor (sklearn)",
    "best_params": best_params,
    "features": list(X.columns),
    "metrics": {
        "train": metrics_train,
        "val":   metrics_val,
        "test":  metrics_test
    }
}
pd.DataFrame([model_info]).to_json("models/model_info.json", orient="records", indent=2)

print("\nModel saved → models/gbr_model.pkl")
print("\n✓ Training complete!")
