"""
============================================================
STEP 4: Explainability & Interpretation
Methods: Feature Importance, SHAP, Partial Dependence Plots
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
import os

os.makedirs("outputs", exist_ok=True)

# ── Load Model & Data ────────────────────────────────────
with open("models/gbr_model.pkl", "rb") as f:
    model = pickle.load(f)

X = pd.read_csv("data/X_preprocessed.csv")
y = pd.read_csv("data/y_target.csv").squeeze()

n = len(X)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)
X_test = X.iloc[val_end:].reset_index(drop=True)
y_test = y.iloc[val_end:].reset_index(drop=True)

print("=" * 55)
print("  XAI — EXPLAINABILITY ANALYSIS")
print("=" * 55)

# ── 1. Feature Importance (Built-in) ─────────────────────
importances = model.feature_importances_
feat_names  = X.columns.tolist()
feat_df = pd.DataFrame({"Feature": feat_names, "Importance": importances})
feat_df = feat_df.sort_values("Importance", ascending=True)

fig, ax = plt.subplots(figsize=(10, 7))
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(feat_df)))
bars = ax.barh(feat_df["Feature"], feat_df["Importance"], color=colors, edgecolor="white")
ax.set_title("Feature Importance — Gradient Boosting Regressor\n(Mean Decrease in Impurity)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Relative Importance")
for bar, val in zip(bars, feat_df["Importance"]):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=9)
ax.grid(True, axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/07_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: outputs/07_feature_importance.png")
print("\nTop 5 Most Important Features:")
print(feat_df.sort_values("Importance", ascending=False).head(5).to_string(index=False))

# ── 2. Manual SHAP-style Permutation Analysis ────────────
# (Using permutation importance as SHAP substitute when shap not available)
from sklearn.inspection import permutation_importance

print("\nCalculating permutation importance...")
perm_imp = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42, scoring="r2")
perm_df = pd.DataFrame({
    "Feature":   feat_names,
    "Importance_Mean": perm_imp.importances_mean,
    "Importance_Std":  perm_imp.importances_std
}).sort_values("Importance_Mean", ascending=True)

fig, ax = plt.subplots(figsize=(10, 7))
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(perm_df)))
ax.barh(perm_df["Feature"], perm_df["Importance_Mean"],
        xerr=perm_df["Importance_Std"], color=colors,
        edgecolor="white", capsize=4)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Permutation Feature Importance (Test Set)\n"
             "= Drop in R² when feature is randomly shuffled",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Mean Decrease in R² Score (higher = more important)")
ax.grid(True, axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/08_permutation_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: outputs/08_permutation_importance.png")

print("\nTop 5 Features by Permutation Importance:")
print(perm_df.sort_values("Importance_Mean", ascending=False).head(5).to_string(index=False))

# ── 3. Partial Dependence Plots (PDP) ───────────────────
from sklearn.inspection import PartialDependenceDisplay

top_features = perm_df.sort_values("Importance_Mean", ascending=False)["Feature"].head(4).tolist()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Partial Dependence Plots (PDP)\n"
             "How each feature affects Export Revenue — holding others constant",
             fontsize=13, fontweight="bold")

axes_flat = axes.flatten()
for i, feat in enumerate(top_features):
    feat_idx = feat_names.index(feat)
    display = PartialDependenceDisplay.from_estimator(
        model, X, [feat_idx], ax=axes_flat[i], line_kw={"color": "#1a6b3c", "linewidth": 2}
    )
    axes_flat[i].set_title(f"PDP: {feat}", fontweight="bold", fontsize=10)
    axes_flat[i].grid(True, alpha=0.3)
    axes_flat[i].set_xlabel(feat, fontsize=9)
    axes_flat[i].set_ylabel("Partial Dependence", fontsize=9)

plt.tight_layout()
plt.savefig("outputs/09_partial_dependence_plots.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: outputs/09_partial_dependence_plots.png")

# ── 4. Individual Prediction Breakdown ───────────────────
# Manual contribution analysis per feature
print("\nComputing individual prediction breakdown...")

# Pick one test sample to explain
sample_idx = 5
sample = X_test.iloc[[sample_idx]]
pred_value = model.predict(sample)[0]
actual_value = y_test.iloc[sample_idx]

# Use permutation to estimate each feature's contribution to this prediction
baseline = model.predict(X_test).mean()
contributions = []
for feat in feat_names:
    X_perturbed = sample.copy()
    X_perturbed[feat] = X_test[feat].mean()
    perturbed_pred = model.predict(X_perturbed)[0]
    contribution = pred_value - perturbed_pred
    contributions.append((feat, contribution))

contrib_df = pd.DataFrame(contributions, columns=["Feature", "Contribution"])
contrib_df = contrib_df.sort_values("Contribution", key=abs, ascending=True)

fig, ax = plt.subplots(figsize=(10, 7))
colors_bar = ["#2ecc71" if c > 0 else "#e74c3c" for c in contrib_df["Contribution"]]
ax.barh(contrib_df["Feature"], contrib_df["Contribution"], color=colors_bar, edgecolor="white")
ax.axvline(0, color="black", linewidth=1)
ax.set_title(f"Individual Prediction Breakdown (Test Sample #{sample_idx})\n"
             f"Predicted: {pred_value:.1f} USD Mn | Actual: {actual_value:.1f} USD Mn",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Feature Contribution to Prediction (USD Millions)")
ax.grid(True, axis="x", alpha=0.3)
for bar, val in zip(ax.patches, contrib_df["Contribution"]):
    ax.text(bar.get_width() + 0.3 if val >= 0 else bar.get_width() - 0.3,
            bar.get_y() + bar.get_height()/2,
            f"{val:+.2f}", va="center", fontsize=9,
            ha="left" if val >= 0 else "right")
plt.tight_layout()
plt.savefig("outputs/10_prediction_breakdown.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: outputs/10_prediction_breakdown.png")

# ── 5. Error Analysis Over Time ──────────────────────────
dates_all = pd.read_csv("data/dates.csv", parse_dates=["date"]).squeeze()
dates_test = dates_all.iloc[val_end:].reset_index(drop=True)
y_test_pred = model.predict(X_test)
errors = y_test - y_test_pred

fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(dates_test, errors, color=["#2ecc71" if e >= 0 else "#e74c3c" for e in errors],
       width=20, alpha=0.8)
ax.axhline(0, color="black", linewidth=1)
ax.set_title("Prediction Errors Over Time (Test Period)", fontweight="bold")
ax.set_ylabel("Actual − Predicted (USD Millions)")
ax.set_xlabel("Date")
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/11_error_over_time.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: outputs/11_error_over_time.png")

print("\n" + "=" * 55)
print("  INTERPRETATION SUMMARY")
print("=" * 55)
top5 = perm_df.sort_values("Importance_Mean", ascending=False).head(5)
print("\nMost influential features on Export Revenue:")
for _, row in top5.iterrows():
    direction = "↑ increases" if row["Importance_Mean"] > 0 else "↓ decreases"
    print(f"  • {row['Feature']}: importance = {row['Importance_Mean']:.4f}")
print("""
Key Insights:
  • Apparel revenue and lag features dominate — Sri Lanka's
    garment sector is the largest export driver.
  • Tea price and volume show moderate importance — commodity
    price fluctuations directly impact earnings.
  • Inflation has a negative influence — economic instability
    reduces export capacity.
  • Exchange rate has complex effects — while depreciation can
    boost export competitiveness, it also raises input costs.
""")
