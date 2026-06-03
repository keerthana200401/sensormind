import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load data & model ──────────────────────────────────────
print("Loading model and data...")
df = pd.read_parquet('../data/features/train_features.parquet')

feature_cols = joblib.load('saved/feature_cols.joblib')
xgb_model    = joblib.load('saved/xgboost.joblib')

X = df[feature_cols]
y = df['RUL_capped']

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Use a sample for SHAP (faster)
X_sample = X_test.sample(500, random_state=42)
print(f"Running SHAP on {len(X_sample)} samples ✅")

# ── 2. Compute SHAP values ────────────────────────────────────
print("Computing SHAP values (takes ~1 min)...")
explainer   = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_sample)
print("SHAP values computed ✅")

# ── 3. Plot 1 — Feature importance (bar chart) ────────────────
plt.figure(figsize=(12, 8))
shap.summary_plot(
    shap_values,
    X_sample,
    plot_type='bar',
    max_display=20,
    show=False
)
plt.title('Top 20 Most Important Features — XGBoost (SHAP)',
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('../data/processed/plot14_shap_importance.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Plot 14 saved ✅")

# ── 4. Plot 2 — SHAP beeswarm (impact direction) ─────────────
plt.figure(figsize=(12, 8))
shap.summary_plot(
    shap_values,
    X_sample,
    max_display=20,
    show=False
)
plt.title('SHAP Beeswarm — Feature Impact on RUL Prediction',
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('../data/processed/plot15_shap_beeswarm.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Plot 15 saved ✅")

# ── 5. Plot 3 — Single engine explanation ─────────────────────
print("\nGenerating single prediction explanation...")
single_sample = X_sample.iloc[[0]]
single_shap   = shap_values[0]

# Top 10 features for this one prediction
feature_impact = pd.DataFrame({
    'feature': feature_cols,
    'shap_value': single_shap
}).reindex(columns=['feature', 'shap_value'])

top10 = feature_impact.reindex(
    feature_impact['shap_value'].abs().nlargest(10).index
)

colors = ['red' if x < 0 else 'steelblue' for x in top10['shap_value']]

plt.figure(figsize=(12, 6))
plt.barh(top10['feature'], top10['shap_value'], color=colors, edgecolor='white')
plt.axvline(x=0, color='black', linewidth=0.8, linestyle='--')
plt.title('Single Engine Prediction Explanation\nRed = pushes RUL down (worse), Blue = pushes RUL up (better)',
          fontsize=12, fontweight='bold')
plt.xlabel('SHAP Value (impact on RUL prediction)')
plt.tight_layout()
plt.savefig('../data/processed/plot16_shap_single_engine.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Plot 16 saved ✅")

# ── 6. Print top sensors in plain English ─────────────────────
mean_shap = pd.DataFrame({
    'feature'   : feature_cols,
    'importance': np.abs(shap_values).mean(axis=0)
}).sort_values('importance', ascending=False)

print("\n" + "=" * 50)
print("   TOP 10 MOST CRITICAL SENSORS — SENSORMIND")
print("=" * 50)
for i, row in mean_shap.head(10).iterrows():
    print(f"  {row['feature']:<35} {row['importance']:.4f}")
print("=" * 50)
print("\nThese are the sensors the AI watches most closely")
print("to predict when an engine will fail.")

# Save SHAP importance
mean_shap.to_csv('saved/shap_importance.csv', index=False)
print("\nSHAP importance saved to models/saved/shap_importance.csv ✅")