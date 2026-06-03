import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import train_test_split
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load data & models ─────────────────────────────────────
print("Loading data and models...")
df = pd.read_parquet('../data/features/train_features.parquet')

feature_cols = joblib.load('saved/feature_cols.joblib')
scaler       = joblib.load('saved/scaler.joblib')
xgb_model    = joblib.load('saved/xgboost.joblib')
rf_model     = joblib.load('saved/random_forest.joblib')
iso_forest   = joblib.load('saved/isolation_forest.joblib')

X = df[feature_cols]
y = df['RUL_capped']

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_test_scaled = scaler.transform(X_test)

print(f"Test samples: {len(X_test)} ✅")

# ── 2. Evaluation function ────────────────────────────────────
def evaluate_model(name, y_true, y_pred, latency_ms):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    # Score function (NASA standard — penalizes late predictions more)
    diff = y_pred - y_true
    score = np.sum(
        np.where(diff < 0,
                 np.exp(-diff / 13) - 1,
                 np.exp(diff  / 10) - 1)
    )

    return {
        'Model'       : name,
        'RMSE'        : round(rmse, 4),
        'MAE'         : round(mae, 4),
        'R2 Score'    : round(r2, 4),
        'NASA Score'  : round(score, 2),
        'Latency (ms)': round(latency_ms, 2)
    }

results = []

# ── 3. Evaluate XGBoost ───────────────────────────────────────
start = time.time()
xgb_preds = xgb_model.predict(X_test)
xgb_latency = (time.time() - start) * 1000

results.append(evaluate_model('XGBoost', y_test, xgb_preds, xgb_latency))
print("XGBoost evaluated ✅")

# ── 4. Evaluate Random Forest ─────────────────────────────────
start = time.time()
rf_preds = rf_model.predict(X_test)
rf_latency = (time.time() - start) * 1000

results.append(evaluate_model('Random Forest', y_test, rf_preds, rf_latency))
print("Random Forest evaluated ✅")

# ── 5. Isolation Forest (anomaly score, not RUL prediction) ───
start = time.time()
iso_scores = iso_forest.decision_function(X_test_scaled)
iso_latency = (time.time() - start) * 1000

# Convert anomaly score to pseudo-RUL for comparison
iso_preds = (iso_scores - iso_scores.min()) / (
    iso_scores.max() - iso_scores.min()
) * 125

results.append(evaluate_model('Isolation Forest', y_test, iso_preds, iso_latency))
print("Isolation Forest evaluated ✅")

# ── 6. Print comparison table ─────────────────────────────────
results_df = pd.DataFrame(results)

print("\n")
print("=" * 70)
print("           MODEL COMPARISON TABLE — SENSORMIND")
print("=" * 70)
print(results_df.to_string(index=False))
print("=" * 70)

# ── 7. Winner ─────────────────────────────────────────────────
best = results_df.loc[results_df['RMSE'].idxmin(), 'Model']
best_rmse = results_df['RMSE'].min()
best_r2   = results_df.loc[results_df['RMSE'].idxmin(), 'R2 Score']

print(f"\n Best model : {best}")
print(f" RMSE       : {best_rmse}")
print(f" R2 Score   : {best_r2}")
print(f"\n '{best}' will be used for production predictions")

# ── 8. Save results ───────────────────────────────────────────
results_df.to_csv('saved/model_comparison.csv', index=False)
print("\nComparison table saved to models/saved/model_comparison.csv ✅")