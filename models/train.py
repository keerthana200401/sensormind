import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load features ──────────────────────────────────────────
print("Loading features...")
df = pd.read_parquet('../data/features/train_features.parquet')

# ── 2. Define feature columns ─────────────────────────────────
exclude = ['engine_id', 'cycle', 'subset', 'RUL', 'RUL_capped']
feature_cols = [c for c in df.columns if c not in exclude]

X = df[feature_cols]
y = df['RUL_capped']  # regression target

# ── 3. Train/test split ───────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features (needed for Isolation Forest)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"Train size : {X_train.shape}")
print(f"Test size  : {X_test.shape}")
print(f"Features   : {len(feature_cols)}")

# ── 4. Model 1: Isolation Forest ──────────────────────────────
print("\nTraining Isolation Forest...")
iso_forest = IsolationForest(
    n_estimators=100,
    contamination=0.05,  # assume 5% anomalies
    random_state=42,
    n_jobs=-1
)
iso_forest.fit(X_train_scaled)
print("Isolation Forest trained ✅")

# ── 5. Model 2: XGBoost ───────────────────────────────────────
print("\nTraining XGBoost...")
xgb_model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50
)
print("XGBoost trained ✅")

# ── 6. Model 3: Random Forest ─────────────────────────────────
print("\nTraining Random Forest...")
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
print("Random Forest trained ✅")

# ── 7. Save all models ────────────────────────────────────────
os.makedirs('saved', exist_ok=True)

joblib.dump(iso_forest, 'saved/isolation_forest.joblib')
joblib.dump(xgb_model,  'saved/xgboost.joblib')
joblib.dump(rf_model,   'saved/random_forest.joblib')
joblib.dump(scaler,     'saved/scaler.joblib')
joblib.dump(feature_cols, 'saved/feature_cols.joblib')

print("\n All models saved to models/saved/ ✅")
print("isolation_forest.joblib")
print("xgboost.joblib")
print("random_forest.joblib")
print("scaler.joblib")
print("feature_cols.joblib")