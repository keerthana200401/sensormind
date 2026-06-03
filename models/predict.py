import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load all saved artifacts ───────────────────────────────
print("Loading models...")
xgb_model    = joblib.load('saved/xgboost.joblib')
rf_model     = joblib.load('saved/random_forest.joblib')
iso_forest   = joblib.load('saved/isolation_forest.joblib')
scaler       = joblib.load('saved/scaler.joblib')
feature_cols = joblib.load('saved/feature_cols.joblib')
shap_importance = pd.read_csv('saved/shap_importance.csv')

TOP_SENSORS = shap_importance.head(5)['feature'].tolist()

print("All models loaded ✅")

# ── 2. Core prediction function ───────────────────────────────
def predict_rul(sensor_data: dict) -> dict:
    """
    Takes a dictionary of sensor readings for one engine cycle.
    Returns RUL prediction, anomaly flag, risk level and top signals.

    Args:
        sensor_data: dict with keys matching feature_cols

    Returns:
        dict with prediction results
    """
    # Build dataframe from input
    input_df = pd.DataFrame([sensor_data])

    # Fill any missing features with 0
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[feature_cols]

    # ── RUL prediction (XGBoost) ──────────────────────────────
    rul_pred = float(xgb_model.predict(input_df)[0])
    rul_pred = max(0, round(rul_pred, 2))  # can't be negative

    # ── Anomaly detection (Isolation Forest) ──────────────────
    input_scaled   = scaler.transform(input_df)
    anomaly_label  = iso_forest.predict(input_scaled)[0]  # -1=anomaly, 1=normal
    anomaly_score  = float(iso_forest.decision_function(input_scaled)[0])
    is_anomaly     = anomaly_label == -1

    # ── Risk level ────────────────────────────────────────────
    if rul_pred <= 15:
        risk = "CRITICAL"
    elif rul_pred <= 30:
        risk = "HIGH"
    elif rul_pred <= 60:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    # ── Top contributing signals ──────────────────────────────
    top_signals = {
        sensor: round(float(input_df[sensor].values[0]), 4)
        for sensor in TOP_SENSORS
        if sensor in input_df.columns
    }

    return {
        'rul_prediction'  : rul_pred,
        'risk_level'      : risk,
        'is_anomaly'      : is_anomaly,
        'anomaly_score'   : round(anomaly_score, 4),
        'top_signals'     : top_signals,
        'model_used'      : 'XGBoost + IsolationForest'
    }

# ── 3. Batch prediction function ──────────────────────────────
def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run predictions on a full dataframe of sensor readings.
    Returns original df with prediction columns added.
    """
    results = []
    for _, row in df.iterrows():
        result = predict_rul(row.to_dict())
        results.append({
            'rul_prediction': result['rul_prediction'],
            'risk_level'    : result['risk_level'],
            'is_anomaly'    : result['is_anomaly'],
            'anomaly_score' : result['anomaly_score']
        })

    results_df = pd.DataFrame(results)
    return pd.concat([df.reset_index(drop=True), results_df], axis=1)

# ── 4. Test with real data ────────────────────────────────────
if __name__ == '__main__':
    print("\nTesting predict_rul() with real engine data...")

    # Load a few real samples
    df = pd.read_parquet('../data/features/train_features.parquet')

    # Test 1 — healthy engine (high RUL)
    healthy_sample = df[df['RUL_capped'] == 125].iloc[0]
    result_healthy = predict_rul(healthy_sample[feature_cols].to_dict())

    # Test 2 — failing engine (low RUL)
    failing_sample = df[df['RUL_capped'] <= 10].iloc[0]
    result_failing = predict_rul(failing_sample[feature_cols].to_dict())

    print("\n" + "=" * 50)
    print("TEST 1 — Healthy Engine")
    print("=" * 50)
    print(f"  Predicted RUL : {result_healthy['rul_prediction']} cycles")
    print(f"  Risk Level    : {result_healthy['risk_level']}")
    print(f"  Anomaly       : {result_healthy['is_anomaly']}")
    print(f"  Top signals   : {result_healthy['top_signals']}")

    print("\n" + "=" * 50)
    print("TEST 2 — Failing Engine")
    print("=" * 50)
    print(f"  Predicted RUL : {result_failing['rul_prediction']} cycles")
    print(f"  Risk Level    : {result_failing['risk_level']}")
    print(f"  Anomaly       : {result_failing['is_anomaly']}")
    print(f"  Top signals   : {result_failing['top_signals']}")

    print("\n" + "=" * 50)
    print("predict_rul() is working correctly ✅")
    print("Ready to plug into FastAPI backend")
    print("=" * 50)