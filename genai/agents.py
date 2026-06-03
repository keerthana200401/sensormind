import os
import sys
import joblib
import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from genai.rag_pipeline import generate_alert, explain_prediction
from genai.vectorstore import query_similar_failures, ingest_failure_reports

# ── Load ML models ────────────────────────────────────────────
xgb_model    = joblib.load('../models/saved/xgboost.joblib')
iso_forest   = joblib.load('../models/saved/isolation_forest.joblib')
scaler       = joblib.load('../models/saved/scaler.joblib')
feature_cols = joblib.load('../models/saved/feature_cols.joblib')

# ── Agent 1: Analyst Agent ────────────────────────────────────
class AnalystAgent:
    """
    Runs ML prediction on engine sensor data.
    Identifies risk level and anomalies.
    Searches historical failure database.
    """
    def __init__(self):
        self.name = "ML Analyst Agent"

    def run(self, engine_id: str) -> dict:
        print(f"\n[{self.name}] Analyzing {engine_id}...")

        # Load sample data
        df = pd.read_parquet('../data/features/train_features.parquet')

        # Pick sample based on engine type
        if "critical" in engine_id.lower():
            sample = df[df['RUL_capped'] <= 15].iloc[0]
        elif "healthy" in engine_id.lower():
            sample = df[df['RUL_capped'] >= 120].iloc[0]
        else:
            sample = df.sample(1, random_state=42).iloc[0]

        # Run ML prediction
        input_df     = pd.DataFrame([sample[feature_cols]])
        rul_pred     = float(xgb_model.predict(input_df)[0])
        rul_pred     = max(0, round(rul_pred, 2))
        input_scaled = scaler.transform(input_df)
        is_anomaly   = iso_forest.predict(input_scaled)[0] == -1
        anomaly_score= float(iso_forest.decision_function(input_scaled)[0])

        # Risk level
        if rul_pred <= 15:
            risk = "CRITICAL"
        elif rul_pred <= 30:
            risk = "HIGH"
        elif rul_pred <= 60:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        # Top signals
        top_signals = {
            col: round(float(sample[col]), 4)
            for col in ['sensor_13', 'sensor_15', 'sensor_11']
            if col in sample.index
        }

        prediction = {
            'rul_prediction': rul_pred,
            'risk_level'    : risk,
            'is_anomaly'    : is_anomaly,
            'anomaly_score' : round(anomaly_score, 4),
            'top_signals'   : top_signals,
            'model_used'    : 'XGBoost + IsolationForest'
        }

        # Search similar failures
        query = f"sensor_13 sensor_15 sensor_11 RUL {rul_pred} risk {risk}"
        similar = query_similar_failures(query, n_results=3)

        print(f"[{self.name}] RUL: {rul_pred} cycles | Risk: {risk} | Anomaly: {is_anomaly}")
        print(f"[{self.name}] Similar failures found: {[r['id'] for r in similar]}")

        return {
            'engine_id' : engine_id,
            'prediction': prediction,
            'similar_failures': similar
        }


# ── Agent 2: Explainer Agent ──────────────────────────────────
class ExplainerAgent:
    """
    Takes ML prediction from Analyst Agent.
    Calls Claude API via RAG pipeline.
    Generates plain-English maintenance alert.
    """
    def __init__(self):
        self.name = "Explainer Agent"

    def run(self, analyst_output: dict) -> dict:
        engine_id  = analyst_output['engine_id']
        prediction = analyst_output['prediction']

        print(f"\n[{self.name}] Generating alert for {engine_id}...")

        # Generate alert via Claude API + RAG
        result = generate_alert(prediction, engine_id=engine_id)

        # Generate explanation
        explanation = explain_prediction(
            prediction,
            engine_id = engine_id,
            question  = "Which sensor should I check first and why?"
        )

        print(f"[{self.name}] Alert generated ✅")
        print(f"[{self.name}] Tokens used: {result['input_tokens']} in / {result['output_tokens']} out")

        return {
            'engine_id'  : engine_id,
            'risk_level' : prediction['risk_level'],
            'rul'        : prediction['rul_prediction'],
            'alert'      : result['alert'],
            'explanation': explanation,
            'references' : result['similar_failures']
        }


# ── Pipeline orchestrator ─────────────────────────────────────
class SensorMindPipeline:
    """
    Orchestrates Analyst + Explainer agents sequentially.
    This is the full end-to-end SensorMind pipeline.
    """
    def __init__(self):
        self.analyst  = AnalystAgent()
        self.explainer= ExplainerAgent()

    def run(self, engine_id: str) -> dict:
        print(f"\n{'='*60}")
        print(f"  SENSORMIND PIPELINE — {engine_id}")
        print(f"{'='*60}")

        # Step 1: Analyst runs ML
        analyst_output = self.analyst.run(engine_id)

        # Step 2: Explainer generates alert
        final_output = self.explainer.run(analyst_output)

        return final_output


# ── Test ──────────────────────────────────────────────────────
if __name__ == '__main__':
    ingest_failure_reports()

    pipeline = SensorMindPipeline()

    # Test 1 — Critical engine
    result_critical = pipeline.run("Engine_CRITICAL_007")

    print(f"\n{'='*60}")
    print("  FINAL MAINTENANCE ALERT")
    print(f"{'='*60}")
    print(f"Engine   : {result_critical['engine_id']}")
    print(f"RUL      : {result_critical['rul']} cycles")
    print(f"Risk     : {result_critical['risk_level']}")
    print(f"References: {result_critical['references']}")
    print(f"\n--- ALERT ---")
    print(result_critical['alert'])
    print(f"\n--- EXPLANATION ---")
    print(result_critical['explanation'])

    # Test 2 — Healthy engine
    result_healthy = pipeline.run("Engine_HEALTHY_042")

    print(f"\n{'='*60}")
    print(f"Engine   : {result_healthy['engine_id']}")
    print(f"RUL      : {result_healthy['rul']} cycles")
    print(f"Risk     : {result_healthy['risk_level']}")
    print(f"\n--- ALERT ---")
    print(result_healthy['alert'])

    print(f"\n{'='*60}")
    print("Week 3 Complete ✅")
    print("Sensor data → ML → ChromaDB → Claude → Alert")
    print(f"{'='*60}")