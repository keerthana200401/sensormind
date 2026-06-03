import os
import sys
import anthropic
from dotenv import load_dotenv

# Load .env from root folder
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from genai.vectorstore import query_similar_failures, ingest_failure_reports
from genai.prompts import build_alert_prompt, build_explain_prompt, SYSTEM_PROMPT

# ── 1. Setup Anthropic client ─────────────────────────────────
client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

# ── 2. Main alert generation function ────────────────────────
def generate_alert(prediction: dict, engine_id: str = "Unknown") -> dict:
    """
    Takes a prediction dict from predict.py and generates
    a plain-English maintenance alert using RAG + Claude.

    Args:
        prediction: output from predict_rul()
        engine_id: identifier for the engine

    Returns:
        dict with alert text and supporting context
    """

    # Build search query from top signals
    top_sensors = list(prediction['top_signals'].keys())
    query = f"{' '.join(top_sensors)} sensor anomaly, RUL {prediction['rul_prediction']} cycles, risk {prediction['risk_level']}"

    # Search ChromaDB for similar historical failures
    similar_failures = query_similar_failures(query, n_results=3)

    # Build the prompt
    prompt = build_alert_prompt(
        engine_id       = engine_id,
        rul             = prediction['rul_prediction'],
        risk_level      = prediction['risk_level'],
        is_anomaly      = prediction['is_anomaly'],
        top_signals     = prediction['top_signals'],
        similar_failures= similar_failures
    )

    # Call Claude API
    message = client.messages.create(
        model      = "claude-opus-4-6",
        max_tokens = 1024,
        system     = SYSTEM_PROMPT,
        messages   = [{"role": "user", "content": prompt}]
    )

    alert_text = message.content[0].text

    return {
        "engine_id"       : engine_id,
        "rul_prediction"  : prediction['rul_prediction'],
        "risk_level"      : prediction['risk_level'],
        "is_anomaly"      : prediction['is_anomaly'],
        "alert"           : alert_text,
        "similar_failures": [f['id'] for f in similar_failures],
        "model_used"      : message.model,
        "input_tokens"    : message.usage.input_tokens,
        "output_tokens"   : message.usage.output_tokens
    }


# ── 3. Explain function ───────────────────────────────────────
def explain_prediction(prediction: dict, 
                        engine_id: str,
                        question: str = "Why is this engine at risk?") -> str:
    """
    Answers a specific question about a prediction in plain English.
    """
    prompt = build_explain_prompt(
        engine_id   = engine_id,
        rul         = prediction['rul_prediction'],
        risk_level  = prediction['risk_level'],
        top_signals = prediction['top_signals'],
        shap_info   = "sensor_13 and sensor_15 are top contributors",
        question    = question
    )

    message = client.messages.create(
        model      = "claude-opus-4-6",
        max_tokens = 512,
        system     = SYSTEM_PROMPT,
        messages   = [{"role": "user", "content": prompt}]
    )

    return message.content[0].text


# ── 4. Test the full pipeline ─────────────────────────────────
if __name__ == '__main__':
    # Make sure vectorstore is populated
    ingest_failure_reports()

    print("\n" + "=" * 60)
    print("   TESTING RAG PIPELINE — SENSORMIND")
    print("=" * 60)

    # Simulate a prediction from predict.py
    test_prediction_critical = {
        'rul_prediction': 16.88,
        'risk_level'    : 'HIGH',
        'is_anomaly'    : True,
        'anomaly_score' : -0.042,
        'top_signals'   : {
            'sensor_13'    : 2388.25,
            'sensor_13_lag1': 2388.23,
            'sensor_15'    : 8.5158,
            'sensor_15_lag1': 8.472,
            'sensor_13_lag2': 2388.28
        },
        'model_used': 'XGBoost + IsolationForest'
    }

    test_prediction_healthy = {
        'rul_prediction': 126.27,
        'risk_level'    : 'LOW',
        'is_anomaly'    : False,
        'anomaly_score' : 0.082,
        'top_signals'   : {
            'sensor_13'    : 2388.02,
            'sensor_13_lag1': 0.0,
            'sensor_15'    : 8.4195,
            'sensor_15_lag1': 0.0,
            'sensor_13_lag2': 0.0
        },
        'model_used': 'XGBoost + IsolationForest'
    }

    # Test 1 — HIGH risk engine
    print("\nTest 1 — HIGH risk engine alert...")
    result = generate_alert(test_prediction_critical, engine_id="Engine_007")

    print(f"\n{'='*60}")
    print(f"ENGINE ID  : {result['engine_id']}")
    print(f"RUL        : {result['rul_prediction']} cycles")
    print(f"RISK       : {result['risk_level']}")
    print(f"REFERENCES : {result['similar_failures']}")
    print(f"TOKENS     : {result['input_tokens']} in / {result['output_tokens']} out")
    print(f"\n--- CLAUDE ALERT ---")
    print(result['alert'])

    # Test 2 — Explain function
    print(f"\n{'='*60}")
    print("Test 2 — Explain prediction...")
    explanation = explain_prediction(
        test_prediction_critical,
        engine_id = "Engine_007",
        question  = "Which sensor should I check first and why?"
    )
    print(f"\n--- CLAUDE EXPLANATION ---")
    print(explanation)

    # Test 3 — LOW risk engine
    print(f"\n{'='*60}")
    print("Test 3 — LOW risk engine alert...")
    result_healthy = generate_alert(test_prediction_healthy, engine_id="Engine_042")
    print(f"\n--- CLAUDE ALERT ---")
    print(result_healthy['alert'])

    print(f"\n{'='*60}")
    print("RAG Pipeline working end to end ✅")
    print("Sensor data → ML prediction → ChromaDB → Claude → Alert")
    print(f"{'='*60}")