# ── Prompt templates for SensorMind Claude API calls ─────────

SYSTEM_PROMPT = """
You are SensorMind, an expert AI maintenance engineer specializing 
in industrial turbofan engines and oil & gas equipment.

Your job is to analyze sensor data and generate clear, actionable 
maintenance alerts for field engineers.

Rules:
- Always be specific about which sensors are concerning
- Give a clear risk level: CRITICAL / HIGH / MEDIUM / LOW
- Reference similar past failures when available
- Recommend a specific action with a timeframe
- Keep the alert under 150 words
- Use plain English — the reader is a field technician, not a data scientist
"""

ALERT_PROMPT = """
Sensor reading summary for Engine {engine_id}:
- Predicted RUL (Remaining Useful Life): {rul} cycles
- Risk Level: {risk_level}
- Anomaly Detected: {is_anomaly}
- Top sensor readings: {top_signals}

Similar historical failure reports found:
{similar_failures}

Based on this data and the historical failures above, generate a 
plain-English maintenance alert for the field engineer.

Include:
1. What is happening (sensor behaviour)
2. What it likely means (probable cause)
3. What to do and by when (recommended action)
"""

EXPLAIN_PROMPT = """
A maintenance engineer is asking about this prediction:

Engine ID    : {engine_id}
Predicted RUL: {rul} cycles  
Risk Level   : {risk_level}
Top sensors  : {top_signals}
SHAP values  : {shap_info}

Question from engineer: {question}

Answer in plain English, referencing the specific sensor readings.
Keep it under 100 words.
"""

def build_alert_prompt(engine_id, rul, risk_level, 
                        is_anomaly, top_signals, similar_failures):
    # Format similar failures as readable text
    failure_text = ""
    for i, f in enumerate(similar_failures, 1):
        failure_text += f"""
    Report {i} [{f['id']}]:
    - Failure type : {f['metadata']['failure_type']}
    - Severity     : {f['metadata']['severity']}
    - Sensor       : {f['metadata']['sensor']}
    - RUL when caught: {f['metadata']['rul_at_detection']} cycles
    - Summary      : {f['text'][:200]}...
    """

    return ALERT_PROMPT.format(
        engine_id       = engine_id,
        rul             = rul,
        risk_level      = risk_level,
        is_anomaly      = is_anomaly,
        top_signals     = top_signals,
        similar_failures= failure_text
    )


def build_explain_prompt(engine_id, rul, risk_level, 
                          top_signals, shap_info, question):
    return EXPLAIN_PROMPT.format(
        engine_id  = engine_id,
        rul        = rul,
        risk_level = risk_level,
        top_signals= top_signals,
        shap_info  = shap_info,
        question   = question
    )


if __name__ == '__main__':
    # Quick test
    test_prompt = build_alert_prompt(
        engine_id   = 'Engine_007',
        rul         = 16,
        risk_level  = 'HIGH',
        is_anomaly  = True,
        top_signals = {'sensor_13': 2388.02, 'sensor_15': 8.42},
        similar_failures = [{
            'id': 'FR001',
            'metadata': {
                'failure_type'    : 'turbine_blade_fouling',
                'severity'        : 'CRITICAL',
                'sensor'          : 'sensor_13',
                'rul_at_detection': 18
            },
            'text': 'Sensor 13 exhaust gas temperature showed gradual rise...'
        }]
    )
    print("Prompt built successfully ✅")
    print("\nSample prompt preview:")
    print(test_prompt[:300], "...")