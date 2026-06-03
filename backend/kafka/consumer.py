import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import json
import pandas as pd
import joblib
from kafka import KafkaConsumer
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

from backend.db.models import Base
from backend.db.crud import save_prediction, save_sensor_reading

# ── Database setup ────────────────────────────────────────────
DB_URL = (
    f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)
engine       = create_engine(DB_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

# ── Load ML models ────────────────────────────────────────────
print("Loading ML models...")
xgb_model    = joblib.load('../../models/saved/xgboost.joblib')
iso_forest   = joblib.load('../../models/saved/isolation_forest.joblib')
scaler       = joblib.load('../../models/saved/scaler.joblib')
feature_cols = joblib.load('../../models/saved/feature_cols.joblib')

# Load feature data for filling missing cols
df_features  = pd.read_parquet('../../data/features/train_features.parquet')
print("Models loaded ✅")

# ── Setup consumer ────────────────────────────────────────────
consumer = KafkaConsumer(
    'sensor-stream',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    auto_offset_reset='latest',
    group_id='sensormind-consumer'
)

print("Consumer listening on topic: sensor-stream")
print("Waiting for messages...\n")

# ── Process messages ──────────────────────────────────────────
def process_message(message: dict) -> dict:
    """
    Takes a Kafka message, runs ML prediction,
    saves to DB and returns result.
    """
    engine_id = message.get('engine_id', 'Unknown')

    # Build feature vector — fill missing cols from sample
    sample = df_features.sample(1).iloc[0].copy()
    for key in ['sensor_13', 'sensor_15', 'sensor_11',
                'sensor_2',  'sensor_4']:
        if key in message:
            sample[key] = message[key]

    input_df      = pd.DataFrame([sample[feature_cols]])
    rul_pred      = float(xgb_model.predict(input_df)[0])
    rul_pred      = max(0, round(rul_pred, 2))
    input_scaled  = scaler.transform(input_df)
    is_anomaly    = iso_forest.predict(input_scaled)[0] == -1
    anomaly_score = float(iso_forest.decision_function(input_scaled)[0])

    if rul_pred <= 15:   risk = "CRITICAL"
    elif rul_pred <= 30: risk = "HIGH"
    elif rul_pred <= 60: risk = "MEDIUM"
    else:                risk = "LOW"

    top_signals = {
        'sensor_13': round(float(sample['sensor_13']), 4),
        'sensor_15': round(float(sample['sensor_15']), 4),
        'sensor_11': round(float(sample['sensor_11']), 4),
    }

    return {
        'engine_id'     : engine_id,
        'rul_prediction': rul_pred,
        'risk_level'    : risk,
        'is_anomaly'    : is_anomaly,
        'anomaly_score' : round(anomaly_score, 4),
        'top_signals'   : top_signals,
        'model_used'    : 'XGBoost + IsolationForest'
    }


# ── Main consumer loop ────────────────────────────────────────
for msg in consumer:
    message = msg.value

    try:
        result = process_message(message)
        db     = SessionLocal()

        # Save sensor reading
        save_sensor_reading(db, {
            'engine_id': message['engine_id'],
            'cycle'    : message.get('cycle', 0),
            'sensor_13': message.get('sensor_13'),
            'sensor_15': message.get('sensor_15'),
            'sensor_11': message.get('sensor_11'),
            'raw_data' : message
        })

        # Save prediction
        save_prediction(db, result)
        db.close()

        # Print alert if risky
        alert_icon = {
            'CRITICAL': '🔴',
            'HIGH'    : '🟠',
            'MEDIUM'  : '🟡',
            'LOW'     : '🟢'
        }.get(result['risk_level'], '⚪')

        print(f"{alert_icon} [{result['engine_id']}] "
              f"RUL: {result['rul_prediction']} cycles | "
              f"Risk: {result['risk_level']} | "
              f"Anomaly: {result['is_anomaly']}")

        # Trigger Claude alert for HIGH/CRITICAL
        if result['risk_level'] in ['CRITICAL', 'HIGH']:
            print(f"  ⚠️  HIGH RISK — triggering Claude alert...")
            from genai.rag_pipeline import generate_alert
            from genai.vectorstore import ingest_failure_reports
            ingest_failure_reports()
            alert = generate_alert(result, engine_id=result['engine_id'])
            print(f"  📢 ALERT: {alert['alert'][:150]}...")

    except Exception as e:
        print(f"Error processing message: {e}")