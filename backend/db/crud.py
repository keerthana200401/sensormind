from sqlalchemy.orm import Session
from backend.db.models import Prediction, Alert, SensorStream


# ── Predictions ───────────────────────────────────────────────
def save_prediction(db: Session, prediction_data: dict) -> Prediction:
    pred = Prediction(
        engine_id      = prediction_data.get('engine_id'),
        rul_prediction = prediction_data.get('rul_prediction'),
        risk_level     = prediction_data.get('risk_level'),
        is_anomaly     = prediction_data.get('is_anomaly'),
        anomaly_score  = prediction_data.get('anomaly_score'),
        sensor_13      = prediction_data.get('top_signals', {}).get('sensor_13'),
        sensor_15      = prediction_data.get('top_signals', {}).get('sensor_15'),
        sensor_11      = prediction_data.get('top_signals', {}).get('sensor_11'),
        model_used     = prediction_data.get('model_used')
    )
    db.add(pred)
    db.commit()
    db.refresh(pred)
    return pred


def get_predictions(db: Session, limit: int = 50) -> list:
    return db.query(Prediction)\
             .order_by(Prediction.created_at.desc())\
             .limit(limit).all()


def get_predictions_by_engine(db: Session, engine_id: str) -> list:
    return db.query(Prediction)\
             .filter(Prediction.engine_id == engine_id)\
             .order_by(Prediction.created_at.desc()).all()


# ── Alerts ────────────────────────────────────────────────────
def save_alert(db: Session, alert_data: dict,
               prediction_id: int = None) -> Alert:
    alert = Alert(
        engine_id       = alert_data.get('engine_id'),
        prediction_id   = prediction_id,
        alert_text      = alert_data.get('alert'),
        risk_level      = alert_data.get('risk_level'),
        similar_failures= str(alert_data.get('similar_failures', [])),
        input_tokens    = alert_data.get('input_tokens'),
        output_tokens   = alert_data.get('output_tokens')
    )
    db.add(alert)
    db.commit()
    db.refresh(alert)
    return alert


def get_alerts(db: Session, limit: int = 50) -> list:
    return db.query(Alert)\
             .order_by(Alert.created_at.desc())\
             .limit(limit).all()


def get_critical_alerts(db: Session) -> list:
    return db.query(Alert)\
             .filter(Alert.risk_level.in_(['CRITICAL', 'HIGH']))\
             .order_by(Alert.created_at.desc()).all()


# ── Sensor stream ─────────────────────────────────────────────
def save_sensor_reading(db: Session, reading: dict) -> SensorStream:
    stream = SensorStream(
        engine_id = reading.get('engine_id'),
        cycle     = reading.get('cycle'),
        sensor_13 = reading.get('sensor_13'),
        sensor_15 = reading.get('sensor_15'),
        sensor_11 = reading.get('sensor_11'),
        raw_data  = reading.get('raw_data', {})
    )
    db.add(stream)
    db.commit()
    db.refresh(stream)
    return stream