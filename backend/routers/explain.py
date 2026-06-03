import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
import joblib
import pandas as pd

from backend.schemas import SensorInput, ExplainInput, AlertResponse, ExplainResponse
from backend.db import crud

router = APIRouter()

xgb_model    = joblib.load(os.path.join(os.path.dirname(__file__), '../../models/saved/xgboost.joblib'))
iso_forest   = joblib.load(os.path.join(os.path.dirname(__file__), '../../models/saved/isolation_forest.joblib'))
scaler       = joblib.load(os.path.join(os.path.dirname(__file__), '../../models/saved/scaler.joblib'))
feature_cols = joblib.load(os.path.join(os.path.dirname(__file__), '../../models/saved/feature_cols.joblib'))


def get_db():
    from backend.main import SessionLocal
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/explain", response_model=AlertResponse)
def explain_engine(payload: SensorInput, db: Session = Depends(get_db)):
    from genai.rag_pipeline import generate_alert
    from genai.vectorstore import ingest_failure_reports

    ingest_failure_reports()

    df     = pd.read_parquet(
        os.path.join(os.path.dirname(__file__), '../../data/features/train_features.parquet')
    )
    sample = df.sample(1, random_state=42).iloc[0]

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

    result = generate_alert(prediction, engine_id=payload.engine_id)

    pred_data = {**prediction, 'engine_id': payload.engine_id}
    saved_pred = crud.save_prediction(db, pred_data)
    saved_alert = crud.save_alert(db, result, prediction_id=saved_pred.id)

    return AlertResponse(
        engine_id       = result['engine_id'],
        rul_prediction  = rul_pred,
        risk_level      = risk,
        alert           = result['alert'],
        similar_failures= result['similar_failures'],
        prediction_id   = saved_pred.id
    )


@router.post("/explain/question", response_model=ExplainResponse)
def explain_question(payload: ExplainInput, db: Session = Depends(get_db)):
    from genai.rag_pipeline import explain_prediction

    df     = pd.read_parquet(
        os.path.join(os.path.dirname(__file__), '../../data/features/train_features.parquet')
    )
    sample = df.sample(1, random_state=42).iloc[0]

    input_df   = pd.DataFrame([sample[feature_cols]])
    rul_pred   = float(xgb_model.predict(input_df)[0])
    top_signals= {
        col: round(float(sample[col]), 4)
        for col in ['sensor_13', 'sensor_15', 'sensor_11']
        if col in sample.index
    }

    if rul_pred <= 15:   risk = "CRITICAL"
    elif rul_pred <= 30: risk = "HIGH"
    elif rul_pred <= 60: risk = "MEDIUM"
    else:                risk = "LOW"

    prediction = {
        'rul_prediction': rul_pred,
        'risk_level'    : risk,
        'is_anomaly'    : False,
        'top_signals'   : top_signals
    }

    explanation = explain_prediction(
        prediction,
        engine_id = payload.engine_id,
        question  = payload.question
    )

    return ExplainResponse(
        engine_id  = payload.engine_id,
        question   = payload.question,
        explanation= explanation
    )