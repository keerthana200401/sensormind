import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List

from backend.schemas import HistoryResponse
from backend.db import crud

router = APIRouter()


def get_db():
    from backend.main import SessionLocal
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/history", response_model=List[HistoryResponse])
def get_history(limit: int = 50, db: Session = Depends(get_db)):
    return crud.get_predictions(db, limit=limit)


@router.get("/history/{engine_id}", response_model=List[HistoryResponse])
def get_engine_history(engine_id: str, db: Session = Depends(get_db)):
    return crud.get_predictions_by_engine(db, engine_id)


@router.get("/alerts")
def get_alerts(db: Session = Depends(get_db)):
    alerts = crud.get_alerts(db, limit=50)
    return [
        {
            "id"        : a.id,
            "engine_id" : a.engine_id,
            "alert_text": a.alert_text,
            "risk_level": a.risk_level,
            "created_at": a.created_at
        }
        for a in alerts
    ]


@router.get("/alerts/critical")
def get_critical_alerts(db: Session = Depends(get_db)):
    alerts = crud.get_critical_alerts(db)
    return [
        {
            "id"        : a.id,
            "engine_id" : a.engine_id,
            "alert_text": a.alert_text,
            "risk_level": a.risk_level,
            "created_at": a.created_at
        }
        for a in alerts
    ]