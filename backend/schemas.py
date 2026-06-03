from pydantic import BaseModel
from typing import Optional, Dict, List
from datetime import datetime


# ── Request schemas ───────────────────────────────────────────
class SensorInput(BaseModel):
    engine_id: str
    sensor_data: Optional[Dict[str, float]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "engine_id": "Engine_007",
                "sensor_data": None
            }
        }


class ExplainInput(BaseModel):
    engine_id : str
    question  : Optional[str] = "Why is this engine at risk?"


# ── Response schemas ──────────────────────────────────────────
class PredictionResponse(BaseModel):
    engine_id      : str
    rul_prediction : float
    risk_level     : str
    is_anomaly     : bool
    anomaly_score  : float
    top_signals    : Dict[str, float]
    model_used     : str
    prediction_id  : Optional[int] = None

    class Config:
        from_attributes = True


class AlertResponse(BaseModel):
    engine_id       : str
    rul_prediction  : float
    risk_level      : str
    alert           : str
    similar_failures: List[str]
    prediction_id   : Optional[int] = None

    class Config:
        from_attributes = True


class ExplainResponse(BaseModel):
    engine_id  : str
    question   : str
    explanation: str

    class Config:
        from_attributes = True


class HistoryResponse(BaseModel):
    id            : int
    engine_id     : str
    rul_prediction: float
    risk_level    : str
    is_anomaly    : bool
    created_at    : datetime

    class Config:
        from_attributes = True