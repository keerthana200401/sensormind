from sqlalchemy import (
    Column, Integer, Float, String,
    Boolean, Text, JSON, TIMESTAMP
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()

class Prediction(Base):
    __tablename__ = "predictions"

    id             = Column(Integer, primary_key=True, index=True)
    engine_id      = Column(String(50), nullable=False)
    rul_prediction = Column(Float, nullable=False)
    risk_level     = Column(String(20), nullable=False)
    is_anomaly     = Column(Boolean, nullable=False)
    anomaly_score  = Column(Float)
    sensor_13      = Column(Float)
    sensor_15      = Column(Float)
    sensor_11      = Column(Float)
    model_used     = Column(String(50))
    created_at     = Column(TIMESTAMP, server_default=func.now())


class Alert(Base):
    __tablename__ = "alerts"

    id              = Column(Integer, primary_key=True, index=True)
    engine_id       = Column(String(50), nullable=False)
    prediction_id   = Column(Integer)
    alert_text      = Column(Text, nullable=False)
    risk_level      = Column(String(20), nullable=False)
    similar_failures= Column(String(200))
    input_tokens    = Column(Integer)
    output_tokens   = Column(Integer)
    created_at      = Column(TIMESTAMP, server_default=func.now())


class SensorStream(Base):
    __tablename__ = "sensor_stream"

    id         = Column(Integer, primary_key=True, index=True)
    engine_id  = Column(String(50), nullable=False)
    cycle      = Column(Integer, nullable=False)
    sensor_13  = Column(Float)
    sensor_15  = Column(Float)
    sensor_11  = Column(Float)
    raw_data   = Column(JSON)
    created_at = Column(TIMESTAMP, server_default=func.now())