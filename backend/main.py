import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from dotenv import load_dotenv

load_dotenv()

from backend.db.models import Base
from backend.routers import predict, explain, history

# ── Database setup ────────────────────────────────────────────
DB_URL = (
    f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

engine       = create_engine(DB_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables automatically
Base.metadata.create_all(bind=engine)

# ── FastAPI app ───────────────────────────────────────────────
app = FastAPI(
    title       = "SensorMind API",
    description = "AI-powered predictive maintenance for industrial engines",
    version     = "1.0.0"
)

# Allow React frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"]
)

# ── Dependency ────────────────────────────────────────────────
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ── Include routers ───────────────────────────────────────────
app.include_router(predict.router, prefix="/api", tags=["Predict"])
app.include_router(explain.router, prefix="/api", tags=["Explain"])
app.include_router(history.router, prefix="/api", tags=["History"])

# ── Health check ──────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "SensorMind API",
        "status" : "running",
        "version": "1.0.0",
        "docs"   : "/docs"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}