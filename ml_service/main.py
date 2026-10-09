from fastapi import FastAPI
from pathlib import Path
import joblib

app = FastAPI(title="Himalayan Guardian ML Service")

MODEL_PATH = Path(__file__).parent / "model" / "glof_risk_model.joblib"
model = None

@app.on_event("startup")
def load_model():
    global model
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)

@app.get("/")
def home():
    return {
        "service": "Himalayan Guardian ML",
        "model_loaded": model is not None
    }
