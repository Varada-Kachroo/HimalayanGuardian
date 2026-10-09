from fastapi import FastAPI, HTTPException
from pathlib import Path
import joblib

app = FastAPI(title="Himalayan Guardian ML Service")

MODEL_PATH = Path(__file__).parent / "model" / "glof_risk_model.joblib"
bundle = None
model = None
features = []


@app.on_event("startup")
def load_model():
    global bundle, model, features

    if MODEL_PATH.exists():
        bundle = joblib.load(MODEL_PATH)
        model = bundle["model"]
        features = bundle["features"]


@app.get("/")
def home():
    return {
        "service": "Himalayan Guardian ML",
        "model_loaded": model is not None
    }


@app.get("/model-info")
def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_name": bundle.get("model_name"),
        "expected_features": features,
        "metrics": bundle.get("metrics"),
        "trained_on": bundle.get("trained_on")
    }


@app.post("/predict")
def predict(data: dict):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    missing = [f for f in features if f not in data]
    if missing:
        raise HTTPException(
            status_code=400,
            detail={"missing_features": missing}
        )

    try:
        import pandas as pd

        row = pd.DataFrame(
            [[data[f] for f in features]],
            columns=features
        )
        score = float(model.predict_proba(row)[0][1])

        return {
            "risk_score": round(score, 4),
            "risk_type": "relative model score, not calibrated flood probability"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
