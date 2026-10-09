from fastapi import FastAPI, HTTPException
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


@app.get("/model-info")
def model_info():
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check the model file path."
        )

    features = getattr(model, "feature_names_in_", None)
    feature_count = getattr(model, "n_features_in_", None)

    return {
        "model_type": type(model).__name__,
        "expected_features": (
            list(features) if features is not None else None
        ),
        "number_of_features": (
            int(feature_count) if feature_count is not None else None
        ),
        "classes": (
            model.classes_.tolist()
            if hasattr(model, "classes_")
            else None
        )
    }
