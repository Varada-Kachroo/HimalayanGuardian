from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import pandas as pd
import json

app = FastAPI(title="Himalayan Guardian API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA = Path(__file__).parent / "data"

@app.get("/")
def home():
    return {"message": "Himalayan Guardian API is running"}

@app.get("/lakes")
def get_lakes():
    path = DATA / "lake_risk.csv"
    if not path.exists():
        return {"error": "lake_risk.csv not uploaded"}
    return pd.read_csv(path).fillna("").to_dict(orient="records")

@app.get("/forecast")
def get_forecast():
    path = DATA / "forecast.csv"
    if not path.exists():
        return {"error": "forecast.csv not uploaded"}
    return pd.read_csv(path).fillna("").to_dict(orient="records")

@app.get("/summary")
def get_summary():
    path = DATA / "summary.json"
    if not path.exists():
        return {"error": "summary.json not uploaded"}
    with open(path) as f:
        return json.load(f)
