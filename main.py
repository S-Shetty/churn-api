from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import os
from typing import Dict, Any

from churnexplainer import ExplainedModel

app = FastAPI()

# --- Load the model ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "telco_linear"

try:
    churn_predictor = ExplainedModel(MODEL_NAME)

    # Check if model loaded correctly
    if churn_predictor.categoricalencoder is None or churn_predictor.pipeline is None:
        raise ValueError("Model not properly loaded. Components missing.")

except Exception as e:
    print(f"[ERROR] Failed to load model '{MODEL_NAME}': {e}")
    churn_predictor = None  # This ensures predict() will still fail gracefully

# --- Prediction Endpoint ---
@app.post("/predict")
def predict(data: Dict[str, Any]):
    if churn_predictor is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        pred = churn_predictor.predict_df(pd.DataFrame([data]))[0][0]
        return {"prediction": int(pred)}
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
