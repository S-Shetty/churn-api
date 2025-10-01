from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import os
from typing import Dict, Any  # ✅ Add this line

from churnexplainer import ChurnExplainer


app = FastAPI()

# --- Load the model ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "telco_linear", "telco_linear.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# --- Prediction Endpoint ---
@app.post("/predict")
def predict(data: Dict[str, Any]):
    try:
        pred = churn_predictor.predict(data)
        return {"prediction": int(pred)}
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

