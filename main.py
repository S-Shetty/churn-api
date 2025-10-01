from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import os
from typing import Dict, Any

from churnexplainer import ExplainedModel  # ✅ Correct class

app = FastAPI()

churn_predictor = ExplainedModel("telco_linear")  # ✅ Initialize model properly

@app.post("/predict")
def predict(data: Dict[str, Any]):
    try:
        pred, _ = churn_predictor.predict_df(pd.DataFrame([data]))  # ✅ Correct method
        return {"prediction": int(pred[0])}
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
