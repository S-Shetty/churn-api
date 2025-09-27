from fastapi import FastAPI, Request
from pydantic import BaseModel
import pandas as pd
import logging

from inference import ModelService

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

# Initialize FastAPI app
app = FastAPI(title="Churn Prediction API")

# Initialize model service
# Initialize model service
model_service = ModelService(model_name="telco_linear")


# Request model for Pydantic validation
class ChurnRequest(BaseModel):
    data: list

@app.get("/")
def root():
    return {"message": "Churn Prediction API is running"}

@app.post("/predict")
def predict_churn(request: ChurnRequest):
    try:
        logger.info("[INFO] Received data for prediction")

        df_raw = pd.DataFrame(request.data)
        df_casted = pd.DataFrame([
            model_service.model.cast_dct(row) for row in df_raw.to_dict(orient="records")
        ])

        predictions = model_service.predict(df_casted)

        return {"predictions": predictions}

    except Exception as e:
        logger.error(f"[ERROR] Prediction failed: {e}")
        return {"error": str(e)}

@app.post("/explain")
def explain_churn(request: ChurnRequest):
    try:
        logger.info("[INFO] Received data for explanation")

        df_raw = pd.DataFrame(request.data)
        row = model_service.model.cast_dct(df_raw.iloc[0].to_dict())

        probability, explanation = model_service.explain(row)

        return {
            "probability": probability,
            "explanation": explanation
        }

    except Exception as e:
        logger.error(f"[ERROR] Explanation failed: {e}")
        return {"error": str(e)}
