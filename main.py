from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import logging

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load model (will be initialized in startup event)
model = None

# Sample input format using Pydantic
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

# Triggered when the API starts
@app.on_event("startup")
def load_model():
    global model
    try:
        model_path = "models/telco_linear/telco_linear.pkl"
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            logger.info("✅ Model loaded successfully.")
        else:
            logger.error(f"❌ Model file not found at: {model_path}")
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")

@app.get("/")
def root():
    return {"message": "Churn Prediction API is running"}

@app.post("/predict")
def predict(data: CustomerData):
    try:
        if model is None:
            raise ValueError("Model not loaded")
        
        df = pd.DataFrame([data.dict()])
        prediction = model.predict(df)
        return {"predictions": prediction.tolist()}
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})
