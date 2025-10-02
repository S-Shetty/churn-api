from fastapi import FastAPI
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.get("/")
def root():
    logger.info("Health check endpoint hit.")
    return {"message": "Churn Prediction API is running"}
