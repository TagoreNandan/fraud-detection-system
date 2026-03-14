import logging
import os

import logging

from fastapi import FastAPI

from app.model_loader import load_model
from app.schemas.transaction import Transaction

app = FastAPI(title="Fraud Detection API")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.on_event("startup")
def startup_event():
    logger.info("Starting Fraud Detection API")
    model = load_model()
    if model is None:
        logger.warning("Model not available at startup")
    else:
        logger.info("Model ready for predictions")


@app.get("/")
def root():
    return {"message": "Fraud Detection API Running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: Transaction):
    logger.info("Prediction request received")
    model = load_model()
    if model is None:
        return {"error": "Model not available"}

    try:
        import pandas as pd

        data = pd.DataFrame([payload.dict()])
        prediction = model.predict(data)[0]
        proba = model.predict_proba(data)[0][1]
        return {
            "prediction": "Fraud" if int(prediction) == 1 else "Legitimate",
            "fraud_probability": float(proba),
        }
    except Exception as exc:
        logger.warning("Prediction failed: %s", exc)
        return {"error": "Prediction failed"}