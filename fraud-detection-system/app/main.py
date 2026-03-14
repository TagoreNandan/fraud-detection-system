import logging
import os

import joblib
from fastapi import FastAPI

app = FastAPI(title="Fraud Detection API")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL = None
SCALER = None


def _get_model_paths():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "model")
    model_path = os.path.join(model_dir, "fraud_model.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    return model_path, scaler_path


def load_model():
    """Load model artifacts lazily and reuse them for requests."""
    global MODEL, SCALER
    if MODEL is None:
        model_path, scaler_path = _get_model_paths()
        try:
            MODEL = joblib.load(model_path)
            logger.info("Model loaded successfully")
        except Exception as exc:
            logger.warning("Model not loaded: %s", exc)
            MODEL = None

        try:
            SCALER = joblib.load(scaler_path)
            logger.info("Scaler loaded successfully")
        except Exception as exc:
            logger.warning("Scaler not loaded: %s", exc)
            SCALER = None
    return MODEL


@app.get("/")
def root():
    return {"status": "ok", "service": "fraud-detection-api"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: dict):
    model = load_model()
    if model is None:
        return {"error": "Model not available"}

    try:
        import pandas as pd

        data = pd.DataFrame([payload])
        prediction = model.predict(data)[0]
        proba = model.predict_proba(data)[0][1]
        return {
            "prediction": "Fraud" if int(prediction) == 1 else "Legitimate",
            "fraud_probability": float(proba),
        }
    except Exception as exc:
        logger.warning("Prediction failed: %s", exc)
        return {"error": "Prediction failed"}
