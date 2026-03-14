import logging
import os

import joblib

logger = logging.getLogger(__name__)

MODEL_CACHE = None
SCALER_CACHE = None


def load_model():
    global MODEL_CACHE, SCALER_CACHE

    if MODEL_CACHE is not None:
        return MODEL_CACHE

    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_dir = os.path.join(base_dir, "model")
    pipeline_path = os.path.join(model_dir, "fraud_pipeline.pkl")
    model_path = os.path.join(model_dir, "fraud_model.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")

    try:
        if os.path.exists(pipeline_path):
            MODEL_CACHE = joblib.load(pipeline_path)
            logger.info("Model loaded successfully")
            return MODEL_CACHE

        if os.path.exists(model_path) and os.path.exists(scaler_path):
            MODEL_CACHE = joblib.load(model_path)
            SCALER_CACHE = joblib.load(scaler_path)
            logger.info("Model loaded successfully")
            return MODEL_CACHE

        raise FileNotFoundError("Model artifacts not found in backend/model")
    except Exception as exc:
        logger.error("Model load failed: %s", exc)
        return None
