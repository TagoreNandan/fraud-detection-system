import logging
from pathlib import Path

import joblib

logger = logging.getLogger(__name__)

MODEL = None
SCALER = None

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "model.pkl"
SCALER_PATH = BASE_DIR / "model" / "scaler.pkl"


def load_model():
    """Load model artifacts lazily and reuse them for requests."""
    global MODEL, SCALER
    if MODEL is None:
        try:
            MODEL = joblib.load(MODEL_PATH)
            logger.info("Model loaded successfully")
        except Exception as exc:
            logger.warning("Model not loaded: %s", exc)
            MODEL = None

    if SCALER is None:
        try:
            SCALER = joblib.load(SCALER_PATH)
            logger.info("Scaler loaded successfully")
        except Exception as exc:
            logger.warning("Scaler not loaded: %s", exc)
            SCALER = None

    return MODEL
