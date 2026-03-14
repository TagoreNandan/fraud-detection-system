import logging
from pathlib import Path

import joblib

logger = logging.getLogger(__name__)

pipeline = None

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "fraud_pipeline.pkl"


def get_pipeline():
    """Load the trained pipeline lazily and reuse it for requests."""
    global pipeline
    if pipeline is None:
        try:
            pipeline = joblib.load(MODEL_PATH)
            logger.info("Fraud model pipeline loaded successfully")
        except Exception as exc:
            logger.warning("Model not loaded: %s", exc)
            print("Model not loaded:", exc)
            pipeline = None
    return pipeline
