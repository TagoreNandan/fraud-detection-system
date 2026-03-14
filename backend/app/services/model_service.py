import logging
from pathlib import Path

import joblib

logger = logging.getLogger(__name__)

model_pipeline = None


def get_model():
    """Load the trained pipeline lazily and reuse it for requests."""
    global model_pipeline
    if model_pipeline is None:
        base_dir = Path(__file__).resolve().parent.parent
        model_path = base_dir / "model" / "fraud_pipeline.pkl"
        model_pipeline = joblib.load(model_path)
        logger.info("Fraud model pipeline loaded successfully")
    return model_pipeline
