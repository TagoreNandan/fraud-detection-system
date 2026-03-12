import logging
import os

import joblib

logger = logging.getLogger(__name__)

_pipeline = None


def _get_pipeline_path() -> str:
    """Resolve the model path from the current file location."""
    services_dir = os.path.dirname(os.path.abspath(__file__))
    app_dir = os.path.dirname(services_dir)
    backend_dir = os.path.dirname(app_dir)
    return os.path.join(backend_dir, "model", "fraud_pipeline.pkl")


def get_pipeline():
    """Load the trained pipeline once and reuse it for all requests."""
    global _pipeline
    if _pipeline is None:
        pipeline_path = _get_pipeline_path()
        logger.info("Loading model from %s", pipeline_path)
        try:
            _pipeline = joblib.load(pipeline_path)
            logger.info("Model loaded successfully")
        except FileNotFoundError as exc:
            logger.error("Model file not found at %s", pipeline_path)
            raise FileNotFoundError(
                "Model file not found. Ensure fraud_pipeline.pkl is included in deployment."
            ) from exc
    return _pipeline
