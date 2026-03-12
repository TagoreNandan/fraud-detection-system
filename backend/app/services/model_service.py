import logging
from pathlib import Path

import joblib

logger = logging.getLogger(__name__)

_pipeline = None


def _get_pipeline_path() -> Path:
    """Resolve the model path from the current file location."""
    return Path(__file__).resolve().parents[2] / "model" / "fraud_pipeline.pkl"


def get_pipeline():
    """Load the trained pipeline once and reuse it for all requests."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    pipeline_path = _get_pipeline_path()
    if not pipeline_path.exists():
        logger.error("Model loading failed: missing artifact at %s", pipeline_path)
        raise RuntimeError("Model artifact missing in deployment")

    logger.info("Loading fraud detection model...")
    try:
        _pipeline = joblib.load(pipeline_path)
        logger.info("Model loaded successfully")
    except Exception as exc:
        logger.exception("Model loading failed")
        raise RuntimeError("Model artifact missing in deployment") from exc
    return _pipeline
