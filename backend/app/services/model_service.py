import logging
import os

import joblib

logger = logging.getLogger(__name__)

_pipeline = None

PIPELINE_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "model",
        "fraud_pipeline.pkl",
    )
)


def get_pipeline():
    """Load the trained pipeline once and reuse it for all requests."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    print("Model loading from:", PIPELINE_PATH)
    if not os.path.exists(PIPELINE_PATH):
        logger.error("Model loading failed: missing artifact at %s", PIPELINE_PATH)
        raise RuntimeError("Model artifact missing in deployment")

    logger.info("Loading fraud detection model...")
    try:
        _pipeline = joblib.load(PIPELINE_PATH)
        logger.info("Model loaded successfully")
    except Exception as exc:
        logger.exception("Model loading failed")
        raise RuntimeError("Model artifact missing in deployment") from exc
    return _pipeline
