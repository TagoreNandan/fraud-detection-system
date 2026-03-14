import logging
from pathlib import Path

import joblib
from fastapi import HTTPException

logger = logging.getLogger(__name__)

_model = None

PIPELINE_PATH = Path(__file__).resolve().parents[2] / "model" / "fraud_pipeline.pkl"


def load_model():
    """Load the trained pipeline lazily and reuse it for requests."""
    global _model
    if _model is not None:
        return _model

    if not PIPELINE_PATH.exists():
        logger.warning("Model file missing at %s", PIPELINE_PATH)
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Model not available. Deployment may still be starting."
            },
        )

    try:
        _model = joblib.load(PIPELINE_PATH)
        logger.info("Fraud model pipeline loaded successfully")
        return _model
    except Exception:
        logger.exception("Model loading failed")
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Model not available. Deployment may still be starting."
            },
        )
