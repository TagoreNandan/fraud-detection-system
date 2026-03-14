import logging

import numpy as np
import shap

from .model_service import load_model

_explainer = None
logger = logging.getLogger(__name__)


def _build_background_sample() -> np.ndarray:
    """Create a small synthetic background sample for SHAP."""
    rng = np.random.default_rng(42)
    return rng.normal(size=(100, 30))


def get_explainer():
    """Create the SHAP explainer once using a small background sample."""
    global _explainer
    if _explainer is None:
        pipeline = load_model()
        background_sample = _build_background_sample()
        logger.info("Creating SHAP explainer with synthetic background sample")
        _explainer = shap.Explainer(pipeline.predict_proba, background_sample)
    return _explainer
