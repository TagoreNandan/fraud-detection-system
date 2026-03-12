from pathlib import Path

import joblib

BACKEND_DIR = Path(__file__).resolve().parents[2]
PIPELINE_PATH = BACKEND_DIR / "model" / "fraud_pipeline.pkl"

_pipeline = None


def get_pipeline():
    """Load the trained pipeline once and reuse it for all requests."""
    global _pipeline
    if _pipeline is None:
        _pipeline = joblib.load(PIPELINE_PATH)
    return _pipeline
