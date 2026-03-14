import os

import joblib

PIPELINE_PATH = "backend/model/fraud_pipeline.pkl"

_pipeline = None


def get_pipeline():
    """Load the trained pipeline once and reuse it for all requests."""
    global _pipeline
    if _pipeline is None:
        if not os.path.exists(PIPELINE_PATH):
            raise RuntimeError(
                f"Model file not found at {PIPELINE_PATH}. "
                "Train the model locally and commit the pipeline file."
            )
        _pipeline = joblib.load(PIPELINE_PATH)
        print("Fraud model pipeline loaded successfully")
    return _pipeline
