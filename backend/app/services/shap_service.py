from pathlib import Path

import pandas as pd
import shap

from .model_service import get_pipeline

_explainer = None


def get_explainer():
    """Create the SHAP explainer once using a small background sample."""
    global _explainer
    if _explainer is None:
        pipeline = get_pipeline()
        root_dir = Path(__file__).resolve().parents[3]
        background_path = root_dir / "data" / "creditcard.csv"
        background_df = pd.read_csv(background_path).drop("Class", axis=1)
        background_sample = background_df.sample(n=min(100, len(background_df)), random_state=42)
        _explainer = shap.Explainer(pipeline.predict_proba, background_sample)
    return _explainer
