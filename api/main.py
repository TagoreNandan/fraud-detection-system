from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import sys
from pathlib import Path
import shap
import numpy as np

# Add project root to sys.path so imports work when running from api folder
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from database.db import init_db, save_transaction

# Load pipeline (NOT scaler, NOT model separately)
pipeline = joblib.load(os.path.join(BASE_DIR, "model", "fraud_pipeline.pkl"))

# SHAP explains individual predictions by showing how each feature pushes the
# model toward Fraud or Legitimate. Explainability builds trust, helps with
# audits, and makes it easier to debug model behavior in production.
if hasattr(pipeline, "named_steps"):
    # Assume the last step is the model, and earlier steps are preprocessing
    model = pipeline.steps[-1][1]
    preprocessor = pipeline[:-1]
else:
    model = pipeline
    preprocessor = None

# TreeExplainer works well with XGBoost-style models inside a pipeline
explainer = shap.TreeExplainer(model)

# Create app
app = FastAPI(title="Fraud Detection API")


@app.on_event("startup")
def startup_event():
    # Create the database and table on app startup
    init_db()


class Transaction(BaseModel):
    Time: float
    Amount: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float


@app.get("/")
def home():
    return {"message": "Fraud Detection API running"}


@app.post("/predict")
def predict(transaction: Transaction):

    data = pd.DataFrame([transaction.dict()])

    prediction = pipeline.predict(data)[0]
    probability = pipeline.predict_proba(data)[0][1]

    # Prepare data for SHAP (use the same preprocessing as the model)
    if preprocessor is not None:
        model_input = preprocessor.transform(data)
        if hasattr(preprocessor, "get_feature_names_out"):
            feature_names = preprocessor.get_feature_names_out()
        else:
            feature_names = data.columns
    else:
        model_input = data
        feature_names = data.columns

    # SHAP values show how each feature contributes to this prediction.
    # This improves transparency for fraud decisions.
    shap_values = explainer(model_input)
    values = shap_values.values[0]

    # Pick the top 3 features with the largest absolute impact
    top_indices = np.argsort(np.abs(values))[::-1][:3]
    top_risk_factors = []

    for i in top_indices:
        # Pipeline prefixes like "scaler__" or "remainder__" are not user friendly.
        # Strip them so the API returns clean feature names.
        raw_name = str(feature_names[i])
        clean_name = raw_name.split("__")[-1]

        top_risk_factors.append(
            {
                "feature": clean_name,
                "impact": float(values[i])
            }
        )

    # Save transaction to the database for monitoring
    save_transaction(
        amount=transaction.Amount,
        prediction="Fraud" if prediction == 1 else "Legitimate",
        fraud_probability=float(probability)
    )

    return {
        "prediction": "Fraud" if prediction == 1 else "Legitimate",
        "fraud_probability": float(probability),
        "explanation": {
            "top_risk_factors": top_risk_factors
        }
    }