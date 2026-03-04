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

from database.db import init_db, save_transaction, reset_transactions

# Load pipeline (NOT scaler, NOT model separately)
pipeline = joblib.load(os.path.join(BASE_DIR, "model", "fraud_pipeline.pkl"))

# SHAP explainer is created on startup so it is ready for predictions.
explainer = None

# Create app
app = FastAPI(title="Fraud Detection API")


@app.on_event("startup")
def startup_event():
    # Create the database and table on app startup
    init_db()
    # Archive old data and reset counts on each API run
    reset_transactions(archive=True)
    # Load SHAP explainer once at startup
    global explainer
    # Use a small background dataset so SHAP can build a masker
    background_path = os.path.join(BASE_DIR, "data", "creditcard.csv")
    background_df = pd.read_csv(background_path).drop("Class", axis=1)
    background_sample = background_df.sample(n=min(100, len(background_df)), random_state=42)
    explainer = shap.Explainer(pipeline.predict_proba, background_sample)


class Transaction(BaseModel):
    # Input schema for a single transaction sent to /predict
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
    # Simple health check endpoint
    return {"message": "Fraud Detection API running"}


@app.post("/predict")
def predict(transaction: Transaction):
    # Convert the incoming JSON into a single-row DataFrame
    data = pd.DataFrame([transaction.dict()])
    # Model prediction and fraud probability
    prediction = pipeline.predict(data)[0]
    probability = pipeline.predict_proba(data)[0][1]

    # SHAP values show how each feature contributes to this prediction.
    # This improves transparency for fraud decisions.
    shap_values = explainer(data)
    values = shap_values.values

    # If SHAP returns values for both classes, use the Fraud class (index 1)
    if values.ndim == 3:
        values = values[0, :, 1]
    else:
        values = values[0]

    # SHAP may provide feature names; fall back to input columns if missing.
    feature_names = shap_values.feature_names if shap_values.feature_names is not None else data.columns

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
                "impact": float(abs(values[i]))
            }
        )

    # Save transaction to the database for monitoring
    save_transaction(
        amount=transaction.Amount,
        prediction="Fraud" if prediction == 1 else "Legitimate",
        fraud_probability=float(probability)
    )

    # Return prediction and the top SHAP risk factors
    return {
        "prediction": "Fraud" if prediction == 1 else "Legitimate",
        "fraud_probability": float(probability),
        "top_risk_factors": top_risk_factors
    }