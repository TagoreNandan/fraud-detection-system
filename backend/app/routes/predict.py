import numpy as np
import pandas as pd
from fastapi import APIRouter

from app.db.database import get_metrics, get_recent_transactions, reset_transactions, save_transaction
from app.schemas.transaction import BatchTransaction, Transaction
from app.services.model_service import get_pipeline
from app.services.shap_service import get_explainer

router = APIRouter()


@router.post("/predict")
def predict(transaction: Transaction):
    """Predict fraud for a single transaction and return SHAP explanations."""
    data = pd.DataFrame([transaction.dict()])

    pipeline = get_pipeline()
    if pipeline is None:
        return {"error": "Model not available"}
    prediction = pipeline.predict(data)[0]
    probability = pipeline.predict_proba(data)[0][1]

    explainer = get_explainer()
    shap_values = explainer(data)
    values = shap_values.values

    if values.ndim == 3:
        values = values[0, :, 1]
    else:
        values = values[0]

    feature_names = shap_values.feature_names if shap_values.feature_names is not None else data.columns
    top_indices = np.argsort(np.abs(values))[::-1][:3]
    top_risk_factors = []

    for i in top_indices:
        raw_name = str(feature_names[i])
        clean_name = raw_name.split("__")[-1]
        top_risk_factors.append(
            {
                "feature": clean_name,
                "impact": float(abs(values[i]))
            }
        )

    save_transaction(
        amount=transaction.Amount,
        prediction="Fraud" if prediction == 1 else "Legitimate",
        fraud_probability=float(probability)
    )

    return {
        "prediction": "Fraud" if prediction == 1 else "Legitimate",
        "fraud_probability": float(probability),
        "top_risk_factors": top_risk_factors
    }


@router.get("/transactions")
def transactions(limit: int = 200):
    """Return the most recent transactions for the dashboard."""
    return get_recent_transactions(limit)


@router.get("/metrics")
def metrics():
    """Return dashboard metrics."""
    return get_metrics()


@router.post("/reset")
def reset():
    """Clear all transactions from the database."""
    reset_transactions(archive=False)
    return {"status": "success", "message": "Database reset to 0 transactions"}


@router.post("/batch-predict")
def batch_predict(batch: BatchTransaction):
    """Predict fraud for a list of transactions."""
    if not batch.transactions:
        return {"results": []}

    data = pd.DataFrame([t.dict() for t in batch.transactions])
    pipeline = get_pipeline()
    if pipeline is None:
        return {"error": "Model not available"}

    predictions = pipeline.predict(data)
    probabilities = pipeline.predict_proba(data)[:, 1]

    results = []
    for i, t in enumerate(batch.transactions):
        prediction_label = "Fraud" if predictions[i] == 1 else "Legitimate"
        prob = float(probabilities[i])

        save_transaction(
            amount=t.Amount,
            prediction=prediction_label,
            fraud_probability=prob
        )

        results.append({
            "prediction": prediction_label,
            "fraud_probability": prob,
        })

    return {"results": results}
