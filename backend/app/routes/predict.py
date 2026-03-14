import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from backend.app.db.database import get_metrics, get_recent_transactions, reset_transactions, save_transaction
from backend.app.schemas.transaction import BatchTransaction, Transaction
from backend.app.model_loader import load_model

router = APIRouter()

FEATURE_ORDER = [
    "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
    "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18",
    "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27",
    "V28", "Amount"
]

logger = logging.getLogger(__name__)


@router.post("/predict")
def predict(transaction: Transaction):
    """Predict fraud for a single transaction and return SHAP explanations."""
    logger.info("Prediction request received")
    data_dict = transaction.dict()
    data = pd.DataFrame([[data_dict[col] for col in FEATURE_ORDER]], columns=FEATURE_ORDER)

    pipeline = load_model()
    if pipeline is None:
        return {"error": "Model not available"}
    prediction = pipeline.predict(data)[0]
    probability = pipeline.predict_proba(data)[0][1]

    # SHAP disabled for stability.

    save_transaction(
        amount=transaction.Amount,
        prediction="Fraud" if prediction == 1 else "Legitimate",
        fraud_probability=float(probability)
    )

    label = "Fraud" if prediction == 1 else "Legitimate"
    print("Prediction successful")
    return {
        "prediction": label,
        "fraud_probability": float(probability),
        "explanation": "SHAP disabled for stability"
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

    data_rows = [[t.dict()[col] for col in FEATURE_ORDER] for t in batch.transactions]
    data = pd.DataFrame(data_rows, columns=FEATURE_ORDER)
    pipeline = load_model()
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


@router.post("/batch_predict")
async def batch_predict_csv(file: UploadFile = File(...)):
    pipeline = load_model()
    if pipeline is None:
        return {"error": "Model not available"}

    try:
        df = pd.read_csv(file.file)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid CSV file: {exc}")

    required_columns = set(["Time", "Amount"] + [f"V{i}" for i in range(1, 29)])
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns: {', '.join(missing)}")

    df = df[FEATURE_ORDER]
    predictions = pipeline.predict(df)
    probabilities = pipeline.predict_proba(df)[:, 1]

    df["fraud_prediction"] = ["Fraud" if p == 1 else "Legitimate" for p in predictions]
    df["fraud_probability"] = probabilities
    df["risk_band"] = np.where(
        probabilities < 0.3,
        "Safe",
        np.where(probabilities <= 0.7, "Suspicious", "High Risk")
    )

    output_dir = Path(__file__).resolve().parents[2] / "reports" / "batch_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    output_path = output_dir / filename
    df.to_csv(output_path, index=False)

    fraud_count = int((df["fraud_prediction"] == "Fraud").sum())
    avg_probability = float(probabilities.mean()) if len(probabilities) else 0.0
    preview_rows = df[df["fraud_prediction"] == "Fraud"].head(5).to_dict(orient="records")

    return {
        "rows_processed": int(len(df)),
        "fraud_count": fraud_count,
        "avg_probability": avg_probability,
        "filename": filename,
        "fraud_preview": preview_rows,
    }


@router.get("/download/{filename}")
def download_batch_output(filename: str):
    output_dir = Path(__file__).resolve().parents[2] / "reports" / "batch_outputs"
    file_path = output_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=file_path, filename=filename)
