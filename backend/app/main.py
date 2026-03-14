import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Lazy loading avoids startup crashes when model files are missing in Railway.
from app.model_loader import load_model

app = FastAPI(title="Fraud Detection API")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "ok", "service": "fraud-detection-api"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: dict):
    # Prediction stays safe if model artifacts are unavailable.
    model = load_model()
    if model is None:
        return {"error": "Model not available"}

    try:
        import pandas as pd

        data = pd.DataFrame([payload])
        prediction = model.predict(data)[0]
        proba = model.predict_proba(data)[0][1]
        return {
            "prediction": "Fraud" if int(prediction) == 1 else "Legitimate",
            "fraud_probability": float(proba),
        }
    except Exception as exc:
        logger.warning("Prediction failed: %s", exc)
        return {"error": "Prediction failed"}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)