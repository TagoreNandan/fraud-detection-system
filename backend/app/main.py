import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .db.database import init_db, reset_transactions
from .routes.predict import router as predict_router
from .services.model_service import get_pipeline
from .services.shap_service import get_explainer

BASE_DIR = Path(__file__).resolve().parents[2]
FRONTEND_DIR = BASE_DIR / "frontend" / "public"

app = FastAPI(title="Fraud Detection API")

logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    try:
        # Create the database and table on app startup
        init_db()
        # Archive old data and reset counts on each API run
        reset_transactions(archive=True)
        # Load the model once so it is ready for prediction requests
        get_pipeline()
        # Warm the SHAP explainer at startup so it is ready for requests
        get_explainer()
    except Exception:
        logger.exception("Startup initialization failed; continuing to serve API")


app.include_router(predict_router)


@app.get("/health")
def health():
    return {"status": "ok"}

# Mount frontend at root (MUST BE AT THE END so it doesn't shadow /predict)
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")