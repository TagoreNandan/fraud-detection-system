from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .db.database import init_db, reset_transactions
from .routes.predict import router as predict_router
from .services.shap_service import get_explainer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = PROJECT_ROOT / "frontend" / "public"

app = FastAPI(title="Fraud Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    # Create the database and table on app startup
    init_db()
    # Archive old data and reset counts on each API run
    reset_transactions(archive=True)
    # Warm the SHAP explainer at startup so it is ready for requests
    get_explainer()


app.include_router(predict_router)

# Mount frontend at root (MUST BE AT THE END so it doesn't shadow /predict)
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")