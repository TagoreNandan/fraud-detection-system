from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .routes.predict import router as predict_router
from .services.model_service import get_pipeline

BASE_DIR = Path(__file__).resolve().parents[2]
FRONTEND_DIR = BASE_DIR / "frontend" / "public"

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
    print("Fraud API Started")
    # Load the model once so it is ready for prediction requests
    get_pipeline()


app.include_router(predict_router)


@app.get("/health")
def health():
    return {"status": "ok"}

# Mount frontend at root (MUST BE AT THE END so it doesn't shadow /predict)
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")