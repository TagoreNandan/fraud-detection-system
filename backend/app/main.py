import logging
import os
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from backend.app.db.database import init_db
from backend.app.model_loader import load_model
from backend.app.routes.predict import router

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
def startup():
    init_db()
    print("DB initialised")
    try:
        load_model()
    except Exception as exc:
        logger.warning("Model not available at startup: %s", exc)
    logger.info("Server startup complete")


@app.get("/")
def root():
    return {"status": "Fraud Detection API Running"}


@app.get("/health")
def health():
    return {"status": "ok"}