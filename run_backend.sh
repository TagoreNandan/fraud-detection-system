#!/bin/bash
source .venv/bin/activate
python -m uvicorn backend.app.main:app --reload --port 8000
