FROM python:3.11-slim

WORKDIR /app

COPY backend /app/backend

RUN pip install --no-cache-dir -r /app/backend/requirements.txt

EXPOSE $PORT

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
