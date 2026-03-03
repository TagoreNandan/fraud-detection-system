# Fraud Detection System (Production-Style)

This project is a beginner-friendly, production-style fraud detection platform.
It includes a FastAPI backend, a Streamlit dashboard, SQLite logging, and a batch
prediction script for CSV files.

## Project Structure

```
fraud-detection-system/
│
├── api/
│   └── main.py
│
├── model/
│   ├── fraud_pipeline.pkl
│   └── batch_predict.py
│
├── database/
│   └── db.py
│
├── dashboard/
│   └── app.py
│
└── requirements.txt
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the API

From the project root:

```bash
cd api
uvicorn main:app --reload
```

The API will run at:
`http://127.0.0.1:8000/predict`

## Run the Dashboard

From the project root:

```bash
cd dashboard
streamlit run app.py
```

The dashboard will run at:
`http://localhost:8502`

## Run Batch Predictions

From the project root:

```bash
python model/batch_predict.py
```

This will read `data/creditcard.csv` (if it exists) and write
`data/creditcard_predictions.csv`.

## What Each File Does

- [api/main.py](api/main.py)
	- FastAPI backend for predictions.
	- Loads the trained pipeline and exposes `/predict`.
	- Logs each prediction to SQLite via `database/db.py`.

- [dashboard/app.py](dashboard/app.py)
	- Streamlit UI for testing transactions.
	- Calls the API safely and shows user-friendly errors.
	- Displays monitoring metrics from the SQLite database.

- [database/db.py](database/db.py)
	- Creates the SQLite database and `transactions` table.
	- Saves each transaction with timestamp, amount, and prediction.

- [model/batch_predict.py](model/batch_predict.py)
	- Runs batch predictions on a CSV file.
	- Saves results with prediction and probability columns.

- [model/fraud_pipeline.pkl](model/fraud_pipeline.pkl)
	- Trained ML pipeline used by the API and batch script.

- [requirements.txt](requirements.txt)
	- Python dependencies needed to run the project.
