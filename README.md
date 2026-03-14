# Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange)
![MIT License](https://img.shields.io/badge/License-MIT-green)

## Overview
This project predicts fraudulent credit card transactions using a trained XGBoost model. It provides a FastAPI backend for real-time predictions, SHAP explanations for transparency, and SQLite logging for traceability.

## Quick Demo
- Simulate or upload transactions and send them to the API.
- The model predicts Fraud vs Legitimate in milliseconds.
- SHAP highlights the top risk-driving features for each prediction.
- All predictions are stored in SQLite for auditing.
- The frontend displays live metrics, trends, and explanations.

## System Architecture
See [architecture/system_design.md](architecture/system_design.md) for the latest diagram.
```

## Model Performance
Metrics from the latest training run:
- ROC-AUC: 0.9732
- Precision: 0.53
- Recall: 0.86

## Model Comparison Experiment
To compare multiple models fairly, a separate script trains Logistic Regression, Random Forest, and XGBoost using the same train/test split, then plots all ROC curves on one chart.

Run the comparison script:
```bash
python ml/compare_models.py
```
This saves the ROC chart to reports/roc_comparison.png.

Latest comparison results:

| Model | ROC-AUC | Fraud Recall (Class=1) |
| --- | --- | --- |
| Logistic Regression | 0.97 | 0.9184 |
| Random Forest | 0.96 | 0.7551 |
| XGBoost | 0.97 | 0.8469 |



## Model Comparison

We benchmarked multiple models:

- Logistic Regression
- Random Forest
- XGBoost

Interestingly, Logistic Regression achieved the highest ROC-AUC and fraud recall.

This indicates that the dataset (PCA-transformed features) is largely linearly separable.

Therefore Logistic Regression was selected as the final production model due to:

- Strong performance
- Lower latency
- Better interpretability
- Simpler deployment


## API Example
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Time":0.0,"Amount":149.62,"V1":-1.359807,"V2":-0.072781,"V3":2.536346,"V4":1.378155,"V5":-0.338321,"V6":0.462388,"V7":0.239599,"V8":0.098698,"V9":0.363787,"V10":0.090794,"V11":-0.551600,"V12":-0.617801,"V13":-0.991390,"V14":-0.311169,"V15":1.468177,"V16":-0.470401,"V17":0.207971,"V18":0.025791,"V19":0.403993,"V20":0.251412,"V21":-0.018307,"V22":0.277838,"V23":-0.110474,"V24":0.066928,"V25":0.128539,"V26":-0.189115,"V27":0.133558,"V28":-0.021053}'
```

## How to Run the Project
1. Install requirements:
	```bash
	pip install -r requirements.txt
	```
2. Start the API:
	```bash
	python -m uvicorn backend.app.main:app --reload --port 8000
	```

## Local Run (Frontend + Backend)
1. Install requirements:
	```bash
	pip install -r requirements.txt
	```
2. Start the backend:
	```bash
	bash run_backend.sh
	```
3. Start the frontend:
	```bash
	bash run_frontend.sh
	```
4. Open:
	```bash
	http://localhost:5500
	```

### Reset Database
```bash
bash reset_db.sh
```

## Project Structure
```
fraud-detection-system
│
├── backend/             # FastAPI service + model artifacts
│   ├── app/             # API routes, schemas, services, and DB
│   ├── model/           # Trained pipeline used by the API
│   ├── requirements.txt # Backend-only dependencies
│   └── Dockerfile       # Container for API
│
├── frontend/            # Static web UI assets
│   ├── public/          # HTML/CSS/JS served by FastAPI
│   └── src/             # Reserved for future frontend build
│
├── ml/                  # Training and batch inference scripts
│   ├── train.py
│   ├── batch_predict.py
│   ├── compare_models.py
│   └── generate_test_data.py
│
├── data/                # Datasets and synthetic samples
│   ├── creditcard.csv
│   └── synthetic_transactions.csv
│
├── reports/             # Model evaluation artifacts
│   └── roc_comparison.png
│
├── notebooks/           # Experiments and exploration
│   └── experimentation.ipynb
│
├── architecture/        # System design diagrams
│   └── system_design.png
│
├── docker-compose.yml   # Local container orchestration
├── .env.example         # Environment variable template
├── requirements.txt
└── README.md
```

## 🌐 Deployment
Architecture flow:
User → Vercel Frontend → Render FastAPI → ML Model

## Future Improvements
- Real-time transaction streaming with Kafka
- Graph-based fraud detection models
- Automatic model retraining pipeline
- Docker deployment
- Cloud deployment (AWS / GCP)
- Real-time fraud alerting system

