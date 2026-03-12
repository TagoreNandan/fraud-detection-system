from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

from xgboost import XGBClassifier

# Train an XGBoost pipeline for fraud detection and save it as a single file
print("Loading dataset...")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load dataset
df = pd.read_csv(PROJECT_ROOT / "data" / "creditcard.csv")

print("Dataset shape:", df.shape)

# Features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Columns to scale (keep scale consistent for Amount and Time)
scale_columns = ["Amount", "Time"]

# Preprocessor: scale selected columns, pass the rest through unchanged
preprocessor = ColumnTransformer(
    transformers=[
        ("scaler", StandardScaler(), scale_columns)
    ],
    remainder="passthrough"
)

# Create pipeline: preprocessing + XGBoost classifier
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
        random_state=42,
        eval_metric="logloss"
    ))
])

print("Splitting dataset...")

# Train-test split (stratified to keep fraud ratio stable)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("Training samples:", X_train.shape)
print("Testing samples:", X_test.shape)

print("Training pipeline...")

# Train pipeline
pipeline.fit(X_train, y_train)

# Evaluate using ROC-AUC, confusion matrix, and precision/recall
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print("\nModel Evaluation")
print("================")

print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save pipeline for API and batch prediction usage
print("\nSaving pipeline...")

pipeline_path = PROJECT_ROOT / "backend" / "model" / "fraud_pipeline.pkl"
joblib.dump(pipeline, pipeline_path)

print(f"Pipeline saved as {pipeline_path}")