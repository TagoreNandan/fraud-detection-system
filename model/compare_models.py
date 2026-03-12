import os

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


def build_preprocessor(scale_columns):
    return ColumnTransformer(
        transformers=[("scaler", StandardScaler(), scale_columns)],
        remainder="passthrough"
    )


def build_models(scale_pos_weight):
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric="logloss"
        )
    }


def main():
    print("Loading dataset...")
    df = pd.read_csv("../data/creditcard.csv")
    print("Dataset shape:", df.shape)

    X = df.drop("Class", axis=1)
    y = df["Class"]

    scale_columns = ["Amount", "Time"]
    preprocessor = build_preprocessor(scale_columns)

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    models = build_models(scale_pos_weight)

    roc_curves = {}
    roc_auc_scores = {}
    recall_scores = {}

    print("Training and evaluating models...")
    for name, model in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])

        pipeline.fit(X_train, y_train)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_curves[name] = (fpr, tpr)

        roc_auc_scores[name] = roc_auc_score(y_test, y_proba)

        report = classification_report(y_test, pipeline.predict(X_test), output_dict=True)
        recall_scores[name] = report["1"]["recall"]

    print("\nROC-AUC Scores")
    print("================")
    for name, score in roc_auc_scores.items():
        print(f"{name} ROC-AUC: {score:.2f}")

    best_by_auc = max(roc_auc_scores, key=roc_auc_scores.get)
    print("\nBest model by ROC-AUC:", best_by_auc)

    print("\nFraud Recall (Class=1)")
    print("=======================")
    for name, score in recall_scores.items():
        print(f"{name} Recall: {score:.4f}")

    print("\nSaving ROC comparison plot...")
    os.makedirs("../reports", exist_ok=True)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    for name, (fpr, tpr) in roc_curves.items():
        plt.plot(fpr, tpr, label=name)

    plt.plot([0, 1], [0, 1], "k--", label="Baseline")
    plt.title("Fraud Detection Model ROC Comparison")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("../reports/model_comparison_roc.png", dpi=150)
    plt.close()

    print("Saved plot to ../reports/model_comparison_roc.png")


if __name__ == "__main__":
    main()
