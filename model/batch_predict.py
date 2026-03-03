import argparse
from pathlib import Path

import joblib
import pandas as pd


def run_batch_prediction(input_csv: Path, output_csv: Path) -> None:
    """Load a CSV, run predictions, and save results to a new CSV."""
    # Load data
    data = pd.read_csv(input_csv)

    # Remove the label column if it exists
    if "Class" in data.columns:
        data = data.drop(columns=["Class"])

    # Load trained pipeline
    pipeline_path = (Path(__file__).resolve().parent / "fraud_pipeline.pkl").resolve()
    pipeline = joblib.load(pipeline_path)

    # Predict fraud class and probability
    predictions = pipeline.predict(data)
    probabilities = pipeline.predict_proba(data)[:, 1]

    # Save results with original data
    result = data.copy()
    result["prediction"] = ["Fraud" if p == 1 else "Legitimate" for p in predictions]
    result["fraud_probability"] = probabilities

    result.to_csv(output_csv, index=False)

    print(f"Saved predictions to: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch fraud predictions")
    parser.add_argument(
        "--input",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "data" / "creditcard.csv"),
        help="Path to input CSV file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "data" / "creditcard_predictions.csv"),
        help="Path to output CSV file"
    )
    args = parser.parse_args()

    run_batch_prediction(Path(args.input), Path(args.output))
