import streamlit as st
import requests
import pandas as pd
import random
import sqlite3
from pathlib import Path
import joblib

# CONFIG
API_URL = "http://127.0.0.1:8000/predict"
API_TIMEOUT_SECONDS = 5
DB_PATH = (Path(__file__).resolve().parent / ".." / "database" / "fraud.db").resolve()

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="💳",
    layout="wide"
)

# LOAD DATASET FOR REALISTIC SIMULATION
@st.cache_data
def load_data():
    return pd.read_csv("../data/creditcard.csv")

df = load_data()


def load_recent_transactions(limit: int = 200) -> pd.DataFrame:
    """Load recent transactions from SQLite for dashboard metrics."""
    if not DB_PATH.exists():
        return pd.DataFrame()

    with sqlite3.connect(DB_PATH) as connection:
        query = """
            SELECT timestamp, amount, prediction, fraud_probability, device, location
            FROM transactions
            ORDER BY id DESC
            LIMIT ?
        """
        return pd.read_sql_query(query, connection, params=(limit,))

# SIDEBAR
st.sidebar.title("🏦 Fraud Detection System")

page = st.sidebar.radio(
    "Navigation",
    ["Transaction Simulator", "Fraud Monitoring Dashboard", "Batch Fraud Detection"]
)

# PAGE 1 — TRANSACTION SIMULATOR
if page == "Transaction Simulator":

    st.title("💳 Transaction Simulator")

    st.markdown("Simulate a real-world transaction")

    col1, col2 = st.columns(2)

    with col1:
        amount = st.slider("Transaction Amount (₹)", 10, 100000, 500)

        location = st.selectbox(
            "Location",
            ["Same City", "Different City", "International"]
        )

    with col2:
        device = st.selectbox(
            "Device",
            ["Mobile", "Desktop", "ATM"]
        )

        time_period = st.selectbox(
            "Time",
            ["Morning", "Afternoon", "Evening", "Night"]
        )

    if st.button("🔍 Analyze Transaction", use_container_width=True):

        # Pick realistic transaction sample
        sample = df.sample(1).drop("Class", axis=1)

        sample["Amount"] = amount
        sample["Time"] = random.uniform(0, 172800)

        transaction = sample.iloc[0].to_dict()

        # Call the API safely so the app never crashes if the API is offline
        try:
            response = requests.post(
                API_URL,
                json=transaction,
                timeout=API_TIMEOUT_SECONDS
            )
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the API. Please make sure FastAPI is running.")
            st.stop()
        except requests.exceptions.Timeout:
            st.error("The API took too long to respond. Please try again.")
            st.stop()
        except requests.exceptions.HTTPError:
            st.error("The API returned an error. Please check the API logs.")
            st.stop()
        except requests.exceptions.RequestException:
            st.error("Unexpected error while calling the API.")
            st.stop()

        prediction = result.get("prediction")
        probability = result.get("fraud_probability")
        explanation = result.get("explanation", {})
        top_risk_factors = explanation.get("top_risk_factors", [])

        if prediction is None or probability is None:
            st.error("API response is missing required fields.")
            st.stop()

        st.divider()

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Prediction")

            if prediction == "Fraud":
                st.error("⚠️ Fraud Detected")
            else:
                st.success("✅ Legitimate Transaction")

            st.metric(
                "Fraud Probability",
                f"{probability:.4f}"
            )

            st.progress(probability)

        with col2:

            st.subheader("Transaction Details")

            st.write(f"Amount: ₹{amount}")
            st.write(f"Location: {location}")
            st.write(f"Device: {device}")
            st.write(f"Time: {time_period}")

        st.divider()

        # SHAP explains why the model predicted Fraud or Legitimate.
        # This helps users understand and trust the model's decision.
        st.subheader("Top Risk Factors")

        if len(top_risk_factors) == 0:
            st.info("No explanation returned by the API.")
        else:
            for item in top_risk_factors:
                feature = item.get("feature", "Unknown")
                impact = item.get("impact", 0.0)
                st.write(f"- {feature} (impact: {impact:.4f})")

# PAGE 2 — FRAUD MONITORING DASHBOARD
elif page == "Fraud Monitoring Dashboard":

    st.title("📊 Fraud Monitoring Dashboard")

    history = load_recent_transactions()

    if len(history) == 0:

        st.info("No transactions yet. Run a few predictions first.")

    else:

        total = len(history)
        frauds = len(history[history["prediction"] == "Fraud"])
        legit = total - frauds

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Transactions", total)
        col2.metric("Frauds Detected", frauds)
        col3.metric(
            "Fraud Rate",
            f"{(frauds/total)*100:.2f}%"
        )

        st.divider()

        st.subheader("Fraud Probability Over Time")

        chart_data = history.copy()
        chart_data["timestamp"] = pd.to_datetime(chart_data["timestamp"])

        st.line_chart(
            chart_data.set_index("timestamp")["fraud_probability"]
        )

        st.divider()

        st.subheader("Fraud vs Legitimate")

        counts = pd.DataFrame(
            {
                "prediction": ["Fraud", "Legitimate"],
                "count": [frauds, legit]
            }
        )
        st.bar_chart(counts.set_index("prediction")["count"])

        st.divider()

        st.subheader("Recent Transactions")

        st.dataframe(
            history,
            use_container_width=True
        )

# PAGE 3 — BATCH FRAUD DETECTION
elif page == "Batch Fraud Detection":

    st.title("📁 Batch Fraud Detection")
    st.markdown("Upload a CSV file and get fraud predictions for all rows.")

    # Choose how predictions are made
    mode = st.radio(
        "Prediction Mode",
        ["API only", "Local model only"],
        horizontal=True
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:

        # Required columns for the model
        required_columns = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]

        # Load the CSV safely
        try:
            batch_data = pd.read_csv(uploaded_file)
        except Exception:
            st.error("Invalid CSV file. Please upload a valid CSV.")
            st.stop()

        # Check for missing columns
        missing = [col for col in required_columns if col not in batch_data.columns]
        if len(missing) > 0:
            st.error(f"Missing columns: {', '.join(missing)}")
            st.stop()

        # Load local model only when the user chooses local mode
        local_pipeline = None
        if mode == "Local model only":
            local_pipeline_path = (Path(__file__).resolve().parent / ".." / "model" / "fraud_pipeline.pkl").resolve()
            if local_pipeline_path.exists():
                local_pipeline = joblib.load(local_pipeline_path)
            else:
                st.error("Local model not found at model/fraud_pipeline.pkl")
                st.stop()

        predictions = []
        probabilities = []

        # Show progress for large files
        total_rows = len(batch_data)
        progress_bar = st.progress(0)

        # Send each row to the API or use the local model
        for index, row in batch_data.iterrows():
            row_data = row[required_columns].to_dict()

            if mode == "API only":
                try:
                    response = requests.post(API_URL, json=row_data, timeout=API_TIMEOUT_SECONDS)
                    response.raise_for_status()
                    result = response.json()
                    prediction = result.get("prediction")
                    probability = result.get("fraud_probability")
                except requests.exceptions.RequestException:
                    st.error("API is not available. Please start FastAPI and try again.")
                    st.stop()
            else:
                local_df = pd.DataFrame([row_data])
                local_pred = local_pipeline.predict(local_df)[0]
                local_prob = local_pipeline.predict_proba(local_df)[0][1]
                prediction = "Fraud" if local_pred == 1 else "Legitimate"
                probability = float(local_prob)

            predictions.append(prediction)
            probabilities.append(probability)

            # Update progress (avoid division by zero)
            if total_rows > 0:
                progress_bar.progress(int(((index + 1) / total_rows) * 100))

        # Add prediction columns to the results
        results = batch_data.copy()
        results["prediction"] = predictions
        results["fraud_probability"] = probabilities

        # Summary metrics
        total = len(results)
        frauds = len(results[results["prediction"] == "Fraud"])
        fraud_rate = (frauds / total) * 100 if total > 0 else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Transactions", total)
        col2.metric("Frauds Detected", frauds)
        col3.metric("Fraud Rate", f"{fraud_rate:.2f}%")

        st.divider()

        # Highlight fraud rows in red
        def highlight_fraud(row):
            if row["prediction"] == "Fraud":
                return ["background-color: #ffcccc"] * len(row)
            return [""] * len(row)

        st.subheader("Batch Results")
        # Pandas Styler has a maximum cell limit, so only style a small preview
        max_styled_rows = 1000
        if len(results) > max_styled_rows:
            st.info(f"Showing a styled preview of the first {max_styled_rows} rows.")
            preview = results.head(max_styled_rows)
            st.dataframe(preview.style.apply(highlight_fraud, axis=1), use_container_width=True)
            st.subheader("Full Results (no styling)")
            st.dataframe(results, use_container_width=True)
        else:
            st.dataframe(results.style.apply(highlight_fraud, axis=1), use_container_width=True)

        st.divider()

        # Download results as CSV
        csv_data = results.to_csv(index=False)
        st.download_button(
            label="Download Results CSV",
            data=csv_data,
            file_name="batch_fraud_predictions.csv",
            mime="text/csv"
        )