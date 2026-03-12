"""
Transaction Simulator

This script generates realistic-looking fake credit card transactions and sends
them to the FastAPI /predict endpoint in a loop. It is designed for learning
and demo purposes, so the code is simple, clear, and fully commented.
"""

import requests
import random
import time
import datetime

API_URL = "http://127.0.0.1:8000/predict"

# Speed modes control the delay between transactions
SPEEDS = {
    "FAST": 0.5,
    "NORMAL": 2.0,
    "SLOW": 5.0
}


def generate_transaction() -> dict:
    """Create a single fake transaction with the required model features."""
    # Time is seconds since first transaction in the original dataset
    time_value = random.uniform(0, 172800)

    # Amount is more likely to be small, but can be large occasionally
    if random.random() < 0.1:
        amount = random.uniform(50000, 200000)
    else:
        amount = random.uniform(10, 5000)

    # Base feature values around 0, with occasional unusual spikes
    transaction = {
        "Time": time_value,
        "Amount": amount
    }

    for i in range(1, 29):
        feature_name = f"V{i}"

        # Occasionally create unusual values that may increase fraud risk
        if random.random() < 0.05:
            value = random.uniform(-8, 8)
        else:
            value = random.uniform(-2, 2)

        transaction[feature_name] = value

    return transaction


def send_transaction(transaction: dict) -> dict:
    """Send one transaction to the API and return the parsed response."""
    try:
        response = requests.post(API_URL, json=transaction, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Could not connect to the API"}
    except requests.exceptions.Timeout:
        return {"error": "API request timed out"}
    except requests.exceptions.RequestException:
        return {"error": "Unexpected API error"}


def run_simulator(speed_mode: str = "NORMAL") -> None:
    """Run the simulator continuously with a chosen speed mode."""
    delay_seconds = SPEEDS.get(speed_mode.upper(), SPEEDS["NORMAL"])

    print(f"Starting simulator in {speed_mode.upper()} mode ({delay_seconds}s delay)")
    print("Press Ctrl+C to stop.\n")

    while True:
        transaction = generate_transaction()
        result = send_transaction(transaction)

        now = datetime.datetime.now().strftime("%H:%M:%S")
        amount = transaction["Amount"]

        if "error" in result:
            print(f"[{now}] Amount: ₹{amount:.0f} -> ERROR ({result['error']})")
        else:
            prediction = result.get("prediction", "Unknown")
            probability = result.get("fraud_probability", 0.0)
            print(f"[{now}] Amount: ₹{amount:.0f} -> {prediction} ({probability:.2f})")

        time.sleep(delay_seconds)


if __name__ == "__main__":
    # Change this to FAST, NORMAL, or SLOW if you want a different speed
    run_simulator(speed_mode="NORMAL")
