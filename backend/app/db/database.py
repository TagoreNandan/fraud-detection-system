import sqlite3
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "transactions.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            amount REAL,
            prediction TEXT,
            fraud_probability REAL,
            device TEXT,
            location TEXT
        )
        """
    )

    conn.commit()
    conn.close()


def save_transaction(amount, prediction, fraud_probability, device="web", location="unknown"):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    timestamp = datetime.now().isoformat()

    cursor.execute(
        """
        INSERT INTO transactions(timestamp, amount, prediction, fraud_probability, device, location)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (timestamp, float(amount), prediction, float(fraud_probability), device, location)
    )

    conn.commit()
    conn.close()


def reset_transactions(archive: bool = False) -> str:
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM transactions")
    conn.commit()
    conn.close()
    return ""


def get_recent_transactions(limit: int = 200) -> list[dict]:
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT timestamp, amount, prediction, fraud_probability, device, location
        FROM transactions
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,)
    )
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_metrics() -> dict:
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM transactions")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM transactions WHERE prediction = 'Fraud'")
    frauds = cursor.fetchone()[0]

    conn.close()
    return {
        "total_transactions": total,
        "frauds_detected": frauds,
        "fraud_rate": (frauds / total) if total > 0 else 0.0,
    }