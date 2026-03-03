import sqlite3
from pathlib import Path
from datetime import datetime

# Database file stored inside the database folder
DB_PATH = Path(__file__).resolve().parent / "fraud.db"


def init_db() -> None:
	"""Create the database and transactions table if they do not exist."""
	with sqlite3.connect(DB_PATH) as connection:
		cursor = connection.cursor()
		cursor.execute(
			"""
			CREATE TABLE IF NOT EXISTS transactions (
				id INTEGER PRIMARY KEY AUTOINCREMENT,
				timestamp TEXT NOT NULL,
				amount REAL NOT NULL,
				prediction TEXT NOT NULL,
				fraud_probability REAL NOT NULL,
				device TEXT NOT NULL,
				location TEXT NOT NULL
			)
			"""
		)
		connection.commit()


def save_transaction(
	amount: float,
	prediction: str,
	fraud_probability: float,
	device: str = "Unknown",
	location: str = "Unknown"
) -> None:
	"""Insert a new transaction row with an automatic timestamp."""
	timestamp = datetime.now().isoformat(timespec="seconds")

	with sqlite3.connect(DB_PATH) as connection:
		cursor = connection.cursor()
		cursor.execute(
			"""
			INSERT INTO transactions (
				timestamp,
				amount,
				prediction,
				fraud_probability,
				device,
				location
			) VALUES (?, ?, ?, ?, ?, ?)
			""",
			(timestamp, float(amount), prediction, float(fraud_probability), device, location)
		)
		connection.commit()
