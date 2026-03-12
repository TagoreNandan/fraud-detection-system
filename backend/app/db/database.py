import sqlite3
import csv
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "transactions.db"
ARCHIVE_DIR = DATA_DIR / "archives"


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


def archive_transactions() -> str:
	"""Save all transactions to a CSV file and return the file path."""
	init_db()
	ARCHIVE_DIR.mkdir(exist_ok=True)
	archive_path = ARCHIVE_DIR / f"transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

	with sqlite3.connect(DB_PATH) as connection:
		cursor = connection.cursor()
		cursor.execute("SELECT * FROM transactions")
		rows = cursor.fetchall()
		headers = [description[0] for description in cursor.description]

		with open(archive_path, "w", newline="") as csv_file:
			writer = csv.writer(csv_file)
			writer.writerow(headers)
			writer.writerows(rows)

	return str(archive_path)


def clear_transactions() -> None:
	"""Delete all transactions and reset the auto-increment counter."""
	init_db()
	with sqlite3.connect(DB_PATH) as connection:
		cursor = connection.cursor()
		cursor.execute("DELETE FROM transactions")
		cursor.execute("DELETE FROM sqlite_sequence WHERE name='transactions'")
		connection.commit()


def reset_transactions(archive: bool = True) -> str:
	"""Archive transactions (optional) and clear the table. Returns archive path."""
	archive_path = ""
	if archive:
		archive_path = archive_transactions()
	clear_transactions()
	return archive_path


def get_recent_transactions(limit: int = 200) -> list[dict]:
	"""Load recent transactions from SQLite for dashboard metrics."""
	init_db()
	with sqlite3.connect(DB_PATH) as connection:
		connection.row_factory = sqlite3.Row
		cursor = connection.cursor()
		cursor.execute(
			"""
			SELECT timestamp, amount, prediction, fraud_probability, device, location
			FROM transactions
			ORDER BY id DESC
			LIMIT ?
			""",
			(limit,)
		)
		return [dict(row) for row in cursor.fetchall()]


def get_metrics() -> dict:
	"""Load total transaction count and fraud count."""
	init_db()
	with sqlite3.connect(DB_PATH) as connection:
		cursor = connection.cursor()
		cursor.execute("SELECT COUNT(*) FROM transactions")
		total = cursor.fetchone()[0]
		
		cursor.execute("SELECT COUNT(*) FROM transactions WHERE prediction = 'Fraud'")
		frauds = cursor.fetchone()[0]
		
		return {
			"total_transactions": total,
			"frauds_detected": frauds,
			"fraud_rate": (frauds / total) if total > 0 else 0.0
		}
