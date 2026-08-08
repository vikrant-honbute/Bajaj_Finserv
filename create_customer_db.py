import csv
import sqlite3
from pathlib import Path


SOURCE = Path(__file__).with_name("Customer_Data_.csv")
DATABASE = Path(__file__).with_name("customer_data.db")

NUMERIC_TYPES = {
    "Loan_amount": "INTEGER",
    "Interest": "REAL",
    "Tenure": "INTEGER",
    "Principal_amount": "INTEGER",
    # These columns contain both numeric values and YYYY-MM values in the source.
    "DPD_30": "TEXT",
    "DPD_90": "TEXT",
    "App_score": "REAL",
    "Imputed_Income": "INTEGER",
    "PIncode": "INTEGER",
    "EMI_amount": "INTEGER",
    "Age": "INTEGER",
}


def convert_value(column, value):
    if not value or not value.strip():
        return None
    if NUMERIC_TYPES.get(column) == "INTEGER":
        return int(value)
    if NUMERIC_TYPES.get(column) == "REAL":
        return float(value)
    return value


with SOURCE.open("r", encoding="utf-8-sig", newline="") as source_file:
    reader = csv.DictReader(source_file)
    columns = reader.fieldnames
    quoted_columns = ", ".join(
        f'"{column}" {NUMERIC_TYPES.get(column, "TEXT")}'
        for column in columns
    )
    placeholders = ", ".join("?" for _ in columns)

    with sqlite3.connect(DATABASE) as connection:
        connection.execute("DROP TABLE IF EXISTS customers")
        connection.execute(f"CREATE TABLE customers ({quoted_columns})")
        connection.executemany(
            f"INSERT INTO customers VALUES ({placeholders})",
            (
                [convert_value(column, row[column]) for column in columns]
                for row in reader
            ),
        )

print(f"Created {DATABASE.name} with {len(columns)} columns")
