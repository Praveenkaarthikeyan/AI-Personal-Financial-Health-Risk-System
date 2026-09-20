# ============================================================
# database.py
# Purpose: SQLite database layer for multi-user support.
#          Defines the schema (users, financial_records,
#          predictions) and provides simple CRUD functions that
#          the rest of the app (app.py, prediction.py) calls
#          instead of writing raw SQL everywhere.
#
# Uses SQLite because it's a single file, needs no separate
# server, and ships built into Python - ideal for a student
# project. The functions here are written so the underlying
# database could later be swapped for MySQL/PostgreSQL without
# changing how the rest of the app calls them.
# ============================================================

import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join("database", "financial_app.db")


def get_connection():
    """Opens a connection to the SQLite database file.
    Creates the 'database' folder if it doesn't exist yet."""
    os.makedirs("database", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row   # lets us access columns by name, e.g. row["income"]
    conn.execute("PRAGMA foreign_keys = ON")  # enforce foreign key relationships
    return conn


def init_db():
    """Creates all tables if they don't already exist. Safe to call
    every time the app starts - CREATE TABLE IF NOT EXISTS won't
    wipe existing data."""
    conn = get_connection()
    cur = conn.cursor()

    # ----------------------------------------------------------
    # users table: one row per customer
    # ----------------------------------------------------------
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT,
            phone TEXT,
            created_at TEXT NOT NULL
        )
    """)

    # ----------------------------------------------------------
    # financial_records table: one row per financial snapshot.
    # user_id links back to the users table (foreign key).
    # ----------------------------------------------------------
    cur.execute("""
        CREATE TABLE IF NOT EXISTS financial_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            monthly_income REAL,
            monthly_expenses REAL,
            monthly_savings REAL,
            existing_emi REAL,
            other_debt REAL,
            loan_amount REAL,
            loan_tenure_months INTEGER,
            credit_history REAL,
            dependents INTEGER,
            gender TEXT,
            married TEXT,
            education TEXT,
            self_employed TEXT,
            property_area TEXT,
            employment_status TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)

    # ----------------------------------------------------------
    # predictions table: one row per prediction result.
    # financial_record_id links back to financial_records.
    # ----------------------------------------------------------
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            financial_record_id INTEGER NOT NULL,
            loan_prediction TEXT,
            loan_approval_probability REAL,
            health_score REAL,
            health_category TEXT,
            risk_level TEXT,
            risk_probability REAL,
            cluster_label TEXT,
            is_anomaly INTEGER,
            created_at TEXT NOT NULL,
            FOREIGN KEY (financial_record_id) REFERENCES financial_records(id) ON DELETE CASCADE
        )
    """)

    conn.commit()
    conn.close()


# ================================================================
# USER CRUD (Create, Read, Update, Delete)
# ================================================================

def add_user(name: str, email: str = "", phone: str = "") -> int:
    """Adds a new user, returns the new user's id."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO users (name, email, phone, created_at) VALUES (?, ?, ?, ?)",
        (name, email, phone, datetime.now().isoformat())
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def get_all_users() -> list:
    """Returns every user as a list of dicts."""
    conn = get_connection()
    rows = conn.execute("SELECT * FROM users ORDER BY name").fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_user(user_id: int) -> dict:
    """Returns a single user by id, or None if not found."""
    conn = get_connection()
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def search_users(keyword: str) -> list:
    """Searches users by name or email (case-insensitive partial match)."""
    conn = get_connection()
    pattern = f"%{keyword}%"
    rows = conn.execute(
        "SELECT * FROM users WHERE name LIKE ? OR email LIKE ?",
        (pattern, pattern)
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def update_user(user_id: int, name: str = None, email: str = None, phone: str = None):
    """Updates only the fields provided (None fields are left unchanged)."""
    existing = get_user(user_id)
    if not existing:
        return False
    conn = get_connection()
    conn.execute(
        "UPDATE users SET name = ?, email = ?, phone = ? WHERE id = ?",
        (
            name if name is not None else existing["name"],
            email if email is not None else existing["email"],
            phone if phone is not None else existing["phone"],
            user_id
        )
    )
    conn.commit()
    conn.close()
    return True


def delete_user(user_id: int):
    """Deletes a user AND all their financial_records/predictions,
    thanks to ON DELETE CASCADE set up in the schema."""
    conn = get_connection()
    conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()


# ================================================================
# FINANCIAL RECORD operations
# ================================================================

def add_financial_record(user_id: int, record: dict) -> int:
    """Adds a new financial snapshot for a user. 'record' is a dict
    with keys matching the financial_records columns. Returns the
    new record's id."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO financial_records (
            user_id, monthly_income, monthly_expenses, monthly_savings,
            existing_emi, other_debt, loan_amount, loan_tenure_months,
            credit_history, dependents, gender, married, education,
            self_employed, property_area, employment_status, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id,
        record.get("monthly_income"), record.get("monthly_expenses"),
        record.get("monthly_savings"), record.get("existing_emi"),
        record.get("other_debt"), record.get("loan_amount"),
        record.get("loan_tenure_months"), record.get("credit_history"),
        record.get("dependents"), record.get("gender"), record.get("married"),
        record.get("education"), record.get("self_employed"),
        record.get("property_area"), record.get("employment_status"),
        datetime.now().isoformat()
    ))
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def get_records_for_user(user_id: int) -> list:
    """Returns all financial records for a user, oldest first -
    used for the History/Trend feature."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM financial_records WHERE user_id = ? ORDER BY created_at ASC",
        (user_id,)
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


# ================================================================
# PREDICTION operations
# ================================================================

def add_prediction(financial_record_id: int, result: dict) -> int:
    """Saves a prediction result linked to a financial record."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO predictions (
            financial_record_id, loan_prediction, loan_approval_probability,
            health_score, health_category, risk_level, risk_probability,
            cluster_label, is_anomaly, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        financial_record_id,
        result.get("loan_prediction"), result.get("loan_approval_probability"),
        result.get("health_score"), result.get("health_category"),
        result.get("risk_level"), result.get("risk_probability"),
        result.get("cluster_label"), int(result.get("is_anomaly", 0)),
        datetime.now().isoformat()
    ))
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def get_predictions_for_user(user_id: int) -> list:
    """Returns every prediction ever made for a user, joined with
    the financial record it was based on - newest first. Used for
    the Financial History dashboard page."""
    conn = get_connection()
    rows = conn.execute("""
        SELECT p.*, f.monthly_income, f.monthly_expenses, f.monthly_savings,
               f.created_at as record_date
        FROM predictions p
        JOIN financial_records f ON p.financial_record_id = f.id
        WHERE f.user_id = ?
        ORDER BY p.created_at DESC
    """, (user_id,)).fetchall()
    conn.close()
    return [dict(row) for row in rows]


# ------------------------------------------------------------
# Quick manual test - only runs when this file is executed
# directly (python src/database.py), not when imported.
# ------------------------------------------------------------
if __name__ == "__main__":
    init_db()
    print("Database initialized at:", DB_PATH)

    # Add a test user
    uid = add_user("Test User", "test@example.com", "9999999999")
    print(f"Added user with id: {uid}")

    # Add a test financial record
    rid = add_financial_record(uid, {
        "monthly_income": 40000, "monthly_expenses": 20000,
        "monthly_savings": 5000, "existing_emi": 3000, "other_debt": 0,
        "loan_amount": 200, "loan_tenure_months": 60,
        "credit_history": 1.0, "dependents": 2, "gender": "Male",
        "married": "Yes", "education": "Graduate", "self_employed": "No",
        "property_area": "Urban", "employment_status": "Salaried"
    })
    print(f"Added financial record with id: {rid}")

    # Add a test prediction
    pid = add_prediction(rid, {
        "loan_prediction": "Approved", "loan_approval_probability": 0.82,
        "health_score": 72.2, "health_category": "Good",
        "risk_level": "Low Risk", "risk_probability": 0.15,
        "cluster_label": "Balanced", "is_anomaly": 0
    })
    print(f"Added prediction with id: {pid}")

    print("\nAll users:", get_all_users())
    print("\nRecords for user:", get_records_for_user(uid))
    print("\nPredictions for user:", get_predictions_for_user(uid))