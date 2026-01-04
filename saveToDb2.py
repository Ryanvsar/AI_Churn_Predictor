"""
------------------------------------------------------
Author: Ryan Vandersar
Date (DD-MM-YYYY): 26-12-2025
------------------------------------------------------
Program Title: Churn Prediction AI
------------------------------------------------------
File Description:
- Loads advanced simulated usage data into SQLite
- Works with usageSimulation2.py
------------------------------------------------------
"""

import os
import sqlite3
from dbConfig import DB_PATH
from usageSimulation2 import generate_users


def save_to_db2():
    # Resolve absolute paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    schema_path = os.path.join(base_dir, "schema.sql")

    # Load schema
    with open(schema_path, "r") as f:
        schema_sql = f.read()

    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create tables
    cursor.executescript(schema_sql)
    conn.commit()

    # Generate simulated data
    users, churned_user_ids, at_risk_user_ids = generate_users()

    # -----------------------------
    # Insert users
    # -----------------------------
    for user in users.values():
        cursor.execute(
            """
            INSERT OR REPLACE INTO users (userId, churnLabel)
            VALUES (?, ?)
            """,
            (user["userId"], user["churnLabel"])
        )

    # -----------------------------
    # Insert sessions
    # -----------------------------
    for user in users.values():
        for session in user["sessions"]:
            cursor.execute(
                """
                INSERT OR REPLACE INTO sessions (
                    sessionId,
                    userId,
                    sessionDate,
                    sessionLength,
                    sessionEvents
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session["sessionId"],
                    user["userId"],
                    session["sessionDate"],
                    session["sessionLength"],
                    session["sessionEvents"]
                )
            )

    conn.commit()
    conn.close()

    print("\n--- Database Load Complete ---")
    print(f"Users inserted: {len(users)}")

    return users, churned_user_ids, at_risk_user_ids