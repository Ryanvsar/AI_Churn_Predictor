"""
------------------------------------------------------
Author: Ryan Vandersar
Date (DD-MM-YYYY): 22-12-2025
------------------------------------------------------
Program Title: Churn Prediction AI
------------------------------------------------------
File Description: Works with schema.sql to fill the db
------------------------------------------------------
"""

import os
import sqlite3
from dbConfig import DB_PATH
from usageSimulation import generate_users


def save_to_db():
    # Absolute path to schema.sql
    base_dir = os.path.dirname(os.path.abspath(__file__))
    schema_path = os.path.join(base_dir, "schema.sql")

    # Load schema
    with open(schema_path, "r") as f:
        schema_sql = f.read()

    # CONNECT USING THE SINGLE SOURCE OF TRUTH
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create tables
    cursor.executescript(schema_sql)
    conn.commit()

    # Generate simulated users + known churn labels
    users, churn_user_ids = generate_users()

    # Insert users
    for user in users.values():
        cursor.execute(
            """
            INSERT OR IGNORE INTO users (userId, churnPred)
            VALUES (?, ?)
            """,
            (user["userId"], user["churnPred"])
        )

    # Insert sessions
    for user in users.values():
        for session in user["sessions"]:
            cursor.execute(
                """
                INSERT OR IGNORE INTO sessions (
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
    return users, churn_user_ids
