from __future__ import annotations

import os
import sqlite3
from typing import Dict, List, Tuple


def _table_columns(conn: sqlite3.Connection, table: str) -> List[Dict]:
    cur = conn.execute(f"PRAGMA table_info({table});")
    cols = []
    for row in cur.fetchall():
        # cid, name, type, notnull, dflt_value, pk
        cols.append(
            {
                "name": row[1],
                "type": row[2],
                "notnull": bool(row[3]),
                "pk": bool(row[5]),
            }
        )
    return cols


def _list_tables(conn: sqlite3.Connection) -> List[str]:
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    return [r[0] for r in cur.fetchall()]


def validate_sqlite_db_path(db_path: str) -> Tuple[bool, str]:
    if not db_path.strip():
        return False, "dbPath is empty."
    if not os.path.exists(db_path):
        return False, f"Database file not found: {db_path}"
    if not os.path.isfile(db_path):
        return False, f"dbPath is not a file: {db_path}"
    return True, ""


def inspect_db(db_path: str) -> Dict:
    ok, err = validate_sqlite_db_path(db_path)
    if not ok:
        return {
            "ok": False,
            "error": err,
            "tables": [],
            "capabilities": {},
        }

    conn = sqlite3.connect(db_path)
    try:
        tables = _list_tables(conn)
        tables_info = []
        for t in tables:
            cols = _table_columns(conn, t)
            tables_info.append({"name": t, "columns": cols})

        table_set = set(tables)
        has_users = "users" in table_set
        has_sessions = "sessions" in table_set
        has_feature_usage = "feature_usage" in table_set
        has_session_events = "session_events" in table_set

        has_churn_labels = False
        notes: List[str] = []

        if has_users:
            users_cols = {c["name"] for c in _table_columns(conn, "users")}
            if "churnLabel" in users_cols:
                has_churn_labels = True
            else:
                notes.append("`users` table exists but lacks `churnLabel`; supervised training will be unavailable.")
        else:
            notes.append("Missing `users` table; supervised training will be unavailable.")

        if not has_sessions:
            notes.append("Missing `sessions` table; cannot compute churn features.")

        if not has_feature_usage and not has_session_events:
            notes.append("No feature usage tables found; feature-depth explanations will be limited.")

        capabilities = {
            "hasUsersTable": has_users,
            "hasSessionsTable": has_sessions,
            "hasChurnLabels": has_churn_labels,
            "hasFeatureUsageTable": has_feature_usage,
            "hasSessionEventsTable": has_session_events,
            "notes": notes,
        }

        return {"ok": True, "error": None, "tables": tables_info, "capabilities": capabilities}
    finally:
        conn.close()

