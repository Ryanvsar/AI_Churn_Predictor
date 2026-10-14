import sqlite3

import pandas as pd

from backend.connectors.source import open_source
from backend.schema.detect import SchemaMap, detect_schema
from backend.schema.introspect import profile_database
from tests.conftest import LEGACY_DB


def detect(spec):
    src = open_source(spec, use_cache=False)
    return detect_schema(src, profile_database(src))


def acts(smap):
    return {a.table: a for a in smap.enabled_activities()}


def test_legacy_schema():
    m = detect(str(LEGACY_DB))
    assert (m.user_table, m.user_id_column) == ("users", "userId")
    a = acts(m)
    assert set(a) == {"sessions"}
    assert a["sessions"].time_format == "yyyymmdd"
    assert set(a["sessions"].measures) == {"sessionLength", "sessionEvents"}
    assert m.label and m.label.column == "churnLabel" and m.label.positive_values == ["1"]


def test_learning_app(sample_dir):
    m = detect(str(sample_dir / "learning_app.db"))
    assert (m.user_table, m.user_id_column) == ("learners", "learner_id")
    a = acts(m)
    assert set(a) == {"lessons", "achievements"}
    assert a["lessons"].time_column == "completed_at" and a["lessons"].time_format == "iso"
    assert {"xp_earned", "duration_sec", "mistakes"} <= set(a["lessons"].measures)
    assert "lesson_id" not in a["lessons"].measures
    assert a["lessons"].categoricals == ["lesson_type"]
    assert m.signup_column == "signup_date"
    assert "plan" in m.static_categoricals
    assert "email" not in m.static_categoricals  # personal data is never used
    assert m.label is None


def test_navigation_app_without_foreign_keys(sample_dir):
    m = detect(str(sample_dir / "navigation_app.db"))
    assert (m.user_table, m.user_id_column) == ("drivers", "id")
    a = acts(m)
    assert a["trips"].user_column == "driver_id" and a["trips"].time_format == "unix_ms"
    assert a["reports"].time_format == "unix_s"
    assert {"distance_km", "duration_min"} <= set(a["trips"].measures)
    assert m.signup_column == "joined_at" and m.signup_format == "unix_s"
    assert "country" in m.static_categoricals
    assert "phone" not in m.static_categoricals


def test_saas_app_one_join_hop(sample_dir):
    m = detect(str(sample_dir / "saas_app.db"))
    assert m.user_table == "accounts"
    a = acts(m)
    assert a["sessions"].user_column == "account_id"
    ev = a["events"]
    assert ev.via is not None and ev.via.table == "sessions" and ev.via.user_column == "account_id"
    assert ev.categoricals == ["event_name"]
    assert m.static_numerics == ["company_size"]


def test_leaky_and_label_columns(tmp_path):
    db = tmp_path / "app.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, status TEXT, total_orders INTEGER,
                                last_login TEXT, country TEXT, created_at TEXT);
        CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER, ordered_at TEXT, amount REAL);
    """)
    for i in range(1, 60):
        conn.execute("INSERT INTO customers VALUES (?,?,?,?,?,?)",
                     (i, "cancelled" if i % 5 == 0 else "active", i, "2025-03-01", "US" if i % 2 else "DE", "2025-01-01"))
        for k in range(3):
            conn.execute("INSERT INTO orders VALUES (NULL,?,?,?)", (i, f"2025-02-{10 + k:02d} 10:00:00", 9.5 + k))
    conn.commit()
    conn.close()
    m = detect(str(db))
    assert m.user_table == "customers"
    assert m.label.column == "status" and m.label.positive_values == ["cancelled"]
    assert "total_orders" not in m.static_numerics  # reflects current state -> would leak
    assert m.signup_column == "created_at"
    assert acts(m)["orders"].measures == ["amount"]


def test_sqlalchemy_url_and_csv_folder(sample_dir, tmp_path):
    # Same database through a URL (how Postgres/MySQL are connected) ...
    m = detect(f"sqlite:///{sample_dir / 'navigation_app.db'}")
    assert set(acts(m)) == {"trips", "reports"}
    # ... and exported to a folder of CSV files (types and keys must be inferred from values).
    conn = sqlite3.connect(sample_dir / "learning_app.db")
    for t in ("learners", "lessons"):
        pd.read_sql(f"SELECT * FROM {t} LIMIT 20000", conn).to_csv(tmp_path / f"{t}.csv", index=False)
    conn.close()
    m = detect(str(tmp_path))
    assert m.user_table == "learners"
    assert acts(m)["lessons"].user_column == "learner_id"
    assert acts(m)["lessons"].time_format == "iso"


def test_schema_map_roundtrip(sample_dir):
    m = detect(str(sample_dir / "saas_app.db"))
    again = SchemaMap.from_dict(m.to_dict())
    assert again.to_dict() == m.to_dict()
