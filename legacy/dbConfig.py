import os

# The legacy scripts live in legacy/ but read and write the demo database in the repo root.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "app_usage.db")
