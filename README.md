# Project_001_Churn
This program will take information VIA a database and utilize Machine learning to predict if a user is going to churn based on usage metrics. The program will provide an alert if a user is predicted to churn.  

--------- Project Outline ---------

1. Use python to create users, simulate sessions, and inject churn behaviour patterns

2. Create database using SQLite.
    Create tables; track sessions, events, features used, etc. 

3. Create SQL Queries

4. Use python to convert query results to dataframes and calculate trends

5. Train Logistic Regression ML model to spot and predict churn
   
6. Label churn behaviours

7. Send an alert containing the users that are predicted to churn in the near future

## New (revamp): Local web app (Option A)

This repo now includes:
- **FastAPI backend** (`backend/app.py`) that loads a local SQLite database, runs churn analysis, and serves results.
- **React frontend UI** served by the backend at `http://localhost:8000/ui` (no Node/npm required for v1).

### 1) Setup (Windows / PowerShell)

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

### 2) Generate demo database (optional)

This will create `app_usage.db` in the repo folder:

```bash
python -c "from saveToDb2 import save_to_db2; save_to_db2()"
```

### 3) Run the backend + UI

Start the API server:

```bash
python -m uvicorn backend.app:app --reload --port 8000
```

If you see `WinError 10048`, port 8000 is already in use. Either stop the other process, or run on another port:

```bash
python -m uvicorn backend.app:app --reload --port 8001
```

Open the UI in your browser:
- `http://localhost:8000/ui`

In the UI:
- Paste the DB path (for demo: `D:\\Coding\\churn\\Project_001_Churn\\app_usage.db`)
- Click **Validate DB**
- Click **Run Churn Analysis**

### Notes
- **Explainability (SHAP)** requires the `shap` package to be installed (`pip install -r requirements.txt`). If it isn’t installed, the UI still works but the “Why risky” panel will show a fallback message.
- The database schema supports optional tables for richer product-depth analytics:
  - `feature_usage`
  - `session_events`
