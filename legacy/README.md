# Legacy scripts

The original command-line prototype, kept for reference. They only work with the fixed demo schema
(`schema.sql`: `users`, `sessions`, `feature_usage`) and are not used by the web app.

```bash
cd legacy
python main.py   # regenerates ../app_usage.db with usageSimulation2 and runs the original XGBoost script
```

Note: `main.py` overwrites `app_usage.db` in the repository root.
For any other database use the web app (`uvicorn backend.app:app`) or `backend.pipeline.run.run_analysis`.
