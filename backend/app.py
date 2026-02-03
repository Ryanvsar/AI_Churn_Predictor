from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.db.sqlite_validate import inspect_db, validate_sqlite_db_path
from backend.db.sessions import load_user_sessions_timeseries
from backend.db.windows import derive_observation_window
from backend.explain.shap_explain import (
    ShapArtifacts,
    top_factors_for_user,
    why_risky_summary,
)
from backend.jobs.runner import run_job
from backend.jobs.store import JOBS, JobRecord
from backend.model.train_and_score import ModelConfig, train_and_score
from backend.schemas.api import (
    DbValidateRequest,
    DbValidateResponse,
    JobStatusResponse,
    RunAnalysisRequest,
    RunAnalysisResponse,
    UserDetailsResponse,
    UserExplainResponse,
    UsersListResponse,
    AggregatesResponse,
    UserTimeSeriesResponse,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = REPO_ROOT / "frontend"

app = FastAPI(title="Churn Prediction Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    # Local-only dev: the UI can be served by this backend or a separate dev server.
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="ui")


@app.get("/health")
def health():
    return {"ok": True, "time": datetime.utcnow().isoformat() + "Z"}


@app.post("/db/validate", response_model=DbValidateResponse)
def db_validate(payload: DbValidateRequest):
    ok, err = validate_sqlite_db_path(payload.dbPath)
    if not ok:
        raise HTTPException(status_code=400, detail=err)

    info = inspect_db(payload.dbPath)
    if not info["ok"]:
        raise HTTPException(status_code=400, detail=info["error"] or "DB validation failed.")

    return {
        "ok": True,
        "dbPath": payload.dbPath,
        "tables": info["tables"],
        "capabilities": info["capabilities"],
    }


def _analysis_job_impl(job: JobRecord, payload: RunAnalysisRequest):
    job.progress = 0.1

    info = inspect_db(payload.dbPath)
    if not info["ok"]:
        raise RuntimeError(info["error"] or "DB validation failed.")
    caps = info["capabilities"]
    if not caps.get("hasSessionsTable", False):
        raise RuntimeError("Required table `sessions` not found.")

    job.progress = 0.25

    # If the caller didn't provide an observation window, derive a safe default that avoids
    # leaking the "future churn period" into training on the demo dataset.
    obs_start = payload.observationStart
    obs_end = payload.observationEnd
    if obs_start is None or obs_end is None:
        w = derive_observation_window(payload.dbPath, observation_days=60, prediction_days=40)
        if w is not None:
            obs_start = w.observation_start
            obs_end = w.observation_end

    model_cfg = ModelConfig(threshold=payload.threshold)
    out = train_and_score(
        db_path=payload.dbPath,
        observation_start=obs_start,
        observation_end=obs_end,
        config=model_cfg,
    )
    df_scored = out["df_scored"]
    features = out["features"]

    job.progress = 0.7

    shap_art: ShapArtifacts | None = None
    if out["supervised"] and out["model"] is not None:
        # SHAP is optional at runtime; if not installed, we keep the API working without explanations.
        try:
            from backend.explain.shap_explain import compute_shap_for_xgb

            # Recompute scaled X using same scaler stats (we stored scaler only, not X_scaled).
            # For v1 we approximate: use raw feature values then scale/noise similarly here.
            import numpy as np
            from sklearn.preprocessing import StandardScaler

            X = df_scored[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            np.random.seed(model_cfg.random_state)
            X_noisy = X.to_numpy(dtype=float) + np.random.normal(0, 0.02, X.shape)
            scaler: StandardScaler = out["scaler"]
            X_scaled = scaler.transform(X_noisy)

            shap_art = compute_shap_for_xgb(out["model"], X_scaled, features, max_rows=5000)
        except Exception:
            shap_art = None

    job.progress = 0.95
    summary = {
        "observationStart": obs_start,
        "observationEnd": obs_end,
        "usersTotal": int(df_scored.shape[0]),
        "usersScored": int(df_scored.shape[0]),
        "featuresUsed": list(features),
        "supervised": bool(out["supervised"]),
    }
    artifacts = {
        "dbPath": payload.dbPath,
        "threshold": payload.threshold,
        "observationStart": obs_start,
        "observationEnd": obs_end,
        "capabilities": caps,
        "df_scored": df_scored,
        "features": features,
        "shap": shap_art,
        "global_importance": (shap_art.global_importance if shap_art else []),
    }

    return {"summary": summary, "artifacts": artifacts}


@app.post("/analysis/run", response_model=RunAnalysisResponse)
def analysis_run(payload: RunAnalysisRequest, background: BackgroundTasks):
    ok, err = validate_sqlite_db_path(payload.dbPath)
    if not ok:
        raise HTTPException(status_code=400, detail=err)

    job_id = str(uuid.uuid4())
    job = JobRecord(jobId=job_id)
    JOBS[job_id] = job
    background.add_task(run_job, job, _analysis_job_impl, payload)
    return {"jobId": job_id}


@app.get("/analysis/jobs/{job_id}", response_model=JobStatusResponse)
def analysis_job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    return {
        "jobId": job.jobId,
        "status": job.status,
        "progress": job.progress,
        "startedAt": job.startedAt,
        "finishedAt": job.finishedAt,
        "error": job.error,
        "summary": job.summary,
    }


@app.get("/analysis/{job_id}/users", response_model=UsersListResponse)
def analysis_users(job_id: str, top: int = 100, threshold: float = 0.0, sort: str = "prob_desc"):
    job = JOBS.get(job_id)
    if not job or job.status != "succeeded":
        raise HTTPException(status_code=404, detail="Job not found or not complete.")

    df = job.artifacts.get("df_scored")
    if df is None:
        raise HTTPException(status_code=500, detail="Job results unavailable.")

    dfu = df[["userId", "churnProbability", "riskTier"]].copy()
    dfu = dfu[dfu["churnProbability"] >= float(threshold)]

    if sort == "prob_asc":
        dfu = dfu.sort_values("churnProbability", ascending=True)
    else:
        dfu = dfu.sort_values("churnProbability", ascending=False)

    top = max(1, min(int(top), 50000))
    dfu = dfu.head(top)

    users = [
        {
            "userId": int(r["userId"]),
            "churnProbability": float(r["churnProbability"]),
            "riskTier": r["riskTier"],
        }
        for _, r in dfu.iterrows()
    ]
    return {"jobId": job_id, "total": int(dfu.shape[0]), "users": users}


@app.get("/analysis/{job_id}/users/{user_id}", response_model=UserDetailsResponse)
def analysis_user_details(job_id: str, user_id: int):
    job = JOBS.get(job_id)
    if not job or job.status != "succeeded":
        raise HTTPException(status_code=404, detail="Job not found or not complete.")

    df = job.artifacts.get("df_scored")
    if df is None:
        raise HTTPException(status_code=500, detail="Job results unavailable.")

    row = df[df["userId"] == int(user_id)]
    if row.empty:
        raise HTTPException(status_code=404, detail="User not found in results.")
    r = row.iloc[0].to_dict()
    metrics = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in r.items() if k not in {"userId"}}

    return {
        "jobId": job_id,
        "userId": int(user_id),
        "churnProbability": float(r["churnProbability"]),
        "riskTier": r["riskTier"],
        "metrics": metrics,
    }


@app.get("/analysis/{job_id}/users/{user_id}/explain", response_model=UserExplainResponse)
def analysis_user_explain(job_id: str, user_id: int):
    job = JOBS.get(job_id)
    if not job or job.status != "succeeded":
        raise HTTPException(status_code=404, detail="Job not found or not complete.")

    df = job.artifacts.get("df_scored")
    shap_art = job.artifacts.get("shap")
    features = job.artifacts.get("features") or []
    if df is None:
        raise HTTPException(status_code=500, detail="Job results unavailable.")

    row = df[df["userId"] == int(user_id)]
    if row.empty:
        raise HTTPException(status_code=404, detail="User not found in results.")
    user_row = row.iloc[0]

    if shap_art is None:
        # Graceful fallback when SHAP isn't installed or supervised training wasn't possible.
        return {
            "jobId": job_id,
            "userId": int(user_id),
            "churnProbability": float(user_row["churnProbability"]),
            "topFactors": [],
            "whyRiskySummary": "Explainability is not available (model not supervised or SHAP not installed).",
        }

    # Map row index to shap row position (we computed SHAP for the same df order)
    idx = int(row.index[0])
    shap_row = shap_art.shap_values[idx]
    top = top_factors_for_user(user_row, shap_row, shap_art.feature_names, top_k=8)
    return {
        "jobId": job_id,
        "userId": int(user_id),
        "churnProbability": float(user_row["churnProbability"]),
        "topFactors": top,
        "whyRiskySummary": why_risky_summary(top),
    }


@app.get("/analysis/{job_id}/users/{user_id}/timeseries", response_model=UserTimeSeriesResponse)
def analysis_user_timeseries(job_id: str, user_id: int):
    job = JOBS.get(job_id)
    if not job or job.status != "succeeded":
        raise HTTPException(status_code=404, detail="Job not found or not complete.")

    db_path = job.artifacts.get("dbPath")
    if not db_path:
        raise HTTPException(status_code=500, detail="Job DB path unavailable.")

    df = load_user_sessions_timeseries(
        db_path=db_path,
        user_id=int(user_id),
        observation_start=job.artifacts.get("observationStart"),
        observation_end=job.artifacts.get("observationEnd"),
    )
    points = [
        {"date": r["date"], "sessionLength": float(r["sessionLength"]), "sessionEvents": float(r["sessionEvents"])}
        for _, r in df.iterrows()
    ]
    return {"jobId": job_id, "userId": int(user_id), "points": points}


@app.get("/analysis/{job_id}/aggregates", response_model=AggregatesResponse)
def analysis_aggregates(job_id: str):
    job = JOBS.get(job_id)
    if not job or job.status != "succeeded":
        raise HTTPException(status_code=404, detail="Job not found or not complete.")

    df = job.artifacts.get("df_scored")
    if df is None:
        raise HTTPException(status_code=500, detail="Job results unavailable.")

    # Numeric metrics only, excluding probability itself (we include it as a metric too).
    numeric_cols = [
        c
        for c in df.columns
        if c not in {"userId", "riskTier"}
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    def group_summary(name: str, dfg):
        means = dfg[numeric_cols].mean(numeric_only=True).fillna(0.0).to_dict()
        means = {k: float(v) for k, v in means.items()}
        return {"name": name, "count": int(dfg.shape[0]), "metricsMean": means}

    groups = [
        group_summary("AllUsers", df),
        group_summary("HighRisk", df[df["riskTier"] == "High"]),
        group_summary("MediumRisk", df[df["riskTier"] == "Medium"]),
        group_summary("LowRisk", df[df["riskTier"] == "Low"]),
    ]

    return {
        "jobId": job_id,
        "groups": groups,
        "globalFeatureImportance": job.artifacts.get("global_importance", []),
    }

