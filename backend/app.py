from __future__ import annotations

import json
import math
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from backend.connectors.source import SourceError, open_source
from backend.data.timeseries import user_daily_series
from backend.explain.contrib import top_factors_for_user, why_risky_summary
from backend.jobs.runner import run_job
from backend.jobs.store import JOBS, JobRecord
from backend.pipeline.run import AnalysisOptions, AnalysisResult, run_analysis
from backend.schema.detect import SchemaMap, detect_schema
from backend.schema.introspect import profile_database
from backend.schemas.api import (
    AggregatesResponse,
    DbValidateRequest,
    DbValidateResponse,
    JobStatusResponse,
    MappingInfo,
    MappingSaveRequest,
    ModelInfoResponse,
    RunAnalysisRequest,
    RunAnalysisResponse,
    UploadResponse,
    UserDetailsResponse,
    UserExplainResponse,
    UsersListResponse,
    UserTimeSeriesResponse,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = REPO_ROOT / "frontend"
UPLOAD_DIR = REPO_ROOT / "uploads"
MAPPINGS_DIR = REPO_ROOT / "mappings"
UPLOAD_SUFFIXES = {".db", ".sqlite", ".sqlite3", ".db3", ".csv", ".tsv", ".parquet", ".xlsx", ".xls", ".zip"}

app = FastAPI(title="Churn Prediction Backend", version="2.0.0")

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


def _clean(v: Any) -> Any:
    """JSON-safe values: NaN/inf -> None, numpy scalars -> python."""
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return f if math.isfinite(f) else None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if v is pd.NA or v is pd.NaT:
        return None
    return v


@app.get("/")
def root():
    return RedirectResponse(url="/ui/")


@app.get("/health")
def health():
    return {"ok": True, "time": datetime.now(timezone.utc).isoformat()}


# ------------------------------------------------------------------------------------------------
# Data sources
# ------------------------------------------------------------------------------------------------

@app.post("/db/validate", response_model=DbValidateResponse)
def db_validate(payload: DbValidateRequest):
    try:
        source = open_source(payload.source)
        profile = profile_database(source)
        smap = detect_schema(source, profile)
    except SourceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001 - report unexpected database errors to the UI
        raise HTTPException(status_code=400, detail=f"Could not read the database: {e}")

    acts = smap.enabled_activities()
    caps = {
        "hasUsersTable": smap.user_table is not None,
        "hasSessionsTable": bool(acts),
        "hasChurnLabels": smap.label is not None,
        "hasFeatureUsageTable": any(a.categoricals for a in acts),
        "hasSessionEventsTable": len(acts) > 1,
        "activityTables": [a.table for a in acts],
        "notes": smap.notes,
    }
    return {
        "ok": bool(acts),
        "dbPath": payload.source,
        "source": source.display,
        "dialect": source.dialect,
        "tables": profile.to_dict()["tables"],
        "capabilities": caps,
        "schemaMap": smap.to_dict(),
    }


@app.post("/db/upload", response_model=UploadResponse)
async def db_upload(file: UploadFile = File(...)):
    name = Path(file.filename or "upload").name
    suffix = Path(name).suffix.lower()
    if suffix not in UPLOAD_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(UPLOAD_SUFFIXES)}")
    UPLOAD_DIR.mkdir(exist_ok=True)
    safe = re.sub(r"[^0-9A-Za-z._-]+", "_", name)
    dest = UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{safe}"
    with open(dest, "wb") as out:
        shutil.copyfileobj(file.file, out)
    return {"path": str(dest), "filename": name, "sizeBytes": dest.stat().st_size}


def _mapping_path(name: str) -> Path:
    safe = re.sub(r"[^0-9A-Za-z_-]+", "_", name).strip("_")
    if not safe:
        raise HTTPException(status_code=400, detail="Invalid mapping name.")
    return MAPPINGS_DIR / f"{safe}.json"


@app.get("/mappings")
def list_mappings():
    if not MAPPINGS_DIR.exists():
        return {"mappings": []}
    return {"mappings": sorted(p.stem for p in MAPPINGS_DIR.glob("*.json"))}


@app.get("/mappings/{name}", response_model=MappingInfo)
def get_mapping(name: str):
    path = _mapping_path(name)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Mapping not found.")
    return {"name": path.stem, "schemaMap": json.loads(path.read_text())}


@app.post("/mappings", response_model=MappingInfo)
def save_mapping(payload: MappingSaveRequest):
    try:
        smap = SchemaMap.from_dict(payload.schemaMap)
    except TypeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid schema map: {e}")
    MAPPINGS_DIR.mkdir(exist_ok=True)
    path = _mapping_path(payload.name)
    path.write_text(json.dumps(smap.to_dict(), indent=2))
    return {"name": path.stem, "schemaMap": smap.to_dict()}


# ------------------------------------------------------------------------------------------------
# Analysis jobs
# ------------------------------------------------------------------------------------------------

def _analysis_job_impl(job: JobRecord, payload: RunAnalysisRequest):
    def progress(p: float, msg: str):
        job.progress = float(p)
        job.message = msg

    opts = AnalysisOptions(
        source=payload.source,
        schema_map=payload.schemaMap,
        churn_window_days=payload.churnWindowDays,
        label_source=payload.labelSource,
        threshold=payload.threshold,
        high_risk_threshold=max(payload.highRiskThreshold, payload.threshold),
        max_users=payload.maxUsers,
        observation_start=payload.observationStart,
        observation_end=payload.observationEnd,
    )
    res: AnalysisResult = run_analysis(opts, progress)
    df = res.df_scored
    churn = res.churn.to_dict()
    summary = {
        "observationStart": str(res.data.min_ts.date()) if res.data.min_ts is not None else None,
        "observationEnd": churn["scoring_date"],
        "usersTotal": int(df.shape[0]),
        "usersScored": int(df.shape[0]),
        "usersActive": int((df["status"] == "active").sum()),
        "featuresUsed": list(res.features),
        "supervised": bool(res.supervised),
        "churnDefinition": churn,
        "metrics": _clean(res.metrics),
        "notes": res.notes,
    }
    artifacts = {
        "result": res,
        "row_index": {uid: i for i, uid in enumerate(df["userId"].tolist())},
        "threshold": payload.threshold,
    }
    return {"summary": summary, "artifacts": artifacts}


@app.post("/analysis/run", response_model=RunAnalysisResponse)
def analysis_run(payload: RunAnalysisRequest, background: BackgroundTasks):
    try:
        open_source(payload.source)
    except SourceError as e:
        raise HTTPException(status_code=400, detail=str(e))

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
        "message": job.message,
        "startedAt": job.startedAt,
        "finishedAt": job.finishedAt,
        "error": job.error,
        "summary": job.summary,
    }


def _result(job_id: str) -> tuple[JobRecord, AnalysisResult]:
    job = JOBS.get(job_id)
    if not job or job.status != "succeeded":
        raise HTTPException(status_code=404, detail="Job not found or not complete.")
    res = job.artifacts.get("result")
    if res is None:
        raise HTTPException(status_code=500, detail="Job results unavailable.")
    return job, res


def _user_row(job: JobRecord, res: AnalysisResult, user_id: str) -> tuple[int, pd.Series]:
    idx = job.artifacts["row_index"].get(str(user_id))
    if idx is None:
        raise HTTPException(status_code=404, detail="User not found in results.")
    return idx, res.df_scored.iloc[idx]


def _known_label(row: pd.Series):
    v = row.get("knownLabel")
    return None if v is None or pd.isna(v) else int(v)


@app.get("/analysis/{job_id}/users", response_model=UsersListResponse)
def analysis_users(
    job_id: str,
    top: int = 100,
    threshold: float = 0.0,
    sort: str = "prob_desc",
    includeChurned: bool = True,
):
    _, res = _result(job_id)
    df = res.df_scored
    cols = ["userId", "churnProbability", "riskTier", "status"] + (["knownLabel"] if "knownLabel" in df else [])
    dfu = df[cols]
    if not includeChurned:
        dfu = dfu[dfu["status"] != "churned"]
    dfu = dfu[dfu["churnProbability"] >= float(threshold)]
    dfu = dfu.sort_values("churnProbability", ascending=(sort == "prob_asc"), kind="stable")
    total = int(dfu.shape[0])
    dfu = dfu.head(max(1, min(int(top), 50000)))

    users = [
        {
            "userId": str(r.userId),
            "churnProbability": float(r.churnProbability),
            "riskTier": r.riskTier,
            "status": r.status,
            "knownLabel": None if not hasattr(r, "knownLabel") or pd.isna(r.knownLabel) else int(r.knownLabel),
        }
        for r in dfu.itertuples(index=False)
    ]
    return {"jobId": job_id, "total": total, "users": users}


@app.get("/analysis/{job_id}/users/{user_id}", response_model=UserDetailsResponse)
def analysis_user_details(job_id: str, user_id: str):
    job, res = _result(job_id)
    _, row = _user_row(job, res, user_id)
    skip = {"userId", "churnProbability", "riskTier", "status", "knownLabel"}
    metrics = {k: _clean(v) for k, v in row.items() if k not in skip}
    return {
        "jobId": job_id,
        "userId": str(user_id),
        "churnProbability": float(row["churnProbability"]),
        "riskTier": row["riskTier"],
        "status": row["status"],
        "knownLabel": _known_label(row),
        "metrics": metrics,
        "metricLabels": {k: res.feature_labels.get(k, k) for k in metrics},
    }


@app.get("/analysis/{job_id}/users/{user_id}/explain", response_model=UserExplainResponse)
def analysis_user_explain(job_id: str, user_id: str):
    job, res = _result(job_id)
    idx, row = _user_row(job, res, user_id)

    if res.contributions is None:
        return {
            "jobId": job_id,
            "userId": str(user_id),
            "churnProbability": float(row["churnProbability"]),
            "topFactors": [],
            "whyRiskySummary": "Explanations need a trained model; this run used the cadence-based heuristic "
                               "because there was not enough labelled history.",
        }
    if row["status"] == "churned":
        why = (
            "Already churned according to the label column in your database."
            if res.churn.label_source == "explicit"
            else f"Already churned: no activity for {int(row['recency_days'])} days "
                 f"(churn window is {res.churn.window_days} days)."
        )
        return {
            "jobId": job_id,
            "userId": str(user_id),
            "churnProbability": float(row["churnProbability"]),
            "topFactors": [],
            "whyRiskySummary": why,
        }
    top = top_factors_for_user(row, res.contributions[idx], res.features, res.feature_labels, top_k=8)
    return {
        "jobId": job_id,
        "userId": str(user_id),
        "churnProbability": float(row["churnProbability"]),
        "topFactors": _clean(top),
        "whyRiskySummary": why_risky_summary(top),
    }


@app.get("/analysis/{job_id}/users/{user_id}/timeseries", response_model=UserTimeSeriesResponse)
def analysis_user_timeseries(job_id: str, user_id: str, days: int = 0):
    job, res = _result(job_id)
    _user_row(job, res, user_id)
    n_days = int(days) if days and days > 0 else max(res.churn.lookback_days, 60)
    names, points = user_daily_series(res.data, str(user_id), res.scoring_day, min(n_days, 730))
    return {
        "jobId": job_id,
        "userId": str(user_id),
        "measureNames": names,
        "scoringDate": res.churn.scoring_date,
        "points": points,
    }


@app.get("/analysis/{job_id}/aggregates", response_model=AggregatesResponse)
def analysis_aggregates(job_id: str):
    _, res = _result(job_id)
    df = res.df_scored
    numeric_cols = [
        c for c in df.columns
        if c not in {"userId", "riskTier", "status", "knownLabel"} and pd.api.types.is_numeric_dtype(df[c])
    ]

    def group_summary(name: str, dfg: pd.DataFrame):
        means = dfg[numeric_cols].mean(numeric_only=True).to_dict()
        return {"name": name, "count": int(dfg.shape[0]),
                "metricsMean": {k: float(v) for k, v in means.items() if pd.notna(v) and math.isfinite(v)}}

    active = df[df["status"] != "churned"]
    groups = [
        group_summary("AllUsers", df),
        group_summary("HighRisk", active[active["riskTier"] == "High"]),
        group_summary("MediumRisk", active[active["riskTier"] == "Medium"]),
        group_summary("LowRisk", active[active["riskTier"] == "Low"]),
    ]
    if (df["status"] == "churned").any():
        groups.append(group_summary("AlreadyChurned", df[df["status"] == "churned"]))

    labels = {c: res.feature_labels.get(c, c) for c in numeric_cols}
    labels["churnProbability"] = "Churn probability"
    return {
        "jobId": job_id,
        "groups": groups,
        "metricLabels": labels,
        "globalFeatureImportance": res.global_importance,
    }


@app.get("/analysis/{job_id}/model", response_model=ModelInfoResponse)
def analysis_model(job_id: str):
    _, res = _result(job_id)
    return {
        "jobId": job_id,
        "supervised": res.supervised,
        "churnDefinition": res.churn.to_dict(),
        "metrics": _clean(res.metrics),
        "schemaMap": res.schema_map.to_dict(),
        "notes": res.notes,
    }
