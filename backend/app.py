from __future__ import annotations

import json
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from backend import service
from backend.connectors.source import SourceError, open_source
from backend.jobs.runner import run_job
from backend.jobs.store import JOBS, JobRecord
from backend.schema.detect import SchemaMap
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
SITE_DIR = REPO_ROOT / "site"
UPLOAD_DIR = REPO_ROOT / "uploads"
MAPPINGS_DIR = REPO_ROOT / "mappings"
UPLOAD_SUFFIXES = {".db", ".sqlite", ".sqlite3", ".db3", ".csv", ".tsv", ".parquet", ".xlsx", ".xls", ".zip"}

app = FastAPI(title="Churn Prediction Backend", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    # Local-only dev: the UI can be served by this backend or a separate dev server.
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    # `engine: server` tells the web UI to use this API (database URLs, large files) instead of
    # running the pipeline inside the browser.
    return {"ok": True, "engine": "server", "time": datetime.now(timezone.utc).isoformat()}


@app.get("/engine.json")
def engine_info():
    # Overrides the static site/engine.json ({"engine": "browser"}) so the UI uses this server.
    return {"engine": "server"}


# ------------------------------------------------------------------------------------------------
# Data sources
# ------------------------------------------------------------------------------------------------

@app.post("/db/validate", response_model=DbValidateResponse)
def db_validate(payload: DbValidateRequest):
    try:
        return service.validate_source(payload.source)
    except SourceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001 - report unexpected database errors to the UI
        raise HTTPException(status_code=400, detail=f"Could not read the database: {e}")


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

    an = service.analyze(
        {
            "source": payload.source,
            "schema_map": payload.schemaMap,
            "churn_window_days": payload.churnWindowDays,
            "label_source": payload.labelSource,
            "threshold": payload.threshold,
            "high_risk_threshold": max(payload.highRiskThreshold, payload.threshold),
            "max_users": payload.maxUsers,
            "observation_start": payload.observationStart,
            "observation_end": payload.observationEnd,
        },
        progress,
    )
    return {"summary": service.job_summary(an), "artifacts": {"analysis": an}}


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


def _analysis(job_id: str) -> service.Analysis:
    job = JOBS.get(job_id)
    if not job or job.status != "succeeded":
        raise HTTPException(status_code=404, detail="Job not found or not complete.")
    an = job.artifacts.get("analysis")
    if an is None:
        raise HTTPException(status_code=500, detail="Job results unavailable.")
    return an


def _per_user(fn, job_id: str, user_id: str, *args):
    an = _analysis(job_id)
    try:
        return fn(an, user_id, *args)
    except KeyError:
        raise HTTPException(status_code=404, detail="User not found in results.")


@app.get("/analysis/{job_id}/result")
def analysis_result(job_id: str):
    """Everything the dashboard shows in one payload (overview, all users, model info)."""
    return {"jobId": job_id, **service.full_result(_analysis(job_id))}


@app.get("/analysis/{job_id}/export.csv", response_class=PlainTextResponse)
def analysis_export(job_id: str):
    return PlainTextResponse(
        service.export_csv(_analysis(job_id)),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="churn_predictions.csv"'},
    )


@app.get("/analysis/{job_id}/users", response_model=UsersListResponse)
def analysis_users(
    job_id: str,
    top: int = 100,
    threshold: float = 0.0,
    sort: str = "prob_desc",
    includeChurned: bool = True,
):
    out = service.users_list(_analysis(job_id), top, threshold, sort, includeChurned)
    return {"jobId": job_id, **out}


@app.get("/analysis/{job_id}/users/{user_id}", response_model=UserDetailsResponse)
def analysis_user_details(job_id: str, user_id: str):
    return {"jobId": job_id, **_per_user(service.user_details, job_id, user_id)}


@app.get("/analysis/{job_id}/users/{user_id}/explain", response_model=UserExplainResponse)
def analysis_user_explain(job_id: str, user_id: str):
    return {"jobId": job_id, **_per_user(service.user_explain, job_id, user_id)}


@app.get("/analysis/{job_id}/users/{user_id}/timeseries", response_model=UserTimeSeriesResponse)
def analysis_user_timeseries(job_id: str, user_id: str, days: int = 0):
    return {"jobId": job_id, **_per_user(service.user_timeseries, job_id, user_id, days)}


@app.get("/analysis/{job_id}/users/{user_id}/bundle")
def analysis_user_bundle(job_id: str, user_id: str):
    """Everything the user panel shows in one payload."""
    return service.clean(_per_user(service.user_bundle, job_id, user_id))


@app.get("/analysis/{job_id}/aggregates", response_model=AggregatesResponse)
def analysis_aggregates(job_id: str):
    return {"jobId": job_id, **service.aggregates(_analysis(job_id))}


@app.get("/analysis/{job_id}/model", response_model=ModelInfoResponse)
def analysis_model(job_id: str):
    return {"jobId": job_id, **service.model_info(_analysis(job_id))}


# ------------------------------------------------------------------------------------------------
# Web UI (the same static site that is deployed to Vercel)
# ------------------------------------------------------------------------------------------------

@app.get("/ui")
def ui_redirect():
    return RedirectResponse(url="/")


if SITE_DIR.exists():
    app.mount("/", StaticFiles(directory=str(SITE_DIR), html=True), name="site")
