from __future__ import annotations

import traceback
from datetime import datetime
from typing import Any, Dict

from backend.jobs.store import JobRecord


def run_job(job: JobRecord, fn, *args, **kwargs) -> None:
    job.status = "running"
    job.startedAt = datetime.utcnow()
    job.progress = 0.01

    try:
        result: Dict[str, Any] = fn(job, *args, **kwargs)
        job.artifacts = result.get("artifacts", {})
        job.summary = result.get("summary", {})
        job.progress = 1.0
        job.status = "succeeded"
    except Exception as e:  # noqa: BLE001 (keep wide catch for job boundary)
        job.status = "failed"
        job.error = f"{e}\n{traceback.format_exc()}"
        job.progress = 1.0
    finally:
        job.finishedAt = datetime.utcnow()

