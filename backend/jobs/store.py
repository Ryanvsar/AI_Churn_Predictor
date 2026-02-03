from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class JobRecord:
    jobId: str
    status: str = "queued"  # queued|running|succeeded|failed
    progress: float = 0.0
    startedAt: Optional[datetime] = None
    finishedAt: Optional[datetime] = None
    error: Optional[str] = None
    summary: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)  # non-JSON python objects (df/model/explainer)


JOBS: Dict[str, JobRecord] = {}

