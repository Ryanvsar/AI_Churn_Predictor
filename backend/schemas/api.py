from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


JobStatus = Literal["queued", "running", "succeeded", "failed"]


class ErrorResponse(BaseModel):
    detail: str


class DbValidateRequest(BaseModel):
    dbPath: str = Field(..., min_length=1)


class DbTableColumn(BaseModel):
    name: str
    type: Optional[str] = None
    notnull: Optional[bool] = None


class DbTableInfo(BaseModel):
    name: str
    columns: List[DbTableColumn]


class DbCapabilities(BaseModel):
    hasUsersTable: bool
    hasSessionsTable: bool
    hasChurnLabels: bool
    hasFeatureUsageTable: bool
    hasSessionEventsTable: bool
    notes: List[str] = Field(default_factory=list)


class DbValidateResponse(BaseModel):
    ok: bool
    dbPath: str
    tables: List[DbTableInfo] = Field(default_factory=list)
    capabilities: DbCapabilities


class RunAnalysisRequest(BaseModel):
    dbPath: str = Field(..., min_length=1)
    observationStart: Optional[int] = None  # YYYYMMDD
    observationEnd: Optional[int] = None  # YYYYMMDD
    threshold: float = 0.4


class RunAnalysisResponse(BaseModel):
    jobId: str


class JobSummary(BaseModel):
    observationStart: Optional[int] = None
    observationEnd: Optional[int] = None
    usersTotal: int = 0
    usersScored: int = 0
    featuresUsed: List[str] = Field(default_factory=list)


class JobStatusResponse(BaseModel):
    jobId: str
    status: JobStatus
    progress: float = 0.0
    startedAt: Optional[datetime] = None
    finishedAt: Optional[datetime] = None
    error: Optional[str] = None
    summary: JobSummary = Field(default_factory=JobSummary)


class UserRow(BaseModel):
    userId: int
    churnProbability: float
    riskTier: Literal["Low", "Medium", "High"]


class UsersListResponse(BaseModel):
    jobId: str
    total: int
    users: List[UserRow]


class UserDetailsResponse(BaseModel):
    jobId: str
    userId: int
    churnProbability: float
    riskTier: Literal["Low", "Medium", "High"]
    metrics: Dict[str, Any]


class ShapFactor(BaseModel):
    featureKey: str
    displayName: str
    shapValue: float
    featureValue: float
    direction: Literal["increases_risk", "decreases_risk"]


class UserExplainResponse(BaseModel):
    jobId: str
    userId: int
    churnProbability: float
    topFactors: List[ShapFactor]
    whyRiskySummary: str


class TimeSeriesPoint(BaseModel):
    date: str  # YYYY-MM-DD
    sessionLength: float
    sessionEvents: float


class UserTimeSeriesResponse(BaseModel):
    jobId: str
    userId: int
    points: List[TimeSeriesPoint]


class AggregateGroup(BaseModel):
    name: str
    count: int
    metricsMean: Dict[str, float]


class GlobalFeatureImportanceRow(BaseModel):
    featureKey: str
    importance: float


class AggregatesResponse(BaseModel):
    jobId: str
    groups: List[AggregateGroup]
    globalFeatureImportance: List[GlobalFeatureImportanceRow] = Field(default_factory=list)

