from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator


JobStatus = Literal["queued", "running", "succeeded", "failed"]
RiskTier = Literal["Low", "Medium", "High"]


class ErrorResponse(BaseModel):
    detail: str


class SourceRequest(BaseModel):
    """A data source: SQLite/CSV/Excel path, a directory, a .zip, or a SQLAlchemy URL."""

    source: Optional[str] = None
    dbPath: Optional[str] = None  # backwards-compatible alias of `source`

    @model_validator(mode="after")
    def _one_source(self):
        if not (self.source or self.dbPath):
            raise ValueError("Provide `source` (file path or database URL).")
        if not self.source:
            self.source = self.dbPath
        return self


class DbValidateRequest(SourceRequest):
    pass


class DbTableColumn(BaseModel):
    name: str
    type: Optional[str] = None
    notnull: Optional[bool] = None
    pk: Optional[bool] = None
    kind: Optional[str] = None
    timeFormat: Optional[str] = None
    nUnique: Optional[int] = None
    examples: List[str] = Field(default_factory=list)


class DbForeignKey(BaseModel):
    columns: List[str]
    refTable: str
    refColumns: List[str]


class DbTableInfo(BaseModel):
    name: str
    rowCount: Optional[int] = None
    isView: bool = False
    primaryKey: List[str] = Field(default_factory=list)
    foreignKeys: List[DbForeignKey] = Field(default_factory=list)
    columns: List[DbTableColumn]


class DbCapabilities(BaseModel):
    hasUsersTable: bool
    hasSessionsTable: bool  # i.e. at least one activity table was found
    hasChurnLabels: bool
    hasFeatureUsageTable: bool  # i.e. categorical activity detail is available
    hasSessionEventsTable: bool  # i.e. more than one activity table
    activityTables: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class DbValidateResponse(BaseModel):
    ok: bool
    dbPath: str
    source: str
    dialect: str
    tables: List[DbTableInfo] = Field(default_factory=list)
    capabilities: DbCapabilities
    schemaMap: Dict[str, Any]
    summary: Dict[str, Any] = Field(default_factory=dict)


class UploadResponse(BaseModel):
    path: str
    filename: str
    sizeBytes: int


class MappingSaveRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    schemaMap: Dict[str, Any]


class MappingInfo(BaseModel):
    name: str
    schemaMap: Dict[str, Any]


class RunAnalysisRequest(SourceRequest):
    threshold: float = Field(0.4, ge=0.0, le=1.0)
    highRiskThreshold: float = Field(0.7, ge=0.0, le=1.0)
    schemaMap: Optional[Dict[str, Any]] = None  # override the auto-detected mapping
    churnWindowDays: Optional[int] = Field(None, ge=1, le=365)  # override the inferred churn window
    labelSource: Literal["auto", "explicit", "derived"] = "auto"
    maxUsers: Optional[int] = Field(None, ge=100)
    observationStart: Optional[Union[int, str]] = None  # YYYYMMDD or ISO date
    observationEnd: Optional[Union[int, str]] = None  # "as of" date; YYYYMMDD or ISO date


class RunAnalysisResponse(BaseModel):
    jobId: str


class JobSummary(BaseModel):
    observationStart: Optional[str] = None
    observationEnd: Optional[str] = None
    usersTotal: int = 0
    usersScored: int = 0
    usersActive: int = 0
    featuresUsed: List[str] = Field(default_factory=list)
    supervised: bool = False
    churnDefinition: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    notes: List[str] = Field(default_factory=list)


class JobStatusResponse(BaseModel):
    jobId: str
    status: JobStatus
    progress: float = 0.0
    message: str = ""
    startedAt: Optional[datetime] = None
    finishedAt: Optional[datetime] = None
    error: Optional[str] = None
    summary: JobSummary = Field(default_factory=JobSummary)


class UserRow(BaseModel):
    userId: str
    churnProbability: float
    riskTier: RiskTier
    status: str = "active"
    knownLabel: Optional[int] = None
    recencyDays: Optional[int] = None
    topReason: Optional[str] = None


class UsersListResponse(BaseModel):
    jobId: str
    total: int
    users: List[UserRow]


class UserDetailsResponse(BaseModel):
    jobId: str
    userId: str
    churnProbability: float
    riskTier: RiskTier
    status: str = "active"
    knownLabel: Optional[int] = None
    metrics: Dict[str, Any]
    metricLabels: Dict[str, str] = Field(default_factory=dict)


class ShapFactor(BaseModel):
    featureKey: str
    displayName: str
    shapValue: float
    featureValue: Optional[float] = None
    direction: Literal["increases_risk", "decreases_risk"]


class UserExplainResponse(BaseModel):
    jobId: str
    userId: str
    churnProbability: float
    topFactors: List[ShapFactor]
    whyRiskySummary: str


class TimeSeriesPoint(BaseModel):
    date: str  # YYYY-MM-DD
    activityCount: float
    measures: Dict[str, float] = Field(default_factory=dict)


class UserTimeSeriesResponse(BaseModel):
    jobId: str
    userId: str
    measureNames: List[str] = Field(default_factory=list)
    scoringDate: Optional[str] = None
    points: List[TimeSeriesPoint]


class AggregateGroup(BaseModel):
    name: str
    count: int
    metricsMean: Dict[str, float]


class GlobalFeatureImportanceRow(BaseModel):
    featureKey: str
    displayName: Optional[str] = None
    importance: float


class AggregatesResponse(BaseModel):
    jobId: str
    groups: List[AggregateGroup]
    metricLabels: Dict[str, str] = Field(default_factory=dict)
    globalFeatureImportance: List[GlobalFeatureImportanceRow] = Field(default_factory=list)


class ModelInfoResponse(BaseModel):
    jobId: str
    supervised: bool
    churnDefinition: Dict[str, Any]
    metrics: Dict[str, Any]
    schemaMap: Dict[str, Any]
    notes: List[str] = Field(default_factory=list)
