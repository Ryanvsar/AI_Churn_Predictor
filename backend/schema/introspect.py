"""Read the structure of any SQLAlchemy-reachable database and profile a sample of every table."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import MetaData, Table, func, inspect, select

from backend.connectors.source import DataSource
from backend.schema.timeparse import detect_time_format

SAMPLE_ROWS = 2000


@dataclass
class ColumnProfile:
    name: str
    sql_type: str
    nullable: bool = True
    is_pk: bool = False
    kind: str = "other"  # numeric | bool | string | datetime | other
    null_frac: float = 0.0
    n_unique: int = 0
    unique_frac: float = 0.0
    avg_len: float = 0.0
    time_format: Optional[str] = None
    time_parse_rate: float = 0.0
    examples: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.sql_type,
            "notnull": not self.nullable,
            "pk": self.is_pk,
            "kind": self.kind,
            "nullFrac": round(self.null_frac, 4),
            "nUnique": self.n_unique,
            "uniqueFrac": round(self.unique_frac, 4),
            "timeFormat": self.time_format,
            "examples": self.examples,
        }


@dataclass
class ForeignKey:
    columns: List[str]
    ref_table: str
    ref_columns: List[str]


@dataclass
class TableProfile:
    name: str
    row_count: int
    columns: Dict[str, ColumnProfile]
    pk: List[str]
    fks: List[ForeignKey]
    is_view: bool = False
    sample: Optional[pd.DataFrame] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "rowCount": self.row_count,
            "isView": self.is_view,
            "primaryKey": self.pk,
            "foreignKeys": [
                {"columns": fk.columns, "refTable": fk.ref_table, "refColumns": fk.ref_columns} for fk in self.fks
            ],
            "columns": [c.to_dict() for c in self.columns.values()],
        }


@dataclass
class DatabaseProfile:
    dialect: str
    tables: Dict[str, TableProfile]

    def to_dict(self) -> dict:
        return {"dialect": self.dialect, "tables": [t.to_dict() for t in self.tables.values()]}


def reflect_table(source: DataSource, name: str) -> Table:
    return Table(name, MetaData(), autoload_with=source.engine)


def _kind(series: pd.Series, sql_type: str) -> str:
    st = sql_type.upper()
    if pd.api.types.is_bool_dtype(series) or "BOOL" in st:
        return "bool"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    non_null = series.dropna()
    if non_null.empty:
        return "other"
    if any(t in st for t in ("INT", "REAL", "FLOAT", "DOUBLE", "NUMERIC", "DECIMAL")):
        if pd.to_numeric(non_null, errors="coerce").notna().mean() > 0.98:
            return "numeric"
    if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
        # Numbers stored as text (common in CSV imports and loose schemas).
        if pd.to_numeric(non_null.astype("string"), errors="coerce").notna().mean() > 0.98:
            return "numeric"
        return "string"
    return "other"


def profile_column(series: pd.Series, name: str, sql_type: str, nullable: bool, is_pk: bool) -> ColumnProfile:
    n = len(series)
    non_null = series.dropna()
    kind = _kind(series, sql_type)
    try:
        n_unique = int(non_null.nunique())
    except TypeError:
        n_unique = int(non_null.astype(str).nunique())
    prof = ColumnProfile(
        name=name,
        sql_type=sql_type,
        nullable=nullable,
        is_pk=is_pk,
        kind=kind,
        null_frac=float(1 - len(non_null) / n) if n else 0.0,
        n_unique=n_unique,
        unique_frac=float(n_unique / len(non_null)) if len(non_null) else 0.0,
        examples=[str(v)[:40] for v in non_null.drop_duplicates().head(3).tolist()],
    )
    if kind == "string":
        prof.avg_len = float(non_null.astype(str).str.len().mean()) if len(non_null) else 0.0
    if kind in ("numeric", "string", "datetime", "other"):
        fmt, rate = detect_time_format(series, name=name, sql_type=sql_type)
        if fmt is not None:
            prof.time_format, prof.time_parse_rate = fmt, rate
            if kind != "numeric" or fmt != "excel":
                prof.kind = "datetime"
    return prof


def profile_database(source: DataSource, sample_rows: int = SAMPLE_ROWS) -> DatabaseProfile:
    insp = inspect(source.engine)
    names = [(n, False) for n in insp.get_table_names()]
    try:
        names += [(n, True) for n in insp.get_view_names()]
    except NotImplementedError:
        pass

    tables: Dict[str, TableProfile] = {}
    with source.engine.connect() as conn:
        for name, is_view in names:
            if name.startswith("sqlite_"):
                continue
            try:
                table = Table(name, MetaData(), autoload_with=conn)
            except Exception:  # noqa: BLE001 - skip objects we cannot reflect (permissions etc.)
                continue
            row_count = int(conn.execute(select(func.count()).select_from(table)).scalar() or 0)
            sample = pd.read_sql(select(table).limit(sample_rows), conn)
            pk_cols = [c.name for c in table.primary_key.columns]
            fks = [
                ForeignKey(
                    columns=[e.parent.name for e in fk.elements],
                    ref_table=fk.elements[0].target_fullname.split(".")[-2] if fk.elements else "",
                    ref_columns=[e.target_fullname.split(".")[-1] for e in fk.elements],
                )
                for fk in table.foreign_key_constraints
                if fk.elements
            ]
            cols = {}
            for col in table.columns:
                series = sample[col.name] if col.name in sample.columns else pd.Series([], dtype=object)
                cols[col.name] = profile_column(
                    series, col.name, str(col.type), bool(col.nullable), col.name in pk_cols
                )
            tables[name] = TableProfile(
                name=name, row_count=row_count, columns=cols, pk=pk_cols, fks=fks, is_view=is_view, sample=sample
            )
    return DatabaseProfile(dialect=source.dialect, tables=tables)


def count_matching_keys(source: DataSource, table: str, column: str, values: List) -> int:
    """How many of `values` exist in table.column (used to link tables without declared FKs)."""
    if not values:
        return 0
    t = reflect_table(source, table)
    col = t.c[column]
    try:
        pytype = col.type.python_type
    except NotImplementedError:
        pytype = None
    if pytype is int:
        vals = [int(float(v)) for v in values if _is_intlike(v)]
    elif pytype is str:
        vals = [str(v) for v in values]
    else:
        vals = list(values)
    if not vals:
        return 0
    with source.engine.connect() as conn:
        found = conn.execute(select(func.count(func.distinct(col))).where(col.in_(vals))).scalar()
    return int(found or 0)


def _is_intlike(v) -> bool:
    try:
        return float(v) == int(float(v))
    except (TypeError, ValueError):
        return False


def numeric_values(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
