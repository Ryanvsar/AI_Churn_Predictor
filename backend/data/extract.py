"""Pull the mapped columns out of the source database into canonical pandas frames."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import select

from backend.connectors.source import DataSource
from backend.schema.detect import ActivityMap, SchemaMap
from backend.schema.introspect import reflect_table
from backend.schema.timeparse import has_time_of_day, parse_time

CHUNK_ROWS = 250_000
RESERVED = {"user_id", "ts"}


@dataclass
class ActivityFrame:
    name: str
    df: pd.DataFrame  # user_id (str), ts (datetime64[ns]), measures (float), categoricals (string)
    measures: List[str]
    categoricals: List[str]
    has_time_of_day: bool


@dataclass
class ExtractedData:
    activities: Dict[str, ActivityFrame]
    users: pd.DataFrame  # user_id, [signup_ts], static columns, [label]
    static_categoricals: List[str] = field(default_factory=list)
    static_numerics: List[str] = field(default_factory=list)
    has_label: bool = False
    min_ts: Optional[pd.Timestamp] = None
    max_ts: Optional[pd.Timestamp] = None


def normalize_ids(s: pd.Series) -> pd.Series:
    """Make user ids comparable across tables: 5, 5.0 and '5' all become '5'."""
    if pd.api.types.is_float_dtype(s):
        vals = s.dropna()
        if len(vals) and np.all(np.isclose(vals, np.round(vals))):
            return s.round().astype("Int64").astype("string")
    if pd.api.types.is_integer_dtype(s):
        return s.astype("Int64").astype("string")
    out = s.astype("string").str.strip()
    return out.str.replace(r"^(-?\d+)\.0+$", r"\1", regex=True)


def _safe_label(name: str) -> str:
    return f"_{name}" if name in RESERVED else name


def _activity_select(source: DataSource, a: ActivityMap):
    t = reflect_table(source, a.table)
    cols = [t.c[a.time_column].label("ts")]
    cols += [t.c[m].label(_safe_label(m)) for m in a.measures]
    cols += [t.c[c].label(_safe_label(c)) for c in a.categoricals]
    if a.via is None:
        return select(t.c[a.user_column].label("user_id"), *cols)
    p = reflect_table(source, a.via.table)
    join = t.join(p, t.c[a.via.local_column] == p.c[a.via.remote_column])
    return select(p.c[a.via.user_column].label("user_id"), *cols).select_from(join)


def extract_activity(source: DataSource, a: ActivityMap) -> ActivityFrame:
    stmt = _activity_select(source, a)
    measures = [_safe_label(m) for m in a.measures]
    cats = [_safe_label(c) for c in a.categoricals]
    parts = []
    with source.engine.connect() as conn:
        for chunk in pd.read_sql(stmt, conn, chunksize=CHUNK_ROWS):
            chunk["ts"] = parse_time(chunk["ts"], a.time_format)
            chunk = chunk.dropna(subset=["ts", "user_id"])
            chunk["user_id"] = normalize_ids(chunk["user_id"])
            for m in measures:
                chunk[m] = pd.to_numeric(chunk[m], errors="coerce").astype(float)
            for c in cats:
                chunk[c] = chunk[c].astype("string").str.strip()
            parts.append(chunk)
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["user_id", "ts", *measures, *cats])
    df["ts"] = df["ts"].astype("datetime64[ns]")
    df = df.sort_values(["user_id", "ts"], kind="stable").reset_index(drop=True)
    return ActivityFrame(
        name=a.table, df=df, measures=measures, categoricals=cats, has_time_of_day=has_time_of_day(df["ts"])
    )


def _label_values(raw: pd.Series, kind: str, positive: List[str]) -> pd.Series:
    if kind == "not_null":
        return raw.notna().astype(float)
    norm = raw.astype("string").str.strip().str.lower().str.replace(r"\.0+$", "", regex=True)
    out = norm.isin([p.lower() for p in positive]).astype(float)
    return out.where(raw.notna())


def extract(
    source: DataSource,
    smap: SchemaMap,
    max_users: Optional[int] = None,
    seed: int = 42,
    use_label: bool = True,
) -> ExtractedData:
    activities = {}
    for a in smap.enabled_activities():
        af = extract_activity(source, a)
        if not af.df.empty:
            activities[a.table] = af
    if not activities:
        raise ValueError("No activity rows could be read from the mapped activity tables.")

    seen = pd.Index(pd.concat([af.df["user_id"] for af in activities.values()]).unique())

    statics_c, statics_n = list(smap.static_categoricals), list(smap.static_numerics)
    if smap.user_table:
        ut = reflect_table(source, smap.user_table)
        cols = [ut.c[smap.user_id_column].label("user_id")]
        if smap.signup_column:
            cols.append(ut.c[smap.signup_column].label("signup_raw"))
        cols += [ut.c[c].label(f"static_c::{c}") for c in statics_c]
        cols += [ut.c[c].label(f"static_n::{c}") for c in statics_n]
        if use_label and smap.label and smap.label.table == smap.user_table:
            cols.append(ut.c[smap.label.column].label("label_raw"))
        with source.engine.connect() as conn:
            users = pd.read_sql(select(*cols), conn)
        users["user_id"] = normalize_ids(users["user_id"])
        users = users.dropna(subset=["user_id"]).drop_duplicates("user_id")
    else:
        users = pd.DataFrame({"user_id": seen.astype("string")})

    if use_label and smap.label and smap.label.table != smap.user_table:
        lt = reflect_table(source, smap.label.table)
        with source.engine.connect() as conn:
            lab = pd.read_sql(
                select(lt.c[smap.label.user_column].label("user_id"), lt.c[smap.label.column].label("label_raw")), conn
            )
        lab["user_id"] = normalize_ids(lab["user_id"])
        users = users.merge(lab.drop_duplicates("user_id"), on="user_id", how="left")

    if "signup_raw" in users.columns:
        users["signup_ts"] = parse_time(users.pop("signup_raw"), smap.signup_format or "iso")
    for c in statics_n:
        users[f"static_n::{c}"] = pd.to_numeric(users[f"static_n::{c}"], errors="coerce").astype(float)
    for c in statics_c:
        users[f"static_c::{c}"] = users[f"static_c::{c}"].astype("string")
    has_label = False
    if "label_raw" in users.columns:
        users["label"] = _label_values(users.pop("label_raw"), smap.label.kind, smap.label.positive_values)
        has_label = bool(users["label"].notna().any())

    # Only users with at least one activity can be scored.
    users = users[users["user_id"].isin(seen)].reset_index(drop=True)
    if max_users and len(users) > max_users:
        users = users.sample(n=int(max_users), random_state=seed).reset_index(drop=True)
    keep = set(users["user_id"])
    for af in activities.values():
        af.df = af.df[af.df["user_id"].isin(keep)].reset_index(drop=True)

    min_ts = min(af.df["ts"].min() for af in activities.values())
    max_ts = max(af.df["ts"].max() for af in activities.values())
    return ExtractedData(
        activities=activities,
        users=users,
        static_categoricals=statics_c,
        static_numerics=statics_n,
        has_label=has_label,
        min_ts=min_ts,
        max_ts=max_ts,
    )


def clip_time(data: ExtractedData, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> ExtractedData:
    """Restrict activity to [start, end] (inclusive by day) — used for 'as of' analyses and backtests."""
    if start is None and end is None:
        return data
    for af in data.activities.values():
        m = pd.Series(True, index=af.df.index)
        if start is not None:
            m &= af.df["ts"] >= start.normalize()
        if end is not None:
            m &= af.df["ts"] < end.normalize() + pd.Timedelta(days=1)
        af.df = af.df[m].reset_index(drop=True)
    data.activities = {k: v for k, v in data.activities.items() if not v.df.empty}
    if not data.activities:
        raise ValueError("No activity in the selected date range.")
    seen = set(pd.concat([af.df["user_id"] for af in data.activities.values()]).unique())
    data.users = data.users[data.users["user_id"].isin(seen)].reset_index(drop=True)
    data.min_ts = min(af.df["ts"].min() for af in data.activities.values())
    data.max_ts = max(af.df["ts"].max() for af in data.activities.values())
    return data
