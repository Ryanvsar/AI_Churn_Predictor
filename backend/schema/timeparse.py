"""
Detect and normalise the many ways apps store time.

Supported formats (returned as the `fmt` string):
- "native"   : database DATE/DATETIME/TIMESTAMP values or pandas datetimes
- "iso"      : strings parseable as dates/datetimes (ISO 8601, "YYYY-MM-DD HH:MM:SS", ...)
- "yyyymmdd" : integers like 20250131 (the original demo schema)
- "unix_s" / "unix_ms" / "unix_us" : epoch timestamps in seconds / milliseconds / microseconds
- "excel"    : Excel serial day numbers (only accepted when the column name looks time-like)

All parsed values are converted to timezone-naive UTC `datetime64[ns]`.
"""
from __future__ import annotations

import re
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd

TIME_NAME_RE = re.compile(
    r"(date|time|_at$|_on$|at$|timestamp|^ts$|_ts$|ts_|day$|^day|created|started|start|ended|occurred|logged|when|period|month|epoch)",
    re.IGNORECASE,
)

# Valid range for parsed timestamps (filters out ids / small numbers that "parse").
MIN_TS = pd.Timestamp("1990-01-01")
MAX_TS = pd.Timestamp("2100-01-01")


def is_time_like_name(name: str) -> bool:
    return bool(TIME_NAME_RE.search(name))


def _to_naive_utc(s: pd.Series) -> pd.Series:
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    return s.astype("datetime64[ns]")


def _numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    return pd.to_numeric(series, errors="coerce")


def _in_range_rate(parsed: pd.Series, n: int) -> float:
    if n == 0:
        return 0.0
    ok = parsed.notna() & (parsed >= MIN_TS) & (parsed <= MAX_TS)
    return float(ok.sum()) / float(n)


def parse_time(series: pd.Series, fmt: str) -> pd.Series:
    """Convert `series` to naive-UTC datetime64 using a detected format. Unparseable -> NaT."""
    if fmt == "native":
        if pd.api.types.is_datetime64_any_dtype(series):
            out = _to_naive_utc(series)
        else:
            out = parse_time(series, "iso")
    elif fmt == "iso":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = pd.to_datetime(series.astype("string"), errors="coerce", utc=True, format="mixed")
        out = _to_naive_utc(out)
    elif fmt == "yyyymmdd":
        num = _numeric(series)
        as_int = num.round().astype("Int64").astype("string")
        out = pd.to_datetime(as_int, format="%Y%m%d", errors="coerce")
    elif fmt in ("unix_s", "unix_ms", "unix_us"):
        unit = {"unix_s": "s", "unix_ms": "ms", "unix_us": "us"}[fmt]
        num = _numeric(series)
        out = pd.to_datetime(num, unit=unit, errors="coerce")
    elif fmt == "excel":
        num = _numeric(series)
        out = pd.to_datetime(num, unit="D", origin="1899-12-30", errors="coerce")
    else:
        raise ValueError(f"Unknown time format: {fmt}")
    out = out.astype("datetime64[ns]")
    return out.where((out >= MIN_TS) & (out <= MAX_TS))


def detect_time_format(series: pd.Series, name: str = "", sql_type: str = "") -> Tuple[Optional[str], float]:
    """
    Return (format, parse_rate) for the best interpretation of `series` as time, or (None, 0.0).
    Numeric columns are only considered when the column name looks time-like, to avoid treating
    ids or measures as timestamps.
    """
    s = series.dropna()
    n = len(s)
    if n == 0:
        return None, 0.0
    name_hint = is_time_like_name(name)
    sql_type = (sql_type or "").upper()

    if pd.api.types.is_datetime64_any_dtype(s) or any(t in sql_type for t in ("DATE", "TIME")):
        rate = _in_range_rate(parse_time(s, "native"), n)
        if rate > 0:
            return "native", rate

    if pd.api.types.is_bool_dtype(s):
        return None, 0.0

    num = _numeric(s)
    numeric_rate = float(num.notna().sum()) / n
    if numeric_rate >= 0.98:
        if not name_hint:
            return None, 0.0
        num = num.dropna()
        integral = bool(np.all(np.isclose(num, np.round(num))))
        med = float(np.median(num))
        candidates = []
        if integral and 19000101 <= med <= 21001231:
            candidates.append("yyyymmdd")
        if 1e8 <= med < 1e11:
            candidates.append("unix_s")
        elif 1e11 <= med < 1e14:
            candidates.append("unix_ms")
        elif 1e14 <= med < 1e17:
            candidates.append("unix_us")
        if 25000 <= med <= 75000:
            candidates.append("excel")
        best, best_rate = None, 0.0
        for fmt in candidates:
            rate = _in_range_rate(parse_time(s, fmt), n)
            if rate > best_rate:
                best, best_rate = fmt, rate
        return (best, best_rate) if best_rate >= 0.9 else (None, 0.0)

    # Strings: require something date-ish so free text / categories are not parsed.
    str_vals = s.astype("string")
    looks_dateish = str_vals.str.contains(r"\d{1,4}[-/.:]\d{1,2}", regex=True, na=False).mean()
    if looks_dateish < 0.9:
        return None, 0.0
    rate = _in_range_rate(parse_time(s, "iso"), n)
    return ("iso", rate) if rate >= 0.9 else (None, 0.0)


def has_time_of_day(ts: pd.Series) -> bool:
    """True when timestamps carry a meaningful time-of-day component (not all midnight)."""
    t = ts.dropna()
    if t.empty:
        return False
    sample = t.iloc[: min(len(t), 5000)]
    return bool((sample != sample.dt.normalize()).mean() > 0.2)
