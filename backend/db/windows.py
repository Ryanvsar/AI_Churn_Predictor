from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Tuple


@dataclass(frozen=True)
class DerivedWindows:
    observation_start: int  # YYYYMMDD
    observation_end: int  # YYYYMMDD


def _yyyymmdd_to_dt(x: int) -> datetime:
    return datetime.strptime(str(int(x)), "%Y%m%d")


def _dt_to_yyyymmdd(dt: datetime) -> int:
    return int(dt.strftime("%Y%m%d"))


def get_min_max_session_date(db_path: str) -> Tuple[Optional[int], Optional[int]]:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute("SELECT MIN(sessionDate), MAX(sessionDate) FROM sessions;")
        row = cur.fetchone()
        if not row:
            return None, None
        return (int(row[0]) if row[0] is not None else None, int(row[1]) if row[1] is not None else None)
    finally:
        conn.close()


def derive_observation_window(
    db_path: str,
    observation_days: int = 60,
    prediction_days: int = 40,
) -> Optional[DerivedWindows]:
    """
    Heuristic default:
    - If the DB spans at least (observation_days + prediction_days), assume the last `prediction_days`
      are a 'future label' period and set observation_end to (max_date - prediction_days).
    - Otherwise, use the max_date as observation_end.
    - Observation_start = observation_end - (observation_days - 1)
    """
    mn, mx = get_min_max_session_date(db_path)
    if mn is None or mx is None:
        return None

    mn_dt = _yyyymmdd_to_dt(mn)
    mx_dt = _yyyymmdd_to_dt(mx)
    span_days = (mx_dt - mn_dt).days + 1

    obs_days = max(int(observation_days), 7)
    pred_days = max(int(prediction_days), 0)

    if span_days >= (obs_days + pred_days):
        obs_end_dt = mx_dt - timedelta(days=pred_days)
    else:
        obs_end_dt = mx_dt

    obs_start_dt = obs_end_dt - timedelta(days=obs_days - 1)
    if obs_start_dt < mn_dt:
        obs_start_dt = mn_dt

    return DerivedWindows(
        observation_start=_dt_to_yyyymmdd(obs_start_dt),
        observation_end=_dt_to_yyyymmdd(obs_end_dt),
    )

