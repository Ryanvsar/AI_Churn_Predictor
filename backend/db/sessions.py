from __future__ import annotations

import sqlite3
from typing import Optional

import pandas as pd


def load_user_sessions_timeseries(
    db_path: str,
    user_id: int,
    observation_start: Optional[int] = None,
    observation_end: Optional[int] = None,
) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        where = ["userId = ?"]
        params = [int(user_id)]
        if observation_start is not None:
            where.append("sessionDate >= ?")
            params.append(int(observation_start))
        if observation_end is not None:
            where.append("sessionDate <= ?")
            params.append(int(observation_end))
        where_sql = " AND ".join(where)

        q = f"""
        SELECT sessionDate, sessionLength, sessionEvents
        FROM sessions
        WHERE {where_sql}
        ORDER BY sessionDate ASC
        ;
        """
        df = pd.read_sql_query(q, conn, params=params)

        if observation_start is None or observation_end is None:
            # If we don't know the intended window, return only observed session points.
            if df.empty:
                return df
            df["date"] = pd.to_datetime(df["sessionDate"].astype(str), format="%Y%m%d", errors="coerce").dt.strftime(
                "%Y-%m-%d"
            )
            df["sessionLength"] = df["sessionLength"].astype(float)
            df["sessionEvents"] = df["sessionEvents"].astype(float)
            return df[["date", "sessionLength", "sessionEvents"]]

        # Build a full daily series for the modeling window so inactivity is visible in charts.
        start_dt = pd.to_datetime(str(int(observation_start)), format="%Y%m%d", errors="coerce")
        end_dt = pd.to_datetime(str(int(observation_end)), format="%Y%m%d", errors="coerce")
        if pd.isna(start_dt) or pd.isna(end_dt):
            return pd.DataFrame(columns=["date", "sessionLength", "sessionEvents"])

        if df.empty:
            idx = pd.date_range(start=start_dt, end=end_dt, freq="D")
            out = pd.DataFrame({"date": idx.strftime("%Y-%m-%d"), "sessionLength": 0.0, "sessionEvents": 0.0})
            return out

        df["dt"] = pd.to_datetime(df["sessionDate"].astype(str), format="%Y%m%d", errors="coerce")
        df["sessionLength"] = df["sessionLength"].astype(float)
        df["sessionEvents"] = df["sessionEvents"].astype(float)

        daily = df.groupby("dt").agg(sessionLength=("sessionLength", "mean"), sessionEvents=("sessionEvents", "mean"))

        idx = pd.date_range(start=start_dt, end=end_dt, freq="D")
        daily = daily.reindex(idx).fillna(0.0)
        daily.index.name = "dt"
        out = daily.reset_index()
        out["date"] = out["dt"].dt.strftime("%Y-%m-%d")
        return out[["date", "sessionLength", "sessionEvents"]]
    finally:
        conn.close()

