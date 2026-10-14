"""Daily activity series for one user, across all mapped activity tables."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from backend.data.extract import ExtractedData
from backend.labels.churn_definition import daynum_to_date, to_daynum


def user_daily_series(data: ExtractedData, user_id: str, end_day: int, days: int) -> Tuple[List[str], List[Dict]]:
    """
    Return (measure_names, points) with one point per calendar day in [end_day-days+1, end_day],
    including zero-activity days so breaks are visible in charts.
    Measures are daily means, named "<table>.<column>".
    """
    start = end_day - days + 1
    idx = np.arange(start, end_day + 1)
    count = pd.Series(0.0, index=idx)
    measures: Dict[str, pd.Series] = {}
    for name, af in data.activities.items():
        df = af.df[af.df["user_id"] == str(user_id)]
        if df.empty:
            for m in af.measures:
                measures[f"{name}.{m}"] = pd.Series(0.0, index=idx)
            continue
        d = to_daynum(df["ts"])
        sel = (d >= start) & (d <= end_day)
        df, d = df.loc[sel], d[sel]
        count = count.add(pd.Series(1.0, index=d).groupby(level=0).sum(), fill_value=0.0)
        for m in af.measures:
            s = pd.Series(df[m].to_numpy(dtype=float), index=d).groupby(level=0).mean()
            measures[f"{name}.{m}"] = s.reindex(idx).fillna(0.0)
    count = count.reindex(idx).fillna(0.0)
    points = [
        {
            "date": daynum_to_date(int(day)),
            "activityCount": float(count[day]),
            "measures": {k: float(v[day]) for k, v in measures.items()},
        }
        for day in idx
    ]
    return list(measures), points
