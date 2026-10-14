"""
Define churn from the data itself.

Most app databases have no churn label, so churn is defined behaviourally:

    a user has churned when they have no activity for H days in a row

H (the churn window) is learned from the app's own rhythm: it is the 95th percentile of the gaps
between a user's consecutive active days, snapped to a natural value (3, 7, 14, 30, ... days).
A daily-habit app ends up with ~7 days, a weekly-commute app with ~14, a monthly tool with ~30-60.

Training examples come from "time-travel" snapshots: at each past cutoff date we compute features
from data up to the cutoff only and label whether the user went silent for the next H days.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

SNAP_DAYS = [3, 7, 14, 21, 30, 45, 60, 90, 120, 180]
EPOCH = np.datetime64("1970-01-01", "D")


def to_daynum(ts: pd.Series) -> np.ndarray:
    return ts.to_numpy(dtype="datetime64[ns]").astype("datetime64[D]").astype(np.int64)


def daynum_to_date(d: int) -> str:
    return str(EPOCH + np.timedelta64(int(d), "D"))


@dataclass
class ChurnDefinition:
    window_days: int  # H: inactivity that counts as churn, also the prediction horizon
    lookback_days: int  # L: how much history the features look at
    auto: bool
    label_source: str  # derived | explicit
    median_gap_days: float
    p95_gap_days: float
    scoring_date: str
    label_cutoff_date: Optional[str] = None  # explicit mode: feature cutoff used for the label
    snapshot_dates: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def describe(self) -> str:
        if self.label_source == "explicit":
            return (f"Churn = the label column in your database; features are taken as of {self.label_cutoff_date}, "
                    f"before churners' activity starts to drop.")
        return f"Churn = no activity for {self.window_days} days" + (" (inferred from usage rhythm)" if self.auto else "")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["description"] = self.describe()
        return d


def activity_days(frames: List[pd.DataFrame]) -> pd.DataFrame:
    """Unique (user_id, day) pairs across all activity sources."""
    parts = [pd.DataFrame({"user_id": f["user_id"].to_numpy(), "day": to_daynum(f["ts"])}) for f in frames]
    days = pd.concat(parts, ignore_index=True).drop_duplicates()
    return days.sort_values(["user_id", "day"], kind="stable").reset_index(drop=True)


def complete_end_day(days: pd.DataFrame) -> int:
    """
    Last day that looks fully recorded. Exports often end part-way through a day (or a few events
    spill past midnight); scoring on such a day would make every user look inactive. A final day
    with under half the activity of the same weekday in previous weeks is treated as partial.
    """
    counts = days.groupby("day").size()
    end = int(counts.index.max())
    for _ in range(3):  # at most a few trailing partial days
        prev = [counts.get(end - 7 * k, 0) for k in range(1, 5)]
        ref = float(np.median([c for c in prev if c > 0] or [0]))
        if ref > 0 and counts.get(end, 0) < 0.5 * ref:
            end -= 1
        else:
            break
    return end


def gap_stats(days: pd.DataFrame) -> tuple:
    gaps = days.groupby("user_id", sort=False)["day"].diff().dropna()
    if gaps.empty:
        return 1.0, 7.0
    return float(gaps.median()), float(gaps.quantile(0.95))


def infer_window(days: pd.DataFrame, min_day: int, max_day: int, override: Optional[int] = None) -> tuple:
    median_gap, p95 = gap_stats(days)
    notes = []
    span = max_day - min_day + 1
    if override:
        return int(override), median_gap, p95, notes
    h = next((s for s in SNAP_DAYS if s >= p95), SNAP_DAYS[-1])
    max_h = max(3, span // 3)
    if h > max_h:
        h = max([s for s in SNAP_DAYS if s <= max_h] or [3])
        notes.append(f"History is short ({span} days); churn window capped at {h} days.")
    return int(h), median_gap, p95, notes


def lookback_for(h: int) -> int:
    return int(min(max(4 * h, 28), 180))


def snapshot_cutoffs(min_day: int, max_day: int, h: int, lookback: int, max_snapshots: int = 12) -> List[int]:
    """Past cutoff days whose full label window (cutoff, cutoff+H] is observed. Most recent first."""
    step = max(h, 7)
    earliest = min_day + max(lookback // 2, 14)
    cuts = []
    c = max_day - h
    while c >= earliest and len(cuts) < max_snapshots:
        cuts.append(int(c))
        c -= step
    return cuts


def derived_labels(days: pd.DataFrame, cutoff: int, h: int) -> pd.DataFrame:
    """
    Users still active at `cutoff` (last activity within H days) and whether they then went silent
    for the next H days (label=1 = churned).
    """
    before = days[days["day"] <= cutoff].groupby("user_id", sort=False)["day"].max()
    eligible = before[before > cutoff - h]
    after = days[(days["day"] > cutoff) & (days["day"] <= cutoff + h)]["user_id"].unique()
    lab = pd.DataFrame({"user_id": eligible.index})
    lab["label"] = (~lab["user_id"].isin(after)).astype(int)
    return lab


def detect_label_cutoff(days: pd.DataFrame, labels: pd.DataFrame, min_day: int, max_day: int) -> Optional[int]:
    """
    For an explicit churn label of unknown timing, find the last day before labelled churners'
    activity starts to fall away from everyone else's. Features must be taken before that point,
    otherwise the model just "sees" the churn that already happened instead of predicting it.
    Returns None when no clear divergence exists.
    """
    lab = labels.dropna(subset=["label"])
    pos = set(lab.loc[lab["label"] == 1, "user_id"])
    neg = set(lab.loc[lab["label"] == 0, "user_id"])
    if len(pos) < 10 or len(neg) < 10:
        return None
    span = max_day - min_day + 1
    if span < 28:
        return None
    idx = np.arange(min_day, max_day + 1)
    d = days[days["user_id"].isin(pos | neg)]
    is_pos = d["user_id"].isin(pos)
    rp = d[is_pos].groupby("day").size().reindex(idx, fill_value=0) / len(pos)
    rn = d[~is_pos].groupby("day").size().reindex(idx, fill_value=0) / len(neg)
    ratio = (rp.rolling(7, center=True, min_periods=3).mean() + 1e-9) / (
        rn.rolling(7, center=True, min_periods=3).mean() + 1e-9
    )
    base = float(ratio.iloc[: max(int(span * 0.4), 7)].median())
    if not np.isfinite(base) or base <= 0:
        return None
    below = (ratio < 0.8 * base).to_numpy()
    not_below = np.where(~below)[0]
    if len(not_below) == 0:
        return None
    last_ok = int(idx[not_below[-1]])
    horizon = max_day - last_ok
    if horizon < 7 or horizon > span * 0.6:
        return None
    return last_ok
