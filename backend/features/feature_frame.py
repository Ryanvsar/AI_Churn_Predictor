"""
Schema-agnostic, cutoff-aware feature engineering.

`build_features(data, cutoff_day, ...)` describes every user using ONLY activity on or before
`cutoff_day`, so the same code builds historical training snapshots and today's scoring frame
without leaking the future.

Feature families (generalised from the original sessions-only pipeline):
- overall engagement: recency, active days in cadence-scaled windows, gaps, streaks,
  decay-weighted activity, trend, recent-vs-baseline ratio, day-of-week / hour-of-day entropy
- per activity table: event counts, recency, events per active day, trend
- per numeric measure: mean / std / sum over the lookback, recent mean, recent-vs-baseline ratio
- per categorical column: breadth, entropy, categories dropped recently, share of top categories
- static: tenure, user attributes (one-hot) and safe numeric attributes
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backend.data.extract import ExtractedData
from backend.labels.churn_definition import to_daynum
from backend.schema.naming import humanize

TOP_CATEGORIES = 5
TOP_STATIC_LEVELS = 8
HISTORY_DAYS = 365  # long-range history used for "vs. usual" comparisons


@dataclass
class FeatureSpec:
    """Things fixed once per analysis so every snapshot gets identical columns."""
    window_days: int
    lookback_days: int
    windows: Tuple[int, ...]
    top_categories: Dict[Tuple[str, str], List[str]] = field(default_factory=dict)
    static_levels: Dict[str, List[str]] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)  # feature key -> display name


def make_spec(data: ExtractedData, window_days: int, lookback_days: int) -> FeatureSpec:
    h, L = int(window_days), int(lookback_days)
    windows = tuple(sorted({w for w in (7, h, 2 * h, L) if 1 <= w <= L}))
    spec = FeatureSpec(window_days=h, lookback_days=L, windows=windows)
    for src, af in data.activities.items():
        for c in af.categoricals:
            vc = af.df[c].dropna().value_counts()
            spec.top_categories[(src, c)] = [str(v) for v in vc.index[:TOP_CATEGORIES]]
    for c in data.static_categoricals:
        vc = data.users[f"static_c::{c}"].dropna().value_counts()
        spec.static_levels[c] = [str(v) for v in vc.index[:TOP_STATIC_LEVELS]]
    return spec


def _entropy_by_user(df: pd.DataFrame, key: str) -> pd.Series:
    cnt = df.groupby(["user_id", key], sort=False, observed=True).size().rename("n").reset_index()
    tot = cnt.groupby("user_id", sort=False)["n"].transform("sum")
    p = cnt["n"] / tot
    cnt["h"] = -p * np.log2(p)
    return cnt.groupby("user_id", sort=False)["h"].sum()


def _streaks(days: pd.DataFrame, cutoff: int, lookback_start: int) -> pd.DataFrame:
    d = days[days["day"] >= lookback_start].sort_values(["user_id", "day"], kind="stable")
    if d.empty:
        return pd.DataFrame(columns=["current_streak", "longest_streak"])
    new_run = (d["user_id"] != d["user_id"].shift()) | (d["day"].diff() != 1)
    run_id = new_run.cumsum()
    runs = d.groupby(run_id, sort=False).agg(user_id=("user_id", "first"), end=("day", "max"), n=("day", "size"))
    longest = runs.groupby("user_id", sort=False)["n"].max()
    last = runs.groupby("user_id", sort=False).tail(1).set_index("user_id")
    current = last["n"].where(last["end"] >= cutoff - 1, 0)
    return pd.DataFrame({"current_streak": current, "longest_streak": longest})


class _Builder:
    def __init__(self, users: pd.Index, spec: FeatureSpec):
        self.out = pd.DataFrame(index=users)
        self.spec = spec

    def add(self, key: str, values, label: str, fill=np.nan):
        s = values if isinstance(values, pd.Series) else pd.Series(values)
        self.out[key] = s.reindex(self.out.index).astype(float).fillna(fill) if fill is not None else s.reindex(self.out.index).astype(float)
        self.spec.labels.setdefault(key, label)


def build_features(
    data: ExtractedData,
    cutoff_day: int,
    spec: FeatureSpec,
    users: Optional[pd.Index] = None,
) -> pd.DataFrame:
    """Return one row per user (index = user_id) using only activity on or before `cutoff_day`."""
    h, L = spec.window_days, spec.lookback_days
    lb_start = cutoff_day - L + 1
    recent_start = cutoff_day - h + 1
    half_life = max(float(h), 3.0)

    # ------------------------------------------------------------------ per-source frames
    srcs = {}
    for name, af in data.activities.items():
        df = af.df
        day = to_daynum(df["ts"])
        keep = day <= cutoff_day
        sdf = df.loc[keep].copy()
        sdf["day"] = day[keep]
        srcs[name] = (af, sdf)

    all_days = pd.concat(
        [pd.DataFrame({"user_id": s["user_id"].to_numpy(), "day": s["day"].to_numpy()}) for _, s in srcs.values()],
        ignore_index=True,
    ).drop_duplicates()
    if users is None:
        users = pd.Index(all_days["user_id"].unique(), name="user_id")
    b = _Builder(users, spec)

    # ------------------------------------------------------------------ overall engagement
    g_all = all_days.groupby("user_id", sort=False)["day"]
    last_day, first_day = g_all.max(), g_all.min()
    b.add("recency_days", cutoff_day - last_day, "Days since last activity")
    b.add("tenure_active_days", cutoff_day - first_day, "Days since first activity")

    lb = all_days[all_days["day"] >= lb_start]
    hist_start = cutoff_day - HISTORY_DAYS + 1
    hist = all_days[all_days["day"] >= hist_start]
    hist_span = np.minimum(HISTORY_DAYS, (cutoff_day - first_day + 1).clip(lower=1))
    b.add("lifetime_active_rate", hist.groupby("user_id", sort=False).size() / hist_span,
          "Share of days active since first activity", fill=0.0)
    for w in spec.windows:
        cnt = lb[lb["day"] > cutoff_day - w].groupby("user_id", sort=False).size()
        b.add(f"active_days_{w}d", cnt, f"Active days, last {w} days", fill=0.0)
    observed = np.minimum(L, (cutoff_day - first_day + 1).reindex(b.out.index).fillna(1).clip(lower=1))
    b.add("active_day_rate", b.out[f"active_days_{L}d"] / observed, f"Share of days active (last {L} days)", fill=0.0)
    recent_rate = b.out[f"active_days_{h}d"] / min(h, L)
    b.add("recent_vs_baseline_activity", recent_rate / (b.out["active_day_rate"] + 1e-6),
          f"Activity last {h} days vs usual rate", fill=0.0)

    gaps = lb.sort_values(["user_id", "day"]).groupby("user_id", sort=False)["day"].diff()
    gdf = pd.DataFrame({"user_id": lb.sort_values(["user_id", "day"])["user_id"].to_numpy(), "gap": gaps.to_numpy()}).dropna()
    gg = gdf.groupby("user_id", sort=False)["gap"]
    b.add("gap_mean_days", gg.mean(), "Average days between active days")
    b.add("gap_std_days", gg.std(), "Variability of days between activity")
    b.add("gap_max_days", gg.max(), f"Longest break (last {L} days)")
    typical = gg.median()
    b.add("recency_vs_typical_gap", b.out["recency_days"] / (typical.reindex(b.out.index) + 1.0),
          "Current break vs user's typical break")

    st = _streaks(all_days, cutoff_day, lb_start)
    b.add("current_streak", st["current_streak"], "Current daily streak", fill=0.0)
    b.add("longest_streak", st["longest_streak"], f"Longest daily streak (last {L} days)", fill=0.0)

    lam = math.log(2.0) / half_life
    w = np.exp(-lam * (cutoff_day - lb["day"].to_numpy()))
    b.add("decay_weighted_activity", pd.Series(w, index=lb.index).groupby(lb["user_id"].to_numpy(), sort=False).sum(),
          f"Recency-weighted activity (half-life {int(half_life)}d)", fill=0.0)

    mid = cutoff_day - L // 2
    first_half = lb[lb["day"] <= mid].groupby("user_id", sort=False).size()
    second_half = lb[lb["day"] > mid].groupby("user_id", sort=False).size()
    fh = first_half.reindex(b.out.index).fillna(0)
    sh = second_half.reindex(b.out.index).fillna(0)
    b.add("activity_trend", (sh - fh) / (fh + 1.0), "Activity trend (2nd half vs 1st half)", fill=0.0)

    dow = lb.assign(dow=(lb["day"] + 3) % 7)  # 1970-01-01 was a Thursday -> Monday = 0
    b.add("weekday_entropy", _entropy_by_user(dow, "dow"), "Day-of-week variety", fill=0.0)
    b.add("weekend_share", dow.assign(we=dow["dow"] >= 5).groupby("user_id", sort=False)["we"].mean(),
          "Share of activity on weekends", fill=0.0)

    # ------------------------------------------------------------------ per activity source
    multi = len(srcs) > 1
    for name, (af, sdf) in srcs.items():
        p = f"{name}__"
        tag = f" ({humanize(name)})"
        slb = sdf[sdf["day"] >= lb_start]
        srec = slb[slb["day"] >= recent_start]
        sprev = slb[slb["day"] < recent_start]
        sold = sdf[(sdf["day"] < lb_start) & (sdf["day"] >= hist_start)]
        n_lb = slb.groupby("user_id", sort=False).size()
        n_rec = srec.groupby("user_id", sort=False).size()
        if multi:
            b.add(p + "recency_days", cutoff_day - sdf.groupby("user_id", sort=False)["day"].max(),
                  "Days since last activity" + tag, fill=float(L * 2))
        for wd in spec.windows:
            cnt = slb[slb["day"] > cutoff_day - wd].groupby("user_id", sort=False).size()
            b.add(p + f"events_{wd}d", cnt, f"Events in last {wd} days{tag}", fill=0.0)
        act_days = slb.groupby("user_id", sort=False)["day"].nunique()
        b.add(p + "events_per_active_day", n_lb / act_days, "Events per active day" + tag)
        prev_days = max(L - h, 1)
        b.add(p + "recent_vs_baseline_events",
              (n_rec.reindex(b.out.index).fillna(0) / h) /
              (sprev.groupby("user_id", sort=False).size().reindex(b.out.index).fillna(0) / prev_days + 1e-3),
              f"Event rate last {h} days vs before{tag}", fill=0.0)
        sh_ = slb[slb["day"] > mid].groupby("user_id", sort=False).size().reindex(b.out.index).fillna(0)
        fh_ = slb[slb["day"] <= mid].groupby("user_id", sort=False).size().reindex(b.out.index).fillna(0)
        b.add(p + "events_trend", (sh_ - fh_) / (fh_ + 1.0), "Event trend" + tag, fill=0.0)

        if af.has_time_of_day and not slb.empty:
            hrs = slb.assign(hour=slb["ts"].dt.hour)
            b.add(p + "hour_entropy", _entropy_by_user(hrs, "hour"), "Time-of-day variety" + tag, fill=0.0)

        for m in af.measures:
            mt = f" {humanize(m)}{tag}"
            g = slb.groupby("user_id", sort=False)[m]
            b.add(p + f"{m}__mean", g.mean(), f"Avg{mt}")
            b.add(p + f"{m}__std", g.std(), f"Variability of{mt}")
            b.add(p + f"{m}__sum", g.sum(), f"Total{mt} (last {L} days)", fill=0.0)
            rm = srec.groupby("user_id", sort=False)[m].mean()
            pm = sprev.groupby("user_id", sort=False)[m].mean()
            b.add(p + f"{m}__recent_mean", rm, f"Avg{mt}, last {h} days")
            b.add(p + f"{m}__recent_vs_baseline", (rm - pm) / (pm.abs() + 1e-6),
                  f"Change in{mt} (last {h} days vs before)")
            om = sold.groupby("user_id", sort=False)[m].mean()
            b.add(p + f"{m}__vs_history", (g.mean() - om) / (om.abs() + 1e-6),
                  f"Change in{mt} (last {L} days vs earlier history)")

        for c in af.categoricals:
            ct = f" {humanize(c)}{tag}"
            sub = slb.dropna(subset=[c])
            b.add(p + f"{c}__distinct", sub.groupby("user_id", sort=False)[c].nunique(), f"Distinct{ct}", fill=0.0)
            b.add(p + f"{c}__entropy", _entropy_by_user(sub, c), f"Variety of{ct}", fill=0.0)
            prev_sets = sub[sub["day"] < recent_start].groupby(["user_id", c], sort=False, observed=True).size()
            rec_sets = sub[sub["day"] >= recent_start].groupby(["user_id", c], sort=False, observed=True).size()
            dropped = prev_sets.index.difference(rec_sets.index)
            dropped_cnt = pd.Series(1, index=dropped).groupby(level=0).sum() if len(dropped) else pd.Series(dtype=float)
            b.add(p + f"{c}__dropped", dropped_cnt, f"{humanize(c).capitalize()} values stopped recently{tag}", fill=0.0)
            tot = sub.groupby("user_id", sort=False).size()
            for v in spec.top_categories.get((name, c), []):
                share = sub[sub[c] == v].groupby("user_id", sort=False).size() / tot
                b.add(p + f"{c}__share__{v}", share, f"Share of{ct} = {v}", fill=0.0)

    # ------------------------------------------------------------------ static attributes
    u = data.users.set_index("user_id")
    if "signup_ts" in u.columns:
        signup_day = pd.Series(to_daynum(u["signup_ts"].fillna(pd.Timestamp("1970-01-01"))), index=u.index)
        signup_day = signup_day.where(u["signup_ts"].notna())
        b.add("tenure_days", cutoff_day - signup_day, "Days since signup")
    for c in data.static_numerics:
        b.add(f"user__{c}", u[f"static_n::{c}"], humanize(c).capitalize())
    for c in data.static_categoricals:
        col = u[f"static_c::{c}"].astype("string")
        for v in spec.static_levels.get(c, []):
            b.add(f"user__{c}__is__{v}", (col == v).astype(float), f"{humanize(c).capitalize()} = {v}", fill=0.0)

    out = b.out.replace([np.inf, -np.inf], np.nan)
    out.index.name = "user_id"
    return out
