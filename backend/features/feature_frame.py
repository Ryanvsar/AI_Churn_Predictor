from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _to_datetime_yyyymmdd(series: pd.Series) -> pd.Series:
    # sessions.sessionDate is stored as integer YYYYMMDD in this project.
    return pd.to_datetime(series.astype(str), format="%Y%m%d", errors="coerce")


def _shannon_entropy(counts: np.ndarray) -> float:
    total = float(np.sum(counts))
    if total <= 0:
        return 0.0
    p = counts.astype(float) / total
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))


def _ewm_decay_weights(age_days: pd.Series, half_life_days: float) -> pd.Series:
    # Weight = 0.5^(age/half_life) = exp(-ln(2) * age / half_life)
    half_life_days = max(float(half_life_days), 1e-6)
    lam = math.log(2.0) / half_life_days
    return np.exp(-lam * age_days.astype(float))


@dataclass(frozen=True)
class FeatureConfig:
    half_life_days: float = 14.0
    dropoff_recent_days: int = 14
    rolling_windows_days: Tuple[int, ...] = (7, 14, 30)
    core_features: Tuple[str, ...] = ()


def load_sessions(
    conn: sqlite3.Connection,
    observation_start: Optional[int],
    observation_end: Optional[int],
) -> pd.DataFrame:
    where = []
    if observation_start is not None:
        where.append(f"sessionDate >= {int(observation_start)}")
    if observation_end is not None:
        where.append(f"sessionDate <= {int(observation_end)}")
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    q = f"""
    SELECT userId, sessionId, sessionDate, sessionLength, sessionEvents
    FROM sessions
    {where_sql}
    ;
    """
    df = pd.read_sql_query(q, conn)
    if df.empty:
        return df
    df["session_dt"] = _to_datetime_yyyymmdd(df["sessionDate"])
    return df


def load_feature_usage(
    conn: sqlite3.Connection,
    observation_start: Optional[int],
    observation_end: Optional[int],
) -> pd.DataFrame:
    # Optional table (may not exist). Expected columns:
    # userId, sessionId, sessionDate, featureKey, count
    tables = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='feature_usage';",
        conn,
    )
    if tables.empty:
        return pd.DataFrame()

    where = []
    if observation_start is not None:
        where.append(f"sessionDate >= {int(observation_start)}")
    if observation_end is not None:
        where.append(f"sessionDate <= {int(observation_end)}")
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    q = f"""
    SELECT userId, sessionId, sessionDate, featureKey, count
    FROM feature_usage
    {where_sql}
    ;
    """
    df = pd.read_sql_query(q, conn)
    if df.empty:
        return df
    df["session_dt"] = _to_datetime_yyyymmdd(df["sessionDate"])
    return df


def build_user_feature_frame(
    db_path: str,
    observation_start: Optional[int] = None,
    observation_end: Optional[int] = None,
    config: FeatureConfig = FeatureConfig(),
) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        df_sess = load_sessions(conn, observation_start, observation_end)
        if df_sess.empty:
            return pd.DataFrame(columns=["userId"])

        # Define observation window endpoints based on actual data if not provided.
        obs_start_dt = df_sess["session_dt"].min()
        obs_end_dt = df_sess["session_dt"].max()
        if pd.isna(obs_start_dt) or pd.isna(obs_end_dt):
            return pd.DataFrame(columns=["userId"])

        # ------------- Core aggregates -------------
        agg = df_sess.groupby("userId").agg(
            total_sessions=("sessionId", "count"),
            avg_session_length=("sessionLength", "mean"),
            std_session_length=("sessionLength", "std"),
            avg_session_events=("sessionEvents", "mean"),
            std_session_events=("sessionEvents", "std"),
            first_session_dt=("session_dt", "min"),
            last_session_dt=("session_dt", "max"),
        )
        agg = agg.reset_index()
        agg[["std_session_length", "std_session_events"]] = agg[
            ["std_session_length", "std_session_events"]
        ].fillna(0.0)

        # Recency (days since last session)
        agg["recency_days"] = (obs_end_dt - agg["last_session_dt"]).dt.days.astype(float)

        # Sessions per day (normalized by observed range length)
        obs_days = max(int((obs_end_dt - obs_start_dt).days) + 1, 1)
        agg["sessions_per_day"] = agg["total_sessions"] / float(obs_days)

        # Engagement rate (events per minute-ish; your units are arbitrary, keep ratio)
        agg["engagement_rate"] = agg["avg_session_events"] / (agg["avg_session_length"] + 1.0)

        # Volatility signals
        agg["engagement_volatility"] = agg["std_session_length"].astype(float)
        agg["events_volatility"] = agg["std_session_events"].astype(float)

        # ------------- Gap variance / session cadence -------------
        df_sorted = df_sess.sort_values(["userId", "session_dt"])
        df_sorted["gap_days"] = (
            df_sorted.groupby("userId")["session_dt"].diff().dt.days.astype(float)
        )
        gap_stats = (
            df_sorted.groupby("userId")["gap_days"]
            .agg(session_gap_mean_days="mean", session_gap_var_days="var")
            .reset_index()
        )
        gap_stats[["session_gap_mean_days", "session_gap_var_days"]] = gap_stats[
            ["session_gap_mean_days", "session_gap_var_days"]
        ].fillna(0.0)
        agg = agg.merge(gap_stats, on="userId", how="left").fillna(
            {"session_gap_mean_days": 0.0, "session_gap_var_days": 0.0}
        )

        # ------------- Weekend vs weekday usage ratio -------------
        df_sess["dow"] = df_sess["session_dt"].dt.dayofweek  # Mon=0, Sun=6
        df_sess["is_weekend"] = df_sess["dow"].isin([5, 6]).astype(int)
        wk = (
            df_sess.groupby("userId")["is_weekend"]
            .agg(weekend_sessions="sum", total="count")
            .reset_index()
        )
        wk["weekday_sessions"] = wk["total"] - wk["weekend_sessions"]
        agg = agg.merge(wk[["userId", "weekend_sessions", "weekday_sessions"]], on="userId", how="left")
        agg["weekend_weekday_ratio"] = agg["weekend_sessions"] / (agg["weekday_sessions"] + 1.0)

        # ------------- Time-of-day consistency (entropy proxy) -------------
        # No timestamp-of-day exists in current schema (only YYYYMMDD). We provide:
        # - day_of_week_entropy: consistency across weekdays (lower entropy = more consistent schedule)
        dow_counts = (
            df_sess.groupby(["userId", "dow"])["sessionId"]
            .count()
            .rename("cnt")
            .reset_index()
        )
        # Pivot to 7 bins, compute entropy
        dow_pivot = dow_counts.pivot(index="userId", columns="dow", values="cnt").fillna(0.0)
        dow_entropy = dow_pivot.apply(lambda row: _shannon_entropy(row.to_numpy()), axis=1).rename(
            "day_of_week_entropy"
        )
        agg = agg.merge(dow_entropy.reset_index(), on="userId", how="left").fillna(
            {"day_of_week_entropy": 0.0}
        )

        # ------------- Rolling windows -------------
        for w in config.rolling_windows_days:
            w_start = obs_end_dt - pd.Timedelta(days=int(w) - 1)
            m = df_sess["session_dt"] >= w_start
            df_w = df_sess.loc[m]
            g = df_w.groupby("userId").agg(
                **{
                    f"session_count_last_{w}d": ("sessionId", "count"),
                    f"avg_session_length_last_{w}d": ("sessionLength", "mean"),
                    f"avg_session_events_last_{w}d": ("sessionEvents", "mean"),
                }
            )
            g = g.reset_index()
            agg = agg.merge(g, on="userId", how="left")

        # Fill rolling nulls with 0
        for w in config.rolling_windows_days:
            agg[f"session_count_last_{w}d"] = agg[f"session_count_last_{w}d"].fillna(0).astype(int)
            agg[f"avg_session_length_last_{w}d"] = agg[f"avg_session_length_last_{w}d"].fillna(0.0)
            agg[f"avg_session_events_last_{w}d"] = agg[f"avg_session_events_last_{w}d"].fillna(0.0)

        # ------------- Decay-weighted session frequency / events -------------
        df_sess["age_days"] = (obs_end_dt - df_sess["session_dt"]).dt.days.astype(float)
        df_sess["decay_w"] = _ewm_decay_weights(df_sess["age_days"], config.half_life_days)
        df_sess["w_events"] = df_sess["sessionEvents"].astype(float) * df_sess["decay_w"]
        df_sess["w_len"] = df_sess["sessionLength"].astype(float) * df_sess["decay_w"]
        dec = (
            df_sess.groupby("userId")
            .agg(
                decay_weight_sum=("decay_w", "sum"),
                decay_weighted_events=("w_events", "sum"),
                decay_weighted_length=("w_len", "sum"),
            )
            .reset_index()
        )
        dec["decay_session_frequency"] = dec["decay_weight_sum"] / float(obs_days)
        dec["decay_avg_events"] = dec["decay_weighted_events"] / (dec["decay_weight_sum"] + 1e-9)
        dec["decay_avg_length"] = dec["decay_weighted_length"] / (dec["decay_weight_sum"] + 1e-9)
        agg = agg.merge(
            dec[["userId", "decay_session_frequency", "decay_avg_events", "decay_avg_length"]],
            on="userId",
            how="left",
        ).fillna(
            {"decay_session_frequency": 0.0, "decay_avg_events": 0.0, "decay_avg_length": 0.0}
        )

        # ------------- Trend features (first vs last half) -------------
        midpoint = obs_start_dt + (obs_end_dt - obs_start_dt) / 2
        first_half = df_sess[df_sess["session_dt"] <= midpoint]
        last_half = df_sess[df_sess["session_dt"] > midpoint]

        fh = first_half.groupby("userId").agg(
            fh_avg_length=("sessionLength", "mean"),
            fh_count=("sessionId", "count"),
        )
        lh = last_half.groupby("userId").agg(
            lh_avg_length=("sessionLength", "mean"),
            lh_count=("sessionId", "count"),
        )
        t = fh.join(lh, how="outer").fillna(0.0).reset_index()
        agg = agg.merge(t, on="userId", how="left").fillna(0.0)
        agg["length_trend"] = (agg["lh_avg_length"] - agg["fh_avg_length"]) / (agg["fh_avg_length"] + 1.0)
        agg["frequency_trend"] = (agg["lh_count"] - agg["fh_count"]) / (agg["fh_count"] + 1.0)

        # ------------- Expanded consistency score -------------
        agg["consistency_score"] = 1.0 / (agg["std_session_length"] + 1.0)
        agg["consistency_score_expanded"] = (
            (1.0 / (agg["std_session_length"] + 1.0))
            + (1.0 / (agg["std_session_events"] + 1.0))
            + (1.0 / (agg["session_gap_var_days"] + 1.0))
        ) / 3.0

        # ------------- Feature usage / product depth (optional) -------------
        df_fu = load_feature_usage(conn, observation_start, observation_end)
        if df_fu.empty:
            agg["feature_breadth"] = 0.0
            agg["feature_depth_total"] = 0.0
            agg["core_feature_adoption_rate"] = 0.0
            agg["feature_dropoff_count"] = 0.0
            agg["feature_usage_entropy"] = 0.0
        else:
            df_fu["count"] = df_fu["count"].fillna(1).astype(float)
            breadth = df_fu.groupby("userId")["featureKey"].nunique().rename("feature_breadth")
            depth_total = df_fu.groupby("userId")["count"].sum().rename("feature_depth_total")

            core = set(config.core_features)
            if len(core) > 0:
                used_core = (
                    df_fu[df_fu["featureKey"].isin(core)]
                    .groupby("userId")["featureKey"]
                    .nunique()
                    .rename("used_core_unique")
                )
                adoption = (used_core / float(len(core))).rename("core_feature_adoption_rate")
            else:
                adoption = pd.Series(dtype=float, name="core_feature_adoption_rate")

            recent_start = obs_end_dt - pd.Timedelta(days=int(config.dropoff_recent_days) - 1)
            hist = df_fu.groupby(["userId", "featureKey"])["count"].sum().reset_index()
            recent = (
                df_fu[df_fu["session_dt"] >= recent_start]
                .groupby(["userId", "featureKey"])["count"]
                .sum()
                .reset_index()
            )
            hist_set = hist.groupby("userId")["featureKey"].apply(set).rename("hist_set")
            recent_set = recent.groupby("userId")["featureKey"].apply(set).rename("recent_set")
            drop = pd.concat([hist_set, recent_set], axis=1).fillna(value={})

            def _drop_cnt(row) -> float:
                hs = row.get("hist_set") or set()
                rs = row.get("recent_set") or set()
                return float(len(hs - rs))

            dropoff = drop.apply(_drop_cnt, axis=1).rename("feature_dropoff_count")

            # Feature usage entropy: distribution over feature keys
            fe_counts = df_fu.groupby(["userId", "featureKey"])["count"].sum().rename("cnt").reset_index()
            pivot = fe_counts.pivot(index="userId", columns="featureKey", values="cnt").fillna(0.0)
            fu_entropy = pivot.apply(lambda row: _shannon_entropy(row.to_numpy()), axis=1).rename(
                "feature_usage_entropy"
            )

            agg = agg.merge(breadth.reset_index(), on="userId", how="left")
            agg = agg.merge(depth_total.reset_index(), on="userId", how="left")
            agg = agg.merge(adoption.reset_index(), on="userId", how="left") if not adoption.empty else agg
            agg = agg.merge(dropoff.reset_index(), on="userId", how="left")
            agg = agg.merge(fu_entropy.reset_index(), on="userId", how="left")

            agg["feature_breadth"] = agg["feature_breadth"].fillna(0.0)
            agg["feature_depth_total"] = agg["feature_depth_total"].fillna(0.0)
            if "core_feature_adoption_rate" in agg.columns:
                agg["core_feature_adoption_rate"] = agg["core_feature_adoption_rate"].fillna(0.0)
            else:
                agg["core_feature_adoption_rate"] = 0.0
            agg["feature_dropoff_count"] = agg["feature_dropoff_count"].fillna(0.0)
            agg["feature_usage_entropy"] = agg["feature_usage_entropy"].fillna(0.0)

        # ------------- Final cleanup / types -------------
        # Keep dates as ISO strings for potential debugging but not as model features by default.
        agg["first_session_date"] = agg["first_session_dt"].dt.strftime("%Y-%m-%d")
        agg["last_session_date"] = agg["last_session_dt"].dt.strftime("%Y-%m-%d")
        agg = agg.drop(columns=["first_session_dt", "last_session_dt"], errors="ignore")

        return agg
    finally:
        conn.close()


def default_feature_columns(include_optional: bool = True) -> List[str]:
    cols = [
        "avg_session_length",
        "std_session_length",
        "avg_session_events",
        "std_session_events",
        "total_sessions",
        "sessions_per_day",
        "recency_days",
        "engagement_rate",
        "engagement_volatility",
        "events_volatility",
        "session_gap_mean_days",
        "session_gap_var_days",
        "weekend_weekday_ratio",
        "day_of_week_entropy",
        "decay_session_frequency",
        "decay_avg_events",
        "decay_avg_length",
        "length_trend",
        "frequency_trend",
        "consistency_score",
        "consistency_score_expanded",
    ]
    if include_optional:
        cols += [
            "feature_breadth",
            "feature_depth_total",
            "core_feature_adoption_rate",
            "feature_dropoff_count",
            "feature_usage_entropy",
        ]
    return cols

