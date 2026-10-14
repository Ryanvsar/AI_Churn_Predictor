"""
Framework-free service layer.

Everything the UI needs is produced here as plain JSON-safe dicts, so the exact same code serves
both the FastAPI server (`backend/app.py`) and the in-browser engine (Pyodide web worker).
"""
from __future__ import annotations

import csv
import io
import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import func, select

from backend.connectors.source import open_source
from backend.data.timeseries import user_daily_series
from backend.explain.contrib import top_factors_for_user, why_risky_summary
from backend.pipeline.run import AnalysisOptions, AnalysisResult, run_analysis
from backend.schema.detect import SchemaMap, detect_schema
from backend.schema.introspect import profile_database, reflect_table
from backend.schema.timeparse import parse_time

TIER_ORDER = ["High", "Medium", "Low"]


def clean(v: Any) -> Any:
    """JSON-safe values: NaN/inf -> None, numpy scalars -> python."""
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return f if math.isfinite(f) else None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, dict):
        return {k: clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if v is pd.NA or v is pd.NaT:
        return None
    return v


# ------------------------------------------------------------------------------------------------
# Detection
# ------------------------------------------------------------------------------------------------

def _date_range(source, table: str, column: str, fmt: str) -> Optional[List[str]]:
    """Min/max of a time column (raw min/max order matches time order for all supported formats)."""
    try:
        t = reflect_table(source, table)
        with source.engine.connect() as conn:
            lo, hi = conn.execute(select(func.min(t.c[column]), func.max(t.c[column]))).one()
        parsed = parse_time(pd.Series([lo, hi]), fmt)
        if parsed.isna().any():
            return None
        return [str(parsed.iloc[0].date()), str(parsed.iloc[1].date())]
    except Exception:  # noqa: BLE001 - the range is informational only
        return None


def validate_source(spec: str, display_name: Optional[str] = None) -> Dict:
    source = open_source(spec)
    profile = profile_database(source)
    smap = detect_schema(source, profile)
    acts = smap.enabled_activities()
    tables = profile.tables

    found = []
    for a in acts:
        found.append({
            "table": a.table,
            "rows": tables[a.table].row_count,
            "timeColumn": a.time_column,
            "dateRange": _date_range(source, a.table, a.time_column, a.time_format),
            "measures": a.measures,
            "categories": a.categoricals,
        })
    summary = {
        "userTable": smap.user_table,
        "userCount": tables[smap.user_table].row_count if smap.user_table else None,
        "activities": found,
        "hasLabel": smap.label is not None,
        "labelColumn": f"{smap.label.table}.{smap.label.column}" if smap.label else None,
    }
    return {
        "ok": bool(acts),
        "dbPath": spec,
        "source": display_name or source.display,
        "dialect": source.dialect,
        "tables": profile.to_dict()["tables"],
        "capabilities": {
            "hasUsersTable": smap.user_table is not None,
            "hasSessionsTable": bool(acts),
            "hasChurnLabels": smap.label is not None,
            "hasFeatureUsageTable": any(a.categoricals for a in acts),
            "hasSessionEventsTable": len(acts) > 1,
            "activityTables": [a.table for a in acts],
            "notes": smap.notes,
        },
        "schemaMap": smap.to_dict(),
        "summary": summary,
    }


# ------------------------------------------------------------------------------------------------
# Analysis wrapper
# ------------------------------------------------------------------------------------------------

class Analysis:
    """A finished analysis plus cached per-user lookups."""

    def __init__(self, res: AnalysisResult):
        self.res = res
        self.df = res.df_scored
        self.row_index = {uid: i for i, uid in enumerate(self.df["userId"].tolist())}
        self._reasons: Optional[List[Optional[str]]] = None

    def row(self, user_id: str):
        idx = self.row_index.get(str(user_id))
        if idx is None:
            raise KeyError(f"User {user_id} not found in results.")
        return idx, self.df.iloc[idx]

    def top_reasons(self) -> List[Optional[str]]:
        """Readable name of each user's strongest risk-increasing factor."""
        if self._reasons is None:
            c = self.res.contributions
            if c is None:
                self._reasons = [None] * len(self.df)
            else:
                feats = self.res.features
                body = c[:, :-1]
                best = np.argmax(body, axis=1)
                pos = body[np.arange(len(body)), best] > 0
                labels = self.res.feature_labels
                self._reasons = [
                    labels.get(feats[j], feats[j]) if ok else None for j, ok in zip(best, pos)
                ]
                churned = (self.df["status"] == "churned").to_numpy()
                self._reasons = [None if ch else r for r, ch in zip(self._reasons, churned)]
        return self._reasons


def analyze(options: Dict, progress=lambda p, m: None) -> Analysis:
    keys = AnalysisOptions.__dataclass_fields__
    opts = AnalysisOptions(**{k: v for k, v in options.items() if k in keys})
    return Analysis(run_analysis(opts, progress))


def job_summary(an: Analysis) -> Dict:
    res, df = an.res, an.df
    churn = res.churn.to_dict()
    return clean({
        "observationStart": str(res.data.min_ts.date()) if res.data.min_ts is not None else None,
        "observationEnd": churn["scoring_date"],
        "usersTotal": int(df.shape[0]),
        "usersScored": int(df.shape[0]),
        "usersActive": int((df["status"] == "active").sum()),
        "featuresUsed": list(res.features),
        "supervised": bool(res.supervised),
        "churnDefinition": churn,
        "metrics": res.metrics,
        "notes": res.notes,
    })


def _quality(res: AnalysisResult) -> Dict:
    m = res.metrics
    if not res.supervised or m.get("rocAuc") is None:
        return {"grade": "Estimate", "text": "Not enough history to learn from, so this is a rule-of-thumb estimate."}
    auc, lift = m["rocAuc"], m.get("liftTop10")
    grade = "Good" if auc >= 0.85 else "Fair" if auc >= 0.75 else "Weak"
    text = (f"The 10% of users flagged as riskiest are {lift:.1f}× more likely to churn than average."
            if lift else f"Ranks a churner above a non-churner {auc:.0%} of the time.")
    return {"grade": grade, "text": text, "auc": auc, "liftTop10": lift}


def overview(an: Analysis) -> Dict:
    res, df = an.res, an.df
    active = df[df["status"] != "churned"]
    n_active = int(len(active))
    expected = float(active["churnProbability"].sum())
    tiers = [{"tier": t, "count": int((active["riskTier"] == t).sum())} for t in TIER_ORDER]
    churned = int((df["status"] == "churned").sum())
    hist, edges = np.histogram(active["churnProbability"].to_numpy(), bins=10, range=(0.0, 1.0))
    total_imp = sum(r["importance"] for r in res.global_importance) or 1.0
    drivers = [
        {"key": r["featureKey"], "name": r["displayName"], "share": r["importance"] / total_imp}
        for r in res.global_importance[:6]
    ]
    return clean({
        "activeUsers": n_active,
        "alreadyChurned": churned,
        "expectedChurners": expected,
        "expectedChurnRate": expected / n_active if n_active else 0.0,
        "windowDays": res.churn.window_days,
        "scoringDate": res.churn.scoring_date,
        "tiers": tiers,
        "riskHistogram": [{"from": float(edges[i]), "to": float(edges[i + 1]), "count": int(hist[i])} for i in range(10)],
        "drivers": drivers,
        "quality": _quality(res),
        "churnDescription": res.churn.describe(),
        "labelSource": res.churn.label_source,
        "supervised": res.supervised,
    })


def users_all(an: Analysis) -> List[Dict]:
    """Every user, compact, sorted by risk (the UI filters, searches and exports client-side)."""
    df = an.df
    reasons = an.top_reasons()
    rec = df["recency_days"].to_numpy(dtype=float)
    known = df["knownLabel"].to_numpy(dtype=float) if "knownLabel" in df else None
    out = [
        {
            "userId": uid,
            "risk": float(p),
            "tier": tier,
            "status": status,
            "recencyDays": None if not np.isfinite(r) else int(r),
            "topReason": reason,
            "knownLabel": None if known is None or not np.isfinite(known[i]) else int(known[i]),
        }
        for i, (uid, p, tier, status, r, reason) in enumerate(
            zip(df["userId"], df["churnProbability"], df["riskTier"], df["status"], rec, reasons)
        )
    ]
    out.sort(key=lambda u: (u["status"] == "churned", -u["risk"]))
    return out


def export_csv(an: Analysis) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["user_id", "churn_probability", "risk_level", "status", "days_since_last_activity", "main_reason"])
    for u in users_all(an):  # active users by risk first, then those who already churned
        w.writerow([u["userId"], f"{u['risk']:.4f}", u["tier"] if u["status"] != "churned" else "Churned",
                    u["status"], u["recencyDays"], u["topReason"] or ""])
    return buf.getvalue()


# ------------------------------------------------------------------------------------------------
# Per-user
# ------------------------------------------------------------------------------------------------

def _known_label(row) -> Optional[int]:
    v = row.get("knownLabel")
    return None if v is None or pd.isna(v) else int(v)


def user_details(an: Analysis, user_id: str) -> Dict:
    _, row = an.row(user_id)
    skip = {"userId", "churnProbability", "riskTier", "status", "knownLabel"}
    metrics = {k: clean(v) for k, v in row.items() if k not in skip}
    return {
        "userId": str(user_id),
        "churnProbability": float(row["churnProbability"]),
        "riskTier": row["riskTier"],
        "status": row["status"],
        "knownLabel": _known_label(row),
        "metrics": metrics,
        "metricLabels": {k: an.res.feature_labels.get(k, k) for k in metrics},
    }


def user_explain(an: Analysis, user_id: str, top_k: int = 8) -> Dict:
    idx, row = an.row(user_id)
    res = an.res
    base = {"userId": str(user_id), "churnProbability": float(row["churnProbability"]), "topFactors": []}
    if res.contributions is None:
        return {**base, "whyRiskySummary": "Explanations need a trained model; this run used a rule-of-thumb "
                                           "estimate because there was not enough history."}
    if row["status"] == "churned":
        why = ("Already churned according to the label column in your database."
               if res.churn.label_source == "explicit"
               else f"Already churned: no activity for {int(row['recency_days'])} days "
                    f"(churn means {res.churn.window_days} days without activity).")
        return {**base, "whyRiskySummary": why}
    top = top_factors_for_user(row, res.contributions[idx], res.features, res.feature_labels, top_k=top_k)
    return {**base, "topFactors": clean(top), "whyRiskySummary": why_risky_summary(top)}


def user_timeseries(an: Analysis, user_id: str, days: int = 0) -> Dict:
    an.row(user_id)
    res = an.res
    n = int(days) if days and days > 0 else max(res.churn.lookback_days, 60)
    names, points = user_daily_series(res.data, str(user_id), res.scoring_day, min(n, 730))
    return {"userId": str(user_id), "measureNames": names, "scoringDate": res.churn.scoring_date, "points": points}


KEY_METRICS = ["recency_days", "active_day_rate", "gap_mean_days", "current_streak", "tenure_days",
               "tenure_active_days", "lifetime_active_rate", "recent_vs_baseline_activity"]


def user_bundle(an: Analysis, user_id: str) -> Dict:
    """Everything the user panel shows, in one compact payload."""
    d = user_details(an, user_id)
    e = user_explain(an, user_id, top_k=5)
    h = an.res.churn.window_days
    days = 90 if h >= 21 else 60
    ts = user_timeseries(an, user_id, days)
    pts = ts["points"]
    keys = [k for k in KEY_METRICS if k in d["metrics"]][:6]
    keys += [f["featureKey"] for f in e["topFactors"] if f["featureKey"] not in keys][: 8 - len(keys)]
    return {
        "userId": d["userId"],
        "risk": d["churnProbability"],
        "tier": d["riskTier"],
        "status": d["status"],
        "knownLabel": d["knownLabel"],
        "recencyDays": d["metrics"].get("recency_days"),
        "why": e["whyRiskySummary"],
        "factors": [
            {"name": f["displayName"], "effect": f["shapValue"], "value": f["featureValue"], "key": f["featureKey"]}
            for f in e["topFactors"]
        ],
        "metrics": [{"key": k, "name": d["metricLabels"].get(k, k), "value": d["metrics"].get(k)} for k in keys],
        "series": {
            "start": pts[0]["date"] if pts else None,
            "end": ts["scoringDate"],
            "activity": [p["activityCount"] for p in pts],
            "measures": {m: [p["measures"].get(m, 0.0) for p in pts] for m in ts["measureNames"]},
        },
        "windowDays": h,
    }


# ------------------------------------------------------------------------------------------------
# Aggregates / model (REST API compatibility)
# ------------------------------------------------------------------------------------------------

def users_list(an: Analysis, top: int = 100, threshold: float = 0.0, sort: str = "prob_desc",
               include_churned: bool = True) -> Dict:
    df = an.df
    cols = ["userId", "churnProbability", "riskTier", "status"] + (["knownLabel"] if "knownLabel" in df else [])
    dfu = df[cols].assign(top_reason=an.top_reasons(), recency=df["recency_days"].to_numpy())
    if not include_churned:
        dfu = dfu[dfu["status"] != "churned"]
    dfu = dfu[dfu["churnProbability"] >= float(threshold)]
    dfu = dfu.sort_values("churnProbability", ascending=(sort == "prob_asc"), kind="stable")
    total = int(dfu.shape[0])
    dfu = dfu.head(max(1, min(int(top), 50000)))
    users = [
        {
            "userId": str(r.userId),
            "churnProbability": float(r.churnProbability),
            "riskTier": r.riskTier,
            "status": r.status,
            "knownLabel": None if not hasattr(r, "knownLabel") or pd.isna(r.knownLabel) else int(r.knownLabel),
            "recencyDays": None if pd.isna(r.recency) else int(r.recency),
            "topReason": r.top_reason if isinstance(r.top_reason, str) else None,
        }
        for r in dfu.itertuples(index=False)
    ]
    return {"total": total, "users": users}


def aggregates(an: Analysis) -> Dict:
    res, df = an.res, an.df
    numeric_cols = [
        c for c in df.columns
        if c not in {"userId", "riskTier", "status", "knownLabel"} and pd.api.types.is_numeric_dtype(df[c])
    ]

    def group(name, dfg):
        means = dfg[numeric_cols].mean(numeric_only=True).to_dict()
        return {"name": name, "count": int(dfg.shape[0]),
                "metricsMean": {k: float(v) for k, v in means.items() if pd.notna(v) and math.isfinite(v)}}

    active = df[df["status"] != "churned"]
    groups = [group("AllUsers", df)] + [group(f"{t}Risk", active[active["riskTier"] == t]) for t in TIER_ORDER]
    if (df["status"] == "churned").any():
        groups.append(group("AlreadyChurned", df[df["status"] == "churned"]))
    labels = {c: res.feature_labels.get(c, c) for c in numeric_cols}
    labels["churnProbability"] = "Churn probability"
    return {"groups": groups, "metricLabels": labels, "globalFeatureImportance": res.global_importance}


def model_info(an: Analysis) -> Dict:
    res = an.res
    return clean({
        "supervised": res.supervised,
        "churnDefinition": res.churn.to_dict(),
        "metrics": res.metrics,
        "schemaMap": res.schema_map.to_dict(),
        "notes": res.notes,
    })


def full_result(an: Analysis) -> Dict:
    """Dashboard payload: summary + overview + all users + model info."""
    return {
        "summary": job_summary(an),
        "overview": overview(an),
        "users": users_all(an),
        "model": model_info(an),
    }


def mapping_from_dict(d: Dict) -> SchemaMap:
    return SchemaMap.from_dict(d)
