"""End-to-end churn analysis: source -> schema map -> data -> churn definition -> model -> scores."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from backend.connectors.source import open_source
from backend.data.extract import ExtractedData, clip_time, extract
from backend.features.feature_frame import FeatureSpec, build_features, make_spec
from backend.labels import churn_definition as cd
from backend.model.train_and_score import ModelConfig, TrainedModel, heuristic_probability, risk_tier, train
from backend.schema.detect import SchemaMap, detect_schema, validate_schema_map
from backend.schema.introspect import profile_database
from backend.schema.timeparse import parse_time


@dataclass
class AnalysisOptions:
    source: str
    schema_map: Optional[dict] = None
    churn_window_days: Optional[int] = None
    label_source: str = "auto"  # auto | explicit | derived
    threshold: float = 0.4
    high_risk_threshold: float = 0.7
    max_users: Optional[int] = None
    observation_start: Optional[object] = None  # date-like; YYYYMMDD ints accepted
    observation_end: Optional[object] = None  # "as of" date for scoring


@dataclass
class AnalysisResult:
    df_scored: pd.DataFrame
    features: List[str]
    feature_labels: Dict[str, str]
    contributions: Optional[np.ndarray]  # aligned with df_scored rows, last column = bias
    global_importance: List[Dict]
    churn: cd.ChurnDefinition
    metrics: Dict
    schema_map: SchemaMap
    data: ExtractedData
    scoring_day: int
    supervised: bool
    notes: List[str] = field(default_factory=list)


def _to_ts(x) -> Optional[pd.Timestamp]:
    if x is None or x == "":
        return None
    s = pd.Series([x])
    if isinstance(x, (int, np.integer)) or (isinstance(x, str) and x.isdigit() and len(x) == 8):
        return parse_time(s, "yyyymmdd").iloc[0]
    return parse_time(s, "iso").iloc[0]


def _snapshot_frame(data, spec, cutoff, labels: pd.DataFrame) -> pd.DataFrame:
    feats = build_features(data, cutoff, spec, users=pd.Index(labels["user_id"], name="user_id"))
    feats["label"] = labels.set_index("user_id")["label"].reindex(feats.index).to_numpy()
    feats["snapshot"] = cutoff
    return feats


def run_analysis(opts: AnalysisOptions, progress: Callable[[float, str], None] = lambda p, m: None) -> AnalysisResult:
    notes: List[str] = []
    config = ModelConfig(threshold=opts.threshold, high_risk_threshold=opts.high_risk_threshold)

    progress(0.05, "Reading database structure")
    source = open_source(opts.source)
    profile = profile_database(source)
    if opts.schema_map:
        smap = SchemaMap.from_dict(opts.schema_map)
    else:
        smap = detect_schema(source, profile)
    errs = validate_schema_map(smap, profile)
    if errs:
        raise ValueError("Schema mapping problems: " + "; ".join(errs))

    progress(0.15, "Extracting activity")
    use_label = opts.label_source in ("auto", "explicit")
    data = extract(source, smap, max_users=opts.max_users, use_label=use_label)
    data = clip_time(data, _to_ts(opts.observation_start), _to_ts(opts.observation_end))

    days = cd.activity_days([af.df for af in data.activities.values()])
    min_day, max_day = int(days["day"].min()), int(days["day"].max())
    end_day = cd.complete_end_day(days)
    if end_day < max_day:
        notes.append(f"The last {max_day - end_day} day(s) of data look incomplete; analysing up to {cd.daynum_to_date(end_day)}.")
        data = clip_time(data, None, pd.Timestamp(cd.daynum_to_date(end_day)))
        days = days[days["day"] <= end_day]
        max_day = end_day
    h, median_gap, p95_gap, wnotes = cd.infer_window(days, min_day, max_day, opts.churn_window_days)
    notes += wnotes
    L = cd.lookback_for(h)
    churn = cd.ChurnDefinition(
        window_days=h, lookback_days=L, auto=not opts.churn_window_days, label_source="derived",
        median_gap_days=round(median_gap, 2), p95_gap_days=round(p95_gap, 2),
        scoring_date=cd.daynum_to_date(max_day), notes=notes,
    )

    # ---------------- training data ----------------
    progress(0.3, "Building training snapshots")
    train_df = val_df = None
    validation_kind = ""
    label_cutoff, labels = None, None
    if data.has_label and opts.label_source in ("auto", "explicit"):
        labels = data.users[["user_id", "label"]].dropna()
        label_cutoff = cd.detect_label_cutoff(days, labels, min_day, max_day)
        if label_cutoff is None and opts.label_source == "explicit":
            label_cutoff = max_day - h
            notes.append("Could not find when labelled churners' activity drops; using features as of "
                         f"{cd.daynum_to_date(label_cutoff)} (one churn window before the end).")
        if label_cutoff is not None:
            labels = labels[labels["user_id"].isin(days[days["day"] <= label_cutoff]["user_id"].unique())]
            if labels["label"].nunique() < 2:
                label_cutoff = None
        else:
            notes.append("The label column does not show a clear point where churners drift away "
                         "(it may record churn at many different times), so churn is derived from inactivity instead.")

    if label_cutoff is not None:
        # An explicit label has no fixed horizon, so let features see all history before the cutoff.
        L = int(min(max(label_cutoff - min_day + 1, 28), 180))
        churn.lookback_days = L
        spec = make_spec(data, h, L)
        train_df = _snapshot_frame(data, spec, label_cutoff, labels)
        churn.label_source = "explicit"
        churn.label_cutoff_date = cd.daynum_to_date(label_cutoff)
        validation_kind = "held-out 25% of users (explicit labels)"
    else:
        spec = make_spec(data, h, L)

    if train_df is None:
        cuts = cd.snapshot_cutoffs(min_day, max_day, h, L, config.max_snapshots)
        churn.snapshot_dates = [cd.daynum_to_date(c) for c in cuts]
        frames = []
        for i, c in enumerate(cuts):
            lab = cd.derived_labels(days, c, h)
            if len(lab):
                frames.append(_snapshot_frame(data, spec, c, lab))
            progress(0.3 + 0.25 * (i + 1) / max(len(cuts), 1), f"Snapshot {i + 1}/{len(cuts)}")
        frames = [f for f in frames if f["label"].nunique() == 2 or len(frames) == 1]
        if len(frames) >= 2:
            val_df = frames[0]
            train_df = pd.concat(frames[1:])
            validation_kind = f"most recent snapshot ({churn.snapshot_dates[0]}), trained on {len(frames) - 1} earlier"
        elif len(frames) == 1 and frames[0]["label"].nunique() == 2 and frames[0]["label"].value_counts().min() >= 10:
            train_df = frames[0]
            validation_kind = "held-out 25% of users (single snapshot)"

    # ---------------- score as of the latest date ----------------
    progress(0.6, "Scoring users")
    score_feats = build_features(data, max_day, spec)
    features = [c for c in score_feats.columns]
    feature_labels = dict(spec.labels)
    model: Optional[TrainedModel] = None
    metrics: Dict = {}
    supervised = False
    contributions = None

    if train_df is not None:
        # Drop features that never vary in training (they cannot help and clutter explanations).
        nunique = train_df[features].nunique(dropna=True)
        features = [f for f in features if nunique.get(f, 0) > 1]
        progress(0.65, "Training model")
        try:
            model = train(
                train_df, val_df, features, config,
                baseline_cols=["recency_days", f"active_days_{h}d", f"active_days_{L}d"],
                validation_kind=validation_kind,
            )
            metrics = model.metrics
            supervised = True
        except ValueError as e:
            notes.append(f"Model training skipped: {e}")

    df = score_feats.reset_index()
    recency = df["recency_days"].to_numpy(dtype=float)
    known = None
    if data.has_label:
        known = data.users.set_index("user_id")["label"].reindex(df["user_id"]).to_numpy(dtype=float)
    if churn.label_source == "explicit":
        # The label column says who has already churned; everyone else is predicted.
        status = np.where(known == 1, "churned", np.where(recency >= h, "inactive", "active"))
    else:
        status = np.where(recency >= h, "churned", "active")

    if supervised:
        probs = model.predict(score_feats)
        progress(0.85, "Explaining predictions")
        contributions = model.contributions(score_feats)
    else:
        notes.append("Not enough history to learn from labelled examples; using a cadence-based heuristic score.")
        probs = heuristic_probability(recency, df["gap_mean_days"].to_numpy(dtype=float), h)

    # Already churned users are not part of the prediction problem.
    probs = np.where(status == "churned", 1.0, probs)

    out = pd.DataFrame({
        "userId": df["user_id"].astype(str),
        "churnProbability": probs.astype(float),
        "riskTier": [risk_tier(float(p), config.threshold, config.high_risk_threshold) for p in probs],
        "status": status,
    })
    if known is not None:
        out["knownLabel"] = known
    out = pd.concat([out, score_feats.reset_index(drop=True)[[c for c in score_feats.columns]]], axis=1)

    global_imp: List[Dict] = []
    if contributions is not None:
        active_rows = out["status"].to_numpy() == "active"
        rows = contributions[active_rows] if active_rows.any() else contributions
        mean_abs = np.abs(rows[:, :-1]).mean(axis=0)
        order = np.argsort(-mean_abs)
        global_imp = [
            {"featureKey": model.features[i], "displayName": feature_labels.get(model.features[i], model.features[i]),
             "importance": float(mean_abs[i])}
            for i in order
        ]
        features = list(model.features)

    churn.notes = notes
    progress(0.95, "Done")
    return AnalysisResult(
        df_scored=out, features=features, feature_labels=feature_labels, contributions=contributions,
        global_importance=global_imp, churn=churn, metrics=metrics, schema_map=smap, data=data,
        scoring_day=max_day, supervised=supervised, notes=notes,
    )
