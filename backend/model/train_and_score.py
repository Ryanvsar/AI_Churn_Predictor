from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from backend.features.feature_frame import FeatureConfig, build_user_feature_frame, default_feature_columns


@dataclass(frozen=True)
class ModelConfig:
    threshold: float = 0.4
    high_risk_threshold: float = 0.7
    half_life_days: float = 14.0
    dropoff_recent_days: int = 14
    rolling_windows_days: Tuple[int, ...] = (7, 14, 30)
    core_features: Tuple[str, ...] = ()
    random_state: int = 42


def _load_labels_if_present(conn: sqlite3.Connection) -> pd.DataFrame:
    tables = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='users';",
        conn,
    )
    if tables.empty:
        return pd.DataFrame(columns=["userId", "churnLabel"])

    cols = pd.read_sql_query("PRAGMA table_info(users);", conn)
    if cols.empty or "name" not in cols.columns:
        return pd.DataFrame(columns=["userId", "churnLabel"])
    if "churnLabel" not in set(cols["name"].tolist()):
        return pd.DataFrame(columns=["userId", "churnLabel"])

    return pd.read_sql_query("SELECT userId, churnLabel FROM users;", conn)


def _risk_tier(prob: float, med_threshold: float, high_threshold: float) -> str:
    if prob >= high_threshold:
        return "High"
    if prob >= med_threshold:
        return "Medium"
    return "Low"


def train_and_score(
    db_path: str,
    observation_start: Optional[int],
    observation_end: Optional[int],
    config: ModelConfig,
) -> Dict:
    feat_cfg = FeatureConfig(
        half_life_days=config.half_life_days,
        dropoff_recent_days=config.dropoff_recent_days,
        rolling_windows_days=config.rolling_windows_days,
        core_features=config.core_features,
    )

    df = build_user_feature_frame(
        db_path=db_path,
        observation_start=observation_start,
        observation_end=observation_end,
        config=feat_cfg,
    )
    if df.empty:
        return {
            "df_scored": pd.DataFrame(columns=["userId", "churnProbability", "riskTier"]),
            "features": [],
            "supervised": False,
            "model": None,
            "scaler": None,
            "label_stats": {},
        }

    # Choose feature columns based on what exists in the DF.
    candidate = default_feature_columns(include_optional=True)
    features = [c for c in candidate if c in df.columns]

    X = df[features].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Add small noise to reduce overfitting to synthetic patterns (kept from original script).
    np.random.seed(config.random_state)
    X_noisy = X.to_numpy(dtype=float) + np.random.normal(0, 0.02, X.shape)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_noisy)

    conn = sqlite3.connect(db_path)
    try:
        labels = _load_labels_if_present(conn)
    finally:
        conn.close()

    supervised = False
    model = None
    calibrated = None
    y = None
    label_stats: Dict[str, float] = {}

    if not labels.empty:
        df_l = df.merge(labels, on="userId", how="left")
        if df_l["churnLabel"].notna().any():
            y = df_l["churnLabel"].fillna(0).astype(int)
            if y.nunique() >= 2:
                supervised = True
                neg, pos = np.bincount(y)
                scale_pos_weight = float(neg) / float(max(pos, 1))

                # Explanation model (TreeExplainer works on this).
                model = XGBClassifier(
                    n_estimators=600,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    scale_pos_weight=scale_pos_weight,
                    eval_metric="auc",
                    random_state=config.random_state,
                    n_jobs=-1,
                )
                model.fit(X_scaled, y)

                # Calibrated model for probabilities (reduces extreme 0.999... outputs).
                # In sklearn>=1.8, cv='prefit' is not supported; use CV calibration instead.
                calibrated = CalibratedClassifierCV(
                    estimator=XGBClassifier(
                        n_estimators=600,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        scale_pos_weight=scale_pos_weight,
                        eval_metric="auc",
                        random_state=config.random_state,
                        n_jobs=-1,
                    ),
                    method="sigmoid",
                    cv=3,
                )
                calibrated.fit(X_scaled, y)

                label_stats = {
                    "pos": float(pos),
                    "neg": float(neg),
                    "scale_pos_weight": float(scale_pos_weight),
                }

    if supervised and model is not None:
        if calibrated is not None:
            probs = calibrated.predict_proba(X_scaled)[:, 1].astype(float)
        else:
            probs = model.predict_proba(X_scaled)[:, 1].astype(float)
    else:
        # Fallback risk score (unsupervised-ish): a simple weighted blend of “bad” signals.
        # This is a temporary mode to keep the UI functional when churn labels are absent.
        z = (X - X.mean()) / (X.std() + 1e-6)
        score = (
            0.40 * z.get("recency_days", 0.0)
            + 0.25 * (-z.get("sessions_per_day", 0.0))
            + 0.20 * (-z.get("decay_session_frequency", 0.0))
            + 0.15 * (-z.get("engagement_rate", 0.0))
        )
        s = score.to_numpy(dtype=float)
        # Map to (0,1) by sigmoid
        probs = (1.0 / (1.0 + np.exp(-s))).astype(float)

    df_scored = df[["userId"]].copy()
    df_scored["churnProbability"] = probs
    df_scored["riskTier"] = df_scored["churnProbability"].apply(
        lambda p: _risk_tier(float(p), config.threshold, config.high_risk_threshold)
    )

    # Keep metrics for UI and later SHAP explanations.
    df_scored = df_scored.merge(df.drop(columns=[]), on="userId", how="left")

    return {
        "df_scored": df_scored,
        "features": features,
        "supervised": supervised,
        "model": model,
        "calibrated_model": calibrated,
        "scaler": scaler,
        "label_stats": label_stats,
    }

