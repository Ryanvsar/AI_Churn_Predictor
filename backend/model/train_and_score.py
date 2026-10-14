"""
XGBoost churn model with honest, time-aware validation.

- Training rows come from past snapshots (features as of a cutoff, label from what happened next).
- Validation is the most recent labelled snapshot (or a held-out user split when there is only one),
  so reported metrics measure prediction of the *future*, not fit to the past.
- A small hyper-parameter grid is searched with early stopping on that validation set.
- Probabilities are calibrated (Platt scaling) on the validation predictions.
- The final model is refit on all snapshots and scores users as of the latest date — rows it never
  saw during training.
- Per-user explanations are exact TreeSHAP values from XGBoost's built-in `pred_contribs`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class ModelConfig:
    threshold: float = 0.4
    high_risk_threshold: float = 0.7
    random_state: int = 42
    max_snapshots: int = 12
    quick: bool = False  # smaller search for slow environments (e.g. the in-browser engine)


PARAM_GRID: List[Dict] = [
    {"max_depth": 3, "learning_rate": 0.05, "min_child_weight": 5},
    {"max_depth": 4, "learning_rate": 0.05, "min_child_weight": 3},
    {"max_depth": 5, "learning_rate": 0.05, "min_child_weight": 10},
    {"max_depth": 6, "learning_rate": 0.03, "min_child_weight": 20},
    {"max_depth": 4, "learning_rate": 0.05, "min_child_weight": 5, "balance": True},
    {"max_depth": 3, "learning_rate": 0.1, "min_child_weight": 10, "balance": True},
]
QUICK_GRID: List[Dict] = PARAM_GRID[:2]


def risk_tier(prob: float, med_threshold: float, high_threshold: float) -> str:
    if prob >= high_threshold:
        return "High"
    if prob >= med_threshold:
        return "Medium"
    return "Low"


class Calibrator:
    """Maps raw model probabilities to calibrated ones (Platt scaling by default; isotonic optional)."""

    def __init__(self, method: str):
        self.method = method
        self._iso: Optional[IsotonicRegression] = None
        self._lr: Optional[LogisticRegression] = None

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, raw: np.ndarray, y: np.ndarray) -> "Calibrator":
        if self.method == "isotonic":
            self._iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw, y)
        else:
            self._lr = LogisticRegression(C=1e4).fit(self._logit(raw), y)
        return self

    def transform(self, raw: np.ndarray) -> np.ndarray:
        if self._iso is not None:
            return self._iso.predict(raw)
        return self._lr.predict_proba(self._logit(raw))[:, 1]

    @staticmethod
    def choose(y: np.ndarray) -> str:
        # Sigmoid (Platt) scaling is smooth, so it keeps every user's ranking distinct. Isotonic
        # regression is a step function that ties many users at the same probability, which makes a
        # risk leaderboard hard to act on, so it is only used on request.
        return "sigmoid"


def classification_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> Dict[str, float]:
    y = np.asarray(y).astype(int)
    out = {"n": int(len(y)), "positives": int(y.sum()), "baseRate": float(y.mean()) if len(y) else 0.0}
    if len(np.unique(y)) < 2:
        return out
    pred = (p >= threshold).astype(int)
    k = max(1, int(round(len(y) * 0.1)))
    top = np.argsort(-p)[:k]
    out.update({
        "rocAuc": float(roc_auc_score(y, p)),
        "prAuc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "liftTop10": float(y[top].mean() / max(y.mean(), 1e-9)),
    })
    return out


def _xgb(params: Dict, n_estimators: int, y: np.ndarray, seed: int, early_stopping: bool) -> xgb.XGBClassifier:
    p = dict(params)
    balance = p.pop("balance", False)
    pos = max(int(np.sum(y)), 1)
    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        tree_method="hist",
        eval_metric="logloss",
        early_stopping_rounds=50 if early_stopping else None,
        scale_pos_weight=(len(y) - pos) / pos if balance else 1.0,
        random_state=seed,
        n_jobs=-1,
        **p,
    )


def recency_baseline_auc(X_tr, y_tr, X_va, y_va, cols: List[str]) -> Optional[float]:
    """AUC of a simple logistic regression on recency/frequency, to show the model adds value."""
    cols = [c for c in cols if c in X_tr.columns]
    if not cols or len(np.unique(y_va)) < 2 or len(np.unique(y_tr)) < 2:
        return None
    f = lambda X: np.log1p(X[cols].fillna(0).clip(lower=0).to_numpy(dtype=float))  # noqa: E731
    lr = LogisticRegression(max_iter=1000).fit(f(X_tr), y_tr)
    return float(roc_auc_score(y_va, lr.predict_proba(f(X_va))[:, 1]))


@dataclass
class TrainedModel:
    booster: xgb.XGBClassifier
    calibrator: Calibrator
    features: List[str]
    params: Dict
    metrics: Dict = field(default_factory=dict)

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        return self.booster.predict_proba(X[self.features])[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # A behavioural model is never certain; keep probabilities off the 0%/100% extremes.
        return np.clip(self.calibrator.transform(self.predict_raw(X)), 0.005, 0.99)

    def contributions(self, X: pd.DataFrame) -> np.ndarray:
        """Exact TreeSHAP values in log-odds space; last column is the bias term."""
        dm = xgb.DMatrix(X[self.features], feature_names=self.features)
        return self.booster.get_booster().predict(dm, pred_contribs=True)


def train(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    features: List[str],
    config: ModelConfig,
    baseline_cols: List[str],
    validation_kind: str,
) -> TrainedModel:
    """
    `train_df` / `val_df` hold feature columns plus `label`. When `val_df` is None a stratified
    25% user split of `train_df` is used for validation.
    """
    if val_df is None:
        tr, va = train_test_split(train_df, test_size=0.25, stratify=train_df["label"], random_state=config.random_state)
    else:
        tr, va = train_df, val_df
    X_tr, y_tr = tr[features], tr["label"].to_numpy(dtype=int)
    X_va, y_va = va[features], va["label"].to_numpy(dtype=int)
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2:
        raise ValueError("Training or validation data contains only one class.")

    best = None
    for params in (QUICK_GRID if config.quick else PARAM_GRID):
        m = _xgb(params, 2000, y_tr, config.random_state, early_stopping=True)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        p = m.predict_proba(X_va)[:, 1]
        auc = roc_auc_score(y_va, p)
        if best is None or auc > best[0]:
            best = (auc, params, int(m.best_iteration) + 1, p)
    best_auc, best_params, best_iter, raw_va = best

    method = Calibrator.choose(y_va)
    calibrator = Calibrator(method).fit(raw_va, y_va)

    # Cross-fitted calibrated predictions so the reported Brier/F1 are not optimistic.
    rng = np.random.default_rng(config.random_state)
    fold = rng.integers(0, 2, len(y_va))
    cal_va = np.empty_like(raw_va)
    for k in (0, 1):
        a, b_ = fold == k, fold != k
        if len(np.unique(y_va[b_])) < 2:
            cal_va[a] = calibrator.transform(raw_va[a])
        else:
            cal_va[a] = Calibrator(Calibrator.choose(y_va[b_])).fit(raw_va[b_], y_va[b_]).transform(raw_va[a])

    metrics = classification_metrics(y_va, cal_va, config.threshold)
    metrics["rocAuc"] = float(best_auc)  # ranking quality of the raw model (calibration-independent)
    metrics["prAuc"] = float(average_precision_score(y_va, raw_va))
    metrics["baselineRocAuc"] = recency_baseline_auc(X_tr, y_tr, X_va, y_va, baseline_cols)
    metrics["validation"] = validation_kind
    metrics["trainRows"] = int(len(y_tr))
    metrics["calibration"] = method
    metrics["bestIteration"] = best_iter
    metrics["params"] = {k: v for k, v in best_params.items()}

    # Refit on everything (train + validation) so the newest behaviour is learned too.
    full = pd.concat([tr, va], ignore_index=True)
    y_full = full["label"].to_numpy(dtype=int)
    n_final = max(20, int(best_iter * (len(full) / max(len(tr), 1))))
    final = _xgb(best_params, n_final, y_full, config.random_state, early_stopping=False)
    final.fit(full[features], y_full, verbose=False)
    return TrainedModel(booster=final, calibrator=calibrator, features=features, params=best_params, metrics=metrics)


def heuristic_probability(recency: np.ndarray, mean_gap: np.ndarray, window_days: int) -> np.ndarray:
    """
    Fallback when there is not enough history to learn from: treat each user's activity as a
    Poisson process with their own mean gap g. P(no activity in the next H days) = exp(-H/g),
    combined with the chance they have already quietly left given their current silence r.
    """
    g = np.where(np.isfinite(mean_gap) & (mean_gap > 0), mean_gap, np.nanmedian(mean_gap) if np.isfinite(np.nanmedian(mean_gap)) else 7.0)
    g = np.maximum(g, 1.0)
    p_alive = np.exp(-np.maximum(recency, 0) / (2.0 * g))
    p_return = 1.0 - np.exp(-window_days / g)
    return np.clip(1.0 - p_alive * p_return, 0.0, 1.0)
