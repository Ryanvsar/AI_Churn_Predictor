"""
Score predictions against what really happened next.

Runs the full pipeline on `<app>.db`, then reads `<app>_future.db` (activity after the observed
period, same schema) and checks which active users really went silent for the churn window.

Usage:  python -m simulator.evaluate sample_data/learning_app.db [sample_data/navigation_app.db ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from backend.connectors.source import open_source
from backend.data.extract import extract
from backend.labels.churn_definition import activity_days
from backend.pipeline.run import AnalysisOptions, run_analysis


def evaluate(db_path: str, **opts) -> dict:
    res = run_analysis(AnalysisOptions(source=db_path, **opts))
    future_path = str(Path(db_path).with_name(Path(db_path).stem + "_future.db"))
    fut = extract(open_source(future_path), res.schema_map, use_label=False)
    seen = extract(open_source(db_path), res.schema_map, use_label=False)  # events spilling past the cutoff
    fdays = activity_days([af.df for af in [*fut.activities.values(), *seen.activities.values()]])
    h = res.churn.window_days
    returned = set(fdays[(fdays["day"] > res.scoring_day) & (fdays["day"] <= res.scoring_day + h)]["user_id"])

    act = res.df_scored[res.df_scored["status"] == "active"]
    y = (~act["userId"].isin(returned)).astype(int).to_numpy()
    p = act["churnProbability"].to_numpy()
    k = max(1, int(len(y) * 0.1))
    return {
        "db": db_path,
        "windowDays": h,
        "activeUsers": int(len(y)),
        "actualChurnRate": float(y.mean()),
        "meanPredicted": float(p.mean()),
        "rocAuc": float(roc_auc_score(y, p)),
        "prAuc": float(average_precision_score(y, p)),
        "liftTop10": float(y[np.argsort(-p)[:k]].mean() / max(y.mean(), 1e-9)),
        "validationRocAuc": res.metrics.get("rocAuc"),
        "baselineRocAuc": res.metrics.get("baselineRocAuc"),
        "supervised": res.supervised,
    }


if __name__ == "__main__":
    for path in sys.argv[1:]:
        r = evaluate(path)
        print(", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
