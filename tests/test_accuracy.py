"""Predictions are scored against hidden future activity that the pipeline never sees."""
import pytest

from backend.pipeline.run import AnalysisOptions, run_analysis
from simulator.evaluate import evaluate
from tests.conftest import LEGACY_DB


@pytest.mark.parametrize("app", ["learning_app", "navigation_app", "saas_app"])
def test_future_churn_is_predicted(sample_dir, app):
    r = evaluate(str(sample_dir / f"{app}.db"))
    assert r["supervised"]
    assert r["rocAuc"] >= 0.82, r
    assert r["liftTop10"] >= 3.0, r
    # The model must add value over a recency/frequency-only baseline.
    assert r["validationRocAuc"] >= r["baselineRocAuc"], r
    # Calibrated: average predicted probability close to the real churn rate.
    assert abs(r["meanPredicted"] - r["actualChurnRate"]) < 0.05, r


def test_legacy_database_with_explicit_labels():
    res = run_analysis(AnalysisOptions(source=str(LEGACY_DB)))
    assert res.churn.label_source == "explicit"
    assert res.metrics["rocAuc"] >= 0.9
    assert res.metrics["rocAuc"] > res.metrics["baselineRocAuc"]
    churned = res.df_scored[res.df_scored["knownLabel"] == 1]
    assert (churned["status"] == "churned").all()


def test_override_churn_window_and_inactivity_mode(sample_dir):
    res = run_analysis(AnalysisOptions(source=str(sample_dir / "navigation_app.db"), churn_window_days=21))
    assert res.churn.window_days == 21 and not res.churn.auto
    res = run_analysis(AnalysisOptions(source=str(LEGACY_DB), label_source="derived"))
    assert res.churn.label_source == "derived" and res.supervised


def test_heuristic_fallback_with_short_history(tmp_path):
    import sqlite3
    db = tmp_path / "tiny.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE events (user_id TEXT, event_time TEXT)")
    for u in range(30):
        for d in range(1, 1 + (u % 5) + 1):
            conn.execute("INSERT INTO events VALUES (?, ?)", (f"u{u}", f"2025-01-{d:02d} 12:00:00"))
    conn.commit()
    conn.close()
    res = run_analysis(AnalysisOptions(source=str(db)))
    assert not res.supervised
    assert res.df_scored["churnProbability"].between(0, 1).all()
