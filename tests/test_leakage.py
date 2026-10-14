import pandas as pd

from backend.connectors.source import open_source
from backend.data.extract import extract
from backend.features.feature_frame import build_features, make_spec
from backend.labels import churn_definition as cd
from backend.schema.detect import detect_schema
from backend.schema.introspect import profile_database


def test_features_ignore_rows_after_cutoff(sample_dir):
    src = open_source(str(sample_dir / "learning_app.db"))
    data = extract(src, detect_schema(src, profile_database(src)))
    days = cd.activity_days([af.df for af in data.activities.values()])
    cutoff = int(days["day"].max()) - 30
    spec = make_spec(data, 7, 28)
    full = build_features(data, cutoff, spec)

    # Drop everything after the cutoff and rebuild: features must be identical.
    for af in data.activities.values():
        af.df = af.df[cd.to_daynum(af.df["ts"]) <= cutoff].reset_index(drop=True)
    truncated = build_features(data, cutoff, spec)
    pd.testing.assert_frame_equal(full.sort_index(), truncated.sort_index())


def test_derived_labels_only_use_window_after_cutoff():
    days = pd.DataFrame({
        "user_id": ["a", "a", "b", "b", "c"],
        "day": [10, 12, 10, 25, 1],
    })
    lab = cd.derived_labels(days, cutoff=10, h=7).set_index("user_id")["label"]
    assert lab.to_dict() == {"a": 0, "b": 1}  # c was already inactive at the cutoff -> not eligible


def test_partial_last_day_detected():
    rows = [{"user_id": str(u), "day": d} for d in range(0, 60) for u in range(50)]
    rows += [{"user_id": "0", "day": 60}]  # a single spill-over event on the final day
    days = pd.DataFrame(rows)
    assert cd.complete_end_day(days) == 59


def test_churn_window_follows_usage_rhythm():
    import numpy as np
    rng = np.random.default_rng(0)
    daily = pd.DataFrame([{"user_id": u, "day": d} for u in range(200) for d in range(120) if rng.random() < 0.8])
    weekly = pd.DataFrame([{"user_id": u, "day": d} for u in range(200) for d in range(0, 120, 7)])
    assert cd.infer_window(daily, 0, 119)[0] <= 7
    assert cd.infer_window(weekly, 0, 119)[0] >= 7
    assert cd.infer_window(weekly, 0, 119)[0] > cd.infer_window(daily, 0, 119)[0]
