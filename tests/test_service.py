import csv
import io

import pytest

from backend import service
from tests.conftest import LEGACY_DB


@pytest.fixture(scope="module")
def nav(sample_dir):
    return service.analyze({"source": str(sample_dir / "navigation_app.db"), "quick": True})


def test_detection_summary(sample_dir):
    out = service.validate_source(str(sample_dir / "learning_app.db"), display_name="my.db")
    s = out["summary"]
    assert out["source"] == "my.db"
    assert s["userTable"] == "learners" and s["userCount"] == 1200
    lessons = next(a for a in s["activities"] if a["table"] == "lessons")
    assert lessons["dateRange"][0] == "2025-01-01" and lessons["rows"] > 0


def test_full_result_shapes(nav):
    r = service.full_result(nav)
    o = r["overview"]
    assert o["activeUsers"] + o["alreadyChurned"] == len(r["users"])
    assert sum(t["count"] for t in o["tiers"]) == o["activeUsers"]
    assert 0 < o["expectedChurners"] < o["activeUsers"]
    assert o["drivers"] and abs(sum(d["share"] for d in o["drivers"])) <= 1.0
    assert o["quality"]["grade"] in {"Good", "Fair", "Weak"}

    users = r["users"]
    statuses = [u["status"] for u in users]
    # active users first (by risk), already-churned users last
    first_churned = statuses.index("churned") if "churned" in statuses else len(users)
    assert all(s == "churned" for s in statuses[first_churned:])
    risks = [u["risk"] for u in users[:first_churned]]
    assert risks == sorted(risks, reverse=True)
    assert all(0 < u["risk"] < 1 for u in users[:first_churned])  # never shown as exactly 0% or 100%
    assert all(u["topReason"] for u in users[:20])


def test_user_bundle_and_csv(nav):
    users = service.users_all(nav)
    b = service.user_bundle(nav, users[0]["userId"])
    assert b["factors"] and b["metrics"] and b["why"]
    assert len(b["series"]["activity"]) == 60
    assert set(b["series"]["measures"]) >= {"trips.distance_km"}
    rows = list(csv.reader(io.StringIO(service.export_csv(nav))))
    assert rows[0][0] == "user_id" and len(rows) == len(users) + 1
    with pytest.raises(KeyError):
        service.user_bundle(nav, "nobody")


def test_quick_mode_keeps_quality():
    an = service.analyze({"source": str(LEGACY_DB), "quick": True})
    assert an.res.metrics["rocAuc"] > 0.9
