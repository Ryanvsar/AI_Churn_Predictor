import time
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app import MAPPINGS_DIR, UPLOAD_DIR, app
from tests.conftest import LEGACY_DB

client = TestClient(app)


def wait(job_id):
    for _ in range(600):
        s = client.get(f"/analysis/jobs/{job_id}").json()
        if s["status"] in ("succeeded", "failed"):
            return s
        time.sleep(0.2)
    raise AssertionError("job did not finish")


def test_full_flow_on_legacy_db():
    v = client.post("/db/validate", json={"dbPath": str(LEGACY_DB)})
    assert v.status_code == 200
    body = v.json()
    assert body["capabilities"]["hasSessionsTable"] and body["capabilities"]["hasChurnLabels"]

    job = client.post("/analysis/run", json={"dbPath": str(LEGACY_DB), "threshold": 0.4}).json()["jobId"]
    s = wait(job)
    assert s["status"] == "succeeded", s["error"]
    assert s["summary"]["supervised"] and s["summary"]["metrics"]["rocAuc"] > 0.9

    users = client.get(f"/analysis/{job}/users?top=5&includeChurned=false").json()
    assert len(users["users"]) == 5
    probs = [u["churnProbability"] for u in users["users"]]
    assert probs == sorted(probs, reverse=True)
    uid = users["users"][0]["userId"]

    d = client.get(f"/analysis/{job}/users/{uid}").json()
    assert d["userId"] == uid and "recency_days" in d["metrics"]
    e = client.get(f"/analysis/{job}/users/{uid}/explain").json()
    assert e["topFactors"] and e["whyRiskySummary"]
    ts = client.get(f"/analysis/{job}/users/{uid}/timeseries").json()
    assert ts["measureNames"] == ["sessions.sessionLength", "sessions.sessionEvents"] and ts["points"]
    agg = client.get(f"/analysis/{job}/aggregates").json()
    assert {g["name"] for g in agg["groups"]} >= {"AllUsers", "HighRisk", "MediumRisk", "LowRisk"}
    assert agg["globalFeatureImportance"]
    model = client.get(f"/analysis/{job}/model").json()
    assert model["churnDefinition"]["label_source"] == "explicit"
    assert client.get(f"/analysis/{job}/users/does-not-exist").status_code == 404


def test_mapping_override_and_string_ids(sample_dir):
    v = client.post("/db/validate", json={"source": str(sample_dir / "navigation_app.db")}).json()
    smap = v["schemaMap"]
    smap["activities"] = [a for a in smap["activities"] if a["table"] == "trips"]  # user edit: trips only
    saved = client.post("/mappings", json={"name": "test_nav", "schemaMap": smap})
    assert saved.status_code == 200
    loaded = client.get("/mappings/test_nav").json()["schemaMap"]
    job = client.post("/analysis/run", json={"source": str(sample_dir / "navigation_app.db"),
                                             "schemaMap": loaded, "churnWindowDays": 14}).json()["jobId"]
    s = wait(job)
    assert s["status"] == "succeeded", s["error"]
    assert s["summary"]["churnDefinition"]["window_days"] == 14
    uid = client.get(f"/analysis/{job}/users?top=1").json()["users"][0]["userId"]
    assert len(uid) == 36  # UUID user ids survive end to end
    assert client.get(f"/analysis/{job}/users/{uid}/explain").status_code == 200
    (MAPPINGS_DIR / "test_nav.json").unlink(missing_ok=True)


def test_upload_and_bad_sources(tmp_path):
    with open(LEGACY_DB, "rb") as f:
        up = client.post("/db/upload", files={"file": ("demo.db", f, "application/octet-stream")})
    assert up.status_code == 200
    path = up.json()["path"]
    assert path.startswith(str(UPLOAD_DIR))
    assert client.post("/db/validate", json={"source": path}).status_code == 200
    Path(path).unlink()
    assert client.post("/db/validate", json={"source": "/no/such/file.db"}).status_code == 400
    r = client.post("/db/validate", json={"source": "postgresql://u:p@127.0.0.1:1/x"})
    assert r.status_code == 400
    assert client.post("/db/upload", files={"file": ("x.exe", b"MZ", "application/octet-stream")}).status_code == 400
