"""The static site (deployed to Vercel) must ship the current Python code and valid demo data."""
import json
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app import app
from web.build import BUNDLE, SITE, build_bundle_bytes


def test_python_bundle_is_up_to_date():
    assert BUNDLE.exists(), "run: python -m web.build --bundle"
    assert BUNDLE.read_bytes() == build_bundle_bytes(), "site/py/churn_backend.zip is stale: run python -m web.build --bundle"


def test_demo_payloads():
    for key in ("learning_app", "navigation_app", "saas_app", "app_usage"):
        d = json.loads((SITE / "demo" / f"{key}.json").read_text())
        assert d["result"]["users"] and d["result"]["overview"]["activeUsers"] > 0
        top = next(u for u in d["result"]["users"] if u["status"] != "churned")
        assert top["userId"] in d["bundles"]


def test_site_files_and_engine_switch():
    html = (SITE / "index.html").read_text()
    assert "worker.js" in html and "engine.json" in html
    assert json.loads((SITE / "engine.json").read_text())["engine"] == "browser"
    assert "cdn.jsdelivr.net/pyodide" in (SITE / "worker.js").read_text()
    vercel = json.loads((Path(__file__).resolve().parents[1] / "vercel.json").read_text())
    assert vercel["outputDirectory"] == "site"

    client = TestClient(app)
    assert client.get("/engine.json").json() == {"engine": "server"}
    assert client.get("/").status_code == 200 and "Churn Predictor" in client.get("/").text
