"""
Build the static site in site/ (what gets deployed to Vercel).

    python -m web.build            # rebuild the Python bundle and the demo results
    python -m web.build --bundle   # only rebuild site/py/churn_backend.zip (fast)

- site/py/churn_backend.zip : the backend/ package, minus the FastAPI server, loaded by the in-browser engine
- site/demo/<app>.json      : precomputed results for the example apps shown on the start screen
"""
from __future__ import annotations

import argparse
import io
import json
import random
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SITE = REPO / "site"
BUNDLE = SITE / "py" / "churn_backend.zip"

# Modules the browser engine does not need (web server, request/response models, job queue).
EXCLUDE = {"backend/app.py", "backend/jobs", "backend/schemas"}
FIXED_TIME = (2024, 1, 1, 0, 0, 0)  # deterministic zip bytes so freshness can be tested

DEMO_DETAIL_TOP = 150
DEMO_DETAIL_OTHERS = 40


def bundle_files() -> list[Path]:
    files = []
    for p in sorted((REPO / "backend").rglob("*.py")):
        rel = p.relative_to(REPO).as_posix()
        if "__pycache__" in rel or any(rel == e or rel.startswith(e + "/") for e in EXCLUDE):
            continue
        files.append(p)
    return files


def build_bundle_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in bundle_files():
            info = zipfile.ZipInfo(p.relative_to(REPO).as_posix(), FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, p.read_bytes())
    return buf.getvalue()


def write_bundle() -> None:
    BUNDLE.parent.mkdir(parents=True, exist_ok=True)
    BUNDLE.write_bytes(build_bundle_bytes())
    print(f"wrote {BUNDLE.relative_to(REPO)} ({BUNDLE.stat().st_size / 1024:.0f} KB)")


def _demo_payload(db_path: Path, file_name: str) -> dict:
    from backend import service

    detection = service.validate_source(str(db_path), display_name=file_name)
    an = service.analyze({"source": str(db_path)})
    result = service.full_result(an)
    users = result["users"]
    active = [u["userId"] for u in users if u["status"] != "churned"]
    rng = random.Random(7)
    rest = active[DEMO_DETAIL_TOP:]
    churned = [u["userId"] for u in users if u["status"] == "churned"]
    ids = active[:DEMO_DETAIL_TOP] + rng.sample(rest, min(DEMO_DETAIL_OTHERS, len(rest))) + churned[:10]
    bundles = {uid: service.clean(service.user_bundle(an, uid)) for uid in ids}
    return {
        "fileName": file_name,
        "detection": {"summary": detection["summary"], "schemaMap": detection["schemaMap"]},
        "result": service.clean(result),
        "bundles": bundles,
    }


def write_demos(n_users: int = 1200) -> None:
    from simulator.generate import APPS

    out = SITE / "demo"
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        jobs = []
        for key, gen in APPS.items():
            gen(tmp, n_users=n_users)
            jobs.append((key, tmp / f"{key}.db", f"{key}.db"))
        jobs.append(("app_usage", REPO / "app_usage.db", "app_usage.db"))
        for key, path, name in jobs:
            payload = _demo_payload(path, name)
            dest = out / f"{key}.json"
            dest.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False))
            print(f"wrote {dest.relative_to(REPO)} ({dest.stat().st_size / 1024:.0f} KB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", action="store_true", help="only rebuild the Python bundle")
    args = ap.parse_args()
    write_bundle()
    if not args.bundle:
        write_demos()


if __name__ == "__main__":
    main()
