from pathlib import Path

import pytest

from simulator.generate import APPS

REPO = Path(__file__).resolve().parents[1]
LEGACY_DB = REPO / "app_usage.db"


@pytest.fixture(scope="session")
def sample_dir(tmp_path_factory) -> Path:
    """Three schema-diverse simulated app databases (+ hidden future data), generated once."""
    out = tmp_path_factory.mktemp("sample_data")
    for gen in APPS.values():
        gen(out, n_users=1200)
    return out
