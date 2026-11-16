"""Test fixtures: a clean database and an authenticated API client.

The tests run against the same PostgreSQL server as the development service but
in the ``wetland_test`` database, selected before the app is imported.
"""

from __future__ import annotations

import os

os.environ.setdefault(
    "DATABASE_URL", "postgresql+psycopg://ccdev:ccdev@127.0.0.1:5432/wetland_test"
)
os.environ.setdefault("WETLAND_API_KEY", "test-key")
os.environ.setdefault("WETLAND_ALLOW_RESET", "true")

import pytest  # noqa: E402

from app.db import Base, engine  # noqa: E402
from app.main import app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _schema():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield
    Base.metadata.drop_all(engine)


@pytest.fixture()
def client():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with TestClient(app, headers={"X-API-Key": "test-key"}) as c:
        yield c


@pytest.fixture()
def api(client, monkeypatch):
    import simulator.run_simulator as sim

    class TestApi(sim.Api):
        def __init__(self, base: str = "http://testserver"):
            super().__init__("http://testserver", api_key="test-key")

        def _req(self, method, path, **kwargs):
            response = client.request(method, path, **kwargs)
            if response.status_code >= 400:
                raise RuntimeError(f"{method} {path} -> {response.status_code}: {response.text[:800]}")
            return response.json() if response.content else None

    # load_scenario() instantiates Api(base) internally; redirect it.
    monkeypatch.setattr(sim, "Api", TestApi)
    return TestApi()
