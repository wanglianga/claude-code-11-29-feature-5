"""Contract tests: the served API and the checked-in OpenAPI document."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_openapi_document_is_valid_and_matches_app(client):
    res = client.get("/openapi.json")
    assert res.status_code == 200
    spec = res.json()
    assert spec["openapi"].startswith("3.")
    assert spec["info"]["title"]
    # Every operation must be documented with at least a summary.
    for path, ops in spec["paths"].items():
        for method, op in ops.items():
            assert op.get("summary"), f"{method.upper()} {path} has no summary"
    # Protected paths declare the API-key scheme; the probe does not.
    assert "security" in spec["paths"]["/stations"]["get"]
    assert "security" not in spec["paths"]["/health"]["get"]


def test_checked_in_openapi_is_current(client):
    from app.main import app

    app.openapi_schema = None
    generated = app.openapi()
    on_disk = json.loads((ROOT / "openapi.json").read_text())
    assert generated["paths"].keys() == on_disk["paths"].keys()
    for path, ops in generated["paths"].items():
        assert ops.keys() == on_disk["paths"][path].keys(), path


def test_swagger_ui_is_served(client):
    for path in ("/docs", "/redoc"):
        res = client.get(path)
        assert res.status_code == 200
        assert b"swagger" in res.content.lower() or b"redoc" in res.content.lower()


def test_health_reports_database(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["database"] == "ok"
    assert body["version"]


def test_stats_shape(client):
    body = client.get("/stats").json()
    for key in ("stations", "observations", "uncalibrated_observations", "conflicts"):
        assert key in body
    assert isinstance(body["runs"], dict)
