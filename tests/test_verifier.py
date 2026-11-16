"""Run the independent verifier inside pytest for every bundled scenario."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import verifier.verify as verify_module
from simulator.run_simulator import load_scenario, retransmit_conflict

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


class _ClientShim:
    """Adapts FastAPI's TestClient to the tiny subset of ``requests`` used."""

    def __init__(self, client):
        self.client = client

    def get(self, url, headers=None, params=None, timeout=None):
        path = url.split("testserver", 1)[1]
        return self.client.get(path, params=params)

    def post(self, url, **kwargs):
        path = url.split("testserver", 1)[1]
        return self.client.post(path, **kwargs)


def _run(client, **body):
    res = client.post("/runs", json={"wait": True, **body})
    assert res.status_code == 201, res.text
    return res.json()


def _verify(client, scenario: str, *, expect_revisions=False, expect_duplicates=False):
    truth = SAMPLES / scenario / "truth.json"
    report = truth.parent / "verification.json"
    original = verify_module.requests
    verify_module.requests = _ClientShim(client)
    try:
        code = verify_module.run_checks(
            scenario,
            truth,
            "http://testserver",
            "test-key",
            expect_revisions,
            expect_duplicates,
        )
    finally:
        verify_module.requests = original
    result = json.loads(report.read_text())
    report.unlink(missing_ok=True)  # never leave test artefacts in samples/
    return code, result


@pytest.fixture()
def scenario_truth_cleanup():
    written = []
    yield written
    for path in written:
        path.unlink(missing_ok=True)


def test_verifier_overlapping(client, api):
    load_scenario("overlapping", "http://testserver", truth_dir=SAMPLES / "overlapping")
    _run(client)
    code, result = _verify(client, "overlapping")
    assert code == 0, [c for c in result["checks"] if not c["ok"]]
    assert result["upload_summary"]["records_checked"] > 0


def test_verifier_reboot(client, api):
    load_scenario("reboot", "http://testserver", truth_dir=SAMPLES / "reboot")
    _run(client)
    code, result = _verify(client, "reboot")
    assert code == 0, [c for c in result["checks"] if not c["ok"]]


def test_verifier_collinear(client, api):
    load_scenario("collinear", "http://testserver", truth_dir=SAMPLES / "collinear")
    _run(client)
    code, result = _verify(client, "collinear")
    assert code == 0, [c for c in result["checks"] if not c["ok"]]


def test_verifier_duplicates(client, api):
    baseline = load_scenario(
        "duplicates", "http://testserver", truth_dir=SAMPLES / "duplicates"
    )
    retransmit_conflict(api, baseline["scn"], baseline["raw"])
    _run(client)
    # Truth lives in the repo; the verifier writes its report next to it.
    report = SAMPLES / "duplicates" / "verification.json"
    try:
        code, result = _verify(client, "duplicates", expect_duplicates=True)
    finally:
        report.unlink(missing_ok=True)
    assert code == 0, [c for c in result["checks"] if not c["ok"]]


def test_verifier_revision(client, api):
    load_scenario(
        "revision", "http://testserver", truth_dir=SAMPLES / "revision", phases=("early",)
    )
    _run(client)
    load_scenario(
        "revision", "http://testserver", truth_dir=SAMPLES / "revision", phases=("late",)
    )
    _run(client, mode="revision")
    report = SAMPLES / "revision" / "verification.json"
    try:
        code, result = _verify(client, "revision", expect_revisions=True)
    finally:
        report.unlink(missing_ok=True)
    assert code == 0, [c for c in result["checks"] if not c["ok"]]


def test_forward_model_is_independent_of_service_code():
    """The verifier must not import the service's physics or calibration."""
    source = (Path(verify_module.__file__)).read_text()
    for forbidden in ("from app", "import app"):
        assert forbidden not in source
