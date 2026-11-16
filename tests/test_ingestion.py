"""Ingestion behaviour: idempotency, conflicts, progress, local-time handling."""

from __future__ import annotations

STATION = {"id": "S1", "name": "S1", "x": 0.0, "y": 0.0, "hearing_radius_m": 2000.0}


def _record(seq: int, local: str = "2026-11-01T00:00:01.000000", **over):
    rec = {
        "source_seq": seq,
        "kind": "sound",
        "class_label": "croak",
        "local_capture_at": local,
        "features": [0.1, 0.2, 0.3],
        "uncertainty_ms": 20.0,
    }
    rec.update(over)
    return rec


def _batch(records, batch_key="b1", progress=None, boot_id="b1"):
    body = {
        "station_id": "S1",
        "boot_id": boot_id,
        "batch_key": batch_key,
        "records": records,
    }
    if progress is not None:
        body["progress"] = progress
    return body


def test_identical_record_is_not_stored_twice(client):
    client.post("/stations", json=STATION)
    first = client.post("/stations/S1/observations", json=_batch([_record(1)]))
    assert first.status_code == 201
    assert first.json()["counts"]["accepted"] == 1

    second = client.post(
        "/stations/S1/observations", json=_batch([_record(1)], batch_key="b2")
    )
    payload = second.json()
    assert payload["counts"]["accepted"] == 0
    assert payload["counts"]["duplicates"] == 1
    assert payload["duplicate_sequences"] == [1]
    obs = client.get("/observations").json()
    assert obs["total"] == 1


def test_batch_replay_is_a_noop(client):
    client.post("/stations", json=STATION)
    body = _batch([_record(1), _record(2, "2026-11-01T00:00:02.000000")])
    first = client.post("/stations/S1/observations", json=body).json()
    replay = client.post("/stations/S1/observations", json=body).json()
    assert replay["replayed"] is True
    assert replay["batch_id"] == first["batch_id"]
    assert replay["counts"]["accepted"] == 0
    assert sorted(replay["original_observation_ids"]) == sorted(first["accepted_observation_ids"])
    assert client.get("/observations").json()["total"] == 2


def test_conflicting_content_never_overwrites(client):
    client.post("/stations", json=STATION)
    client.post("/stations/S1/observations", json=_batch([_record(7, uncertainty_ms=20.0)]))
    conflict = client.post(
        "/stations/S1/observations",
        json=_batch([_record(7, uncertainty_ms=45.0)], batch_key="b2"),
    ).json()
    assert conflict["counts"]["conflicts"] == 1
    assert conflict["counts"]["accepted"] == 0

    obs = client.get("/observations").json()["observations"]
    assert len(obs) == 1
    assert obs[0]["uncertainty_ms"] == 20.0

    conflicts = client.get("/conflicts").json()
    assert len(conflicts) == 1
    assert conflicts[0]["source_seq"] == 7
    assert conflicts[0]["resolved"] is False


def test_same_sequence_in_one_batch_is_duplicate_or_conflict(client):
    client.post("/stations", json=STATION)
    payload = client.post(
        "/stations/S1/observations",
        json=_batch([_record(3), _record(3)]),
    ).json()
    assert payload["counts"]["accepted"] == 1
    assert payload["counts"]["duplicates"] == 1

    client.post(
        "/stations/S1/observations",
        json=_batch(
            [
                _record(9, uncertainty_ms=20.0),
                _record(9, uncertainty_ms=30.0),
            ],
            batch_key="b2",
        ),
    )
    conflicts = client.get("/conflicts").json()
    assert any(c["source_seq"] == 9 for c in conflicts)
    assert client.get("/observations").json()["total"] == 2


def test_sequence_identity_is_per_boot(client):
    client.post("/stations", json=STATION)
    client.post("/stations/S1/observations", json=_batch([_record(1)], boot_id="b1"))
    client.post(
        "/stations/S1/observations",
        json=_batch([_record(1, "2026-11-01T03:00:00.000000")], boot_id="b2", batch_key="b2"),
    )
    assert client.get("/observations").json()["total"] == 2


def test_local_time_with_zone_is_rejected(client):
    client.post("/stations", json=STATION)
    res = client.post(
        "/stations/S1/observations",
        json=_batch([_record(1, local="2026-11-01T00:00:01.000000Z")]),
    )
    assert res.status_code == 422


def test_unknown_station_is_404(client):
    body = _batch([_record(1)])
    body["station_id"] = "NOPE"
    res = client.post("/stations/NOPE/observations", json=body)
    assert res.status_code == 404


def test_progress_reports_are_stored(client):
    client.post("/stations", json=STATION)
    res = client.post(
        "/stations/S1/progress",
        json={
            "boot_id": "b1",
            "progress": {
                "complete_through_local_at": "2026-11-01T01:00:00.000000",
                "complete": True,
                "gaps": [],
            },
        },
    )
    assert res.status_code == 200
    reports = client.get("/stations/S1/progress").json()["reports"]
    assert len(reports) == 1
    assert reports[0]["complete"] is True
    assert reports[0]["declared_local_through"] == "2026-11-01T01:00:00.000000"


def test_calibration_record_requires_known_source(client):
    client.post("/stations", json=STATION)
    rec = _record(
        1,
        kind="calibration",
        class_label="calibration",
        calibration={
            "source_id": "NOPE",
            "emission_reference_at": "2026-11-14T00:00:00.000000Z",
        },
    )
    payload = client.post("/stations/S1/observations", json=_batch([rec])).json()
    assert payload["counts"]["accepted"] == 0
    assert payload["counts"]["rejected"][0]["reason"] == "unknown_calibration_source"


def test_sound_record_rejects_calibration_metadata(client):
    client.post("/stations", json=STATION)
    rec = _record(
        1,
        calibration={
            "source_id": "C1",
            "emission_reference_at": "2026-11-14T00:00:00.000000Z",
        },
    )
    res = client.post("/stations/S1/observations", json=_batch([rec]))
    assert res.status_code == 422


def test_api_key_is_required(client):
    from fastapi.testclient import TestClient

    from app.main import app

    bare = TestClient(app)
    assert bare.get("/stations").status_code == 401
    assert bare.get("/health").status_code == 200


def test_reused_batch_key_with_new_content_is_not_discarded(client):
    """A station that reuses a batch key must not lose an entire upload."""
    client.post("/stations", json=STATION)
    first = client.post(
        "/stations/S1/observations",
        json=_batch([_record(1)], batch_key="reused"),
    ).json()
    assert first["counts"]["accepted"] == 1

    second = client.post(
        "/stations/S1/observations",
        json=_batch([_record(2, "2026-11-01T00:00:05.000000")], batch_key="reused"),
    ).json()
    assert second["replayed"] is False
    assert second["counts"]["accepted"] == 1
    assert client.get("/observations").json()["total"] == 2

    # Resending the second content is a record-level duplicate: nothing new.
    client.post(
        "/stations/S1/observations",
        json=_batch([_record(2, "2026-11-01T00:00:05.000000")], batch_key="reused"),
    )
    assert client.get("/observations").json()["total"] == 2

    # Resending the *first* content under the reused key replays the original
    # batch verbatim and therefore stays a genuine no-op.
    replay = client.post(
        "/stations/S1/observations",
        json=_batch([_record(1)], batch_key="reused"),
    ).json()
    assert replay["replayed"] is True
    assert replay["counts"]["accepted"] == 0
    assert client.get("/observations").json()["total"] == 2
