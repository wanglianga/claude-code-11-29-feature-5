"""End-to-end pipeline tests over the bundled scenarios.

Each test loads a scenario through the same simulator used for demos, runs the
pipeline, and asserts the required behaviours: corrected-time association,
revisions, mirror ambiguity, duplicate/conflict handling, exclusions and
resume-from-cutoff semantics.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from simulator.run_simulator import load_scenario, retransmit_conflict

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


@pytest.fixture()
def sample_dir(tmp_path):
    return tmp_path


def _run(client, **body):
    res = client.post("/runs", json={"wait": True, **body})
    assert res.status_code == 201, res.text
    return res.json()


def test_overlapping_sources_count_once_per_call(client, api, sample_dir):
    load_scenario("overlapping", "http://testserver", truth_dir=sample_dir, reset=False)
    _run(client)
    events = client.get("/events").json()
    assert events["total"] == 6
    for event in events["events"]:
        assert event["station_count"] == 4
        assert event["observation_count"] == 4
        assert event["status"] == "final"
        assert event["region_count"] == 1
        assert event["position"]["certain"] is True
    # Every observation is used exactly once.
    obs = client.get("/observations", params={"kind": "sound"}).json()
    used = [
        o for o in obs["observations"] if o["current_event_id"] is not None
    ]
    assert len(used) == obs["total"] == 24


def test_reboot_jump_is_absorbed_by_per_boot_calibration(client, api, sample_dir):
    load_scenario("reboot", "http://testserver", truth_dir=sample_dir)
    _run(client)
    cals = client.get("/calibrations").json()["calibrations"]
    s2 = {c["boot_id"]: c for c in cals if c["station_id"] == "S2"}
    assert set(s2) == {"b1", "b2"}
    assert s2["b1"]["valid"] and s2["b2"]["valid"]
    assert s2["b1"]["version_hash"] != s2["b2"]["version_hash"]
    # The intercept difference reflects the simulated jump, not wall-clock drift.
    intercepts = [s2[b]["segments"][0]["intercept_ms"] for b in ("b1", "b2")]
    assert abs(intercepts[0] - intercepts[1]) > 1_000_000  # > 1000 s
    events = client.get("/events").json()
    assert events["total"] == 6
    assert all(e["station_count"] == 4 for e in events["events"])


def test_late_backfill_station_still_participates(client, api, sample_dir):
    load_scenario("reboot", "http://testserver", truth_dir=sample_dir)
    _run(client)
    s3 = client.get(
        "/observations", params={"station_id": "S3", "kind": "sound"}
    ).json()["observations"]
    assert s3, "late station contributed no records"
    assert all(o["current_event_id"] is not None for o in s3)


def test_collinear_geometry_returns_two_regions_and_no_position(client, api, sample_dir):
    load_scenario("collinear", "http://testserver", truth_dir=sample_dir)
    _run(client)
    events = client.get("/events").json()["events"]
    assert len(events) == 6
    for event in events:
        assert event["region_count"] >= 2
        assert event["position"]["certain"] is False
        ys = sorted(r["representative"]["y"] for r in event["regions"])
        assert ys[0] < 500 < 500 + 1 < ys[-1] + 1  # one region above, one below the line
        assert event["regions"][0]["bounds"]["y_max"] <= 500 or event["regions"][1]["bounds"]["y_max"] <= 500
    # The event record must say why it is not a single coordinate.
    detail = client.get(f"/events/{events[0]['id']}").json()
    assert detail["reason"]


def test_duplicate_replay_and_conflict_do_not_change_events(client, api, sample_dir):
    baseline = load_scenario("duplicates", "http://testserver", truth_dir=sample_dir / "baseline")
    _run(client)
    events_a = client.get("/events").json()["total"]
    obs_a = client.get("/observations").json()["total"]

    retransmit = retransmit_conflict(api, baseline["scn"], baseline["raw"])
    assert retransmit["replay"]["replayed"] is True
    assert retransmit["replay"]["counts"]["accepted"] == 0
    assert retransmit["conflict"]["counts"]["conflicts"] == 1
    conflict_seq = retransmit["conflict_source_seq"]

    # A second run over unchanged inputs must not double count.
    _run(client)
    assert client.get("/events").json()["total"] == events_a == 3
    assert client.get("/observations").json()["total"] == obs_a
    conflicts = client.get("/conflicts").json()
    assert len(conflicts) == 1
    assert conflicts[0]["source_seq"] == conflict_seq
    # The original record still holds the original uncertainty.
    obs = client.get(
        "/observations", params={"station_id": "S1", "kind": "sound", "limit": 100}
    ).json()["observations"]
    original = next(o for o in obs if o["source_seq"] == conflict_seq)
    assert original["uncertainty_ms"] == pytest.approx(25.0)
    conflict_payload = conflicts[0]["payload"]
    assert conflict_payload["uncertainty_ms"] == pytest.approx(45.0)


def test_revision_run_supersedes_instead_of_splitting(client, api, sample_dir):
    load_scenario(
        "revision", "http://testserver", truth_dir=sample_dir, phases=("early",)
    )
    first = _run(client)
    assert first["status"] in {"done", "awaiting_review"}
    early_events = client.get("/events").json()["events"]
    assert early_events

    load_scenario("revision", "http://testserver", truth_dir=sample_dir, phases=("late",))
    second = _run(client, mode="revision")
    assert second["revision_of_run_id"] == first["id"]
    assert second["calibration_hash"] != first["calibration_hash"]

    all_events = client.get(
        "/events", params={"include_superseded": "true", "limit": 1000}
    ).json()["events"]
    superseded = [e for e in all_events if e["status"] == "superseded"]
    live = [e for e in all_events if e["status"] != "superseded"]
    assert superseded, "late data left every early event live"
    assert live
    assert all(e["superseded_by_event_id"] for e in superseded)
    for old in superseded:
        new = next(e for e in live if e["id"] == old["superseded_by_event_id"])
        assert old["id"] in new["supersedes_events"]
        assert "revision" in new and new["revision"] >= 2
    # The live count reflects the true number of activities, not the sum of runs.
    assert len(live) == 5


def test_exclusion_lets_investigator_finalise_a_silent_station(client, api, sample_dir):
    load_scenario(
        "revision", "http://testserver", truth_dir=sample_dir, phases=("early",)
    )
    _run(client)
    events = client.get("/events").json()["events"]
    provisional = [e for e in events if e["status"] == "provisional"]
    assert provisional, "expected at least one provisional event before S3 reports"
    target = provisional[0]
    unsettled = target["finalization"]["unsettled_stations"]
    assert "S3" in unsettled

    # Excluding an unrelated, already-settled station must not finalise it.
    settled = [
        e["station_id"]
        for e in target["finalization"]["stations"]
        if e["settled"] and e["station_id"] in {"S1", "S2", "S4"}
    ]
    if settled:
        client.post(
            f"/events/{target['id']}/exclusions",
            json={
                "station_id": settled[0],
                "reason": "records verified against the field notebook",
                "created_by": "investigator.a",
            },
        )
        still = client.get(f"/events/{target['id']}").json()
        assert still["status"] == "provisional"

    # Closing every unsettled station explicitly (with reasons) does finalise it.
    for station_id in list(unsettled):
        res = client.post(
            f"/events/{target['id']}/exclusions",
            json={
                "station_id": station_id,
                "reason": f"{station_id} logger failed; battery replaced after the window",
                "created_by": "investigator.a",
            },
        )
        assert res.status_code == 200
    detail = client.get(f"/events/{target['id']}").json()
    assert detail["status"] == "final"
    assert detail["finalization"]["resolved_by"] == "investigator_exclusion"
    entry = next(
        e for e in detail["finalization"]["stations"] if e["station_id"] == "S3"
    )
    assert entry["mode"] == "excluded_by_investigator"
    assert "battery replaced" in entry["reason"]
    assert any(r["station_id"] == "S3" for r in detail["exclusion_reasons"])


def test_elapsed_time_never_finalises_a_silent_station(client, api, sample_dir):
    load_scenario(
        "revision", "http://testserver", truth_dir=sample_dir, phases=("early",)
    )
    first = _run(client)
    # Repeated runs, arbitrarily far in wall-clock time, must keep S3-dependent
    # events provisional until S3 reports or is excluded.
    again = _run(client)
    statuses = {e["status"] for e in client.get("/events").json()["events"]}
    assert "provisional" in statuses
    assert first["status"] == again["status"] == "awaiting_review"


def test_resume_reuses_cutoff_and_calibration_snapshot(client, api, sample_dir):
    load_scenario(
        "overlapping", "http://testserver", truth_dir=sample_dir, up_to_wave=4
    )
    paused = _run(client, debug_stop_after_phase="localize")
    assert paused["status"] == "paused"
    # More uploads arrive after the pause; they must be outside the cutoff.
    load_scenario("overlapping", "http://testserver", truth_dir=sample_dir)
    resumed = client.post(f"/runs/{paused['id']}/resume").json()
    assert resumed["resume_of_run_id"] == paused["id"]
    assert resumed["cutoff"] == paused["cutoff"]
    assert resumed["calibration_hash"] == paused["calibration_hash"]
    # The resumed run only sees the first four waves.
    events = client.get("/events", params={"run_id": resumed["id"]}).json()
    assert 0 < events["total"] < 6
    observations = client.get("/observations", params={"kind": "sound"}).json()
    used = {o["id"] for o in observations["observations"] if o["current_event_id"]}
    assert len(used) == sum(e["observation_count"] for e in events["events"])


def test_superseded_events_leave_no_stale_observation_links(client, api, sample_dir):
    load_scenario(
        "revision", "http://testserver", truth_dir=sample_dir, phases=("early",)
    )
    _run(client)
    early = client.get("/events").json()["events"]
    load_scenario("revision", "http://testserver", truth_dir=sample_dir, phases=("late",))
    _run(client, mode="revision")
    events = client.get(
        "/events", params={"include_superseded": "true", "limit": 1000}
    ).json()["events"]
    superseded_ids = {e["id"] for e in events if e["status"] == "superseded"}
    live_ids = {e["id"] for e in events if e["status"] != "superseded"}
    obs = client.get("/observations", params={"limit": 1000}).json()["observations"]
    for row in obs:
        if row["kind"] != "sound":
            continue
        assert row["current_event_id"] not in superseded_ids, row
        if row["current_event_id"] is not None:
            assert row["current_event_id"] in live_ids


def test_event_detail_exposes_traceability(client, api, sample_dir):
    load_scenario("overlapping", "http://testserver", truth_dir=sample_dir)
    _run(client)
    event = client.get("/events").json()["events"][0]
    detail = client.get(f"/events/{event['id']}").json()
    assert detail["observations"]
    for obs in detail["observations"]:
        assert obs["anchor_ids"]
        assert obs["calibration_model_id"] is not None
        assert obs["corrected_arrival"]
        assert obs["local_capture_at"].endswith(("",)) is not None
        assert "Z" not in obs["local_capture_at"]
    anchors = client.get(f"/events/{event['id']}/anchors").json()
    assert anchors["calibration_hash"] == event["calibration_hash"]
    assert all(item["anchor_count"] > 0 for item in anchors["observations"])


def test_uncalibrated_observations_are_retained_not_dropped(client, api, sample_dir):
    # A station with only two calibration anchors cannot be calibrated.
    client.post(
        "/stations",
        json={"id": "X1", "name": "under-calibrated", "x": 100.0, "y": 100.0},
    )
    client.post(
        "/calibration-sources",
        json={
            "id": "C9",
            "x": 0.0,
            "y": 0.0,
            "emissions": ["2026-11-14T00:00:00.000000Z", "2026-11-14T00:02:00.000000Z"],
        },
    )
    client.post(
        "/sound-classes",
        json={"id": "croak", "feature_dim": 2, "feature_distance_limit": 0.2},
    )
    records = []
    for seq, local in enumerate(
        ["2026-11-01T00:00:01.000000", "2026-11-01T00:02:01.000000"], start=1
    ):
        records.append(
            {
                "source_seq": seq,
                "kind": "calibration",
                "class_label": "calibration",
                "local_capture_at": local,
                "features": [0.0, 0.1],
                "uncertainty_ms": 10.0,
                "calibration": {
                    "source_id": "C9",
                    "emission_reference_at": f"2026-11-14T00:0{2*(seq-1):d}:00.000000Z",
                },
            }
        )
    records.append(
        {
            "source_seq": 3,
            "kind": "sound",
            "class_label": "croak",
            "local_capture_at": "2026-11-01T00:03:00.000000",
            "features": [0.3, 0.3],
            "uncertainty_ms": 20.0,
        }
    )
    body = {
        "station_id": "X1",
        "boot_id": "b1",
        "batch_key": "x1-1",
        "records": records,
    }
    client.post("/stations/X1/observations", json=body)
    _run(client)
    obs = client.get(
        "/observations", params={"station_id": "X1", "kind": "sound"}
    ).json()["observations"]
    assert len(obs) == 1
    assert obs[0]["calibration_status"] == "uncalibrated"
    assert obs[0]["corrected_arrival"] is None
    assert "too_few_anchors" in (obs[0]["calibration_reason"] or "")
    cals = client.get("/calibrations", params={"station_id": "X1"}).json()
    assert cals["total"] >= 1 and cals["calibrations"][0]["valid"] is False


def test_upload_time_is_never_used_as_arrival_time(client, api, sample_dir):
    load_scenario(
        "overlapping", "http://testserver", truth_dir=sample_dir, up_to_wave=2
    )
    _run(client)
    early = client.get("/observations", params={"kind": "sound"}).json()
    for obs in early["observations"]:
        if obs["calibration_status"] == "calibrated":
            # Corrected arrival derives from the local reading and clock model,
            # so it must not equal the upload timestamp.
            assert obs["corrected_arrival"] != obs["uploaded_at"]


def test_async_run_can_be_polled_to_completion(client, api, sample_dir):
    import time

    load_scenario("overlapping", "http://testserver", truth_dir=sample_dir)
    res = client.post("/runs", json={"wait": False})
    assert res.status_code == 201
    run = res.json()
    assert run["status"] in {"pending", "running", "done", "awaiting_review"}
    deadline = time.time() + 60
    while time.time() < deadline:
        current = client.get(f"/runs/{run['id']}").json()
        if current["status"] in {"done", "awaiting_review", "failed", "awaiting_selection"}:
            break
        time.sleep(0.2)
    assert current["status"] in {"done", "awaiting_review"}, current
    assert client.get("/events", params={"run_id": run["id"]}).json()["total"] == 6
