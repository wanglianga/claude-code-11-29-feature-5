"""Tests for the comparability-analysis product.

Covers the interval algebra, coverage derivation, the two comparison methods,
the unknown-never-zero rule, report versioning (new cutoff on late evidence),
exact re-derivation of old reports, and the two end-to-end scenarios
(day/night bias + overlapping stations + empty stratum; night-time failure
reported late).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app import analysis
from app.config import settings
from simulator import analysis_demo

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


# --------------------------------------------------------------------------- #
# Interval algebra
# --------------------------------------------------------------------------- #
def test_union_merges_overlaps():
    assert analysis._union([(1, 2), (2, 3), (5, 6), (0, 1)]) == [(0, 3), (5, 6)]
    assert analysis._union([]) == []


def test_subtract_carves_holes():
    assert analysis._subtract([(0, 10)], [(2, 4), (6, 7)]) == [(0, 2), (4, 6), (7, 10)]
    assert analysis._subtract([(0, 10)], [(0, 10)]) == []
    assert analysis._subtract([(0, 10)], [(-5, 20)]) == []


def test_measure_intersection():
    assert analysis._measure([(0, 10), (20, 30)], 5, 25) == 10.0
    assert analysis._measure([], 0, 10) == 0.0


# --------------------------------------------------------------------------- #
# Period bins
# --------------------------------------------------------------------------- #
def _cfg(**kw):
    base = dict(
        region_a_id="a",
        region_b_id="b",
        habitat_map_id="h",
        reference_composition_id=None,
        class_label=None,
        window_start_us=0,
        window_end_us=4 * 43200 * 1_000_000,
        grid_cell_m=250.0,
        bin_seconds=43200,
        day_start_hour=6,
        night_start_hour=18,
    )
    base.update(kw)
    return analysis.ReportConfig(**base)


def test_bins_require_exact_multiple():
    cfg = _cfg(window_end_us=100_000_000)
    with pytest.raises(ValueError, match="exact multiple"):
        analysis.make_bins(cfg)


def test_bins_classify_day_and_night():
    # 2026-11-14 06:00 UTC + 4 x 12 h bins -> day, night, day, night
    from app.timeutil import to_us

    start = to_us("2026-11-14T06:00:00Z")
    cfg = _cfg(window_start_us=start, window_end_us=start + 4 * 43200 * 1_000_000)
    bins = analysis.make_bins(cfg)
    assert [b.period_class for b in bins] == ["day", "night", "day", "night"]


# --------------------------------------------------------------------------- #
# Definition endpoints & validation
# --------------------------------------------------------------------------- #
def _publish_defs(client, habitat_cells=None):
    client.post(
        "/analysis/regions",
        json={"id": "reg-a", "name": "A", "bbox": {"ix_min": 0, "ix_max": 1, "iy_min": 0, "iy_max": 1}},
    )
    client.post(
        "/analysis/regions",
        json={"id": "reg-b", "name": "B", "cells": [[2, 0], [2, 1], [3, 0], [3, 1]]},
    )
    cells = habitat_cells or [
        {"ix": ix, "iy": iy, "habitat": "reed"} for ix in range(4) for iy in range(2)
    ]
    client.post("/analysis/habitat-maps", json={"id": "hab", "cells": cells})
    client.post(
        "/analysis/reference-compositions",
        json={
            "id": "ref",
            "name": "ref",
            "entries": [
                {"habitat": "reed", "period_class": "day", "weight": 0.5},
                {"habitat": "reed", "period_class": "night", "weight": 0.5},
            ],
        },
    )


def test_region_create_list_get(client):
    res = client.post(
        "/analysis/regions",
        json={"id": "r1", "name": "R1", "bbox": {"ix_min": 0, "ix_max": 1, "iy_min": 2, "iy_max": 3}},
    )
    assert res.status_code == 201, res.text
    body = res.json()
    assert body["cell_count"] == 4
    assert [0, 2] in body["cells"] and [1, 3] in body["cells"]
    assert client.post(
        "/analysis/regions",
        json={"id": "r1", "name": "R1", "bbox": {"ix_min": 0, "ix_max": 0, "iy_min": 0, "iy_max": 0}},
    ).status_code == 409
    assert client.get("/analysis/regions/r1").json()["id"] == "r1"
    assert client.get("/analysis/regions/nope").status_code == 404
    assert len(client.get("/analysis/regions").json()) == 1


def test_region_requires_exactly_one_of_cells_or_bbox(client):
    for payload in (
        {"id": "r1", "name": "R1"},
        {"id": "r1", "name": "R1", "cells": [[0, 0]], "bbox": {"ix_min": 0, "ix_max": 0, "iy_min": 0, "iy_max": 0}},
    ):
        res = client.post("/analysis/regions", json=payload)
        assert res.status_code == 422, res.text


def test_reference_composition_rejects_duplicate_strata(client):
    res = client.post(
        "/analysis/reference-compositions",
        json={
            "id": "ref",
            "name": "ref",
            "entries": [
                {"habitat": "reed", "period_class": "day", "weight": 0.5},
                {"habitat": "reed", "period_class": "day", "weight": 0.5},
            ],
        },
    )
    assert res.status_code == 422


def _report_body(**kw):
    body = {
        "region_a_id": "reg-a",
        "region_b_id": "reg-b",
        "habitat_map_id": "hab",
        "reference_composition_id": "ref",
        "window": {"start": "2026-11-14T06:00:00Z", "end": "2026-11-15T06:00:00Z"},
        "bin_seconds": 43200,
    }
    body.update(kw)
    return body


def test_report_validates_definitions(client):
    _publish_defs(client)
    assert client.post("/analysis/reports", json=_report_body(region_b_id="missing")).status_code == 404
    assert client.post("/analysis/reports", json=_report_body(region_b_id="reg-a")).status_code == 400
    # Overlapping regions are rejected.
    client.post(
        "/analysis/regions",
        json={"id": "reg-c", "name": "C", "cells": [[1, 1], [9, 9]]},
    )
    assert client.post("/analysis/reports", json=_report_body(region_b_id="reg-c")).status_code == 400
    # Grid mismatch between report and region is rejected.
    assert client.post(
        "/analysis/reports", json=_report_body(grid_cell_size_m=500.0)
    ).status_code == 400
    # Window must be an exact multiple of the bin size.
    assert client.post(
        "/analysis/reports",
        json=_report_body(window={"start": "2026-11-14T06:00:00Z", "end": "2026-11-15T06:00:01Z"}),
    ).status_code == 400


def test_report_without_evidence_has_only_unknowns(client):
    """No stations, no events: every stratum is unknown, never zero."""
    _publish_defs(client)
    res = client.post("/analysis/reports", json=_report_body())
    assert res.status_code == 201, res.text
    result = res.json()["result"]
    for s in result["strata"]:
        for side in ("a", "b"):
            reg = s["regions"][side]
            assert reg["status"] == "no_coverage"
            assert reg["rate_per_cell_hour"] is None
    assert result["common_coverage"]["difference"] is None
    assert result["common_coverage"]["note"]
    assert result["standardized"]["difference"] is None
    assert len(result["standardized"]["empty_strata"]) == 2
    assert result["standardized"]["excluded_weight_mass"] == 1.0


def test_report_is_deterministic_for_same_cutoff(client):
    _publish_defs(client)
    body = _report_body(as_of="2026-11-15T00:00:00Z")
    first = client.post("/analysis/reports", json=body).json()
    second = client.post("/analysis/reports", json=body).json()
    assert first["report_hash"] == second["report_hash"]
    assert first["inputs_hash"] == second["inputs_hash"]


def test_refresh_requires_a_later_cutoff(client):
    _publish_defs(client)
    report = client.post(
        "/analysis/reports", json=_report_body(as_of="2026-11-15T00:00:00Z")
    ).json()
    res = client.post(
        f"/analysis/reports/{report['id']}/refresh",
        json={"as_of": "2026-11-14T00:00:00Z"},
    )
    assert res.status_code == 400
    newer = client.post(f"/analysis/reports/{report['id']}/refresh", json={})
    assert newer.status_code == 201
    assert newer.json()["supersedes_report_id"] == report["id"]


# --------------------------------------------------------------------------- #
# End-to-end: day/night bias + overlapping stations + empty stratum
# --------------------------------------------------------------------------- #
@pytest.fixture()
def daynight_report(client, api):
    outcome = analysis_demo.run_daynight("http://testserver", "test-key", reset=True, api=api)
    return outcome


def test_daynight_end_to_end(client, api, daynight_report):
    report = daynight_report["report"]
    result = report["result"]
    assert daynight_report["reproduce"]["reproducible"] is True

    # 36 events: 24 west (day+night) + 12 east (day only); each contributes once.
    assert result["events"]["included"] == 36
    for item in result["events"]["contributions"]:
        total = sum(c["weight"] for c in item["cells"]) + item["outside_regions_weight"]
        assert abs(total - 1.0) <= 1e-5

    # Naive counts expose the bias; the comparable day rates are equal.
    assert result["naive_counts"]["difference"] == 12.0
    common = result["common_coverage"]
    assert {(s["habitat"], s["period_class"]) for s in common["strata"]} == {
        ("reed", "day"),
        ("water", "day"),
    }
    assert common["difference"] == 0.0
    assert {(s["habitat"], s["period_class"]) for s in common["incomparable_strata"]} == {
        ("reed", "night"),
        ("water", "night"),
        ("mudflat", "day"),
        ("mudflat", "night"),
    }

    # Effort is not doubled where stations overlap: 8 reed cells x 24 h.
    strata = {(s["habitat"], s["period_class"]): s for s in result["strata"]}
    west_reed_day = strata[("reed", "day")]["regions"]["a"]
    assert west_reed_day["effort_cell_hours"] == pytest.approx(192.0, abs=0.05)
    assert west_reed_day["contributions"] == pytest.approx(6.0, abs=1e-6)

    # East nights are unknown everywhere, never zero.
    for key in (("reed", "night"), ("water", "night"), ("mudflat", "day"), ("mudflat", "night")):
        east = strata[key]["regions"]["b"]
        assert east["status"] == "no_coverage"
        assert east["rate_per_cell_hour"] is None

    std = result["standardized"]
    assert std["difference"] == 0.0
    assert std["compared_weight_mass"] == pytest.approx(0.5)
    assert std["excluded_weight_mass"] == pytest.approx(0.5)
    assert len(std["empty_strata"]) == 4
    assert std["common_coverage_crosscheck"]["difference"] == 0.0


def test_daynight_verifier(client, api, daynight_report):
    import verifier.verify_analysis as va

    class _Shim:
        def get(self, url, headers=None, params=None, timeout=None):
            return client.get(url.split("testserver", 1)[1], params=params)

        def post(self, url, headers=None, json=None, timeout=None):
            return client.post(url.split("testserver", 1)[1], json=json)

    original = va.requests
    va.requests = _Shim()
    out_path = SAMPLES / "daynight" / "verification_analysis.json"
    try:
        code = va.run_checks("daynight", SAMPLES / "daynight" / "truth.json", "http://testserver", "test-key")
    finally:
        va.requests = original
        out_path.unlink(missing_ok=True)
    assert code == 0


# --------------------------------------------------------------------------- #
# End-to-end: night-time failure reported late => versioned reports
# --------------------------------------------------------------------------- #
@pytest.fixture()
def nightfailure_reports(client, api):
    return analysis_demo.run_nightfailure("http://testserver", "test-key", reset=True, api=api)


def test_nightfailure_versions(client, api, nightfailure_reports):
    v1 = nightfailure_reports["report_v1"]
    v2 = nightfailure_reports["report_v2"]
    assert v1["report_hash"] != v2["report_hash"]
    assert v2["supersedes_report_id"] == v1["id"]
    assert nightfailure_reports["reproduce_v1"]["reproducible"] is True
    assert nightfailure_reports["reproduce_v2"]["reproducible"] is True

    r1, r2 = v1["result"], v2["result"]
    night1 = next(s for s in r1["strata"] if s["habitat"] == "reed" and s["period_class"] == "night")
    night2 = next(s for s in r2["strata"] if s["habitat"] == "reed" and s["period_class"] == "night")
    # v1 believed the nights were covered and scored a false zero.
    assert night1["regions"]["a"]["status"] == "covered"
    assert night1["regions"]["a"]["rate_per_cell_hour"] == 0.0
    assert r1["common_coverage"]["difference"] < -0.003
    # v2 marks them unknown; the comparable difference vanishes.
    assert night2["regions"]["a"]["status"] == "no_coverage"
    assert night2["regions"]["a"]["rate_per_cell_hour"] is None
    assert r2["common_coverage"]["difference"] == 0.0
    assert {(s["habitat"], s["period_class"]) for s in r2["standardized"]["empty_strata"]} == {
        ("reed", "night")
    }

    compare = nightfailure_reports["compare"]
    assert compare["same_inputs"] is True
    assert compare["same_result"] is False
    lost = [c for c in compare["stratum_changes"] if c["change"] == "coverage_lost"]
    assert lost and lost[0]["period_class"] == "night"
    evidence = compare["evidence_arrived_between_cutoffs"]
    assert evidence["progress_reports_declaring_gaps"] == 3


def test_nightfailure_verifier(client, api, nightfailure_reports):
    import verifier.verify_analysis as va

    class _Shim:
        def get(self, url, headers=None, params=None, timeout=None):
            return client.get(url.split("testserver", 1)[1], params=params)

        def post(self, url, headers=None, json=None, timeout=None):
            return client.post(url.split("testserver", 1)[1], json=json)

    original = va.requests
    va.requests = _Shim()
    out_path = SAMPLES / "nightfailure" / "verification_analysis.json"
    try:
        code = va.run_checks(
            "nightfailure", SAMPLES / "nightfailure" / "truth.json", "http://testserver", "test-key"
        )
    finally:
        va.requests = original
        out_path.unlink(missing_ok=True)
    assert code == 0


def test_analysis_verifier_is_independent_of_service_code():
    source = Path("verifier/verify_analysis.py").read_text()
    for forbidden in ("from app", "import app"):
        assert forbidden not in source


def test_old_report_reproduces_after_events_are_superseded(client, api):
    """A report frozen before a revision run must re-derive identically after
    the run superseded its events: supersession is judged by creation time,
    not by the mutable status column."""
    from simulator.run_simulator import load_scenario

    load_scenario("revision", "http://testserver", truth_dir=SAMPLES / "revision", phases=("early",))
    client.post("/runs", json={"wait": True})
    client.post(
        "/analysis/regions",
        json={"id": "west", "name": "W", "bbox": {"ix_min": 0, "ix_max": 3, "iy_min": 4, "iy_max": 7}},
    )
    client.post(
        "/analysis/regions",
        json={"id": "east", "name": "E", "bbox": {"ix_min": 4, "ix_max": 7, "iy_min": 4, "iy_max": 7}},
    )
    client.post(
        "/analysis/habitat-maps",
        json={
            "id": "hab",
            "cells": [
                {"ix": ix, "iy": iy, "habitat": "reed"}
                for ix in range(8)
                for iy in range(4, 8)
            ],
        },
    )
    report = client.post(
        "/analysis/reports",
        json={
            "region_a_id": "west",
            "region_b_id": "east",
            "habitat_map_id": "hab",
            "class_label": "trill",
            "window": {"start": "2026-11-14T00:00:00Z", "end": "2026-11-14T02:00:00Z"},
            "bin_seconds": 3600,
        },
    )
    assert report.status_code == 201, report.text
    report = report.json()
    assert report["result"]["events"]["included"] > 0

    # Late data + a revision run supersede the events the report counted.
    load_scenario("revision", "http://testserver", truth_dir=SAMPLES / "revision", phases=("late",))
    client.post("/runs", json={"wait": True, "mode": "revision"})
    superseded = client.get("/events", params={"include_superseded": "true", "limit": 1000}).json()
    assert any(e["status"] == "superseded" for e in superseded["events"])

    # The old report still re-derives exactly from its own cutoff.
    repro = client.post(f"/analysis/reports/{report['id']}/reproduce").json()
    assert repro["reproducible"] is True
    assert repro["recomputed_hash"] == report["report_hash"]
