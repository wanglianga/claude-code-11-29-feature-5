"""Clock calibration and localisation unit tests."""

from __future__ import annotations

import math

import pytest

from app import calibration as calib
from app.config import Settings, default_area
from app.localize import StationPoint, localize


def _settings(**over) -> Settings:
    base = Settings()
    return base.__class__(**{**base.__dict__, **over})


def _anchors(
    slope: float, intercept_us: float, reference_times: list[int], station_id="S1", boot="b1"
) -> list[calib.Anchor]:
    return [
        calib.Anchor(
            observation_id=100 + i,
            source_id=f"cal{i}",
            local_us=int(round(slope * t + intercept_us)),
            reference_arrival_us=t,
            propagation_us=0.0,
        )
        for i, t in enumerate(reference_times)
    ]


REF0 = 1_794_600_000_000_000


def test_piecewise_linear_map_recovers_slope_and_intercept():
    cfg = _settings()
    slope, intercept = 1.0000123, -123456789
    times = [REF0 + i * 30_000_000 for i in range(40)]
    fit = calib.fit_calibration("S1", "b1", _anchors(slope, intercept, times), cfg)
    assert fit.valid, fit.reason
    assert fit.segments
    assert fit.segments[0].slope == pytest.approx(slope, abs=1e-8)
    result = calib.apply_calibration(fit, int(round(slope * (REF0 + 20_000_000) + intercept)), cfg)
    assert result.status == "calibrated"
    assert result.reference_arrival_us == pytest.approx(REF0 + 20_000_000, abs=2000)


def test_too_few_anchors_leaves_observations_uncalibrated():
    cfg = _settings()
    fit = calib.fit_calibration(
        "S1", "b1", _anchors(1.0, 0, [REF0, REF0 + 30_000_000]), cfg
    )
    assert not fit.valid
    assert fit.reason.startswith("too_few_anchors")
    result = calib.apply_calibration(fit, REF0, cfg)
    assert result.status == "uncalibrated"
    assert result.reference_arrival_us is None


def test_short_anchor_span_is_rejected():
    cfg = _settings()
    times = [REF0 + i * 1_000_000 for i in range(12)]  # 11 s span
    fit = calib.fit_calibration("S1", "b1", _anchors(1.0, 0, times), cfg)
    assert not fit.valid
    assert "anchor_span_too_short" in fit.reason


def test_large_residual_is_rejected():
    cfg = _settings()
    times = [REF0 + i * 60_000_000 for i in range(20)]
    anchors = _anchors(1.0, 0, times)
    # Add an inconsistent cluster that no single line can explain.
    anchors += [
        calib.Anchor(
            observation_id=900 + i,
            source_id=f"bad{i}",
            local_us=t + (i - 4) * 20_000_000 // 4,
            reference_arrival_us=t,
            propagation_us=0,
        )
        for i, t in enumerate(REF0 + j * 60_000_000 for j in range(10, 16))
    ]
    fit = calib.fit_calibration("S1", "b1", anchors, cfg)
    assert not fit.valid
    assert "residual_too_large" in fit.reason or "no_plausible" in fit.reason


def test_implausible_clock_rate_is_rejected():
    cfg = _settings()
    times = [REF0 + i * 60_000_000 for i in range(12)]
    fit = calib.fit_calibration("S1", "b1", _anchors(1.5, 0, times), cfg)
    assert not fit.valid
    assert "implausible_clock_rate" in fit.reason


def test_two_boots_get_distinct_models():
    cfg = _settings()
    times = [REF0 + i * 60_000_000 for i in range(12)]
    b1 = calib.fit_calibration("S1", "b1", _anchors(1.0, 0, times), cfg)
    # A 20-minute jump in the counter: intercept differs by 1200 s.
    b2 = calib.fit_calibration("S1", "b2", _anchors(1.0, 1_200_000_000, times), cfg)
    assert b1.valid and b2.valid
    assert b1.version_hash() != b2.version_hash()
    local = int(round(1.0 * (REF0 + 300_000_000) + 1_200_000_000))
    ref_b2 = calib.apply_calibration(b2, local, cfg).reference_arrival_us
    ref_b1 = calib.apply_calibration(b1, local, cfg).reference_arrival_us
    assert abs((ref_b1 - ref_b2) - 1_200_000_000) < 5000


def test_drift_is_modelled_by_segments():
    cfg = _settings(cal_split_rms_ms=1.0, cal_split_min_gain=0.05)
    times = [REF0 + i * 60_000_000 for i in range(40)]
    anchors = []
    for i, t in enumerate(times):
        # Rate changes halfway: 0 ppm then 300 ppm, a visible kink.
        drift = 0.0 if i < 20 else 300e-6
        local = int(round(t + drift * (t - times[0])))
        anchors.append(
            calib.Anchor(i + 1, f"c{i}", local, t, 0.0)
        )
    fit = calib.fit_calibration("S1", "b1", anchors, cfg)
    assert fit.valid, fit.reason
    assert len(fit.segments) >= 2


def test_extrapolation_far_outside_span_is_uncalibrated():
    cfg = _settings(cal_coverage_margin_s=60)
    times = [REF0 + i * 30_000_000 for i in range(10)]
    fit = calib.fit_calibration("S1", "b1", _anchors(1.0, 0, times), cfg)
    assert fit.valid
    far_local = int(times[-1] + 3_600_000_000)
    result = calib.apply_calibration(fit, far_local, cfg)
    assert result.status == "uncalibrated"
    assert "outside_calibrated_span" in result.reason


# --------------------------------------------------------------------------- #
# Localisation
# --------------------------------------------------------------------------- #
REF_US = 1_794_600_000_000_000
SPEED = 343.0


def _stations(points):
    return [StationPoint(f"S{i+1}", x, y, 3000.0) for i, (x, y) in enumerate(points)]


def _arrivals(points, source, speed=SPEED):
    sx, sy = source
    return [REF_US + math.hypot(x - sx, y - sy) / speed * 1e6 for x, y in points]


def test_four_stations_localise_within_a_cell():
    cfg = _settings()
    points = [(0, 0), (1200, 0), (1200, 900), (0, 900)]
    stations = _stations(points)
    source = (500.0, 400.0)
    res = localize(stations, _arrivals(points, source), [5_000.0] * 4, cfg, default_area())
    assert res.feasible
    assert len(res.regions) == 1
    region = res.regions[0]
    assert math.hypot(region.representative["x"] - source[0], region.representative["y"] - source[1]) < 120
    assert res.position_certain


def test_collinear_stations_return_both_mirrors():
    cfg = _settings()
    points = [(0, 500), (600, 500), (1200, 500), (1800, 500)]
    stations = _stations(points)
    source = (900.0, 150.0)
    res = localize(stations, _arrivals(points, source), [5_000.0] * 4, cfg, default_area())
    assert res.feasible
    assert len(res.regions) == 2
    assert not res.position_certain
    assert res.geometry["collinear"] is True
    ys = sorted(r.bounds["y_min"] + (r.bounds["y_max"] - r.bounds["y_min"]) / 2 for r in res.regions)
    assert ys[0] < 400 < 600 < ys[1]
    assert any("collinear" in w for w in res.warnings)


def test_two_stations_are_not_certain():
    cfg = _settings()
    points = [(0, 0), (1200, 0)]
    stations = _stations(points)
    res = localize(stations, _arrivals(points, (600.0, 300.0)), [5_000.0] * 2, cfg, default_area())
    assert res.feasible
    assert not res.position_certain


def test_inconsistent_arrivals_have_no_feasible_region():
    cfg = _settings()
    points = [(0, 0), (1200, 0), (1200, 900), (0, 900)]
    stations = _stations(points)
    arrivals = _arrivals(points, (500.0, 400.0))
    arrivals[3] += 5_000_000  # 5 s late: impossible in a 2 km area
    res = localize(stations, arrivals, [5_000.0] * 4, cfg, default_area())
    assert not res.feasible
    assert res.reason == "no_feasible_cell_in_area"


def test_hearing_radius_excludes_far_stations():
    cfg = _settings()
    points = [(0, 0), (1200, 0), (1200, 900), (0, 900)]
    stations = [StationPoint(f"S{i+1}", x, y, 100.0) for i, (x, y) in enumerate(points)]
    res = localize(stations, _arrivals(points, (500.0, 400.0)), [5_000.0] * 4, cfg, default_area())
    assert not res.feasible
