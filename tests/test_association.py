"""Candidate grouping, the exclusive solver, and pipeline semantics."""

from __future__ import annotations

import math

from app import association, solver
from app.association import Candidate, ObsInput
from app.config import default_area, settings
from app.localize import StationPoint, localize

REF = 1_794_600_000_000_000
SPEED = 343.0


def _station_points(points, radius=4000.0):
    return {f"S{i+1}": StationPoint(f"S{i+1}", x, y, radius) for i, (x, y) in enumerate(points)}


def _obs(oid, station, points, source, features, uncertainty_us=20_000, class_label="croak"):
    idx = int(station[1:]) - 1
    x, y = points[idx]
    arrival = REF + math.hypot(x - source[0], y - source[1]) / SPEED * 1e6
    return ObsInput(
        observation_id=oid,
        station_id=station,
        boot_id="b1",
        class_label=class_label,
        corrected_arrival_us=int(round(arrival)),
        uncertainty_us=uncertainty_us,
        features=list(features),
        calibration_model_id=1,
        anchor_ids=["C1@x"],
    )


def test_one_sound_heard_by_four_stations_is_one_clique():
    points = [(0, 0), (1200, 0), (1200, 900), (0, 900)]
    stations = _station_points(points)
    source = (500.0, 400.0)
    observations = [
        _obs(i + 1, f"S{i+1}", points, source, [0.5, 0.5, 0.5]) for i in range(4)
    ]
    result = association.build_candidates(
        observations, stations, {"croak": 0.2}, settings, default_area()
    )
    top = result.candidates[0]
    assert top.explained == 4
    assert len({o for o in top.observation_ids}) == 4


def test_distant_pairs_cannot_chain_into_one_event():
    """A-B and B-C may be locally compatible, but A-C is not; no (A,B,C) clique."""
    points = [(0, 0), (2000, 0)]
    stations = _station_points(points)
    # S1 hears at 0 s; S2 hears at 5.83 s (2 km apart) -> compatible.
    a = ObsInput(1, "S1", "b1", "croak", REF, 5_000, [0.5, 0.5], 1, [])
    b = ObsInput(2, "S2", "b1", "croak", REF + 5_830_000, 5_000, [0.5, 0.5], 1, [])
    # A third record 30 s later at S1 cannot belong to any of them.
    c = ObsInput(3, "S1", "b1", "croak", REF + 30_000_000, 5_000, [0.5, 0.5], 1, [])
    result = association.build_candidates(
        [a, b, c], stations, {"croak": 0.2}, settings, default_area()
    )
    for cand in result.candidates:
        assert 3 not in cand.observation_ids
    assert any(set(c.observation_ids) == {1, 2} for c in result.candidates)


def test_feature_distance_blocks_association():
    points = [(0, 0), (1200, 0)]
    stations = _station_points(points)
    source = (600.0, 300.0)
    a = _obs(1, "S1", points, source, [0.0, 0.0])
    b = _obs(2, "S2", points, source, [0.9, 0.9])
    result = association.build_candidates([a, b], stations, {"croak": 0.2}, settings, default_area())
    assert result.candidates == []


def test_same_station_twice_is_never_one_event():
    points = [(0, 0), (1200, 0)]
    stations = _station_points(points)
    source = (600.0, 300.0)
    a = _obs(1, "S1", points, source, [0.5, 0.5])
    b = _obs(2, "S1", points, source, [0.5, 0.5])
    result = association.build_candidates([a, b], stations, {"croak": 0.2}, settings, default_area())
    assert result.candidates == []


def _candidate(cid, obs_ids, points, source, features, class_label="croak", unc=20_000):
    stations = _station_points(points)
    station_ids = [f"S{i+1}" for i in range(len(points))]
    arrivals = []
    for (x, y) in points:
        arrivals.append(REF + math.hypot(x - source[0], y - source[1]) / SPEED * 1e6)
    loc = localize(
        [stations[s] for s in station_ids], arrivals, [float(unc)] * len(points),
        settings, default_area(),
    )
    assert loc.feasible
    r = loc.residuals_by_observation(
        loc.regions[0].representative["x"], loc.regions[0].representative["y"]
    )
    rss = int(round(sum(v * v for v in r)))
    return Candidate(
        candidate_id=cid,
        class_label=class_label,
        observation_ids=obs_ids,
        station_ids=station_ids,
        localization=loc,
        residual_sq_us2=rss,
        residual_rms_us=int(rss**0.5),
        emission_us=int(loc.regions[0].emission_us),
        window_start_us=0,
        window_end_us=1,
    )


def test_solver_never_shares_an_observation():
    points = [(0, 0), (1200, 0), (1200, 900)]
    c1 = _candidate("E1", [1, 2, 3], points, (500, 400), [0.5])
    c2 = _candidate("E2", [3, 4, 5], points, (520, 410), [0.5])
    c3 = _candidate("E3", [4, 5, 6], points, (900, 700), [0.5])
    result = solver.solve([c1, c2, c3], {i: 1 for i in range(1, 7)}, settings)
    chosen = set(result.best.observation_ids)
    assert len(chosen) == sum(1 for _ in chosen)  # no repeats inside the best option
    for option in result.options:
        ids = option.observation_ids
        assert len(ids) == len(set(ids))


def test_solver_prefers_more_reliable_observations_then_less_residual():
    points = [(0, 0), (1200, 0), (1200, 900)]
    good = _candidate("E1", [1, 2, 3], points, (500, 400), [0.5])
    weak = _candidate("E2", [4, 5, 6], points, (500, 400), [0.5])
    weights = {1: 1, 2: 1, 3: 1, 4: 0, 5: 0, 6: 0}
    result = solver.solve([good, weak], weights, settings)
    assert result.best.explained_observations == 6
    assert result.best.reliable_observations == 3


def test_solver_reports_exact_ties_for_human_selection():
    points = [(0, 0), (1200, 0)]
    a = _candidate("E1", [1, 2], points, (500, 400), [0.5])
    b = _candidate("E2", [3, 4], points, (500, 400), [0.5])
    c = _candidate("E3", [1, 4], points, (500, 400), [0.5])
    d = _candidate("E4", [2, 3], points, (500, 400), [0.5])
    result = solver.solve([a, b, c, d], {i: 1 for i in range(1, 5)}, settings)
    assert result.tied is True
    assert "ambiguous" in " ".join(result.notes)
    assert len(result.options) >= 2


def test_solver_reports_proven_optimality_empty_input():
    result = solver.solve([], {}, settings)
    assert result.best is None
    assert result.proven_optimal


def test_solver_matches_brute_force_on_random_small_instances():
    """The exact solver must equal exhaustive enumeration, including weights."""
    import itertools
    import random

    class _FakeLoc:
        feasible = True

    def brute(cands, weights):
        best = None
        n = len(cands)
        for mask in range(1 << n):
            used: set[int] = set()
            ok = True
            for i in range(n):
                if mask >> i & 1:
                    s = set(cands[i].observation_ids)
                    if used & s:
                        ok = False
                        break
                    used |= s
            if not ok:
                continue
            key = (
                sum(weights.get(o, 0) for o in used),
                len(used),
                -sum(cands[i].residual_sq_us2 for i in range(n) if mask >> i & 1),
            )
            if best is None or key > best:
                best = key
        return best

    random.seed(1234)
    for trial in range(120):
        n = random.randint(1, 10)
        cands = [
            Candidate(
                candidate_id=f"E{i}",
                class_label="x",
                observation_ids=sorted(random.sample(range(1, 10), random.randint(1, 5))),
                station_ids=[],
                localization=_FakeLoc(),
                residual_sq_us2=random.randint(1, 9000),
                residual_rms_us=0,
                emission_us=0,
                window_start_us=0,
                window_end_us=1,
            )
            for i in range(n)
        ]
        weights = {o: random.randint(0, 2) for o in range(1, 10)}
        expected = brute(cands, weights)
        result = solver.solve(cands, weights, settings)
        assert result.proven_optimal
        got = (
            result.best.reliable_observations,
            result.best.explained_observations,
            -result.best.total_residual_sq_us2,
        )
        assert got == expected, (trial, got, expected)
