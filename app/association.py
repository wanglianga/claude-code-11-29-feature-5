"""Candidate grouping of calibrated arrivals.

Rules enforced here
-------------------
* Only observations whose clock was calibrated are considered.  Upload time is
  never substituted for a missing arrival time.
* A candidate group uses at most one observation per station.
* Pairwise compatibility requires the same sound class, a feature distance
  below the class limit, and a decodable propagation-time difference:
  ``|t_i - t_j| <= distance(station_i, station_j) / speed + u_i + u_j``.
  By the triangle inequality this is a necessary condition for any common
  source, and it cannot be relaxed by chaining: groups are *cliques*, so every
  pair is checked, not just neighbouring pairs along a chain.
* A clique only becomes a candidate event when the grid search finds at least
  one feasible source region for the whole set, i.e. one common emission time
  and position explains every arrival simultaneously.  This is what prevents a
  chain of locally-close records from being merged into one distant event.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

from .config import Area, Settings
from .localize import Localization, StationPoint, localize
from .physics import distance


@dataclass
class ObsInput:
    observation_id: int
    station_id: str
    boot_id: str
    class_label: str
    corrected_arrival_us: int
    uncertainty_us: int
    features: list[float]
    calibration_model_id: int | None
    anchor_ids: list[str]


@dataclass
class Candidate:
    candidate_id: str
    class_label: str
    observation_ids: list[int]
    station_ids: list[str]
    localization: Localization
    residual_sq_us2: int
    residual_rms_us: int
    emission_us: int
    window_start_us: int
    window_end_us: int
    warnings: list[str] = field(default_factory=list)

    @property
    def explained(self) -> int:
        return len(self.observation_ids)


@dataclass
class CandidateSet:
    candidates: list[Candidate]
    cliques_examined: int
    cliques_infeasible: int
    truncated: bool
    notes: list[str]
    skipped_uncalibrated: int
    observations_considered: int


def _feature_distance(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return math.inf
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _compatible(
    a: ObsInput,
    b: ObsInput,
    station_a: StationPoint,
    station_b: StationPoint,
    feature_limit: float,
    speed_mps: float,
    slack_us: float,
) -> bool:
    if a.station_id == b.station_id:
        return False
    dt = abs(a.corrected_arrival_us - b.corrected_arrival_us)
    max_delta = (
        distance(
            station_a.x, station_a.y, station_b.x, station_b.y
        )
        / speed_mps
        * 1_000_000.0
    )
    if dt > max_delta + a.uncertainty_us + b.uncertainty_us + slack_us:
        return False
    if _feature_distance(a.features, b.features) > feature_limit:
        return False
    return True


def _enumerate_cliques(
    nodes: list[ObsInput],
    adjacency: dict[int, set[int]],
    max_size: int,
    max_cliques: int,
) -> tuple[list[list[int]], bool]:
    """Bron-Kerbosch with pivoting, deterministic order, size-capped.

    A clique that reaches ``max_size`` is recorded and not extended, so that a
    large dense component does not silently produce an event with more stations
    than the configured maximum; the note is surfaced to the caller.
    """
    index = {n.observation_id: i for i, n in enumerate(nodes)}
    nbrs = {
        i: {index[o] for o in adjacency[nodes[i].observation_id] if o in index}
        for i in range(len(nodes))
    }
    results: list[list[int]] = []
    truncated = False

    def bk(r: list[int], p: set[int], x: set[int]) -> None:
        nonlocal truncated
        if len(results) >= max_cliques:
            truncated = True
            return
        if not p and not x:
            if len(r) >= 2:
                results.append(sorted(r))
            return
        if len(r) >= max_size:
            if len(r) >= 2:
                results.append(sorted(r))
            return
        pivot = max(p | x, key=lambda v: len(p & nbrs[v])) if (p or x) else None
        candidates = p - (nbrs[pivot] if pivot is not None else set())
        for v in sorted(candidates, key=lambda i: nodes[i].corrected_arrival_us):
            bk(r + [v], p & nbrs[v], x & nbrs[v])
            p = p - {v}
            x = x | {v}
            if len(results) >= max_cliques:
                truncated = True
                return

    bk([], set(range(len(nodes))), set())
    return results, truncated


def _subset_candidates(
    clique: list[ObsInput],
    min_size: int,
    radius: int = 2,
) -> list[list[ObsInput]]:
    """Subsets obtained by dropping up to ``radius`` members.

    Used when a maximal clique has no common source point, which is what an
    overlapping pair of sound sources looks like: one or two shared stations'
    records are genuinely ambiguous between two events, and freeing them lets
    both events form.  Radius 2 covers the case where both members of a pair
    of coincident calls are contaminated at the same two stations.
    """
    out: list[list[ObsInput]] = []
    n = len(clique)
    for drop in range(1, radius + 1):
        if n - drop < min_size:
            break
        for combo in itertools.combinations(range(n), drop):
            keep = [clique[i] for i in range(n) if i not in combo]
            stations = {o.station_id for o in keep}
            if len(stations) != len(keep):
                continue
            out.append(keep)
    return out


def build_candidates(
    observations: list[ObsInput],
    stations: dict[str, StationPoint],
    class_limits: dict[str, float],
    cfg: Settings,
    area: Area,
) -> CandidateSet:
    considered = len(observations)
    notes: list[str] = []
    by_class: dict[str, list[ObsInput]] = {}
    for obs in observations:
        by_class.setdefault(obs.class_label, []).append(obs)

    candidates: dict[tuple[int, ...], Candidate] = {}
    cliques_examined = 0
    cliques_infeasible = 0
    truncated_any = False
    stopped = False

    for class_label in sorted(by_class):
        nodes = sorted(
            by_class[class_label],
            key=lambda o: (o.corrected_arrival_us, o.station_id, o.observation_id),
        )
        limit = class_limits.get(class_label, cfg.feature_distance_limit)
        adjacency: dict[int, set[int]] = {o.observation_id: set() for o in nodes}
        for i in range(len(nodes)):
            a = nodes[i]
            sa = stations.get(a.station_id)
            if sa is None:
                continue
            for j in range(i + 1, len(nodes)):
                b = nodes[j]
                sb = stations.get(b.station_id)
                if sb is None:
                    continue
                if b.corrected_arrival_us - a.corrected_arrival_us > (
                    area_extent_speed(area, cfg.propagation_speed_mps)
                    + a.uncertainty_us
                    + b.uncertainty_us
                    + cfg.association_time_slack_ms * 1000.0
                ):
                    # Sorted arrivals: no later record can be within the global
                    # maximum propagation-time difference.
                    break
                if _compatible(
                    a,
                    b,
                    sa,
                    sb,
                    limit,
                    cfg.propagation_speed_mps,
                    cfg.association_time_slack_ms * 1000.0,
                ):
                    adjacency[a.observation_id].add(b.observation_id)
                    adjacency[b.observation_id].add(a.observation_id)
        if not any(adjacency.values()):
            continue
        remaining = cfg.max_candidate_groups - len(candidates)
        if remaining <= 0:
            truncated_any = True
            stopped = True
            break
        cliques, truncated = _enumerate_cliques(
            nodes, adjacency, cfg.max_group_size, max(remaining * 4, 64)
        )
        truncated_any |= truncated
        for clique_ids in cliques:
            if len(candidates) >= cfg.max_candidate_groups:
                truncated_any = True
                stopped = True
                break
            # _enumerate_cliques returns positions in `nodes`, not observation ids.
            clique = [nodes[pos] for pos in clique_ids]
            cliques_examined += 1
            added = _register_group(
                clique, candidates, stations, cfg, area, notes
            )
            if not added:
                cliques_infeasible += 1
                for subset in _subset_candidates(clique, cfg.min_event_stations):
                    if len(candidates) >= cfg.max_candidate_groups:
                        truncated_any = True
                        stopped = True
                        break
                    _register_group(subset, candidates, stations, cfg, area, notes)
        if stopped:
            break

    if truncated_any:
        notes.append(
            "candidate enumeration hit max_candidate_groups="
            f"{cfg.max_candidate_groups}; later cliques were not evaluated"
        )
    ordered = sorted(
        candidates.values(),
        key=lambda c: (-c.explained, c.residual_sq_us2, tuple(c.observation_ids)),
    )
    for i, cand in enumerate(ordered):
        cand.candidate_id = f"E{i + 1:04d}"
    if notes:
        notes.append(
            "grouping is clique-based: every pair in a candidate must satisfy the "
            "propagation-time-difference bound, so chains of nearby pairs cannot "
            "merge distant sources"
        )
    return CandidateSet(
        candidates=ordered,
        cliques_examined=cliques_examined,
        cliques_infeasible=cliques_infeasible,
        truncated=truncated_any,
        notes=notes,
        skipped_uncalibrated=0,
        observations_considered=considered,
    )


def area_extent_speed(area: Area, speed_mps: float) -> float:
    return math.hypot(area.x_max - area.x_min, area.y_max - area.y_min) / speed_mps * 1_000_000.0


def _register_group(
    group: list[ObsInput],
    candidates: dict[tuple[int, ...], Candidate],
    stations: dict[str, StationPoint],
    cfg: Settings,
    area: Area,
    notes: list[str],
) -> bool:
    if len(group) < cfg.min_event_stations:
        return False
    key = tuple(sorted(o.observation_id for o in group))
    if key in candidates:
        return True
    station_points = []
    arrivals = []
    uncertainties = []
    for obs in sorted(group, key=lambda o: o.observation_id):
        st = stations.get(obs.station_id)
        if st is None:
            return False
        station_points.append(st)
        arrivals.append(obs.corrected_arrival_us)
        uncertainties.append(float(obs.uncertainty_us))
    result = localize(station_points, arrivals, uncertainties, cfg, area)
    if not result.feasible or not result.regions:
        return False
    best = result.regions[0]
    rms = best.representative["rms_residual_us"]
    residual_sq = int(round((rms**2) * len(group)))
    window_start = min(
        obs.corrected_arrival_us - obs.uncertainty_us for obs in group
    )
    window_end = max(
        obs.corrected_arrival_us + obs.uncertainty_us for obs in group
    )
    candidates[key] = Candidate(
        candidate_id="",
        class_label=group[0].class_label,
        observation_ids=[o.observation_id for o in sorted(group, key=lambda o: o.observation_id)],
        station_ids=sorted({o.station_id for o in group}),
        localization=result,
        residual_sq_us2=residual_sq,
        residual_rms_us=int(round(rms)),
        emission_us=int(round(best.emission_us)),
        window_start_us=min(window_start, best.emission_min_us),
        window_end_us=max(window_end, best.emission_max_us),
        warnings=list(result.warnings),
    )
    return True
