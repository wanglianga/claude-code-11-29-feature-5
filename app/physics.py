"""Geometric primitives: propagation delay and feasibility intervals."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .config import Area


@dataclass(frozen=True)
class StationGeometry:
    id: str
    x: float
    y: float
    hearing_radius_m: float


def distance(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def propagation_delay_us(ax: float, ay: float, bx: float, by: float, speed_mps: float) -> float:
    return distance(ax, ay, bx, by) / speed_mps * 1_000_000.0


def area_diagonal_m(area: Area) -> float:
    return math.hypot(area.x_max - area.x_min, area.y_max - area.y_min)


@dataclass
class Feasibility:
    """Result of testing one candidate source point for a group of arrivals."""

    feasible: bool
    emission_us: float | None
    emission_min_us: float | None
    emission_max_us: float | None
    rms_residual_us: float | None
    max_normalized: float | None
    residuals_us: list[float]


def evaluate_point_with_uncertainty(
    x: float,
    y: float,
    stations: list[tuple[float, float]],
    arrivals_us: list[float],
    uncertainties_us: list[float],
    speed_mps: float,
    slack_us: float,
    expand_us: float = 0.0,
) -> Feasibility:
    adjusted: list[float] = []
    for (sx, sy), t in zip(stations, arrivals_us):
        adjusted.append(t - propagation_delay_us(x, y, sx, sy, speed_mps))
    weights = [1.0 / max(u, 1.0) for u in uncertainties_us]
    total_w = sum(weights)
    emission = sum(a * w for a, w in zip(adjusted, weights)) / total_w
    residuals = [a - emission for a in adjusted]
    lo = max(a - (u + slack_us + expand_us) for a, u in zip(adjusted, uncertainties_us))
    hi = min(a + (u + slack_us + expand_us) for a, u in zip(adjusted, uncertainties_us))
    feasible = lo <= hi
    rms = math.sqrt(sum(r * r for r in residuals) / len(residuals))
    max_norm = max(
        abs(r) / max(u + slack_us, 1.0) for r, u in zip(residuals, uncertainties_us)
    )
    return Feasibility(
        feasible=feasible,
        emission_us=emission if feasible else None,
        emission_min_us=lo if feasible else None,
        emission_max_us=hi if feasible else None,
        rms_residual_us=rms,
        max_normalized=max_norm,
        residuals_us=residuals,
    )
