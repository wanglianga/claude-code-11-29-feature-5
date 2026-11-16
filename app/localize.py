"""Coarse-to-fine 2-D grid search for feasible source regions.

The fit error for a candidate point is the dispersion of the per-station
``corrected arrival - propagation delay`` values about their weighted mean.
A point is feasible when every station's adjusted value agrees within that
station's calibrated uncertainty plus a configured slack, when the station is
within its hearing radius, and when at least two stations contribute.

The search is deliberately conservative: a cell is kept when its centre is
feasible after expanding every uncertainty by the time-equivalent of the cell's
half diagonal.  That over-approximates the feasible set by at most one cell, so
no candidate region is silently dropped; the promised precision is the finest
grid cell size actually reached.

Everything here is bounded by the configured observation area and grid
resolution.  The output is a set of candidate regions, never a claimed exact
inversion, and multiple disconnected regions are all returned.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .config import Area, Settings, default_area
from .physics import distance


@dataclass(frozen=True)
class StationPoint:
    id: str
    x: float
    y: float
    hearing_radius_m: float


@dataclass
class GridCell:
    ix: int
    iy: int
    level: int
    res_m: float
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    rms_us: float
    emission_us: float


@dataclass
class Region:
    region_id: str
    cells: list[GridCell]
    bounds: dict
    area_m2: float
    representative: dict
    emission_us: float
    emission_min_us: int
    emission_max_us: int
    extent_m: float
    cells_truncated: bool = False


@dataclass
class Localization:
    feasible: bool
    reason: str | None
    regions: list[Region] = field(default_factory=list)
    grid_resolution_m: float = 0.0
    coarse_cells: int = 0
    leaf_cells: int = 0
    geometry: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    position_certain: bool = False
    station_points: list[StationPoint] = field(default_factory=list)
    arrivals_us: list[float] = field(default_factory=list)
    uncertainties_us: list[float] = field(default_factory=list)
    speed_mps: float = 343.0

    def residuals_by_observation(self, x: float, y: float) -> list[float]:
        """Adjusted-arrival residuals at an arbitrary candidate point."""
        adjusted = [
            t - distance(x, y, st.x, st.y) / self.speed_mps * 1_000_000.0
            for st, t in zip(self.station_points, self.arrivals_us)
        ]
        weights = [1.0 / max(u, 1.0) for u in self.uncertainties_us]
        emission = sum(a * w for a, w in zip(adjusted, weights)) / sum(weights)
        return [a - emission for a in adjusted]


def _cell_centers(cells: list[tuple[int, int]], level: int, coarse: float, area: Area) -> tuple[np.ndarray, np.ndarray]:
    res = coarse / (2**level)
    if not cells:
        return np.array([]), np.array([])
    arr = np.asarray(cells, dtype=float)
    xs = area.x_min + (arr[:, 0] + 0.5) * res
    ys = area.y_min + (arr[:, 1] + 0.5) * res
    # Clip centres of clipped edge cells to the area.
    xs = np.clip(xs, area.x_min, area.x_max)
    ys = np.clip(ys, area.y_min, area.y_max)
    return xs, ys


def _eval_points(
    xs: np.ndarray,
    ys: np.ndarray,
    stations: list[StationPoint],
    arrivals: list[float],
    uncertainties: list[float],
    speed_mps: float,
    slack_us: float,
    expand_us: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo = np.full(xs.shape, -np.inf)
    hi = np.full(xs.shape, np.inf)
    rss = np.zeros(xs.shape)
    weighted_sum = np.zeros(xs.shape)
    weight_sum = np.zeros(xs.shape)
    hear = np.ones(xs.shape, dtype=bool)
    adjusted: list[np.ndarray] = []
    for st, t, u in zip(stations, arrivals, uncertainties):
        dist = np.hypot(xs - st.x, ys - st.y)
        hear &= dist <= st.hearing_radius_m
        delay = dist / speed_mps * 1_000_000.0
        adj = t - delay
        tol = u + slack_us + expand_us
        lo = np.maximum(lo, adj - tol)
        hi = np.minimum(hi, adj + tol)
        weight = 1.0 / max(u, 1.0)
        weighted_sum += adj * weight
        weight_sum += weight
        adjusted.append(adj)
    emission = weighted_sum / weight_sum
    for adj in adjusted:
        rss += (adj - emission) ** 2
    rms = np.sqrt(rss / max(1, len(stations)))
    feasible = hear & (lo <= hi)
    return feasible, emission, rms


def _half_diagonal_us(res_m: float, speed_mps: float) -> float:
    return math.hypot(res_m / 2.0, res_m / 2.0) / speed_mps * 1_000_000.0


def _tri_area(a: StationPoint, b: StationPoint, c: StationPoint) -> float:
    return abs((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)) / 2.0


def geometry_summary(stations: list[StationPoint]) -> dict:
    n = len(stations)
    if n < 2:
        return {"station_count": n, "collinear": True, "max_baseline_m": 0.0, "max_perpendicular_m": 0.0}
    max_base = 0.0
    pair = (stations[0], stations[1])
    for i in range(n):
        for j in range(i + 1, n):
            d = distance(stations[i].x, stations[i].y, stations[j].x, stations[j].y)
            if d > max_base:
                max_base, pair = d, (stations[i], stations[j])
    max_perp = 0.0
    if max_base > 0:
        ax, ay = pair[0].x, pair[0].y
        bx, by = pair[1].x, pair[1].y
        for st in stations:
            perp = abs((bx - ax) * (ay - st.y) - (ax - st.x) * (by - ay)) / max_base
            max_perp = max(max_perp, perp)
    max_triangle = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                max_triangle = max(max_triangle, _tri_area(stations[i], stations[j], stations[k]))
    return {
        "station_count": n,
        "collinear": max_perp <= max(1e-3, 1e-9 * max_base * 1e6),
        "max_baseline_m": round(max_base, 3),
        "max_perpendicular_m": round(max_perp, 6),
        "max_triangle_area_m2": round(max_triangle, 6),
    }


def localize(
    stations: list[StationPoint],
    arrivals_us: list[int | float],
    uncertainties_us: list[float],
    cfg: Settings,
    area: Area | None = None,
    speed_mps: float | None = None,
) -> Localization:
    area = area or default_area()
    speed = speed_mps or cfg.propagation_speed_mps
    slack = cfg.localization_slack_ms * 1000.0
    geometry = geometry_summary(stations)
    if len(stations) < 2:
        return Localization(
            feasible=False,
            reason="fewer_than_two_stations",
            geometry=geometry,
            station_points=stations,
            arrivals_us=[float(a) for a in arrivals_us],
            uncertainties_us=[float(u) for u in uncertainties_us],
            speed_mps=speed,
        )
    coarse = cfg.grid_coarse_m
    nx = max(1, math.ceil((area.x_max - area.x_min) / coarse))
    ny = max(1, math.ceil((area.y_max - area.y_min) / coarse))
    cells = [(ix, iy) for iy in range(ny) for ix in range(nx)]
    expand = _half_diagonal_us(coarse, speed)
    xs, ys = _cell_centers(cells, 0, coarse, area)
    feasible, emission, rms = _eval_points(
        xs, ys, stations, list(arrivals_us), list(uncertainties_us), speed, slack, expand
    )
    if not feasible.any():
        return Localization(
            feasible=False,
            reason="no_feasible_cell_in_area",
            grid_resolution_m=coarse,
            coarse_cells=len(cells),
            geometry=geometry,
            station_points=stations,
            arrivals_us=[float(a) for a in arrivals_us],
            uncertainties_us=[float(u) for u in uncertainties_us],
            speed_mps=speed,
        )
    current = [cells[i] for i in np.nonzero(feasible)[0]]
    current_emission = [float(emission[i]) for i in np.nonzero(feasible)[0]]
    current_rms = [float(rms[i]) for i in np.nonzero(feasible)[0]]
    level = 0
    warnings: list[str] = []
    while level < cfg.grid_refine_levels:
        if len(current) > cfg.grid_max_leaf_cells:
            warnings.append(
                f"refinement_stopped_at_level_{level}: {len(current)} cells exceed "
                f"grid_max_leaf_cells={cfg.grid_max_leaf_cells}; precision is "
                f"{coarse / (2**level):.2f} m, coarser than requested "
                f"{coarse / (2**cfg.grid_refine_levels):.2f} m"
            )
            break
        fine: list[tuple[int, int]] = []
        fine_em: list[float] = []
        fine_rms: list[float] = []
        res = coarse / (2 ** (level + 1))
        half_extent = _half_diagonal_us(res, speed)
        for ix, iy in current:
            children = [
                (2 * ix, 2 * iy),
                (2 * ix + 1, 2 * iy),
                (2 * ix, 2 * iy + 1),
                (2 * ix + 1, 2 * iy + 1),
            ]
            children = [
                (cx, cy)
                for cx, cy in children
                if area.x_min + cx * res < area.x_max and area.y_min + cy * res < area.y_max
            ]
            cxs, cys = _cell_centers(children, level + 1, coarse, area)
            cf, ce, cr = _eval_points(
                cxs, cys, stations, list(arrivals_us), list(uncertainties_us), speed, slack, half_extent
            )
            for k in np.nonzero(cf)[0]:
                fine.append(children[int(k)])
                fine_em.append(float(ce[k]))
                fine_rms.append(float(cr[k]))
        if not fine:
            warnings.append(f"no_feasible_child_at_level_{level + 1}; keeping level {level} cells")
            break
        current, current_emission, current_rms = fine, fine_em, fine_rms
        level += 1

    res = coarse / (2**level)
    cells_by_index = {
        cell: GridCell(
            ix=cell[0],
            iy=cell[1],
            level=level,
            res_m=res,
            x_min=area.x_min + cell[0] * res,
            x_max=min(area.x_min + (cell[0] + 1) * res, area.x_max),
            y_min=area.y_min + cell[1] * res,
            y_max=min(area.y_min + (cell[1] + 1) * res, area.y_max),
            rms_us=current_rms[i],
            emission_us=current_emission[i],
        )
        for i, cell in enumerate(current)
    }
    # Connected components on 8-neighbourhood.
    seen: set[tuple[int, int]] = set()
    components: list[list[tuple[int, int]]] = []
    for cell in cells_by_index:
        if cell in seen:
            continue
        stack = [cell]
        seen.add(cell)
        comp = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            ix, iy = cur
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nb = (ix + dx, iy + dy)
                    if nb in cells_by_index and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
        components.append(comp)

    max_u = max(uncertainties_us) if uncertainties_us else 0.0
    regions: list[Region] = []
    for idx, comp in enumerate(components):
        comp_cells = [cells_by_index[c] for c in comp]
        x_min = min(c.x_min for c in comp_cells)
        x_max = max(c.x_max for c in comp_cells)
        y_min = min(c.y_min for c in comp_cells)
        y_max = max(c.y_max for c in comp_cells)
        area_m2 = sum(
            (c.x_max - c.x_min) * (c.y_max - c.y_min) for c in comp_cells
        )
        best = min(comp_cells, key=lambda c: (c.rms_us, c.ix, c.iy))
        cx = (best.x_min + best.x_max) / 2.0
        cy = (best.y_min + best.y_max) / 2.0
        em_lo = min(c.emission_us for c in comp_cells) - (max_u + slack)
        em_hi = max(c.emission_us for c in comp_cells) + (max_u + slack)
        extent = max(x_max - x_min, y_max - y_min)
        truncated = len(comp_cells) > cfg.max_returned_cells_per_region
        regions.append(
            Region(
                region_id=f"R{idx + 1}",
                cells=comp_cells,
                bounds={
                    "x_min": round(x_min, 3),
                    "x_max": round(x_max, 3),
                    "y_min": round(y_min, 3),
                    "y_max": round(y_max, 3),
                },
                area_m2=round(area_m2, 3),
                representative={
                    "x": round(cx, 3),
                    "y": round(cy, 3),
                    "rms_residual_us": round(best.rms_us, 3),
                },
                emission_us=best.emission_us,
                emission_min_us=int(round(em_lo)),
                emission_max_us=int(round(em_hi)),
                extent_m=round(extent, 3),
                cells_truncated=truncated,
            )
        )
    regions.sort(key=lambda r: (r.area_m2, r.region_id))
    for i, region in enumerate(regions):
        region.region_id = f"R{i + 1}"

    position_certain = (
        len(regions) == 1
        and regions[0].extent_m <= cfg.definite_region_max_extent_m
        and not geometry["collinear"]
        and len(stations) >= 3
    )
    if geometry["collinear"]:
        warnings.append(
            "stations_collinear: source position is mirror-ambiguous across the station line; "
            "no unique coordinate is claimed"
        )
    if len(regions) > 1:
        warnings.append(
            f"{len(regions)} disconnected feasible regions; all are returned and no single "
            "position is claimed"
        )
    if not position_certain and not warnings:
        warnings.append(
            "feasible region exceeds the definite-position extent; reporting candidate area only"
        )
    leaf_total = len(cells_by_index)
    return Localization(
        feasible=True,
        reason=None,
        regions=regions,
        grid_resolution_m=res,
        coarse_cells=len(cells),
        leaf_cells=leaf_total,
        geometry=geometry,
        warnings=warnings,
        position_certain=position_certain,
        station_points=stations,
        arrivals_us=[float(a) for a in arrivals_us],
        uncertainties_us=[float(u) for u in uncertainties_us],
        speed_mps=speed,
    )
