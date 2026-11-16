"""Regional activity-rate comparability analysis.

Raw call counts from two regions cannot be compared directly: night-time
shutdowns, habitat composition and overlapping stations all change the
*opportunity to hear*.  This module turns the evidence the service already
holds (station progress declarations, calibration models, localised events)
into an analysis product that reports, on a comparable sample only:

* **coverage facts** per analysis grid cell x period bin x habitat, with the
  online and calibration status of every contributing station;
* **event contributions** in which one cross-station event contributes exactly
  one unit in total, spread over all of its candidate grid cells when the
  position is uncertain;
* **activity rates** (contributions per covered cell-hour) per region and
  stratum, never an estimate of the true population;
* a **common-coverage comparison** (most conservative: only grid-periods fully
  covered in *both* regions, weighted) and, when a pre-published reference
  composition is supplied, a **standardised** regional difference with its
  weights, variance and every empty stratum disclosed.

Two rules are never broken:

* no coverage on one side means *unknown* — a stratum is listed as
  incomparable/empty, and other strata are never used to fill the gap with a
  zero;
* a report is frozen at a knowledge cutoff (``as_of``): late shutdown or
  calibration information produces a *new* report at a new cutoff, and every
  old report can be re-derived exactly from its cutoff.

All computation is deterministic given the cutoff and the report
configuration; ``report_hash`` commits to the inputs and the result.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from . import calibration as calib
from .config import Settings
from .models import (
    AnalysisRegion,
    CalibrationModel,
    Event,
    HabitatMap,
    Observation,
    ReferenceComposition,
    Station,
    StationProgress,
)
from .pipeline import _fit_from_model
from .timeutil import iso, parse_local_us

UNCLASSIFIED_HABITAT = "unclassified"


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ReportConfig:
    region_a_id: str
    region_b_id: str
    habitat_map_id: str
    reference_composition_id: str | None
    class_label: str | None
    window_start_us: int
    window_end_us: int
    grid_cell_m: float
    bin_seconds: int
    day_start_hour: int
    night_start_hour: int
    methods: tuple[str, ...] = ("common_coverage", "standardized")

    def resolved(self) -> dict:
        return {
            "region_a_id": self.region_a_id,
            "region_b_id": self.region_b_id,
            "habitat_map_id": self.habitat_map_id,
            "reference_composition_id": self.reference_composition_id,
            "class_label": self.class_label,
            "window": {"start": iso(self.window_start_us), "end": iso(self.window_end_us)},
            "grid_cell_m": self.grid_cell_m,
            "bin_seconds": self.bin_seconds,
            "period_classes": {
                "day_start_hour": self.day_start_hour,
                "night_start_hour": self.night_start_hour,
            },
            "methods": list(self.methods),
        }


@dataclass(frozen=True)
class Grid:
    x_min: float
    y_min: float
    cell_m: float

    def cell_of(self, x: float, y: float) -> tuple[int, int]:
        return (
            math.floor((x - self.x_min) / self.cell_m),
            math.floor((y - self.y_min) / self.cell_m),
        )

    def center(self, ix: int, iy: int) -> tuple[float, float]:
        return (
            self.x_min + (ix + 0.5) * self.cell_m,
            self.y_min + (iy + 0.5) * self.cell_m,
        )


@dataclass
class Bin:
    index: int
    start_us: int
    end_us: int
    period_class: str

    @property
    def seconds(self) -> float:
        return (self.end_us - self.start_us) / 1e6


def make_bins(cfg: ReportConfig) -> list[Bin]:
    """Tile the study window and label each bin with its period class.

    The window length must be an exact multiple of the bin size so every bin
    is comparable in duration; the period class is decided by the UTC hour of
    the bin midpoint.
    """
    span_us = cfg.window_end_us - cfg.window_start_us
    bin_us = cfg.bin_seconds * 1_000_000
    if span_us <= 0 or span_us % bin_us != 0:
        raise ValueError(
            "window length must be a positive exact multiple of bin_seconds "
            f"(got {(span_us / 1e6):.0f}s vs bin {cfg.bin_seconds}s)"
        )
    from datetime import datetime, timezone

    bins: list[Bin] = []
    for k in range(span_us // bin_us):
        start = cfg.window_start_us + k * bin_us
        end = start + bin_us
        mid = datetime.fromtimestamp((start + end) / 2 / 1e6, tz=timezone.utc)
        hour = mid.hour
        if cfg.day_start_hour <= cfg.night_start_hour:
            is_day = cfg.day_start_hour <= hour < cfg.night_start_hour
        else:  # wrap-around definition, e.g. day from 20:00 to 08:00
            is_day = hour >= cfg.day_start_hour or hour < cfg.night_start_hour
        bins.append(Bin(k, start, end, "day" if is_day else "night"))
    return bins


# --------------------------------------------------------------------------- #
# Interval algebra (reference-time, seconds as floats)
# --------------------------------------------------------------------------- #
Interval = tuple[float, float]


def _union(intervals: list[Interval]) -> list[Interval]:
    out: list[Interval] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


def _subtract(intervals: list[Interval], holes: list[Interval]) -> list[Interval]:
    out = list(intervals)
    for hs, he in holes:
        nxt: list[Interval] = []
        for start, end in out:
            if he <= start or hs >= end:
                nxt.append((start, end))
                continue
            if start < hs:
                nxt.append((start, hs))
            if he < end:
                nxt.append((he, end))
        out = nxt
    return out


def _measure(intervals: list[Interval], start: float, end: float) -> float:
    return sum(
        min(e, end) - max(s, start)
        for s, e in intervals
        if min(e, end) > max(s, start)
    )


# --------------------------------------------------------------------------- #
# Coverage facts
# --------------------------------------------------------------------------- #
@dataclass
class StationCoverage:
    station_id: str
    intervals_s: list[Interval]  # reference-time seconds
    notes: list[dict] = field(default_factory=list)


def _latest_valid_calibrations(
    db: Session, as_of_us: int
) -> dict[tuple[str, str], CalibrationModel]:
    """The newest valid calibration per boot known at the cutoff."""
    rows = list(
        db.execute(
            select(CalibrationModel).where(
                CalibrationModel.valid.is_(True),
                CalibrationModel.created_at_us <= as_of_us,
            )
        ).scalars()
    )
    best: dict[tuple[str, str], CalibrationModel] = {}
    for row in rows:
        key = (row.station_id, row.boot_id)
        if key not in best or row.created_at_us > best[key].created_at_us:
            best[key] = row
    return best


def station_coverage(
    db: Session, as_of_us: int, cfg: Settings
) -> dict[str, StationCoverage]:
    """Reconstruct each station's listening intervals as known at the cutoff.

    A station covers reference time only where it (a) has a valid calibration
    for the boot (otherwise local readings cannot be placed in reference time
    at all) and (b) has declared, through a progress report, that its upload is
    complete through that instant.  Declared gaps are subtracted.  Elapsed
    wall-clock time never counts as coverage.
    """
    calibrations = _latest_valid_calibrations(db, as_of_us)
    fits = {key: _fit_from_model(model, cfg) for key, model in calibrations.items()}

    # Earliest uploaded record per boot: the earliest instant the station can
    # evidence it was listening.  Anything earlier is unknowable.
    first_local: dict[tuple[str, str], int] = {}
    for station_id, boot_id, local_us in db.execute(
        select(
            Observation.station_id,
            Observation.boot_id,
            Observation.local_capture_us,
        ).where(Observation.ingest_at_us <= as_of_us)
    ):
        key = (station_id, boot_id)
        first_local[key] = min(first_local.get(key, local_us), local_us)

    reports = list(
        db.execute(
            select(StationProgress)
            .where(StationProgress.received_at_us <= as_of_us)
            .order_by(StationProgress.id)
        ).scalars()
    )
    through_local: dict[tuple[str, str], int] = {}
    unmappable_reports: dict[tuple[str, str], int] = {}
    gaps: dict[tuple[str, str], list[dict]] = {}
    for rep in reports:
        key = (rep.station_id, rep.boot_id)
        fit = fits.get(key)
        if fit is not None:
            mapped = calib.apply_calibration(fit, rep.declared_local_us, cfg)
            if mapped.reference_arrival_us is not None:
                # Keep the local reading whose *mapped* instant is latest; an
                # individual unmappable report never erases earlier evidence.
                current = through_local.get(key)
                if current is None or mapped.reference_arrival_us > current[1]:
                    through_local[key] = (rep.declared_local_us, mapped.reference_arrival_us)
            else:
                unmappable_reports[key] = unmappable_reports.get(key, 0) + 1
        for gap in rep.gaps_json or []:
            gaps.setdefault(key, []).append(gap)

    boots = sorted(set(first_local) | set(through_local) | set(unmappable_reports))
    out: dict[str, StationCoverage] = {}
    for station_id, boot_id in boots:
        cov = out.setdefault(station_id, StationCoverage(station_id, []))
        fit = fits.get((station_id, boot_id))
        model = calibrations.get((station_id, boot_id))
        if fit is None or model is None:
            cov.notes.append(
                {
                    "boot_id": boot_id,
                    "status": "uncalibrated",
                    "reason": "no valid calibration known at the cutoff; "
                    "coverage of this boot is unknown, not zero",
                }
            )
            continue
        if (station_id, boot_id) not in through_local:
            note = {
                "boot_id": boot_id,
                "status": "no_progress",
                "reason": "no progress report received before the cutoff",
            }
            if unmappable_reports.get((station_id, boot_id)):
                note = {
                    "boot_id": boot_id,
                    "status": "unmappable_progress",
                    "reason": "progress reports exist but none of their declared "
                    "instants maps into reference time through the valid calibration",
                }
            cov.notes.append(note)
            continue
        declared_local, through_ref_us = through_local[(station_id, boot_id)]
        first = first_local.get((station_id, boot_id), declared_local)
        start_res = calib.apply_calibration(fit, first, cfg)
        start_ref = (
            start_res.reference_arrival_us
            if start_res.reference_arrival_us is not None
            else through_ref_us
        )
        intervals: list[Interval] = [(start_ref / 1e6, through_ref_us / 1e6)]
        holes: list[Interval] = []
        gap_notes = []
        for gap in gaps.get((station_id, boot_id), []):
            try:
                g_from = calib.apply_calibration(
                    fit, parse_local_us(gap["from_local_at"]), cfg
                )
                g_to = calib.apply_calibration(
                    fit, parse_local_us(gap["through_local_at"]), cfg
                )
            except (KeyError, ValueError) as exc:
                gap_notes.append({"gap": gap, "status": "ignored", "reason": str(exc)})
                continue
            if g_from.reference_arrival_us is None or g_to.reference_arrival_us is None:
                gap_notes.append({"gap": gap, "status": "unmappable"})
                continue
            holes.append(
                (g_from.reference_arrival_us / 1e6, g_to.reference_arrival_us / 1e6)
            )
            gap_notes.append(
                {
                    "gap": {
                        "from": iso(g_from.reference_arrival_us),
                        "through": iso(g_to.reference_arrival_us),
                        "reason": gap.get("reason"),
                    },
                    "status": "applied",
                }
            )
        intervals = _subtract(_union(intervals), holes)
        cov.intervals_s.extend(intervals)
        cov.intervals_s = _union(cov.intervals_s)
        cov.notes.append(
            {
                "boot_id": boot_id,
                "status": "online_with_gaps" if holes else "online",
                "calibration_model_id": model.id,
                "calibration_version_hash": model.version_hash,
                "unmappable_progress_reports": unmappable_reports.get(
                    (station_id, boot_id), 0
                ),
                "coverage": [
                    {"from": iso(int(s * 1e6)), "through": iso(int(e * 1e6))}
                    for s, e in intervals
                ],
                "gaps": gap_notes,
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Event contributions
# --------------------------------------------------------------------------- #
def live_events_as_of(db: Session, as_of_us: int, class_label: str | None) -> list[Event]:
    """Events known and still live at the cutoff.

    Supersession is judged by the *creation time* of the superseding event,
    never by the mutable ``status`` column: an event superseded only by a
    younger event created after the cutoff was still live at the cutoff, and
    an old report must re-derive exactly that world.
    """
    rows = list(
        db.execute(select(Event).where(Event.created_at_us <= as_of_us)).scalars()
    )
    by_id = {e.id: e for e in rows}
    out = []
    for ev in rows:
        superseder = by_id.get(ev.superseded_by_event_id or -1)
        if superseder is not None and superseder.created_at_us <= as_of_us:
            continue
        if ev.status == "superseded" and ev.superseded_by_event_id is None:
            continue  # defensive: superseded without a named successor
        if class_label and ev.class_label != class_label:
            continue
        out.append(ev)
    out.sort(key=lambda e: (e.emission_us, e.id))
    return out


def event_cell_weights(event: Event, grid: Grid) -> dict[tuple[int, int], float]:
    """Spread one event over its candidate grid cells.

    The event contributes exactly one unit in total; when the position is
    uncertain every candidate analysis cell carries a share proportional to
    the number of localisation cells falling inside it.
    """
    counts: dict[tuple[int, int], int] = {}
    total = 0
    for region in event.regions_json or []:
        for x0, y0, x1, y1 in region.get("cells", []):
            cell = grid.cell_of((x0 + x1) / 2.0, (y0 + y1) / 2.0)
            counts[cell] = counts.get(cell, 0) + 1
            total += 1
    if total == 0:
        # A localisation always carries cells; this is a defensive fallback so
        # an event can never vanish from the accounting.
        cell = grid.cell_of(event.x, event.y)
        return {cell: 1.0}
    return {cell: n / total for cell, n in counts.items()}


# --------------------------------------------------------------------------- #
# Report generation
# --------------------------------------------------------------------------- #
def _load_definitions(db: Session, cfg: ReportConfig):
    region_a = db.get(AnalysisRegion, cfg.region_a_id)
    region_b = db.get(AnalysisRegion, cfg.region_b_id)
    habitat_map = db.get(HabitatMap, cfg.habitat_map_id)
    reference = (
        db.get(ReferenceComposition, cfg.reference_composition_id)
        if cfg.reference_composition_id
        else None
    )
    missing = [
        name
        for name, obj in (
            ("region_a", region_a),
            ("region_b", region_b),
            ("habitat_map", habitat_map),
        )
        if obj is None
    ]
    if missing:
        raise KeyError(", ".join(missing))
    for name, obj in (("region_a", region_a), ("region_b", region_b), ("habitat_map", habitat_map)):
        if abs(obj.grid_cell_size_m - cfg.grid_cell_m) > 1e-9:
            raise ValueError(
                f"{name} was published on a {obj.grid_cell_size_m} m grid but the report "
                f"uses {cfg.grid_cell_m} m; re-publish on the report grid"
            )
    cells_a = {tuple(c) for c in region_a.cells_json}
    cells_b = {tuple(c) for c in region_b.cells_json}
    overlap = cells_a & cells_b
    if overlap:
        raise ValueError(
            f"regions overlap on {len(overlap)} cells (e.g. {sorted(overlap)[0]}); "
            "a cell cannot belong to both sides of a comparison"
        )
    return region_a, region_b, habitat_map, reference


def generate_report(
    db: Session, cfg: ReportConfig, as_of_us: int, settings: Settings
) -> dict:
    """Compute the full comparability report.  Pure in ``as_of_us``: nothing
    in the result depends on wall-clock time."""
    region_a, region_b, habitat_map, reference = _load_definitions(db, cfg)
    grid = Grid(settings.area_x_min, settings.area_y_min, cfg.grid_cell_m)
    bins = make_bins(cfg)
    cells_a = {tuple(c) for c in region_a.cells_json}
    cells_b = {tuple(c) for c in region_b.cells_json}
    habitat_of = {
        (c["ix"], c["iy"]): c["habitat"] for c in habitat_map.cells_json
    }
    region_of = {**{c: "a" for c in cells_a}, **{c: "b" for c in cells_b}}

    stations = {
        s.id: s for s in db.execute(select(Station)).scalars()
    }
    coverage = station_coverage(db, as_of_us, settings)

    # ---- coverage facts per cell x bin ----------------------------------- #
    tol = settings.analysis_coverage_tolerance_s
    cell_stations: dict[tuple[int, int], list[str]] = {}
    for cell in sorted(cells_a | cells_b):
        cx, cy = grid.center(*cell)
        hearers = [
            sid
            for sid, st in stations.items()
            if math.hypot(st.x - cx, st.y - cy) <= st.hearing_radius_m
        ]
        cell_stations[cell] = hearers

    facts: list[dict] = []
    # effort[region][(habitat, period_class)] -> covered cell-seconds
    effort: dict[str, dict[tuple[str, str], float]] = {"a": {}, "b": {}}
    full_effort: dict[str, dict[tuple[str, str], float]] = {"a": {}, "b": {}}
    fully_covered_facts: dict[tuple[tuple[int, int], int], bool] = {}
    for cell in sorted(cells_a | cells_b):
        habitat = habitat_of.get(cell, UNCLASSIFIED_HABITAT)
        region = region_of[cell]
        hearers = cell_stations[cell]
        for b in bins:
            per_station = []
            union_intervals: list[Interval] = []
            for sid in hearers:
                cov = coverage.get(sid)
                seconds = 0.0
                if cov is not None:
                    seconds = _measure(cov.intervals_s, b.start_us / 1e6, b.end_us / 1e6)
                    union_intervals.extend(cov.intervals_s)
                per_station.append(
                    {
                        "station_id": sid,
                        "covered_seconds": round(seconds, 6),
                        "status": (
                            "online_calibrated"
                            if seconds > 0
                            else ("no_coverage" if cov is not None and cov.intervals_s else "unknown")
                        ),
                    }
                )
            # Overlapping stations share the same opportunity to hear: the
            # union of their intervals, never the sum, is the effort.
            covered = _measure(_union(union_intervals), b.start_us / 1e6, b.end_us / 1e6)
            covered = min(covered, b.seconds)
            # Gap endpoints are mapped through a calibration with a few
            # seconds of uncertainty; slivers at that scale are noise, not
            # coverage, and a bin short by that scale is still complete.
            if covered <= tol:
                covered = 0.0
            full = covered >= b.seconds - tol
            fully_covered_facts[(cell, b.index)] = full
            key = (habitat, b.period_class)
            effort[region][key] = effort[region].get(key, 0.0) + covered
            if full:
                full_effort[region][key] = (
                    full_effort[region].get(key, 0.0) + b.seconds
                )
            facts.append(
                {
                    "cell": list(cell),
                    "habitat": habitat,
                    "region": region,
                    "bin": b.index,
                    "period_class": b.period_class,
                    "bin_seconds": b.seconds,
                    "covered_seconds": round(covered, 6),
                    "fully_covered": full,
                    "stations": per_station,
                }
            )

    # ---- event contributions --------------------------------------------- #
    events = live_events_as_of(db, as_of_us, cfg.class_label)
    counts: dict[str, dict[tuple[str, str], float]] = {"a": {}, "b": {}}
    full_counts: dict[str, dict[tuple[str, str], float]] = {"a": {}, "b": {}}
    contribution_items: list[dict] = []
    outside_window = 0
    for ev in events:
        b = next(
            (b for b in bins if b.start_us <= ev.emission_us < b.end_us), None
        )
        if b is None:
            outside_window += 1
            continue
        weights = event_cell_weights(ev, grid)
        cells_out = []
        outside_weight = 0.0
        for (ix, iy), w in sorted(weights.items()):
            habitat = habitat_of.get((ix, iy), UNCLASSIFIED_HABITAT)
            region = region_of.get((ix, iy))
            if region is None:
                outside_weight += w
                continue
            key = (habitat, b.period_class)
            counts[region][key] = counts[region].get(key, 0.0) + w
            if fully_covered_facts.get(((ix, iy), b.index), False):
                full_counts[region][key] = full_counts[region].get(key, 0.0) + w
            cells_out.append(
                {
                    "cell": [ix, iy],
                    "weight": round(w, 6),
                    "habitat": habitat,
                    "region": region,
                }
            )
        contribution_items.append(
            {
                "event_id": ev.id,
                "class_label": ev.class_label,
                "emission": iso(ev.emission_us),
                "bin": b.index,
                "period_class": b.period_class,
                "cells": cells_out,
                "outside_regions_weight": round(outside_weight, 6),
            }
        )

    # ---- strata table ----------------------------------------------------- #
    habitats = sorted(
        {habitat_of.get(c, UNCLASSIFIED_HABITAT) for c in (cells_a | cells_b)}
        | {e["habitat"] for e in (reference.entries_json if reference else [])}
    )
    period_classes = sorted({b.period_class for b in bins} | {"day", "night"})
    strata: list[dict] = []
    stratum_rate: dict[tuple[str, str], dict] = {}
    for habitat in habitats:
        for period_class in period_classes:
            key = (habitat, period_class)
            row = {"habitat": habitat, "period_class": period_class, "regions": {}}
            for region, region_id in (("a", cfg.region_a_id), ("b", cfg.region_b_id)):
                eff = effort[region].get(key, 0.0) / 3600.0
                full_eff = full_effort[region].get(key, 0.0) / 3600.0
                cnt = counts[region].get(key, 0.0)
                full_cnt = full_counts[region].get(key, 0.0)
                rate = cnt / eff if eff > 0 else None
                var_rate = cnt / (eff * eff) if eff > 0 else None
                full_rate = full_cnt / full_eff if full_eff > 0 else None
                full_var = full_cnt / (full_eff * full_eff) if full_eff > 0 else None
                row["regions"][region] = {
                    "region_id": region_id,
                    "effort_cell_hours": round(eff, 6),
                    "fully_covered_effort_cell_hours": round(full_eff, 6),
                    "contributions": round(cnt, 6),
                    "contributions_fully_covered": round(full_cnt, 6),
                    "rate_per_cell_hour": round(rate, 9) if rate is not None else None,
                    "variance_of_rate": round(var_rate, 12) if var_rate is not None else None,
                    "rate_fully_covered": round(full_rate, 9) if full_rate is not None else None,
                    "variance_fully_covered": round(full_var, 12) if full_var is not None else None,
                    "status": (
                        "covered"
                        if eff > 0
                        else (
                            "no_coverage_with_contributions"
                            if cnt > 0
                            else "no_coverage"
                        )
                    ),
                }
            strata.append(row)
            stratum_rate[key] = row

    result: dict = {
        "as_of": iso(as_of_us),
        "config": cfg.resolved(),
        "regions": {
            "a": {"id": region_a.id, "name": region_a.name, "cell_count": len(cells_a)},
            "b": {"id": region_b.id, "name": region_b.name, "cell_count": len(cells_b)},
        },
        "bins": [
            {
                "index": b.index,
                "start": iso(b.start_us),
                "end": iso(b.end_us),
                "seconds": b.seconds,
                "period_class": b.period_class,
            }
            for b in bins
        ],
        "coverage_facts": {
            "grid_cell_m": cfg.grid_cell_m,
            "station_notes": [
                {"station_id": cov.station_id, "notes": cov.notes}
                for cov in sorted(
                    coverage.values(), key=lambda c: c.station_id
                )
                if cov.notes
            ],
            "facts": facts,
        },
        "events": {
            "included": len(contribution_items),
            "outside_window": outside_window,
            "contributions": contribution_items,
        },
        "strata": strata,
    }

    # ---- naive contrast (deliberately not comparable) -------------------- #
    naive_counts = {
        region: round(sum(counts[region].values()), 6) for region in ("a", "b")
    }
    result["naive_counts"] = {
        "counts": naive_counts,
        "difference": round(naive_counts["a"] - naive_counts["b"], 6),
        "warning": (
            "raw counts are NOT comparable: coverage differs between regions; "
            "use common_coverage or standardized for any conclusion"
        ),
    }

    if "common_coverage" in cfg.methods:
        result["common_coverage"] = _common_coverage(strata, cfg)
    if "standardized" in cfg.methods:
        result["standardized"] = _standardized(strata, reference, result.get("common_coverage"))
    return result


def _common_coverage(strata: list[dict], cfg: ReportConfig) -> dict:
    """Weighted comparison over strata fully covered in BOTH regions.

    Only grid-periods with complete coverage enter; a stratum covered on one
    side only is listed as incomparable and never contributes a zero.
    """
    common, incomparable = [], []
    for row in strata:
        a, b = row["regions"]["a"], row["regions"]["b"]
        key = {"habitat": row["habitat"], "period_class": row["period_class"]}
        if a["fully_covered_effort_cell_hours"] > 0 and b["fully_covered_effort_cell_hours"] > 0:
            weight = min(
                a["fully_covered_effort_cell_hours"],
                b["fully_covered_effort_cell_hours"],
            )
            common.append(
                {
                    **key,
                    "weight": round(weight, 6),
                    "rate_a": a["rate_fully_covered"],
                    "rate_b": b["rate_fully_covered"],
                    "rate_difference": round(
                        a["rate_fully_covered"] - b["rate_fully_covered"], 9
                    ),
                    "variance": round(
                        (a["variance_fully_covered"] or 0.0)
                        + (b["variance_fully_covered"] or 0.0),
                        12,
                    ),
                }
            )
        elif a["fully_covered_effort_cell_hours"] > 0 or b["fully_covered_effort_cell_hours"] > 0:
            incomparable.append(
                {
                    **key,
                    "covered_side": "a" if a["fully_covered_effort_cell_hours"] > 0 else "b",
                    "unknown_side": "b" if a["fully_covered_effort_cell_hours"] > 0 else "a",
                    "reason": "no full coverage on one side; the rate there is unknown, not zero",
                }
            )
    total_w = sum(c["weight"] for c in common)
    if common and total_w > 0:
        difference = sum(c["weight"] * c["rate_difference"] for c in common) / total_w
        variance = sum(c["weight"] ** 2 * c["variance"] for c in common) / total_w**2
        difference_out: float | None = round(difference, 9)
        se_out: float | None = round(math.sqrt(variance), 9)
    else:
        difference_out = se_out = None
    return {
        "method": "common_coverage",
        "description": (
            "rates compared only on grid-periods fully covered in both regions, "
            "weighted by the smaller fully-covered effort of the two sides"
        ),
        "strata": common,
        "difference": difference_out,
        "standard_error": se_out,
        "incomparable_strata": incomparable,
        "note": (
            "no common fully-covered stratum exists; the regions cannot be "
            "compared on this sample"
            if not common
            else None
        ),
    }


def _standardized(
    strata: list[dict], reference: ReferenceComposition | None, common_coverage: dict | None
) -> dict:
    """Standardise stratum rates onto the pre-published reference composition.

    Uses every covered grid-period (broader than the common-coverage sample).
    Strata named in the reference composition but without coverage on either
    side are empty: they are disclosed with their reference weight and never
    filled with a zero borrowed from another stratum.
    """
    if reference is None:
        return {
            "method": "standardized",
            "status": "skipped",
            "reason": "no reference composition configured for this report",
            "common_coverage_crosscheck": common_coverage,
        }
    weights: dict[tuple[str, str], float] = {}
    for entry in reference.entries_json:
        key = (entry["habitat"], entry["period_class"])
        weights[key] = weights.get(key, 0.0) + float(entry["weight"])
    by_key = {
        (row["habitat"], row["period_class"]): row for row in strata
    }
    used, empty, outside_reference = [], [], []
    for key, w in sorted(weights.items()):
        if w <= 0:
            continue
        row = by_key.get(key)
        a = row["regions"]["a"] if row else None
        b = row["regions"]["b"] if row else None
        eff_a = a["effort_cell_hours"] if a else 0.0
        eff_b = b["effort_cell_hours"] if b else 0.0
        entry = {"habitat": key[0], "period_class": key[1], "reference_weight": w}
        if eff_a > 0 and eff_b > 0:
            used.append(
                {
                    **entry,
                    "rate_a": a["rate_per_cell_hour"],
                    "rate_b": b["rate_per_cell_hour"],
                    "variance": round(
                        (a["variance_of_rate"] or 0.0) + (b["variance_of_rate"] or 0.0), 12
                    ),
                }
            )
        else:
            empty.append(
                {
                    **entry,
                    "missing_side": (
                        "both"
                        if eff_a == 0 and eff_b == 0
                        else ("a" if eff_a == 0 else "b")
                    ),
                    "reason": "no coverage on at least one side; treated as unknown, never zero-filled",
                }
            )
    for row in strata:
        key = (row["habitat"], row["period_class"])
        if weights.get(key, 0.0) > 0:
            continue
        if (
            row["regions"]["a"]["effort_cell_hours"] > 0
            or row["regions"]["b"]["effort_cell_hours"] > 0
        ):
            outside_reference.append(
                {
                    "habitat": key[0],
                    "period_class": key[1],
                    "reason": "covered but absent from the reference composition; "
                    "excluded from standardisation",
                }
            )
    total_w = sum(u["reference_weight"] for u in used)
    all_w = sum(w for w in weights.values() if w > 0)
    if used and total_w > 0:
        difference = (
            sum(u["reference_weight"] * (u["rate_a"] - u["rate_b"]) for u in used) / total_w
        )
        variance = (
            sum(u["reference_weight"] ** 2 * u["variance"] for u in used) / total_w**2
        )
        std_a = sum(u["reference_weight"] * u["rate_a"] for u in used) / total_w
        std_b = sum(u["reference_weight"] * u["rate_b"] for u in used) / total_w
        difference_out: float | None = round(difference, 9)
        se_out: float | None = round(math.sqrt(variance), 9)
        std_a_out: float | None = round(std_a, 9)
        std_b_out: float | None = round(std_b, 9)
    else:
        difference_out = se_out = std_a_out = std_b_out = None
    return {
        "method": "standardized",
        "status": "done",
        "reference_composition_id": reference.id,
        "description": (
            "stratum rates standardised onto the pre-published reference "
            "composition over strata covered in both regions"
        ),
        "strata": used,
        "standardized_rate_a": std_a_out,
        "standardized_rate_b": std_b_out,
        "difference": difference_out,
        "standard_error": se_out,
        "reference_weight_total": round(all_w, 6),
        "compared_weight_mass": round(total_w, 6),
        "excluded_weight_mass": round(all_w - total_w, 6),
        "empty_strata": empty,
        "outside_reference_strata": outside_reference,
        "common_coverage_crosscheck": common_coverage,
    }


# --------------------------------------------------------------------------- #
# Hashing
# --------------------------------------------------------------------------- #
def canonical_hash(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def inputs_hash(cfg: ReportConfig) -> str:
    return canonical_hash({"config": cfg.resolved()})[:24]


def report_hash(result: dict) -> str:
    return canonical_hash({"result": result})[:24]
