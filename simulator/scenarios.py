"""Synthetic wetland scenarios for demonstration and validation.

All positions, sources and stations are fictional.  The model is deliberately
simple and documented so an independent checker can recompute arrivals:

* Flat 2-D plane in metres, constant propagation speed (343 m/s by default).
* Each station boot is one affine clock::

      local = boot_start_local + (1 + drift_ppm*1e-6) * (reference - boot_start_ref)

  ``boot_start_local`` is the station-clock counter at power-on and may jump
  arbitrarily between boots.  Drift differs between boots but stays tiny
  (tens of ppm), so a boot needs a piecewise-linear map only if the drift
  changes mid-boot — the default scenarios have one segment per boot.
* Calibration sources sit at known coordinates and emit at known *reference*
  instants.  Every station records each emission it can hear.
* Sound sources are birds/amphibians: a class label, a feature vector, and a
  list of emission instants.

The generator writes the station-side view (local readings, upload batches)
plus a ``truth.json`` that lets a checker recompute arrivals from first
principles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

SPEED_MPS = 343.0
REFERENCE_EPOCH = datetime(2026, 11, 14, 0, 0, 0, tzinfo=timezone.utc)
LOCAL_EPOCH = datetime(2026, 11, 1)


def us(dt: datetime) -> int:
    return int(round(dt.timestamp() * 1_000_000))


def ref_at(seconds: float) -> datetime:
    return REFERENCE_EPOCH + timedelta(seconds=seconds)


def ref_iso(seconds: float) -> str:
    return ref_at(seconds).isoformat(timespec="microseconds").replace("+00:00", "Z")


def local_iso(local_us: int) -> str:
    dt = LOCAL_EPOCH + timedelta(microseconds=local_us)
    return dt.isoformat(timespec="microseconds")


@dataclass
class Boot:
    """One power-on period of a station."""

    boot_id: str
    boot_start_ref_s: float
    boot_start_local_s: float
    drift_ppm: float
    reboot_reason: str | None = None

    def local_us(self, reference_arrival_us: int) -> int:
        a = 1.0 + self.drift_ppm * 1e-6
        return int(
            round(
                self.boot_start_local_s * 1e6
                + a * (reference_arrival_us - us(ref_at(self.boot_start_ref_s)))
            )
        )

    def truth(self) -> dict:
        a = 1.0 + self.drift_ppm * 1e-6
        b = self.boot_start_local_s * 1e6 - a * us(ref_at(self.boot_start_ref_s))
        return {
            "boot_id": self.boot_id,
            "boot_start_ref": ref_iso(self.boot_start_ref_s),
            "boot_start_local": local_iso(int(self.boot_start_local_s * 1e6)),
            "drift_ppm": self.drift_ppm,
            "slope": a,
            "intercept_us": b,
            "reboot_reason": self.reboot_reason,
        }

    def covers(self, reference_s: float) -> bool:
        return reference_s >= self.boot_start_ref_s


@dataclass
class StationSpec:
    id: str
    x: float
    y: float
    radius: float
    boots: list[Boot]
    upload_lag_s: tuple[float, float] = (30.0, 120.0)
    late: bool = False  # uploads only in the late phase
    note: str | None = None
    # Intervals (reference seconds) in which the recorder is actually on.
    # ``None`` means always on.  Offline intervals become declared coverage
    # gaps in the station's progress reports — either with the final wave or,
    # when ``gaps_late`` is set, only in a separate late progress report.
    online_windows_s: list[tuple[float, float]] | None = None
    gaps_late: bool = False

    def boot_at(self, reference_s: float) -> Boot:
        chosen = self.boots[0]
        for boot in self.boots:
            if boot.covers(reference_s) and boot.boot_start_ref_s >= chosen.boot_start_ref_s:
                chosen = boot
        return chosen

    def online_at(self, reference_s: float) -> bool:
        if self.online_windows_s is None:
            return True
        return any(start <= reference_s <= end for start, end in self.online_windows_s)

    def offline_gaps_s(self, lo_s: float, hi_s: float) -> list[tuple[float, float]]:
        """Offline intervals of this station clipped to ``[lo_s, hi_s]``."""
        if self.online_windows_s is None:
            return []
        gaps: list[tuple[float, float]] = []
        cursor = lo_s
        for start, end in sorted(self.online_windows_s):
            if end < lo_s or start > hi_s:
                continue
            if start > cursor:
                gaps.append((cursor, min(start, hi_s)))
            cursor = max(cursor, end)
        if cursor < hi_s:
            gaps.append((cursor, hi_s))
        return [(s, e) for s, e in gaps if e > s]


@dataclass
class SourceSpec:
    id: str
    class_label: str
    x: float
    y: float
    emissions_s: list[float]
    feature: list[float]
    temporal_sigma_ms: float = 0.0


@dataclass
class CalSourceSpec:
    id: str
    x: float
    y: float
    emissions_s: list[float]
    late_from_s: float | None = None  # emissions at/after this instant upload late


@dataclass
class Scenario:
    name: str
    title: str
    description: str
    stations: list[StationSpec]
    sources: list[SourceSpec]
    cal_sources: list[CalSourceSpec]
    area: dict[str, float]
    observation_uncertainty_ms: float = 25.0
    anchor_noise_ms: float = 4.0
    feature_noise: float = 0.015
    closure_margin_s: float = 180.0
    expected: dict = field(default_factory=dict)
    # Analysis-scenario definitions (regions, habitat, reference composition,
    # study window) published to the analysis API after the pipeline run.
    analysis: dict | None = None

    def timeline_end_s(self) -> float:
        sound_end = max((e for s in self.sources for e in s.emissions_s), default=600.0)
        cal_end = max((e for c in self.cal_sources for e in c.emissions_s), default=0.0)
        return max(sound_end, cal_end) + self.closure_margin_s


BOOT_BASE_LOCAL_S = 1_158_000.0  # ~2026-11-14 on the fictional station calendar


def _boots_standard() -> Boot:
    return Boot("b1", -3600.0, BOOT_BASE_LOCAL_S, 35.0)


def build_scenario(name: str) -> Scenario:
    builders = {
        "overlapping": _overlapping,
        "reboot": _reboot,
        "collinear": _collinear,
        "duplicates": _duplicates,
        "revision": _revision,
        "daynight": _daynight,
        "nightfailure": _nightfailure,
    }
    if name not in builders:
        raise KeyError(f"unknown scenario {name!r}; choose from {sorted(builders)}")
    return builders[name]()


def scenario_names() -> list[str]:
    return ["overlapping", "reboot", "collinear", "duplicates", "revision",
            "daynight", "nightfailure"]


# --------------------------------------------------------------------------- #
# 1. Overlapping dual sources
# --------------------------------------------------------------------------- #
def _overlapping() -> Scenario:
    cal = [
        CalSourceSpec("C1", 100.0, -150.0, [-2400.0 + 115.0 * i for i in range(37)]),
        CalSourceSpec("C2", 1650.0, 900.0, [-2280.0 + 110.0 * i for i in range(37)]),
    ]
    stations = [
        StationSpec("S1", 300.0, 100.0, 1700.0, [_boots_standard()], (25.0, 70.0)),
        StationSpec("S2", 1400.0, 150.0, 1700.0, [_boots_standard()], (40.0, 110.0)),
        StationSpec("S3", 1050.0, 950.0, 1700.0, [_boots_standard()], (55.0, 140.0)),
        StationSpec("S4", 550.0, 600.0, 1600.0, [_boots_standard()], (20.0, 60.0)),
    ]
    sources = [
        SourceSpec("A", "chirp", 760.0, 300.0, [120.0, 900.0, 1700.0], [1.0, 0.30, 0.20, 0.90]),
        SourceSpec("B", "chirp", 880.0, 480.0, [123.5, 905.0, 1695.0], [1.0, 0.42, 0.34, 0.86]),
    ]
    return Scenario(
        name="overlapping",
        title="Two chirp sources calling almost simultaneously",
        description=(
            "Two distinct chirp sources 200 m apart call a few seconds apart; S2 and S3 "
            "hear both. One animal heard by four stations must be one event, yet the two "
            "animals must stay two events even though their calls overlap in time. A "
            "station can contribute at most one record to each event."
        ),
        stations=stations,
        sources=sources,
        cal_sources=cal,
        area={"x_min": -300.0, "x_max": 2100.0, "y_min": -500.0, "y_max": 1300.0},
        expected={
            "events": 6,
            "events_with_4_stations": 6,
            "distinct_sources": 2,
            "note": "each source emits 3 calls; every call should be heard by all four stations",
        },
    )


# --------------------------------------------------------------------------- #
# 2. Reboot jump + late backfill + slow clock drift
# --------------------------------------------------------------------------- #
def _reboot() -> Scenario:
    cal = [
        CalSourceSpec("C1", 50.0, 1000.0, [-2400.0 + 100.0 * i for i in range(64)]),
        CalSourceSpec("C2", 1950.0, -250.0, [-2350.0 + 105.0 * i for i in range(62)]),
    ]
    stations = [
        StationSpec("S1", 200.0, 250.0, 2600.0, [_boots_standard()], (25.0, 70.0)),
        StationSpec(
            "S2",
            1250.0,
            100.0,
            1800.0,
            [
                Boot("b1", -3600.0, BOOT_BASE_LOCAL_S, 60.0),
                # Reboot at t=900 s: the clock jumps ~+1500 s and drifts the other way.
                Boot("b2", 900.0, BOOT_BASE_LOCAL_S + 900.0 + 1500.0, -45.0,
                     reboot_reason="power cycle; supercapacitor-backed RTC lost 1500 s"),
            ],
            (40.0, 100.0),
        ),
        StationSpec(
            "S3",
            1200.0,
            900.0,
            2600.0,
            [_boots_standard()],
            (1200.0, 1800.0),
            late=True,
            note="offline for 20 minutes, backfilled hours later",
        ),
        StationSpec("S4", 380.0, 880.0, 2600.0, [_boots_standard()], (20.0, 60.0)),
    ]
    sources = [
        SourceSpec("A", "whistle", 650.0, 250.0, [240.0, 700.0, 1500.0, 2400.0], [0.80, 0.60, 0.45]),
        SourceSpec("B", "whistle", 1150.0, 800.0, [1200.0, 3000.0], [0.62, 0.88, 0.30]),
    ]
    return Scenario(
        name="reboot",
        title="Station restart with a clock jump and a late backfill",
        description=(
            "S2 restarted at t=900 s; its clock jumped by 1500 s and its drift changed "
            "sign. S3 was offline and needs to upload records 20 minutes after the "
            "others. Arrival order and wall-clock elapsed time must not enter the "
            "association: only per-boot calibrated reference time counts, and a station "
            "without a complete progress report keeps its events provisional."
        ),
        stations=stations,
        sources=sources,
        cal_sources=cal,
        area={"x_min": -400.0, "x_max": 2200.0, "y_min": -600.0, "y_max": 1400.0},
        expected={
            "events": 6,
            "rebooted_stations": ["S2"],
            "late_station": "S3",
            "note": "S2's two boots must have separate calibration models",
        },
    )


# --------------------------------------------------------------------------- #
# 3. Collinear stations: genuine mirror ambiguity
# --------------------------------------------------------------------------- #
def _collinear() -> Scenario:
    cal = [
        CalSourceSpec("C1", -50.0, 1200.0, [-2100.0 + 120.0 * i for i in range(32)]),
        CalSourceSpec("C2", 2250.0, -200.0, [-2050.0 + 115.0 * i for i in range(32)]),
    ]
    stations = [
        StationSpec("S1", 150.0, 500.0, 1900.0, [_boots_standard()], (25.0, 70.0)),
        StationSpec("S2", 750.0, 500.0, 1900.0, [_boots_standard()], (40.0, 100.0)),
        StationSpec("S3", 1350.0, 500.0, 1900.0, [_boots_standard()], (50.0, 110.0)),
        StationSpec("S4", 1950.0, 500.0, 1900.0, [_boots_standard()], (30.0, 90.0)),
    ]
    sources = [
        SourceSpec("A", "croak", 900.0, 180.0, [150.0, 800.0, 1450.0], [0.40, 1.00, 0.50]),
        SourceSpec("B", "croak", 900.0, 820.0, [420.0, 1100.0, 1750.0], [0.72, 0.55, 0.80]),
    ]
    return Scenario(
        name="collinear",
        title="Collinear stations cannot resolve the mirror pair",
        description=(
            "All four stations lie exactly on y = 500 m. Reflection across that line "
            "leaves every propagation delay unchanged, so a call at (900, 180) is "
            "timing-indistinguishable from one at (900, 820). The service must return "
            "both disconnected feasible regions and must not report one coordinate. "
            "Counting is unaffected: each call is still one event."
        ),
        stations=stations,
        sources=sources,
        cal_sources=cal,
        area={"x_min": 0.0, "x_max": 2200.0, "y_min": -400.0, "y_max": 1400.0},
        expected={
            "events": 6,
            "collinear": True,
            "min_regions_max": 6,
            "mirror_pair_y": [180.0, 820.0],
            "note": "every event must expose at least two feasible regions",
        },
    )


# --------------------------------------------------------------------------- #
# 4. Retransmission: identical replay + content conflict
# --------------------------------------------------------------------------- #
def _duplicates() -> Scenario:
    cal = [
        CalSourceSpec("C1", -100.0, 400.0, [-1500.0 + 120.0 * i for i in range(30)]),
        CalSourceSpec("C2", 1700.0, 800.0, [-1450.0 + 115.0 * i for i in range(30)]),
    ]
    stations = [
        StationSpec("S1", 150.0, 150.0, 1700.0, [_boots_standard()], (25.0, 70.0)),
        StationSpec("S2", 1750.0, 200.0, 1700.0, [_boots_standard()], (40.0, 90.0)),
        StationSpec("S3", 1150.0, 1300.0, 1700.0, [_boots_standard()], (50.0, 120.0)),
        StationSpec("S4", 600.0, 900.0, 1500.0, [_boots_standard()], (30.0, 85.0)),
    ]
    sources = [
        SourceSpec("A", "drum", 650.0, 450.0, [200.0, 1200.0], [0.60, 0.30, 1.00]),
        SourceSpec("B", "drum", 900.0, 620.0, [1600.0], [0.62, 0.34, 1.00]),
    ]
    return Scenario(
        name="duplicates",
        title="Retransmitted batches: identical replay and a content conflict",
        description=(
            "After the normal uploads, station S1 re-sends its first batch byte-for-byte "
            "(must be counted as a duplicate and change nothing), then re-sends the "
            "record with source_seq 7 carrying a different uncertainty under the same "
            "sequence number (must land in the conflict table and leave the original "
            "untouched). Events and counts must be identical with and without the "
            "retransmission."
        ),
        stations=stations,
        sources=sources,
        cal_sources=cal,
        area={"x_min": -200.0, "x_max": 2000.0, "y_min": -200.0, "y_max": 1600.0},
        feature_noise=0.006,
        expected={
            "events": 3,
            "replay_batch_index": 0,
            "conflict_station": "S1",
            "conflict_uncertainty_ms": 45.0,
        },
    )


# --------------------------------------------------------------------------- #
# 5. Late report + calibration revision => event revisions
# --------------------------------------------------------------------------- #
def _revision() -> Scenario:
    cal = [
        CalSourceSpec("C1", -100.0, 400.0, [-1800.0 + 110.0 * i for i in range(40)]),
        # Emissions of C2 after t=600 arrive only in the late phase, so S1's and
        # S2's calibration gains anchors between the two runs.
        CalSourceSpec("C2", 1700.0, 800.0, [-1780.0 + 105.0 * i for i in range(40)],
                      late_from_s=600.0),
    ]
    stations = [
        StationSpec("S1", 350.0, 150.0, 1600.0, [_boots_standard()], (25.0, 70.0)),
        StationSpec("S2", 1300.0, 300.0, 1600.0, [_boots_standard()], (40.0, 90.0)),
        StationSpec(
            "S3",
            800.0,
            1000.0,
            1600.0,
            [_boots_standard()],
            (1200.0, 1600.0),
            late=True,
            note="slow link; whole boot uploaded in the late phase",
        ),
        StationSpec("S4", 550.0, 820.0, 1500.0, [_boots_standard()], (30.0, 95.0)),
    ]
    sources = [
        SourceSpec("A", "trill", 700.0, 430.0, [250.0, 800.0, 1400.0], [0.70, 0.40, 0.90]),
        SourceSpec("B", "trill", 950.0, 600.0, [1900.0, 2200.0], [0.90, 0.28, 0.65]),
    ]
    return Scenario(
        name="revision",
        title="Late report and calibration revision produce event revisions",
        description=(
            "The first run happens before S3's backfill and before the late calibration "
            "anchors arrive. It publishes provisional events without S3. The second run "
            "sees S3's records and a revised calibration, so events that now include S3 "
            "are published as revisions naming the events they supersede: no activity is "
            "silently split into two while the old count remains."
        ),
        stations=stations,
        sources=sources,
        cal_sources=cal,
        area={"x_min": -300.0, "x_max": 2000.0, "y_min": -400.0, "y_max": 1400.0},
        feature_noise=0.008,
        expected={
            "events_first_run": 5,
            "superseded_by_second_run": 5,
            "late_station": "S3",
            "note": "second run's calibration hash for S1/S2 must differ from the first run",
        },
    )


# --------------------------------------------------------------------------- #
# 6. Day/night biased sampling + overlapping stations + an empty stratum
# --------------------------------------------------------------------------- #
ANALYSIS_WINDOW_START_S = 21_600.0   # 2026-11-14 06:00 UTC
ANALYSIS_WINDOW_END_S = 194_400.0    # 2026-11-16 06:00 UTC
ANALYSIS_BIN_SECONDS = 43_200        # 12 h bins: day1, night1, day2, night2
ANALYSIS_NIGHTS_S = [(64_800.0, 108_000.0), (151_200.0, 194_400.0)]
ANALYSIS_GRID_M = 250.0


def _analysis_cal_sources(west_center: tuple[float, float], east_center: tuple[float, float]) -> list[CalSourceSpec]:
    emissions = [-3600.0 + 1800.0 * i for i in range(115)]  # up to 201600 s
    return [
        CalSourceSpec("C1", west_center[0], west_center[1], list(emissions)),
        CalSourceSpec("C2", east_center[0], east_center[1], list(emissions)),
    ]


def _analysis_sound_times(offsets: list[float]) -> list[float]:
    out: list[float] = []
    for bin_start in (21_600.0, 64_800.0, 108_000.0, 151_200.0):
        out.extend(bin_start + off for off in offsets)
    return sorted(out)


def _habitat_cells() -> list[dict]:
    """Habitat of every analysis cell in both regions (shared by both scenarios
    only in the daynight case; nightfailure overrides to a single habitat)."""
    cells = []
    for ix in range(8):
        for iy in range(4, 8):
            habitat = "reed" if iy in (4, 5) else ("water" if iy == 6 else "mudflat")
            cells.append({"ix": ix, "iy": iy, "habitat": habitat})
    return cells


def _daynight() -> Scenario:
    day_only = [
        (-3_600.0, 64_800.0),
        (108_000.0, 151_200.0),
        (194_400.0, 210_000.0),
    ]
    always = None
    stations = [
        # West block: four stations online around the clock; W1/W3/W4 hearing
        # ranges overlap heavily on the reed cells.
        StationSpec("W1", 300.0, 400.0, 450.0, [_boots_standard()], (25.0, 70.0),
                    online_windows_s=always),
        StationSpec("W2", 650.0, 400.0, 450.0, [_boots_standard()], (30.0, 80.0),
                    online_windows_s=always),
        StationSpec("W3", 500.0, 400.0, 450.0, [_boots_standard()], (35.0, 90.0),
                    online_windows_s=always),
        StationSpec("W4", 500.0, 150.0, 450.0, [_boots_standard()], (20.0, 60.0),
                    online_windows_s=always),
        # East block: every station shuts down at night (18:00-06:00) and says
        # so in its progress gaps.
        StationSpec("E1", 1350.0, 350.0, 400.0, [_boots_standard()], (25.0, 70.0),
                    online_windows_s=day_only, note="scheduled night shutdown"),
        StationSpec("E2", 1800.0, 350.0, 400.0, [_boots_standard()], (30.0, 80.0),
                    online_windows_s=day_only, note="scheduled night shutdown"),
        StationSpec("E3", 1550.0, 350.0, 400.0, [_boots_standard()], (35.0, 90.0),
                    online_windows_s=day_only, note="scheduled night shutdown"),
        StationSpec("E4", 1550.0, 200.0, 400.0, [_boots_standard()], (20.0, 60.0),
                    online_windows_s=day_only, note="scheduled night shutdown"),
    ]
    sources = [
        SourceSpec("WS1", "chirp", 375.0, 275.0, _analysis_sound_times([7200.0, 14400.0, 21600.0]),
                   [1.0, 0.30, 0.20, 0.90]),
        SourceSpec("WS2", "chirp", 625.0, 525.0, _analysis_sound_times([7800.0, 15000.0, 22200.0]),
                   [0.20, 0.90, 0.80, 0.20]),
        SourceSpec("ES1", "chirp", 1375.0, 275.0, _analysis_sound_times([7200.0, 14400.0, 21600.0]),
                   [0.90, 0.25, 0.30, 0.85]),
        SourceSpec("ES2", "chirp", 1625.0, 525.0, _analysis_sound_times([7800.0, 15000.0, 22200.0]),
                   [0.25, 0.85, 0.75, 0.30]),
    ]
    analysis = {
        "window": {"start_s": ANALYSIS_WINDOW_START_S, "end_s": ANALYSIS_WINDOW_END_S},
        "grid_cell_m": ANALYSIS_GRID_M,
        "bin_seconds": ANALYSIS_BIN_SECONDS,
        "class_label": "chirp",
        "regions": [
            {"id": "west", "name": "West block",
             "bbox": {"ix_min": 0, "ix_max": 3, "iy_min": 4, "iy_max": 7}},
            {"id": "east", "name": "East block",
             "bbox": {"ix_min": 4, "ix_max": 7, "iy_min": 4, "iy_max": 7}},
        ],
        "habitat_map": {"id": "hab-daynight", "cells": _habitat_cells()},
        "reference_composition": {
            "id": "ref-daynight",
            "name": "published wetland reference composition",
            "entries": [
                {"habitat": "reed", "period_class": "day", "weight": 0.3},
                {"habitat": "reed", "period_class": "night", "weight": 0.2},
                {"habitat": "water", "period_class": "day", "weight": 0.2},
                {"habitat": "water", "period_class": "night", "weight": 0.1},
                {"habitat": "mudflat", "period_class": "day", "weight": 0.1},
                {"habitat": "mudflat", "period_class": "night", "weight": 0.1},
            ],
        },
        "region_a_id": "west",
        "region_b_id": "east",
        "expected": {
            # Truth emits 6 day + 6 night calls per source; the east night
            # calls are never heard because every east station is off.
            "events_heard": {"west": 24, "east": 12},
            "naive_difference": 12.0,
            "common_strata": [["reed", "day"], ["water", "day"]],
            "incomparable_strata": [
                ["reed", "night"], ["water", "night"],
                ["mudflat", "day"], ["mudflat", "night"],
            ],
            "empty_strata_standardized": [
                ["reed", "night"], ["water", "night"],
                ["mudflat", "day"], ["mudflat", "night"],
            ],
            "max_abs_common_difference": 0.002,
        },
    }
    return Scenario(
        name="daynight",
        title="Day/night biased sampling with overlapping stations and an empty stratum",
        description=(
            "Two blocks with identical true calling rates. West stations listen "
            "around the clock (and their hearing ranges overlap, so effort must "
            "not be double-counted); every east station shuts down at night and "
            "declares the gap. Raw counts say west calls twice as much; the "
            "comparable sample shows the day rates are equal and the east night "
            "rate is unknown. The reference composition also names mudflat, "
            "which no east station can hear: an empty stratum that must be "
            "disclosed, never zero-filled."
        ),
        stations=stations,
        sources=sources,
        cal_sources=_analysis_cal_sources((500.0, 400.0), (1550.0, 350.0)),
        area={"x_min": 0.0, "x_max": 2100.0, "y_min": -1100.0, "y_max": 1100.0},
        expected={
            "events": 36,
            "note": "24 west events (day+night) + 12 east day events; east night calls are lost",
        },
        analysis=analysis,
    )


# --------------------------------------------------------------------------- #
# 7. Night-time recorder failure reported late => report versions
# --------------------------------------------------------------------------- #
def _nightfailure() -> Scenario:
    stations = [
        # North block: all three recorders fail 18:00-06:00 every night, but
        # the failure is silent — the stations keep declaring complete uploads.
        # The gap is only declared later, once field staff notice.
        StationSpec("N1", 500.0, 400.0, 550.0, [_boots_standard()], (25.0, 70.0),
                    online_windows_s=None, gaps_late=True,
                    note="recorder fails at night; gaps declared late"),
        StationSpec("N2", 550.0, 350.0, 550.0, [_boots_standard()], (30.0, 80.0),
                    online_windows_s=None, gaps_late=True,
                    note="recorder fails at night; gaps declared late"),
        StationSpec("N3", 500.0, 150.0, 550.0, [_boots_standard()], (35.0, 90.0),
                    online_windows_s=None, gaps_late=True,
                    note="recorder fails at night; gaps declared late"),
        # South block: healthy stations, online around the clock.
        StationSpec("S1", 1500.0, 400.0, 550.0, [_boots_standard()], (25.0, 70.0)),
        StationSpec("S2", 1550.0, 450.0, 550.0, [_boots_standard()], (30.0, 80.0)),
        StationSpec("S3", 1500.0, 150.0, 550.0, [_boots_standard()], (35.0, 90.0)),
    ]
    # The failure is real: north stations record nothing at night.  What is
    # late is the *declaration* of that fact, not the records.
    failing_windows = [
        (-3_600.0, 64_800.0),
        (108_000.0, 151_200.0),
        (194_400.0, 210_000.0),
    ]
    for st in stations[:3]:
        st.online_windows_s = failing_windows
    sources = [
        SourceSpec("NS1", "croak", 375.0, 275.0, _analysis_sound_times([7200.0, 14400.0, 21600.0]),
                   [0.40, 1.00, 0.50, 0.60]),
        SourceSpec("SS1", "croak", 1375.0, 275.0, _analysis_sound_times([7200.0, 14400.0, 21600.0]),
                   [0.45, 0.95, 0.55, 0.55]),
    ]
    analysis = {
        "window": {"start_s": ANALYSIS_WINDOW_START_S, "end_s": ANALYSIS_WINDOW_END_S},
        "grid_cell_m": ANALYSIS_GRID_M,
        "bin_seconds": ANALYSIS_BIN_SECONDS,
        "class_label": "croak",
        "regions": [
            {"id": "north", "name": "North block",
             "bbox": {"ix_min": 0, "ix_max": 3, "iy_min": 4, "iy_max": 7}},
            {"id": "south", "name": "South block",
             "bbox": {"ix_min": 4, "ix_max": 7, "iy_min": 4, "iy_max": 7}},
        ],
        "habitat_map": {
            "id": "hab-nightfailure",
            "cells": [
                {"ix": ix, "iy": iy, "habitat": "reed"}
                for ix in range(8)
                for iy in range(4, 8)
            ],
        },
        "reference_composition": {
            "id": "ref-nightfailure",
            "name": "reed-only reference composition",
            "entries": [
                {"habitat": "reed", "period_class": "day", "weight": 0.5},
                {"habitat": "reed", "period_class": "night", "weight": 0.5},
            ],
        },
        "region_a_id": "north",
        "region_b_id": "south",
        # Progress reports (per station) that arrive only after report v1:
        # they declare the night-time recorder gaps and force a new cutoff.
        "late_gap_declarations": [
            {
                "station_id": station_id,
                "boot_id": "b1",
                "gaps_s": ANALYSIS_NIGHTS_S,
                "reason": "recorder power fault 18:00-06:00 confirmed by field inspection",
            }
            for station_id in ("N1", "N2", "N3")
        ],
        "expected": {
            "events_heard": {"north": 6, "south": 12},
            # v1 believes the north nights were covered (no gap declared yet):
            # it scores the north night rate as a false zero.
            "v1_night_status": "covered",
            "v1_common_difference_max_abs": -0.003,
            # v2 (after the late gap reports) marks the north nights unknown.
            "v2_night_status": "no_coverage",
            "v2_common_difference_max_abs": 0.002,
            "v2_empty_strata_standardized": [["reed", "night"]],
        },
    }
    return Scenario(
        name="nightfailure",
        title="Night-time recorder failure reported late creates a new report version",
        description=(
            "All north recorders die every night at 18:00 and restart at 06:00, "
            "but keep declaring complete uploads, so report v1 scores the north "
            "night rate as zero (covered, nothing heard) and the comparison "
            "looks significant. When field staff later declare the gaps, a new "
            "report at a new cutoff marks the north nights unknown; the old "
            "report stays exactly reproducible at its own cutoff."
        ),
        stations=stations,
        sources=sources,
        cal_sources=_analysis_cal_sources((500.0, 400.0), (1500.0, 300.0)),
        area={"x_min": 0.0, "x_max": 2100.0, "y_min": -1100.0, "y_max": 1100.0},
        expected={
            "events": 18,
            "note": "6 north day events + 12 south events; north night calls are never recorded",
        },
        analysis=analysis,
    )
