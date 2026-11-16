"""Pausable station simulator.

`--ground-truth-only` writes the station-side view of a scenario without a
server.  The default mode drives a running service: create stations, sound
classes and calibration sources, then POST batches in waves, optionally pausing
mid-scenario so a run can be interrupted and resumed from the same input
cutoff.

Wave semantics
--------------
Uploads are emitted in reference order, but each station's *declared progress*
(`complete_through_local_at`) only advances when the station has actually
finished that boot.  A station whose link is slow (`late=True`) reports nothing
until its late wave, and its earlier silence therefore never counts as
"reported and empty".
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulator.scenarios import (  # noqa: E402
    SPEED_MPS,
    CalSourceSpec,
    Scenario,
    StationSpec,
    build_scenario,
    local_iso,
    ref_iso,
    scenario_names,
    us,
    REFERENCE_EPOCH,
)

API_KEY = "dev-api-key"


def iso_from_us(value: int) -> str:
    dt = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(microseconds=value)
    return dt.isoformat(timespec="microseconds").replace("+00:00", "Z")


# --------------------------------------------------------------------------- #
# Ground truth
# --------------------------------------------------------------------------- #
@dataclass
class TruthEvent:
    source_id: str
    call_index: int
    class_label: str
    x: float
    y: float
    emission_s: float
    heard_by: list[str]

    @property
    def key(self) -> str:
        return f"{self.source_id}#{self.call_index}"


def build_truth(scn: Scenario) -> dict:
    rng = random.Random(7)
    anchors: list[dict] = []
    sounds: list[dict] = []
    events: list[TruthEvent] = []
    rows: list[dict] = []

    def make_row(module, **data) -> dict:
        row = {"module": module, **data}
        rows.append(row)
        return row

    seq: dict[tuple[str, str], int] = {}

    for cal in scn.cal_sources:
        for emission_s in cal.emissions_s:
            emission_us = us(REFERENCE_EPOCH + timedelta(seconds=emission_s))
            for st in scn.stations:
                d = math.hypot(st.x - cal.x, st.y - cal.y)
                if d > st.radius:
                    continue
                if not st.online_at(emission_s):
                    # Recorder off (night shutdown/failure): nothing is
                    # recorded, whether or not the gap has been declared yet.
                    continue
                boot = st.boot_at(emission_s)
                arrival_us = int(round(emission_us + d / SPEED_MPS * 1_000_000.0))
                noisy_us = int(round(arrival_us + rng.gauss(0.0, scn.anchor_noise_ms * 1000.0)))
                make_row(
                    "anchor",
                    station_id=st.id,
                    boot_id=boot.boot_id,
                    kind="calibration",
                    class_label="calibration",
                    local_capture_us=boot.local_us(noisy_us),
                    arrival_us=noisy_us,
                    features=[round(rng.gauss(0.0, 0.02), 6) for _ in range(4)],
                    uncertainty_ms=12.0,
                    cal_source_id=cal.id,
                    emission_us=emission_us,
                    late=bool(cal.late_from_s is not None and emission_s >= cal.late_from_s),
                )
                anchors.append(rows[-1])

    for source in scn.sources:
        for index, emission_s in enumerate(source.emissions_s):
            emission_us = us(REFERENCE_EPOCH + timedelta(seconds=emission_s))
            heard: list[str] = []
            for st in scn.stations:
                d = math.hypot(st.x - source.x, st.y - source.y)
                if d > st.radius:
                    continue
                if not st.online_at(emission_s):
                    continue
                heard.append(st.id)
                boot = st.boot_at(emission_s)
                jitter_us = int(round(rng.gauss(0.0, source.temporal_sigma_ms * 1000.0)))
                arrival_us = int(round(emission_us + d / SPEED_MPS * 1_000_000.0 + jitter_us))
                make_row(
                    "sound",
                    station_id=st.id,
                    boot_id=boot.boot_id,
                    kind="sound",
                    class_label=source.class_label,
                    local_capture_us=boot.local_us(arrival_us),
                    arrival_us=arrival_us,
                    features=[
                        round(v + rng.gauss(0.0, scn.feature_noise), 6) for v in source.feature
                    ],
                    uncertainty_ms=scn.observation_uncertainty_ms,
                    emission_us=emission_us,
                    truth_source=source.id,
                    truth_call_index=index,
                    late=st.late,
                )
                sounds.append(rows[-1])
            events.append(
                TruthEvent(source.id, index, source.class_label, source.x, source.y, emission_s, heard)
            )

    for station_id, boot_id in sorted({(r["station_id"], r["boot_id"]) for r in rows}):
        boot_rows = [r for r in rows if r["station_id"] == station_id and r["boot_id"] == boot_id]
        boot_rows.sort(key=lambda r: r["local_capture_us"])
        for r in boot_rows:
            key = (station_id, boot_id)
            seq[key] = seq.get(key, 0) + 1
            r["source_seq"] = seq[key]

    return {"anchors": anchors, "sounds": sounds, "events": events, "rows": rows}


def to_api_record(row: dict) -> dict:
    out = {
        "source_seq": row["source_seq"],
        "kind": row["kind"],
        "class_label": row["class_label"],
        "local_capture_at": local_iso(row["local_capture_us"]),
        "features": row["features"],
        "uncertainty_ms": row["uncertainty_ms"],
    }
    if row["kind"] == "calibration":
        out["calibration"] = {
            "source_id": row["cal_source_id"],
            "emission_reference_at": iso_from_us(row["emission_us"]),
        }
    return out


def build_waves(scn: Scenario, truth: dict) -> list[dict]:
    """Split each station boot's records into early and late upload waves.

    The early wave carries at most 60% of a boot's records.  A wave declares
    `complete_through` only up to the last record it actually contains, and
    marks the boot complete only on the wave that carries the boot's tail.
    Calibration records explicitly flagged late are withheld from early waves.
    """
    waves: list[dict] = []
    rng = random.Random(11)
    for st in scn.stations:
        for boot in st.boots:
            boot_rows = sorted(
                [r for r in truth["rows"] if r["station_id"] == st.id and r["boot_id"] == boot.boot_id],
                key=lambda r: r["local_capture_us"],
            )
            if not boot_rows:
                continue
            early_pool = [r for r in boot_rows if not r["late"]]
            late_pool = [r for r in boot_rows if r["late"]]
            if st.late:
                early_pool, late_pool = [], boot_rows
            else:
                cut = max(1, math.ceil(len(early_pool) * 0.6))
                early_pool, moved = early_pool[:cut], early_pool[cut:]
                late_pool = sorted(moved + late_pool, key=lambda r: r["local_capture_us"])
            lag = rng.uniform(*st.upload_lag_s)
            if early_pool:
                waves.append(
                    {
                        "station_id": st.id,
                        "boot_id": boot.boot_id,
                        "phase": "early",
                        "upload_s": _max_ref_s(early_pool) + lag,
                        "records": early_pool,
                        "declare_through_ref_s": _max_ref_s(early_pool),
                        "declare_complete": not late_pool,
                    }
                )
            if late_pool:
                waves.append(
                    {
                        "station_id": st.id,
                        "boot_id": boot.boot_id,
                        "phase": "late",
                        "upload_s": _max_ref_s(boot_rows) + max(lag, 600.0),
                        "records": late_pool,
                        "declare_through_ref_s": _max_ref_s(boot_rows) + 600.0,
                        "declare_complete": True,
                    }
                )
    waves.sort(key=lambda w: (w["upload_s"], w["station_id"], w["phase"]))
    _attach_declared_gaps(scn, truth, waves)
    return waves


def _attach_declared_gaps(scn: Scenario, truth: dict, waves: list[dict]) -> None:
    """Declare offline intervals as coverage gaps in the final wave's progress.

    A station with ``online_windows_s`` was off during the complement of those
    windows.  Unless ``gaps_late`` is set, its last upload wave declares the
    gaps (station-local readings, like every progress field); with
    ``gaps_late`` the failure is silent and the gaps only arrive through a
    separate late progress report driven by the analysis demo.
    """
    for st in scn.stations:
        if st.online_windows_s is None or st.gaps_late:
            continue
        boot_waves = [w for w in waves if w["station_id"] == st.id]
        if not boot_waves:
            continue
        final = next((w for w in reversed(boot_waves) if w["declare_complete"]), None)
        if final is None:
            continue
        boot = next(b for b in st.boots if b.boot_id == final["boot_id"])
        first_ref_s = min(
            (r["arrival_us"] - us(REFERENCE_EPOCH)) / 1e6
            for r in truth["rows"]
            if r["station_id"] == st.id
        )
        gaps = []
        for gap_start_s, gap_end_s in st.offline_gaps_s(first_ref_s, final["declare_through_ref_s"]):
            gaps.append(
                {
                    "from_local_at": local_iso(
                        boot.local_us(us(REFERENCE_EPOCH + timedelta(seconds=gap_start_s)))
                    ),
                    "through_local_at": local_iso(
                        boot.local_us(us(REFERENCE_EPOCH + timedelta(seconds=gap_end_s)))
                    ),
                    "reason": "scheduled recorder shutdown (offline interval)",
                }
            )
        if gaps:
            final["gaps"] = gaps


def _max_ref_s(rows: list[dict]) -> float:
    return max((r["arrival_us"] - us(REFERENCE_EPOCH)) / 1e6 for r in rows)


def write_ground_truth(scn: Scenario, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    truth = build_truth(scn)
    waves = build_waves(scn, truth)
    truth_json = {
        "scenario": scn.name,
        "title": scn.title,
        "description": scn.description,
        "propagation_speed_mps": SPEED_MPS,
        "area": scn.area,
        "stations": [
            {
                "id": s.id,
                "x": s.x,
                "y": s.y,
                "hearing_radius_m": s.radius,
                "late": s.late,
                "online_windows_s": s.online_windows_s,
                "boots": [b.truth() for b in s.boots],
            }
            for s in scn.stations
        ],
        "calibration_sources": [
            {
                "id": c.id,
                "x": c.x,
                "y": c.y,
                "emissions": [ref_iso(e) for e in c.emissions_s],
                "emissions_us": [us(REFERENCE_EPOCH + timedelta(seconds=e)) for e in c.emissions_s],
                "late_from": ref_iso(c.late_from_s) if c.late_from_s is not None else None,
            }
            for c in scn.cal_sources
        ],
        "events": [
            {
                "key": e.key,
                "source_id": e.source_id,
                "call_index": e.call_index,
                "class_label": e.class_label,
                "x": e.x,
                "y": e.y,
                "emission_s": e.emission_s,
                "emission": ref_iso(e.emission_s),
                "heard_by": e.heard_by,
            }
            for e in truth["events"]
        ],
        "waves": [
            {
                "station_id": w["station_id"],
                "boot_id": w["boot_id"],
                "phase": w["phase"],
                "upload_s": w["upload_s"],
                "declare_through_ref_s": w["declare_through_ref_s"],
                "declare_complete": w["declare_complete"],
                "gaps": w.get("gaps") or [],
                "records": [to_api_record(r) for r in w["records"]],
                "truth": [
                    {
                        "source_seq": r["source_seq"],
                        "kind": r["kind"],
                        "local_capture_us": r["local_capture_us"],
                        "arrival_us": r["arrival_us"],
                        "truth_source": r.get("truth_source"),
                        "truth_call_index": r.get("truth_call_index"),
                        "emission_us": r.get("emission_us"),
                    }
                    for r in w["records"]
                ],
            }
            for w in waves
        ],
        "expected": scn.expected,
        "analysis": scn.analysis,
    }
    (out_dir / "truth.json").write_text(json.dumps(truth_json, indent=2, default=str))
    return {"truth": truth_json, "waves": waves, "raw": truth}


# --------------------------------------------------------------------------- #
# API driver
# --------------------------------------------------------------------------- #
class Api:
    def __init__(self, base: str, api_key: str = API_KEY):
        self.base = base.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"X-API-Key": api_key})

    def _req(self, method: str, path: str, **kwargs):
        r = self.session.request(method, self.base + path, timeout=300, **kwargs)
        if r.status_code >= 400:
            raise RuntimeError(f"{method} {path} -> {r.status_code}: {r.text[:800]}")
        return r.json() if r.content else None

    def reset(self):
        return self._req("POST", "/admin/reset", params={"confirm": "true"})

    def upsert_station(self, spec: StationSpec):
        body = {
            "id": spec.id,
            "name": f"Station {spec.id}",
            "x": spec.x,
            "y": spec.y,
            "hearing_radius_m": spec.radius,
            "active": True,
            "notes": spec.note or "simulated wetland station",
        }
        existing = {s["id"] for s in self._req("GET", "/stations")}
        if spec.id in existing:
            return self._req("PUT", f"/stations/{spec.id}", json=body)
        return self._req("POST", "/stations", json=body)

    def ensure_class(self, class_label: str, dim: int, limit: float):
        existing = {c["id"] for c in self._req("GET", "/sound-classes")}
        if class_label in existing:
            return
        self._req(
            "POST",
            "/sound-classes",
            json={
                "id": class_label,
                "description": f"simulated {class_label}",
                "feature_dim": dim,
                "feature_distance_limit": limit,
            },
        )

    def ensure_cal_source(self, cal: CalSourceSpec):
        existing = {c["id"] for c in self._req("GET", "/calibration-sources")}
        if cal.id in existing:
            return
        self._req(
            "POST",
            "/calibration-sources",
            json={
                "id": cal.id,
                "name": f"Calibration source {cal.id}",
                "x": cal.x,
                "y": cal.y,
                "emissions": [ref_iso(s) for s in cal.emissions_s],
                "notes": "known position and known emission instants",
            },
        )

    def post_wave(self, wave: dict):
        progress = None
        st_ref = None
        st = None
        if wave["declare_through_ref_s"] is not None:
            st = wave["_station"]
            st_ref = wave["declare_through_ref_s"]
            boot = next(b for b in st.boots if b.boot_id == wave["boot_id"])
            local_through = boot.local_us(us(REFERENCE_EPOCH + timedelta(seconds=st_ref)))
            progress = {
                "complete_through_local_at": local_iso(local_through),
                "complete": wave["declare_complete"],
                "gaps": wave.get("gaps") or [],
            }
        body = {
            "station_id": wave["station_id"],
            "boot_id": wave["boot_id"],
            "batch_key": f"{wave['station_id']}-{wave['boot_id']}-{wave['phase']}",
            "uploaded_at": ref_iso(wave["upload_s"]),
            "records": [to_api_record(r) for r in wave["records"]],
            "progress": progress,
        }
        return self._req("POST", f"/stations/{wave['station_id']}/observations", json=body)

    def post_raw_batch(self, station_id: str, boot_id: str, batch_key: str, records: list[dict]):
        body = {
            "station_id": station_id,
            "boot_id": boot_id,
            "batch_key": batch_key,
            "records": records,
        }
        return self._req("POST", f"/stations/{station_id}/observations", json=body)

    def create_run(self, **kwargs):
        return self._req("POST", "/runs", json={"wait": True, **kwargs})

    def resume(self, run_id: int):
        return self._req("POST", f"/runs/{run_id}/resume")


def _feature_limit(scn: Scenario, class_label: str, pair_limit: float) -> float:
    """Per-class limit: well above same-source scatter, below cross-source gap."""
    spreads = [
        scn.feature_noise * len(s.feature) ** 0.5 * 2.0
        for s in scn.sources
        if s.class_label == class_label
    ]
    same_source = max(spreads + [0.02])
    # The two sources of a class must remain distinguishable, so cap at 70% of
    # their centre distance.
    return max(0.02, min(0.22, min(pair_limit * 0.7, max(same_source * 3.0, 0.03))))


def setup(scn: Scenario, api: Api) -> float:
    for st in scn.stations:
        api.upsert_station(st)
    pair_limit = min(
        (
            math.dist(a.feature, b.feature)
            for a in scn.sources
            for b in scn.sources
            if a.id < b.id and a.class_label == b.class_label
        ),
        default=1.0,
    )
    for class_label in sorted({s.class_label for s in scn.sources}):
        dim = len(next(s for s in scn.sources if s.class_label == class_label).feature)
        api.ensure_class(class_label, dim, limit=_feature_limit(scn, class_label, pair_limit))
    for cal in scn.cal_sources:
        api.ensure_cal_source(cal)
    return pair_limit


def _waves_for(scn: Scenario, truth: dict) -> list[dict]:
    waves = build_waves(scn, truth)
    for w in waves:
        w["_station"] = next(s for s in scn.stations if s.id == w["station_id"])
    return waves


def load_scenario(
    name: str,
    base: str,
    *,
    reset: bool = False,
    up_to_wave: int | None = None,
    pause_after_wave: int | None = None,
    retransmit: bool = False,
    phases: tuple[str, ...] | None = None,
    truth_dir: Path | None = None,
) -> dict:
    scn = build_scenario(name)
    api = Api(base)
    if reset:
        api.reset()
    build = write_ground_truth(scn, truth_dir or Path("samples") / name)
    setup(scn, api)
    waves = _waves_for(scn, build["raw"])
    if phases is not None:
        waves = [w for w in waves if w["phase"] in phases]
    posted = []
    for index, wave in enumerate(waves):
        if up_to_wave is not None and index >= up_to_wave:
            break
        response = api.post_wave(wave)
        posted.append(
            {
                **response,
                "station_id": wave["station_id"],
                "boot_id": wave["boot_id"],
                "batch_key": f"{wave['station_id']}-{wave['boot_id']}-{wave['phase']}",
            }
        )
        if pause_after_wave is not None and index + 1 == pause_after_wave:
            break
    extra = {}
    if retransmit:
        extra["retransmit"] = retransmit_conflict(api, scn, build["raw"])
    return {
        **extra,
        "scenario": name,
        "waves_total": len(waves),
        "waves_posted": len(posted),
        "waves": posted,
        "truth": build["truth"],
        "raw": build["raw"],
        "scn": scn,
    }


def retransmit_conflict(api: Api, scn: Scenario, raw: dict) -> dict:
    """Replay a batch verbatim, then resend one record with different content."""
    replay_target = next(w for w in build_waves(scn, raw) if w["station_id"] == "S1")
    replay_payload = [to_api_record(r) for r in replay_target["records"]]
    replay = api.post_raw_batch(
        "S1", replay_target["boot_id"], f"S1-{replay_target['boot_id']}-{replay_target['phase']}", replay_payload
    )
    # Re-send one *sound* record with different content under the same
    # sequence number: the station believes it corrected its own record.
    conflict_record = next(r for r in replay_target["records"] if r["kind"] == "sound")
    altered = to_api_record(conflict_record)
    altered["uncertainty_ms"] = 45.0
    conflict = api.post_raw_batch(
        "S1", replay_target["boot_id"], f"conflict-resend-{conflict_record['source_seq']}", [altered]
    )
    return {
        "replay": replay,
        "conflict": conflict,
        "conflict_source_seq": conflict_record["source_seq"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pausable wetland station simulator")
    parser.add_argument("--scenario", required=True, choices=scenario_names())
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--reset", action="store_true", help="wipe the service before loading")
    parser.add_argument("--up-to-wave", type=int, help="load only the first N waves")
    parser.add_argument("--pause-after-wave", type=int, help="load N waves then stop")
    parser.add_argument("--retransmit", action="store_true", help="replay and conflict-test after loading")
    parser.add_argument("--phases", help="comma separated subset of early,late")
    parser.add_argument("--truth-dir", type=Path)
    parser.add_argument(
        "--ground-truth-only",
        action="store_true",
        help="write truth.json without contacting the service",
    )
    args = parser.parse_args(argv)
    if args.ground_truth_only:
        scn = build_scenario(args.scenario)
        out = write_ground_truth(scn, args.truth_dir or Path("samples") / args.scenario)
        print(json.dumps({"scenario": args.scenario, "waves": len(out["waves"])}, indent=2))
        return 0
    result = load_scenario(
        args.scenario,
        args.base_url,
        reset=args.reset,
        up_to_wave=args.up_to_wave,
        pause_after_wave=args.pause_after_wave,
        truth_dir=args.truth_dir,
        phases=tuple(args.phases.split(",")) if args.phases else None,
    )
    api = Api(args.base_url)
    extra = {}
    if args.retransmit:
        extra["retransmit"] = retransmit_conflict(api, result["scn"], result["raw"])
    printable = {
        "scenario": result["scenario"],
        "waves_total": result["waves_total"],
        "waves_posted": result["waves_posted"],
        "batches": [
            {
                "station_id": w["station_id"],
                "boot_id": w["boot_id"],
                "batch_key": w["batch_key"],
                "replayed": w["replayed"],
                "counts": w["counts"],
            }
            for w in result["waves"]
        ],
    }
    printable.update(extra)
    print(json.dumps(printable, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
