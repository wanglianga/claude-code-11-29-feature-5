"""Processing pipeline.

Phases: ``calibrate -> collect -> candidates -> localize -> solve -> finalize``.
Every phase boundary writes a checkpoint into ``processing_runs.checkpoint_json``
together with the input cutoff and the calibration snapshot.  A resume loads
the checkpoint instead of recomputing earlier phases and never reads uploads
ingested after the original cutoff, so a run's output cannot mix calibration
parameters from two different snapshots.
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import dataclass
from typing import Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from . import association, calibration as calib, solver
from .config import Area, Settings, settings as global_settings
from .localize import StationPoint
from .models import (
    AuditLog,
    CalibrationModel,
    CalibrationSourceRecord,
    Event,
    EventExclusion,
    EventObservation,
    EventRevisionLink,
    Observation,
    ProcessingRun,
    SoundClass,
    Station,
    StationExclusion,
    StationProgress,
)
from .timeutil import iso, now_us


class PhaseStopped(Exception):
    """Raised by the debug checkpoint hook to pause a run mid-flight."""

    def __init__(self, phase: str):
        super().__init__(phase)
        self.phase = phase


@dataclass
class PipelineContext:
    db: Session
    run: ProcessingRun
    cfg: Settings
    area: Area
    debug_stop_after_phase: str | None = None
    log: Callable[[str], None] = lambda _m: None


def snapshot_settings(cfg: Settings) -> dict:
    snap = cfg.as_dict()
    snap.pop("database_url", None)
    snap.pop("api_key", None)
    snap.pop("samples_dir", None)
    return snap


def create_run(
    db: Session,
    *,
    mode: str = "full",
    cutoff_us: int | None = None,
    created_by: str | None = None,
    resume_of_run_id: int | None = None,
    revision_of_run_id: int | None = None,
    cfg: Settings | None = None,
) -> ProcessingRun:
    cfg = cfg or global_settings
    cutoff = cutoff_us if cutoff_us is not None else now_us()
    run = ProcessingRun(
        batch_id=str(uuid.uuid4()),
        mode=mode,
        status="pending",
        cutoff_us=cutoff,
        params_json=snapshot_settings(cfg),
        calibration_json=None,
        phase=None,
        phase_log=[],
        created_at_us=now_us(),
        resume_of_run_id=resume_of_run_id,
        revision_of_run_id=revision_of_run_id,
        created_by=created_by,
    )
    db.add(run)
    db.flush()
    return run


def _save_checkpoint(run: ProcessingRun, checkpoint: dict) -> None:
    """Persist a checkpoint as a *new* JSON value.

    SQLAlchemy does not track in-place mutation of a JSON column, and assigning
    the same dict object again is a no-op, so every save deep-copies.
    """
    run.checkpoint_json = json.loads(json.dumps(checkpoint, default=str))


def _set_phase(ctx: PipelineContext, phase: str, status: str = "running") -> None:
    ctx.run.phase = phase
    ctx.run.status = status
    log = list(ctx.run.phase_log or [])
    log.append({"phase": phase, "at": iso(now_us())})
    ctx.run.phase_log = log
    ctx.db.flush()
    ctx.log(f"phase={phase} status={status}")
    if ctx.debug_stop_after_phase == phase:
        raise PhaseStopped(phase)


# --------------------------------------------------------------------------- #
# Anchors & calibration
# --------------------------------------------------------------------------- #
def _load_anchor_inputs(
    db: Session, cutoff_us: int
) -> tuple[list[Observation], dict[tuple[str, str], list[calib.Anchor]]]:
    """Resolve calibration observations into anchors.

    An anchor needs: a configured calibration source, an emission instant that
    is one of the declared instants, the station position (to remove the
    propagation delay), and an observation ingested before the cutoff.
    """
    rows = list(
        db.execute(
            select(Observation).where(
                Observation.kind == "calibration",
                Observation.ingest_at_us <= cutoff_us,
            )
        ).scalars()
    )
    sources = {
        s.id: s for s in db.execute(select(CalibrationSourceRecord)).scalars()
    }
    stations = {s.id: s for s in db.execute(select(Station)).scalars()}
    grouped: dict[tuple[str, str], list[calib.Anchor]] = {}
    resolved: list[Observation] = []
    for obs in rows:
        src = sources.get(obs.calibration_source_id or "")
        st = stations.get(obs.station_id)
        if src is None or st is None:
            obs.calibration_status = "uncalibrated"
            obs.calibration_reason = "unknown_calibration_source"
            continue
        declared = {round(a) for a in src.emissions_json}
        if obs.reference_emission_us not in declared:
            obs.calibration_status = "uncalibrated"
            obs.calibration_reason = (
                "emission_instant_not_declared: the record's emission_reference_at is "
                "not one of the configured calibration instants"
            )
            continue
        delay = math.hypot(st.x - src.x, st.y - src.y) / (
            global_settings.propagation_speed_mps
        ) * 1_000_000.0
        anchor = calib.Anchor(
            observation_id=obs.id,
            source_id=f"{src.id}@{iso(obs.reference_emission_us)}",
            local_us=obs.local_capture_us,
            reference_arrival_us=obs.reference_emission_us + delay,
            propagation_us=delay,
        )
        grouped.setdefault((obs.station_id, obs.boot_id), []).append(anchor)
        resolved.append(obs)
    return resolved, grouped


def calibrate_all(
    db: Session, cutoff_us: int, cfg: Settings
) -> tuple[dict, dict[tuple[str, str], CalibrationModel]]:
    cfg_kw = cfg
    _resolved, grouped = _load_anchor_inputs(db, cutoff_us)
    models: dict[tuple[str, str], CalibrationModel] = {}
    snapshot: dict[str, dict] = {}
    boots: set[tuple[str, str]] = set()
    # Only observations ingested before the cutoff may contribute to the
    # calibration snapshot; a late upload cannot retroactively change the
    # parameters used by an already-paused run.
    for station_id, boot_id in db.execute(
        select(Observation.station_id, Observation.boot_id)
        .where(Observation.kind == "sound", Observation.ingest_at_us <= cutoff_us)
        .distinct()
    ):
        boots.add((station_id, boot_id))
    boots |= set(grouped.keys())

    for station_id, boot_id in sorted(boots):
        anchors = sorted(grouped.get((station_id, boot_id), []), key=lambda a: a.local_us)
        fit = calib.fit_calibration(station_id, boot_id, anchors, cfg_kw)
        version = fit.version_hash()
        existing = db.execute(
            select(CalibrationModel).where(
                CalibrationModel.station_id == station_id,
                CalibrationModel.boot_id == boot_id,
                CalibrationModel.version_hash == version,
            )
        ).scalar_one_or_none()
        if existing is None:
            existing = CalibrationModel(
                station_id=station_id,
                boot_id=boot_id,
                version_hash=version,
                anchor_count=len(anchors),
                anchor_ids=[a.source_id for a in anchors],
                segments=[
                    {
                        "index": i,
                        "local_from_us": s.local_from_us,
                        "local_to_us": s.local_to_us,
                        "slope": s.slope,
                        "intercept_us": s.intercept_us,
                        "anchor_ids": [
                            a.source_id for a in anchors if a.observation_id in s.anchor_ids
                        ],
                        "anchor_observation_ids": s.anchor_ids,
                        "rms_residual_us": s.rms_residual_us,
                        "max_residual_us": s.max_residual_us,
                        "se_slope": s.se_slope,
                        "se_intercept_us": s.se_intercept_us,
                    }
                    for i, s in enumerate(fit.segments)
                ]
                if fit.segments
                else None,
                rms_residual_us=int(round(fit.rms_residual_us)) if fit.rms_residual_us is not None else None,
                max_residual_us=int(round(fit.max_residual_us)) if fit.max_residual_us is not None else None,
                uncertainty_us=int(round(fit.base_uncertainty_us)) if fit.base_uncertainty_us else None,
                valid=fit.valid,
                invalid_reason=fit.reason,
                local_anchor_min_us=fit.local_min_us,
                local_anchor_max_us=fit.local_max_us,
                params_json={
                    **fit.params,
                    "dropped_anchor_ids": fit.dropped_anchor_ids,
                    "anchor_span_us": fit.local_span_us,
                    "propagation_speed_mps": cfg.propagation_speed_mps,
                },
                created_at_us=now_us(),
            )
            db.add(existing)
            db.flush()
        models[(station_id, boot_id)] = existing
        snapshot[f"{station_id}/{boot_id}"] = {
            "calibration_model_id": existing.id,
            "version_hash": version,
            "valid": fit.valid,
            "reason": fit.reason,
            "anchor_count": len(anchors),
            "anchor_ids": [a.source_id for a in anchors],
            "dropped_anchor_ids": fit.dropped_anchor_ids,
            "rms_residual_us": existing.rms_residual_us,
            "max_residual_us": existing.max_residual_us,
            "segments": existing.segments,
            "local_anchor_min_us": fit.local_min_us,
            "local_anchor_max_us": fit.local_max_us,
        }

    # Rebuild fits from the persisted snapshot so resumed runs use identical
    # parameters even if the database gained new anchors afterwards.
    fits = {key: _fit_from_model(model, cfg_kw) for key, model in models.items()}
    _apply_calibration_to_observations(db, cutoff_us, fits, cfg_kw, models)
    snapshot_hash = hashlib.sha256(
        json.dumps({k: v["version_hash"] for k, v in sorted(snapshot.items())},
                   sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:24]
    return {"stations": snapshot, "hash": snapshot_hash}, models


def _fit_from_model(model: CalibrationModel, cfg: Settings) -> calib.CalibrationFit:
    segments = []
    for seg in model.segments or []:
        segments.append(
            calib.Segment(
                local_from_us=seg["local_from_us"],
                local_to_us=seg["local_to_us"],
                slope=seg["slope"],
                intercept_us=seg["intercept_us"],
                anchor_ids=seg.get("anchor_observation_ids", []),
                rms_residual_us=seg["rms_residual_us"],
                max_residual_us=seg["max_residual_us"],
                se_slope=seg.get("se_slope", 0.0),
                se_intercept_us=seg.get("se_intercept_us", 0.0),
            )
        )
    fit = calib.CalibrationFit(
        station_id=model.station_id,
        boot_id=model.boot_id,
        valid=model.valid,
        reason=model.invalid_reason,
        segments=segments,
        anchor_ids=model.anchor_ids or [],
        anchor_count=model.anchor_count,
        rms_residual_us=model.rms_residual_us,
        max_residual_us=model.max_residual_us,
        base_uncertainty_us=model.uncertainty_us,
        local_min_us=model.local_anchor_min_us,
        local_max_us=model.local_anchor_max_us,
    )
    fit.params = model.params_json or {}
    return fit


def _apply_calibration_to_observations(
    db: Session,
    cutoff_us: int,
    fits: dict[tuple[str, str], calib.CalibrationFit],
    cfg: Settings,
    models: dict[tuple[str, str], CalibrationModel],
) -> None:
    rows = list(
        db.execute(
            select(Observation).where(Observation.ingest_at_us <= cutoff_us)
        ).scalars()
    )
    for obs in rows:
        if obs.kind == "calibration":
            fit = fits.get((obs.station_id, obs.boot_id))
            if fit and fit.valid:
                obs.calibration_status = "anchor"
                obs.calibration_reason = None
                obs.calibration_model_id = models[(obs.station_id, obs.boot_id)].id
            else:
                obs.calibration_status = "uncalibrated"
                obs.calibration_reason = (
                    fit.reason if fit else "no_calibration_anchors_for_boot"
                )
                obs.calibration_model_id = None
            continue
        fit = fits.get((obs.station_id, obs.boot_id))
        if fit is None:
            obs.calibration_status = "uncalibrated"
            obs.calibration_reason = "no_calibration_anchors_for_boot"
            obs.corrected_arrival_us = None
            obs.corrected_uncertainty_us = None
            obs.calibration_model_id = None
            continue
        result = calib.apply_calibration(fit, obs.local_capture_us, cfg)
        if result.reference_arrival_us is None:
            obs.calibration_status = "uncalibrated"
            obs.calibration_reason = result.reason
            obs.corrected_arrival_us = None
            obs.corrected_uncertainty_us = None
            obs.calibration_model_id = None
            continue
        obs.calibration_status = result.status
        obs.calibration_reason = None
        obs.calibration_model_id = models[(obs.station_id, obs.boot_id)].id
        obs.corrected_arrival_us = result.reference_arrival_us
        # The detector's own timing uncertainty is independent of the clock
        # model, and both add in quadrature; the calibration term is a bound,
        # not the only error source.
        obs.corrected_uncertainty_us = int(
            round(math.hypot(result.uncertainty_us or 0.0, obs.uncertainty_us))
        )
    db.flush()


# --------------------------------------------------------------------------- #
# Collect
# --------------------------------------------------------------------------- #
@dataclass
class Collected:
    observations: list[association.ObsInput]
    stations: dict[str, StationPoint]
    class_limits: dict[str, float]
    skipped_uncalibrated: int
    excluded_station_ids: set[str]
    excluded_observations: int


def collect(
    db: Session, cutoff_us: int, run: ProcessingRun, class_labels: list[str] | None
) -> Collected:
    stations = {
        s.id: StationPoint(s.id, s.x, s.y, s.hearing_radius_m)
        for s in db.execute(select(Station)).scalars()
    }
    class_limits = {
        c.id: (c.feature_distance_limit if c.feature_distance_limit is not None else global_settings.feature_distance_limit)
        for c in db.execute(select(SoundClass)).scalars()
    }
    dims = {
        c.id: c.feature_dim
        for c in db.execute(select(SoundClass)).scalars()
        if c.feature_dim is not None
    }
    stmt = select(Observation).where(
        Observation.kind == "sound", Observation.ingest_at_us <= cutoff_us
    )
    if class_labels:
        stmt = stmt.where(Observation.class_label.in_(class_labels))
    rows = list(db.execute(stmt).scalars())
    excluded = {
        e.station_id
        for e in db.execute(
            select(StationExclusion).where(StationExclusion.active.is_(True))
        ).scalars()
    }
    result: list[association.ObsInput] = []
    skipped = 0
    excluded_obs = 0
    for obs in rows:
        if obs.station_id in excluded:
            excluded_obs += 1
            continue
        if (
            obs.corrected_arrival_us is None
            or obs.corrected_uncertainty_us is None
            or obs.calibration_model_id is None
        ):
            skipped += 1
            continue
        dim = dims.get(obs.class_label)
        if dim is not None and len(obs.features) != dim:
            skipped += 1
            obs.calibration_reason = (
                obs.calibration_reason
                or f"feature_dim_mismatch: expected {dim}, got {len(obs.features)}"
            )
            continue
        model = db.get(CalibrationModel, obs.calibration_model_id)
        result.append(
            association.ObsInput(
                observation_id=obs.id,
                station_id=obs.station_id,
                boot_id=obs.boot_id,
                class_label=obs.class_label,
                corrected_arrival_us=obs.corrected_arrival_us,
                uncertainty_us=obs.corrected_uncertainty_us,
                features=[float(v) for v in obs.features],
                calibration_model_id=obs.calibration_model_id,
                anchor_ids=(model.anchor_ids if model else []),
            )
        )
    return Collected(
        observations=result,
        stations={k: v for k, v in stations.items() if k not in excluded},
        class_limits=class_limits,
        skipped_uncalibrated=skipped,
        excluded_station_ids=excluded,
        excluded_observations=excluded_obs,
    )


# --------------------------------------------------------------------------- #
# Finalisation eligibility
# --------------------------------------------------------------------------- #
@dataclass
class ProgressMap:
    mapped_through: dict[tuple[str, str], int]
    complete_report: dict[tuple[str, str], bool]
    excluded_stations: set[str]
    exclusion_records: list[StationExclusion]


def compute_progress_map(
    db: Session, run: ProcessingRun, cutoff_us: int
) -> ProgressMap:
    """Map each report's declared local progress into reference time.

    A station's completeness only counts if its boot has a valid calibration;
    an uncalibrated progress report cannot bound reference time and therefore
    does not make any event finalisable.  Elapsed time alone never counts.
    """
    models = list(
        db.execute(select(CalibrationModel).where(CalibrationModel.valid.is_(True))).scalars()
    )
    by_boot = {(m.station_id, m.boot_id): m for m in models}
    mapped: dict[tuple[str, str], int] = {}
    complete: dict[tuple[str, str], bool] = {}
    reports = list(
        db.execute(
            select(StationProgress).where(StationProgress.received_at_us <= cutoff_us).order_by(StationProgress.id)
        ).scalars()
    )
    for rep in reports:
        model = by_boot.get((rep.station_id, rep.boot_id))
        if model is None:
            continue
        fit = _fit_from_model(model, global_settings)
        res = calib.apply_calibration(fit, rep.declared_local_us, global_settings)
        if res.reference_arrival_us is None:
            # A progress report whose local instant cannot be mapped to
            # reference time proves nothing and is ignored, not guessed.
            continue
        key = (rep.station_id, rep.boot_id)
        mapped[key] = max(mapped.get(key, -1), res.reference_arrival_us)
        complete[key] = complete.get(key, False) or rep.complete
    exclusions = list(
        db.execute(
            select(StationExclusion).where(StationExclusion.active.is_(True))
        ).scalars()
    )
    return ProgressMap(
        mapped_through=mapped,
        complete_report=complete,
        excluded_stations={e.station_id for e in exclusions},
        exclusion_records=exclusions,
    )


def _exclusion_for_window(
    pm: ProgressMap, station_id: str, window_start_us: int, window_end_us: int
):
    for exc in pm.exclusion_records:
        if exc.station_id != station_id:
            continue
        if exc.from_ref_us is not None and window_end_us < exc.from_ref_us:
            continue
        if exc.to_ref_us is not None and window_start_us > exc.to_ref_us:
            continue
        return exc
    return None


def _finalization_state(
    db: Session,
    pm: ProgressMap,
    candidate: association.Candidate,
    stations: dict[str, StationPoint],
    run: ProcessingRun,
    cfg: Settings,
) -> dict:
    """Decide whether the candidate's window may be finalised.

    An event is finalisable only when every station that could hear it has
    *declared*, through a validly calibrated progress report, that it has
    finished uploading the whole window.  Ways a station can settle:

    * ``reported``: its complete-progress marker is at least window end + margin;
    * ``excluded_by_investigator``: the investigator recorded an exclusion that
      covers the window, with a reason, via the API;
    * ``outside_hearing_range``: the station physically cannot have heard it.

    Elapsed wall-clock time is never a reason.  A station that is silent, late,
    or uncalibrated keeps the event provisional until it reports or is
    explicitly excluded.  A station that *contributes* a record still has to
    report through the window: the record itself can be revised by a later
    calibration.
    """
    margin = cfg.progress_margin_s * 1_000_000
    participating = set(candidate.station_ids)
    # Every station that could hear this event must account for itself, and
    # "could hear" is a geometric question about the feasible regions.
    reachable = {
        station_id
        for station_id, st in stations.items()
        if _station_in_range(st, candidate, cfg)
    }
    per_station: list[dict] = []
    tally: dict[str, int] = {}
    unsettled: list[str] = []
    reasons: list[str] = []

    for station_id, st in sorted(stations.items()):
        mode = "contributing" if station_id in participating else "not_contributing"
        in_range = station_id in reachable
        if not in_range:
            tally["out_of_hearing_range"] = tally.get("out_of_hearing_range", 0) + 1
            per_station.append(
                {
                    "station_id": station_id,
                    "mode": "out_of_hearing_range",
                    "settled": True,
                    "reason": "outside this station's configured hearing radius",
                    "progress_through": None,
                    "window_end": iso(candidate.window_end_us),
                }
            )
            continue

        exclusion = _exclusion_for_window(
            pm, station_id, candidate.window_start_us, candidate.window_end_us
        )
        if exclusion is not None:
            tally["excluded_by_investigator"] = tally.get("excluded_by_investigator", 0) + 1
            per_station.append(
                {
                    "station_id": station_id,
                    "mode": "excluded_by_investigator",
                    "settled": True,
                    "reason": exclusion.reason,
                    "exclusion_id": exclusion.id,
                    "progress_through": None,
                    "window_end": iso(candidate.window_end_us),
                }
            )
            continue

        through = _station_progress_through(pm, station_id, candidate.window_end_us)
        settled = through is not None and through >= candidate.window_end_us + margin
        label = f"{mode}_and_reported_through" if (settled and mode == "contributing") else mode
        tally[label] = tally.get(label, 0) + 1
        if not settled:
            unsettled.append(station_id)
            if through is None:
                reasons.append(
                    f"{station_id}: no calibrated progress report; "
                    "wall-clock time is not accepted as evidence of completeness"
                )
            else:
                reasons.append(
                    f"{station_id}: progress reported only through "
                    f"{((through - candidate.window_end_us) / 1e6):+.0f}s relative to window end"
                )
        per_station.append(
            {
                "station_id": station_id,
                "mode": label,
                "settled": settled,
                "reason": None if settled else "progress_does_not_cover_window",
                "progress_through": iso(through) if through else None,
                "window_end": iso(candidate.window_end_us),
            }
        )

    eligible = not unsettled
    state = {
        "eligible": eligible,
        "status": "final" if eligible else "provisional",
        "window_start": iso(candidate.window_start_us),
        "window_end": iso(candidate.window_end_us),
        "unsettled_stations": unsettled,
        "stations": per_station,
        "modes": tally,
        "rule": (
            "final only when every station in hearing range has declared complete "
            "upload coverage of the window, or was explicitly excluded with a reason"
        ),
    }
    if reasons:
        state["reasons"] = reasons
    return state


def _station_progress_through(pm: ProgressMap, station_id: str, window_end: int) -> int | None:
    """Latest reference instant this station has declared complete upload coverage for.

    A partial-coverage report (``complete=False`` for its boot) still bounds
    coverage up to its declared instant: it says "everything before this local
    reading has been uploaded".  It just does not claim the boot is finished.
    """
    best = None
    for (sid, _boot), mapped in pm.mapped_through.items():
        if sid != station_id:
            continue
        best = mapped if best is None else max(best, mapped)
    return best


def _station_in_range(st: StationPoint, cand: association.Candidate, cfg: Settings) -> bool:
    """Could this station have heard the event at all?

    Uses the *largest* feasible region's far corner, so a station is only
    excused from accounting for the window when no candidate source position is
    within its hearing radius.  Being conservative here means a marginal
    station still has to report, rather than being silently excused.
    """
    for region in cand.localization.regions:
        x0, x1 = region.bounds["x_min"], region.bounds["x_max"]
        y0, y1 = region.bounds["y_min"], region.bounds["y_max"]
        farthest = max(
            math.hypot(st.x - x, st.y - y)
            for x in (x0, x1)
            for y in (y0, y1)
        )
        if farthest <= st.hearing_radius_m:
            return True
        nx = min(max(st.x, x0), x1)
        ny = min(max(st.y, y0), y1)
        if math.hypot(st.x - nx, st.y - ny) <= st.hearing_radius_m:
            return True
    return False


# --------------------------------------------------------------------------- #
# Run execution
# --------------------------------------------------------------------------- #
def execute_run(
    db: Session,
    run: ProcessingRun,
    *,
    class_labels: list[str] | None = None,
    debug_stop_after_phase: str | None = None,
    log: Callable[[str], None] = lambda _m: None,
) -> ProcessingRun:
    cfg = global_settings
    if run.status in {"done", "completed"}:
        return run
    if run.status == "awaiting_selection" and not run.selected_option_id:
        raise ValueError("run is awaiting a human option selection")
    run.status = "running"
    run.started_at_us = run.started_at_us or now_us()
    db.flush()
    checkpoint = dict(run.checkpoint_json or {})
    paused_phases = set(checkpoint.get("paused", []))
    area = _area_from_params(run.params_json)
    ctx = PipelineContext(
        db=db, run=run, cfg=cfg, area=area, debug_stop_after_phase=None, log=log
    )
    stop = debug_stop_after_phase
    if stop and stop in paused_phases:
        stop = None  # already paused there once; don't loop forever

    def maybe_pause(phase: str) -> None:
        if stop == phase:
            checkpoint.setdefault("paused", []).append(phase)
            _save_checkpoint(run, checkpoint)
            run.status = "paused"
            db.flush()
            raise PhaseStopped(phase)

    try:
        if not checkpoint.get("calibrate"):
            _set_phase(ctx, "calibrate")
            snapshot, models = calibrate_all(db, run.cutoff_us, cfg)
            run.calibration_json = {
                **snapshot,
                "models": {
                    f"{station}/{boot}": model.version_hash
                    for (station, boot), model in sorted(models.items())
                },
            }
            run.calibration_hash = snapshot["hash"]
            checkpoint["calibrate"] = {"hash": snapshot["hash"]}
            _save_checkpoint(run, checkpoint)
            db.flush()
            _set_phase(ctx, "calibrate", "done")
        maybe_pause("calibrate")

        if not checkpoint.get("collect"):
            _set_phase(ctx, "collect")
            collected = collect(db, run.cutoff_us, run, class_labels)
            checkpoint["collect"] = {
                # Full input snapshot: a resumed run must see exactly this set
                # and these calibrated arrivals, even if later uploads or a
                # later calibration changed the observation rows meanwhile.
                "observations": [
                    {
                        "observation_id": o.observation_id,
                        "station_id": o.station_id,
                        "boot_id": o.boot_id,
                        "class_label": o.class_label,
                        "corrected_arrival_us": o.corrected_arrival_us,
                        "uncertainty_us": o.uncertainty_us,
                        "features": o.features,
                        "calibration_model_id": o.calibration_model_id,
                        "anchor_ids": o.anchor_ids,
                    }
                    for o in collected.observations
                ],
                "stations": {
                    k: {"id": v.id, "x": v.x, "y": v.y, "hearing_radius_m": v.hearing_radius_m}
                    for k, v in collected.stations.items()
                },
                "class_limits": collected.class_limits,
                "skipped_uncalibrated": collected.skipped_uncalibrated,
                "excluded_stations": sorted(collected.excluded_station_ids),
                "excluded_observations": collected.excluded_observations,
                "class_labels": class_labels,
            }
            _save_checkpoint(run, checkpoint)
            db.flush()
            _set_phase(ctx, "collect", "done")
        else:
            collected = _restore_collected(db, checkpoint["collect"], run.cutoff_us)
        maybe_pause("collect")

        if not checkpoint.get("candidates"):
            _set_phase(ctx, "candidates")
            cand_set = association.build_candidates(
                collected.observations, collected.stations, collected.class_limits, cfg, area
            )
            checkpoint["candidates"] = {
                "candidate_ids": [c.candidate_id for c in cand_set.candidates],
                "candidates": [
                    {
                        "candidate_id": c.candidate_id,
                        "class_label": c.class_label,
                        "observation_ids": c.observation_ids,
                        "station_ids": c.station_ids,
                        "residual_rms_us": c.residual_rms_us,
                        "residual_sq_us2": c.residual_sq_us2,
                        "emission_us": c.emission_us,
                        "region_count": len(c.localization.regions),
                        "position_certain": c.localization.position_certain,
                        "window_start_us": c.window_start_us,
                        "window_end_us": c.window_end_us,
                    }
                    for c in cand_set.candidates
                ],
                "cliques_examined": cand_set.cliques_examined,
                "cliques_infeasible": cand_set.cliques_infeasible,
                "truncated": cand_set.truncated,
                "notes": cand_set.notes,
            }
            _save_checkpoint(run, checkpoint)
            db.flush()
            _set_phase(ctx, "candidates", "done")
        else:
            cand_set = _restore_candidates(checkpoint["candidates"], collected, cfg, area)
        maybe_pause("candidates")

        pm = compute_progress_map(db, run, run.cutoff_us)
        if not checkpoint.get("localize"):
            _set_phase(ctx, "localize")
            geo_notes = [
                f"{c.candidate_id}: {len(c.localization.regions)} feasible regions"
                for c in cand_set.candidates
                if len(c.localization.regions) > 1
            ]
            checkpoint["localize"] = {"notes": geo_notes}
            _save_checkpoint(run, checkpoint)
            db.flush()
            _set_phase(ctx, "localize", "done")
        maybe_pause("localize")

        if run.selected_option_id and checkpoint.get("solve"):
            # An investigator already picked an option; finalisation continues
            # from the stored solve results.
            finalization_by_candidate = checkpoint["solve"]["finalization"]
        elif not checkpoint.get("solve"):
            _set_phase(ctx, "solve")
            weights: dict[int, int] = {}
            finalization_by_candidate: dict[str, dict] = {}
            for cand in cand_set.candidates:
                state = _finalization_state(db, pm, cand, collected.stations, run, cfg)
                finalization_by_candidate[cand.candidate_id] = state
                # Weight 1 means "reliably explained": at least one finalisable
                # event could account for this record, so the record will not
                # move later.  Set once and never reset by a later candidate.
                if state["eligible"]:
                    for oid in cand.observation_ids:
                        weights[oid] = 1
                else:
                    for oid in cand.observation_ids:
                        weights.setdefault(oid, 0)
            result = solver.solve(cand_set.candidates, weights, cfg)
            options = [
                {
                    "option_id": option.option_id,
                    "explained_observations": option.explained_observations,
                    "reliable_observations": option.reliable_observations,
                    "total_residual_us": option.total_residual_sq_us2,
                    "event_count": option.event_count,
                    "candidate_ids": option.candidate_ids,
                }
                for option in result.options
            ]
            run.run_options_json = {
                "options": options,
                "tied": result.tied,
                "proven_optimal": result.proven_optimal,
                "nodes_explored": result.nodes_explored,
                "notes": result.notes,
            }
            checkpoint["solve"] = {
                "finalization": finalization_by_candidate,
                "explained": result.best.explained_observations if result.best else 0,
                "reliable": result.best.reliable_observations if result.best else 0,
                "options": {o["option_id"]: o for o in options},
                "proven_optimal": result.proven_optimal,
                "nodes_explored": result.nodes_explored,
                "components": result.components,
            }
            _save_checkpoint(run, checkpoint)
            if result.tied:
                run.status = "awaiting_selection"
                run.phase = "solve"
                run.selected_option_id = None
                run.summary_json = {
                    "status": "awaiting_selection",
                    "options": options,
                    "notes": result.notes,
                }
                db.flush()
                _audit(db, run, "run.ambiguous", {"options": options})
                return run
            if result.best is None:
                run.status = "done"
                run.phase = "solve"
                run.finished_at_us = now_us()
                if collected.skipped_uncalibrated and not collected.observations:
                    reason = (
                        "no calibrated observations available: every sound record was "
                        "held as uncalibrated (see /observations?calibration_status=uncalibrated)"
                    )
                elif not collected.observations:
                    reason = "no sound observations matched the requested classes before the cutoff"
                else:
                    reason = (
                        "no candidate event packs a set of observations: records had no "
                        "consistent common source region"
                    )
                run.summary_json = {
                    "events_created": 0,
                    "reason": reason,
                    "skipped_uncalibrated": collected.skipped_uncalibrated,
                    "observations_considered": len(collected.observations),
                    "notes": result.notes,
                }
                db.flush()
                return run
            run.selected_option_id = result.best.option_id
            db.flush()
            _set_phase(ctx, "solve", "done")
        else:
            finalization_by_candidate = checkpoint["solve"]["finalization"]
        maybe_pause("solve")

        existing_events = list(
            db.execute(select(Event).where(Event.run_id == run.id)).scalars()
        )
        if existing_events:
            # The run already materialised its events before being paused; do
            # not duplicate them on resume.
            run.finished_at_us = run.finished_at_us or now_us()
            _finish_run(ctx, [e.id for e in existing_events])
            db.flush()
            return run
        if not checkpoint.get("finalize"):
            _set_phase(ctx, "finalize")
            created = _materialize_events(
                db, run, cand_set, finalization_by_candidate, pm, collected, cfg, checkpoint
            )
            checkpoint["finalize"] = {"event_ids": created}
            _save_checkpoint(run, checkpoint)
            run.finished_at_us = now_us()
            run.summary_json = {
                "events_created": len(created),
                "event_ids": created,
                "calibration_hash": run.calibration_hash,
                "skipped_uncalibrated": collected.skipped_uncalibrated,
                "observations_considered": len(collected.observations),
                "candidate_notes": cand_set.notes,
            }
            _finish_run(ctx, created)
            db.flush()
            return run
    except PhaseStopped:
        run.status = "paused"
        _save_checkpoint(run, checkpoint)
        db.flush()
        return run
    except Exception as exc:
        run.status = "failed"
        run.error = f"{type(exc).__name__}: {exc}"
        run.finished_at_us = now_us()
        db.flush()
        raise


def _finish_run(ctx: PipelineContext, created_event_ids: list[int]) -> None:
    """Set the terminal status of a run without raising.

    ``done`` is a completed run; ``awaiting_review`` means at least one event
    is still provisional because a station has not yet reported the interval,
    and the run must be re-executed once it has.
    """
    run = ctx.run
    run.phase = "finalize"
    run.checkpoint_json = {**dict(run.checkpoint_json or {}), "finalize": {"event_ids": created_event_ids}}
    run.finished_at_us = now_us()
    pending = [
        e for e in ctx.db.execute(select(Event).where(Event.id.in_(created_event_ids))).scalars()
        if e.status == "provisional"
    ] if created_event_ids else []
    log = list(run.phase_log or [])
    if pending:
        run.status = "awaiting_review"
        log.append(
            {
                "phase": "finalize",
                "status": "awaiting_review",
                "at": iso(now_us()),
                "provisional_events": [e.id for e in pending],
            }
        )
    else:
        run.status = "done"
        log.append({"phase": "finalize", "status": "done", "at": iso(now_us())})
    run.phase_log = log
    ctx.db.flush()


def _audit(db: Session, run: ProcessingRun, action: str, detail: dict) -> None:
    db.add(
        AuditLog(
            actor=run.created_by or "system",
            action=action,
            target_type="run",
            target_id=str(run.id),
            detail=detail,
            created_at_us=now_us(),
        )
    )


def _area_from_params(params: dict) -> Area:
    return Area(
        params.get("area_x_min", global_settings.area_x_min),
        params.get("area_x_max", global_settings.area_x_max),
        params.get("area_y_min", global_settings.area_y_min),
        params.get("area_y_max", global_settings.area_y_max),
    )


def _restore_collected(db: Session, cp: dict, cutoff_us: int) -> Collected:
    """Restore the exact input snapshot frozen by the original run.

    Nothing is re-read from the observation table, so a resume cannot pick up
    post-cutoff recalibration or new uploads.
    """
    observations = [
        association.ObsInput(
            observation_id=o["observation_id"],
            station_id=o["station_id"],
            boot_id=o["boot_id"],
            class_label=o["class_label"],
            corrected_arrival_us=o["corrected_arrival_us"],
            uncertainty_us=o["uncertainty_us"],
            features=[float(v) for v in o["features"]],
            calibration_model_id=o["calibration_model_id"],
            anchor_ids=list(o.get("anchor_ids") or []),
        )
        for o in cp["observations"]
    ]
    stations = {
        k: StationPoint(v["id"], v["x"], v["y"], v["hearing_radius_m"])
        for k, v in cp["stations"].items()
    }
    class_limits = {k: float(v) for k, v in cp["class_limits"].items()}
    excluded = set(cp.get("excluded_stations", []))
    return Collected(
        observations=observations,
        stations=stations,
        class_limits=class_limits,
        skipped_uncalibrated=cp.get("skipped_uncalibrated", 0),
        excluded_station_ids=excluded,
        excluded_observations=cp.get("excluded_observations", 0),
    )


def _restore_candidates(
    cp: dict, collected: Collected, cfg: Settings, area: Area
) -> association.CandidateSet:
    rebuilt = association.build_candidates(
        collected.observations, collected.stations, collected.class_limits, cfg, area
    )
    expected = cp.get("candidate_ids", [])
    if [c.candidate_id for c in rebuilt.candidates] != expected:  # pragma: no cover
        raise RuntimeError(
            "checkpoint mismatch: rebuilding candidates from the stored input set "
            "produced a different candidate list; refusing to continue with mixed inputs"
        )
    return rebuilt


def _materialize_events(
    db: Session,
    run: ProcessingRun,
    cand_set: association.CandidateSet,
    finalization: dict[str, dict],
    pm: ProgressMap,
    collected: Collected,
    cfg: Settings,
    checkpoint: dict,
) -> list[int]:
    options = (run.run_options_json or {}).get("options", [])
    selected = run.selected_option_id
    chosen: dict | None = None
    for opt in options:
        if opt["option_id"] == selected:
            chosen = opt
            break
    if chosen is None and options:
        chosen = options[0]
        run.selected_option_id = chosen["option_id"]
    if chosen is None:
        return []
    by_id = {c.candidate_id: c for c in cand_set.candidates}
    created: list[Event] = []
    # Revision numbers are allocated per lineage so a second revision of the
    # same activity is revision 3, not a fresh revision 1.
    lineage_versions: dict[int, int] = checkpoint.get("lineage_versions") or {}
    for cid in chosen["candidate_ids"]:
        cand = by_id.get(cid)
        if cand is None:  # pragma: no cover
            continue
        state = finalization.get(cid, {"eligible": False, "status": "provisional"})
        lineage = _lineage_key(db, cand)
        version = int(lineage_versions.get(lineage, 1))
        created.append(_create_event(db, run, cand, state, pm, collected, cfg, version))
        lineage_versions[lineage] = version + 1
    checkpoint["lineage_versions"] = lineage_versions
    db.flush()
    _link_revisions(db, run, created)
    db.flush()
    return [e.id for e in created]


def _lineage_key(db: Session, cand: association.Candidate) -> int:
    """Stable key for an activity: its earliest observation id.

    Re-runs and revisions re-explain the same records, so the earliest record
    identifies the lineage across runs.
    """
    return min(cand.observation_ids) if cand.observation_ids else 0


def _create_event(
    db: Session,
    run: ProcessingRun,
    cand: association.Candidate,
    state: dict,
    pm: ProgressMap,
    collected: Collected,
    cfg: Settings,
    revision: int = 1,
) -> Event:
    region = cand.localization.regions[0]
    t = now_us()
    regions_json = [
        {
            "region_id": r.region_id,
            "bounds": r.bounds,
            "area_m2": r.area_m2,
            "representative": r.representative,
            "emission_us": int(round(r.emission_us)),
            "emission_min_us": r.emission_min_us,
            "emission_max_us": r.emission_max_us,
            "extent_m": r.extent_m,
            "cell_count": len(r.cells),
            "cells_truncated": r.cells_truncated,
            "cells": _region_cells(r, cfg),
        }
        for r in cand.localization.regions
    ]
    event = Event(
        run_id=run.id,
        batch_id=run.batch_id,
        class_label=cand.class_label,
        status=state["status"],
        revision=revision,
        reason=(
            "candidate region exceeds the definite-position extent; report as an area"
            if not cand.localization.position_certain
            else None
        ),
        x=region.representative["x"],
        y=region.representative["y"],
        x_min=region.bounds["x_min"],
        x_max=region.bounds["x_max"],
        y_min=region.bounds["y_min"],
        y_max=region.bounds["y_max"],
        position_precision_m=cand.localization.grid_resolution_m,
        emission_us=cand.emission_us,
        emission_min_us=region.emission_min_us,
        emission_max_us=region.emission_max_us,
        residual_rms_us=cand.residual_rms_us,
        residual_max_us=int(max(abs(r) for r in region_residuals(cand))),
        station_count=len(cand.station_ids),
        observation_count=len(cand.observation_ids),
        region_count=len(cand.localization.regions),
        regions_json=regions_json,
        finalization_json=state,
        calibration_hash=run.calibration_hash or "",
        supersedes_events=None,
        window_start_us=cand.window_start_us,
        window_end_us=cand.window_end_us,
        created_at_us=t,
        updated_at_us=t,
    )
    db.add(event)
    db.flush()
    for oid, residual in zip(cand.observation_ids, region_residuals(cand)):
        obs = db.get(Observation, oid)
        model = db.get(CalibrationModel, obs.calibration_model_id) if obs.calibration_model_id else None
        db.add(
            EventObservation(
                event_id=event.id,
                observation_id=oid,
                station_id=obs.station_id,
                corrected_arrival_us=obs.corrected_arrival_us or 0,
                uncertainty_us=obs.corrected_uncertainty_us or 0,
                residual_us=int(round(residual)),
                calibration_model_id=obs.calibration_model_id,
                anchor_ids=(model.anchor_ids if model else []),
            )
        )
        obs.current_event_id = event.id
    for station_id in cand.station_ids:
        exc = _matching_exclusion(pm, station_id, cand.window_start_us, cand.window_end_us)
        if exc is not None:
            db.add(
                EventExclusion(
                    event_id=event.id,
                    station_id=station_id,
                    exclusion_id=exc.id,
                    reason=exc.reason,
                    created_at_us=t,
                )
            )
    return event


def region_residuals(cand: association.Candidate) -> list[float]:
    """Approximate per-observation residuals at the representative point.

    Recomputed exactly from the representative coordinates so that the stored
    residual can be audited against the event's published position.
    """
    region = cand.localization.regions[0]
    x = region.representative["x"]
    y = region.representative["y"]
    return cand.localization.residuals_by_observation(x, y)


def _region_cells(region, cfg: Settings) -> list[list[float]]:
    cells = [[round(c.x_min, 2), round(c.y_min, 2), round(c.x_max, 2), round(c.y_max, 2)] for c in region.cells]
    if len(cells) > cfg.max_returned_cells_per_region:
        return cells[: cfg.max_returned_cells_per_region]
    return cells


def _matching_exclusion(
    pm: ProgressMap, station_id: str, start_us: int, end_us: int
) -> StationExclusion | None:
    for exc in pm.exclusion_records:
        if exc.station_id != station_id:
            continue
        if exc.from_ref_us is not None and end_us < exc.from_ref_us:
            continue
        if exc.to_ref_us is not None and start_us > exc.to_ref_us:
            continue
        return exc
    return None


def _link_revisions(db: Session, run: ProcessingRun, new_events: list[Event]) -> None:
    """Attach every new event to the prior events it re-interprets.

    A prior event is superseded when at least one of its observations appears
    in a new event, or when the same class and overlapping window is now
    explained by an event that shares a station.  If a prior event's records
    are spread over several new events, it is explicitly recorded as split
    into all of them: the old row is marked superseded and each new event names
    it, so no activity is quietly double-counted.
    """
    if not new_events:
        return
    # Any earlier interpretation that is still live must be superseded when the
    # new run re-explains its records.  This applies to a plain re-run as well
    # as to an explicit revision run: otherwise late data would leave the old
    # event in place and count the same activity twice.
    prior = list(
        db.execute(
            select(Event).where(
                Event.run_id != run.id,
                Event.superseded_by_event_id.is_(None),
                Event.status.in_(["final", "provisional"]),
            )
        ).scalars()
    )
    if not prior:
        return
    obs_by_event: dict[int, set[int]] = {}
    stations_by_event: dict[int, set[str]] = {}
    for ev in prior + new_events:
        links = list(
            db.execute(
                select(EventObservation).where(EventObservation.event_id == ev.id)
            ).scalars()
        )
        obs_by_event[ev.id] = {link.observation_id for link in links}
        stations_by_event[ev.id] = {link.station_id for link in links}
    t = now_us()
    for old in prior:
        old_ids = obs_by_event.get(old.id, set())
        matches: list[Event] = []
        for new in new_events:
            new_ids = obs_by_event.get(new.id, set())
            shared = old_ids & new_ids
            if shared:
                matches.append(new)
                continue
            if (
                new.class_label == old.class_label
                and _windows_overlap(old, new)
                and stations_by_event.get(old.id, set()) & stations_by_event.get(new.id, set())
            ):
                matches.append(new)
        if not matches:
            continue
        matches.sort(key=lambda e: (-len(obs_by_event.get(e.id, set()) & old_ids), e.id))
        old.superseded_by_event_id = matches[0].id
        old.status = "superseded"
        old.updated_at_us = t
        # Records that the new interpretation no longer uses must not keep
        # pointing at a superseded event.
        reassigned = set().union(*(obs_by_event.get(new.id, set()) for new in matches))
        for oid in old_ids - reassigned:
            obs_row = db.get(Observation, oid)
            if obs_row is not None and obs_row.current_event_id == old.id:
                obs_row.current_event_id = None
        detail = "split_across_events" if len(matches) > 1 else "replaced_by_event"
        for new in matches:
            supersedes = list(new.supersedes_events or [])
            if old.id not in supersedes:
                supersedes.append(old.id)
            new.supersedes_events = supersedes
            # A revision is one step above the newest version in its lineage,
            # whether that lineage was superseded by this run or earlier.  Only
            # a revision run raises the number; a plain re-run of the same
            # cutoff is a re-computation, not a new revision of the record.
            if run.mode == "revision":
                new.revision = max(new.revision, int(old.revision) + 1)
            db.add(
                EventRevisionLink(
                    new_event_id=new.id,
                    old_event_id=old.id,
                    reason=(
                        f"{detail}: late report or calibration revision re-associated "
                        f"{len(obs_by_event.get(new.id, set()) & old_ids)} of "
                        f"{len(old_ids)} prior observations"
                    ),
                    created_at_us=t,
                )
            )
    db.flush()


def _windows_overlap(a: Event, b: Event) -> bool:
    return not (a.window_end_us < b.window_start_us or b.window_end_us < a.window_start_us)
