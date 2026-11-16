"""API routers."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from . import analysis, pipeline, schemas
from .config import settings
from .db import Base, engine, get_db
from .ingest import ingest_batch, ingest_progress, parse_local_us_iso
from .models import (
    AnalysisRegion,
    AnalysisReport,
    AuditLog,
    CalibrationModel,
    CalibrationSourceRecord,
    Event,
    EventExclusion,
    EventObservation,
    EventRevisionLink,
    HabitatMap,
    Observation,
    ObservationConflict,
    ProcessingRun,
    ReferenceComposition,
    SoundClass,
    Station,
    StationBoot,
    StationExclusion,
    StationProgress,
)
from .timeutil import iso, now_us, to_us

router = APIRouter()


# --------------------------------------------------------------------------- #
# Health / reset
# --------------------------------------------------------------------------- #
@router.get("/health", response_model=schemas.HealthResponse, tags=["system"], summary="Liveness and database connectivity")
def health(db: Session = Depends(get_db)):
    try:
        db.execute(text("select 1"))
        db_state = "ok"
    except Exception as exc:  # pragma: no cover
        db_state = f"error: {exc}"
    from . import __version__

    return schemas.HealthResponse(status="ok", database=db_state, version=__version__)


@router.post("/admin/reset", tags=["system"], summary="Drop and recreate all tables (development only)")
def reset(confirm: bool = Query(False), db: Session = Depends(get_db)):
    if not settings.allow_reset:
        raise HTTPException(403, "reset is disabled (WETLAND_ALLOW_RESET=false)")
    if not confirm:
        raise HTTPException(
            400, "destructive: pass ?confirm=true to drop all observations, events and runs"
        )
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    return {"reset": True}


# --------------------------------------------------------------------------- #
# Stations
# --------------------------------------------------------------------------- #
@router.post("/stations", response_model=schemas.StationOut, status_code=201, tags=["stations"], summary="Create a station with its synthetic coordinates")
def create_station(body: schemas.StationCreate, db: Session = Depends(get_db)):
    if db.get(Station, body.id):
        raise HTTPException(409, f"station {body.id} already exists")
    station = Station(
        id=body.id,
        name=body.name,
        x=body.x,
        y=body.y,
        hearing_radius_m=body.hearing_radius_m or settings.hearing_radius_m,
        active=body.active,
        notes=body.notes,
        created_at_us=now_us(),
    )
    db.add(station)
    db.commit()
    return _station_out(db, station)


@router.put("/stations/{station_id}", response_model=schemas.StationOut, tags=["stations"], summary="Update a station")
def update_station(station_id: str, body: schemas.StationCreate, db: Session = Depends(get_db)):
    if station_id != body.id:
        raise HTTPException(400, "path station id and body id differ")
    station = db.get(Station, station_id)
    if not station:
        raise HTTPException(404, "station not found")
    station.name = body.name
    station.x = body.x
    station.y = body.y
    station.hearing_radius_m = body.hearing_radius_m or station.hearing_radius_m
    station.active = body.active
    station.notes = body.notes
    db.commit()
    return _station_out(db, station)


def _station_out(db: Session, station: Station) -> schemas.StationOut:
    boots = db.scalar(
        select(func.count(StationBoot.id)).where(StationBoot.station_id == station.id)
    )
    return schemas.StationOut(
        id=station.id,
        name=station.name,
        x=station.x,
        y=station.y,
        hearing_radius_m=station.hearing_radius_m,
        active=station.active,
        notes=station.notes,
        created_at=iso(station.created_at_us),
        boot_count=boots or 0,
    )


@router.get("/stations", response_model=list[schemas.StationOut], tags=["stations"], summary="List stations")
def list_stations(db: Session = Depends(get_db)):
    stations = list(db.execute(select(Station).order_by(Station.id)).scalars())
    return [_station_out(db, s) for s in stations]


@router.get("/stations/{station_id}/progress", tags=["stations"], summary="List a station's declared upload progress")
def station_progress(station_id: str, db: Session = Depends(get_db)):
    if not db.get(Station, station_id):
        raise HTTPException(404, "station not found")
    reports = list(
        db.execute(
            select(StationProgress)
            .where(StationProgress.station_id == station_id)
            .order_by(StationProgress.id.desc())
            .limit(50)
        ).scalars()
    )
    models = {
        m.boot_id: m
        for m in db.execute(
            select(CalibrationModel)
            .where(CalibrationModel.station_id == station_id)
            .order_by(CalibrationModel.id)
        ).scalars()
    }
    out = []
    for rep in reports:
        model = models.get(rep.boot_id)
        out.append(
            {
                # Local clock readings are rendered without a zone: they are
                # counters, and mapping them to reference time is a separate,
                # calibrated operation.
                "id": rep.id,
                "boot_id": rep.boot_id,
                "declared_local_through": parse_local_us_iso(rep.declared_local_us),
                "complete": rep.complete,
                "gaps": rep.gaps_json or [],
                "received_at": iso(rep.received_at_us),
                "calibration_model_id": model.id if model else None,
                "calibration_valid": bool(model.valid) if model else False,
            }
        )
    return {"station_id": station_id, "reports": out}


@router.post("/stations/{station_id}/progress", tags=["stations"], summary="Declare a station's upload progress")
def post_progress(
    station_id: str, body: schemas.ProgressRequest, db: Session = Depends(get_db)
):
    try:
        ingest_progress(db, station_id, body)
    except KeyError:
        raise HTTPException(404, "station not found")
    db.commit()
    return {"recorded": True}


@router.post(
    "/stations/{station_id}/observations",
    response_model=schemas.BatchResponse,
    tags=["ingestion"],
    status_code=201,
)
def post_observations(
    station_id: str, body: schemas.ObservationBatch, db: Session = Depends(get_db)
):
    if station_id != body.station_id:
        raise HTTPException(400, "path station id and body station_id differ")
    if not db.get(Station, station_id):
        raise HTTPException(404, "station not found; create it first")
    try:
        result = ingest_batch(db, body)
    except KeyError:
        db.rollback()
        raise HTTPException(404, "station not found; create it first")
    except ValueError as exc:
        db.rollback()
        raise HTTPException(409, str(exc))
    db.commit()
    counts = schemas.IngestionCounts(
        received=len(body.records),
        accepted=len(result.accepted_ids),
        duplicates=len(result.duplicates),
        conflicts=len(result.conflicts),
        rejected=result.rejected,
    )
    return schemas.BatchResponse(
        batch_id=result.batch_id,
        replayed=result.replayed,
        boot_id=body.boot_id,
        counts=counts,
        accepted_observation_ids=result.accepted_ids,
        duplicate_sequences=result.duplicates,
        conflict_sequences=result.conflicts,
        progress_recorded=result.progress_recorded,
        replay_of_batch_id=result.replay_of_batch_id,
        original_observation_ids=result.original_observation_ids,
    )


# --------------------------------------------------------------------------- #
# Calibration sources & sound classes
# --------------------------------------------------------------------------- #
@router.post(
    "/calibration-sources",
    response_model=schemas.CalibrationSourceOut,
    status_code=201,
    tags=["configuration"],
)
def create_calibration_source(body: schemas.CalibrationSourceCreate, db: Session = Depends(get_db)):
    if db.get(CalibrationSourceRecord, body.id):
        raise HTTPException(409, "calibration source already exists")
    row = CalibrationSourceRecord(
        id=body.id,
        name=body.name,
        x=body.x,
        y=body.y,
        emissions_json=[to_us(e) for e in body.emissions],
        notes=body.notes,
        created_at_us=now_us(),
    )
    db.add(row)
    db.commit()
    return _cal_source_out(row)


def _cal_source_out(row: CalibrationSourceRecord) -> schemas.CalibrationSourceOut:
    return schemas.CalibrationSourceOut(
        id=row.id,
        name=row.name,
        x=row.x,
        y=row.y,
        emissions=[iso(e) for e in row.emissions_json],
        notes=row.notes,
    )


@router.get(
    "/calibration-sources",
    response_model=list[schemas.CalibrationSourceOut],
    tags=["configuration"],
)
def list_calibration_sources(db: Session = Depends(get_db)):
    return [
        _cal_source_out(r)
        for r in db.execute(select(CalibrationSourceRecord).order_by(CalibrationSourceRecord.id)).scalars()
    ]


@router.post("/sound-classes", response_model=schemas.ClassOut, status_code=201, tags=["configuration"], summary="Register a sound class and feature distance limit")
def create_sound_class(body: schemas.ClassCreate, db: Session = Depends(get_db)):
    if db.get(SoundClass, body.id):
        raise HTTPException(409, "sound class already exists")
    row = SoundClass(
        id=body.id,
        description=body.description,
        feature_dim=body.feature_dim,
        feature_distance_limit=body.feature_distance_limit,
        created_at_us=now_us(),
    )
    db.add(row)
    db.commit()
    return schemas.ClassOut(
        id=row.id,
        description=row.description,
        feature_dim=row.feature_dim,
        feature_distance_limit=row.feature_distance_limit,
        observation_count=0,
    )


@router.get("/sound-classes", response_model=list[schemas.ClassOut], tags=["configuration"], summary="List sound classes")
def list_sound_classes(db: Session = Depends(get_db)):
    rows = list(db.execute(select(SoundClass).order_by(SoundClass.id)).scalars())
    counts = dict(
        db.execute(
            select(Observation.class_label, func.count(Observation.id)).group_by(Observation.class_label)
        ).all()
    )
    return [
        schemas.ClassOut(
            id=r.id,
            description=r.description,
            feature_dim=r.feature_dim,
            feature_distance_limit=r.feature_distance_limit,
            observation_count=counts.get(r.id, 0),
        )
        for r in rows
    ]


# --------------------------------------------------------------------------- #
# Observations
# --------------------------------------------------------------------------- #
def _observation_out(obs: Observation) -> schemas.ObservationOut:
    return schemas.ObservationOut(
        id=obs.id,
        station_id=obs.station_id,
        boot_id=obs.boot_id,
        source_seq=obs.source_seq,
        kind=obs.kind,
        class_label=obs.class_label,
        local_capture_at=parse_local_us_iso(obs.local_capture_us),
        uploaded_at=iso(obs.uploaded_at_us),
        ingested_at=iso(obs.ingest_at_us),
        features=obs.features,
        uncertainty_ms=obs.uncertainty_us / 1000.0,
        calibration_status=obs.calibration_status,
        calibration_reason=obs.calibration_reason,
        calibration_model_id=obs.calibration_model_id,
        corrected_arrival=iso(obs.corrected_arrival_us),
        current_event_id=obs.current_event_id,
    )


@router.get("/observations", response_model=schemas.ObservationList, tags=["observations"], summary="List stored observations with calibration state")
def list_observations(
    station_id: str | None = None,
    boot_id: str | None = None,
    class_label: str | None = None,
    kind: str | None = None,
    calibration_status: str | None = None,
    event_id: int | None = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    stmt = select(Observation)
    if station_id:
        stmt = stmt.where(Observation.station_id == station_id)
    if boot_id:
        stmt = stmt.where(Observation.boot_id == boot_id)
    if class_label:
        stmt = stmt.where(Observation.class_label == class_label)
    if kind:
        stmt = stmt.where(Observation.kind == kind)
    if calibration_status:
        stmt = stmt.where(Observation.calibration_status == calibration_status)
    if event_id is not None:
        stmt = stmt.join(EventObservation, EventObservation.observation_id == Observation.id).where(
            EventObservation.event_id == event_id
        )
    total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    rows = list(
        db.execute(stmt.order_by(Observation.id).offset(offset).limit(limit)).scalars()
    )
    return schemas.ObservationList(total=total, observations=[_observation_out(o) for o in rows])


@router.get("/observations/{observation_id}", response_model=schemas.ObservationOut, tags=["observations"], summary="Fetch one observation")
def get_observation(observation_id: int, db: Session = Depends(get_db)):
    obs = db.get(Observation, observation_id)
    if not obs:
        raise HTTPException(404, "observation not found")
    return _observation_out(obs)


@router.get("/conflicts", response_model=list[schemas.ConflictOut], tags=["observations"], summary="List retransmissions that conflicted with stored records")
def list_conflicts(
    station_id: str | None = None,
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    stmt = select(ObservationConflict)
    if station_id:
        stmt = stmt.where(ObservationConflict.station_id == station_id)
    rows = list(
        db.execute(stmt.order_by(ObservationConflict.id.desc()).limit(limit)).scalars()
    )
    return [
        schemas.ConflictOut(
            id=r.id,
            station_id=r.station_id,
            boot_id=r.boot_id,
            source_seq=r.source_seq,
            existing_observation_id=r.existing_observation_id,
            content_hash=r.content_hash,
            payload=r.payload,
            received_at=iso(r.received_at_us),
            resolved=r.resolved,
            resolution_note=r.resolution_note,
        )
        for r in rows
    ]


@router.get("/calibrations", response_model=schemas.CalibrationList, tags=["calibration"], summary="List per-boot clock calibrations")
def list_calibrations(
    station_id: str | None = None,
    valid: bool | None = None,
    db: Session = Depends(get_db),
):
    stmt = select(CalibrationModel).order_by(CalibrationModel.station_id, CalibrationModel.boot_id, CalibrationModel.id)
    if station_id:
        stmt = stmt.where(CalibrationModel.station_id == station_id)
    if valid is not None:
        stmt = stmt.where(CalibrationModel.valid.is_(valid))
    rows = list(db.execute(stmt).scalars())
    return schemas.CalibrationList(
        total=len(rows), calibrations=[_calibration_out(r) for r in rows]
    )


def _calibration_out(row: CalibrationModel) -> schemas.CalibrationOut:
    segments = []
    for seg in row.segments or []:
        segments.append(
            schemas.CalibrationSegmentOut(
                index=seg["index"],
                anchor_count=len(seg.get("anchor_ids", [])),
                anchor_ids=seg.get("anchor_ids", []),
                local_from=parse_local_us_iso(seg["local_from_us"]),
                local_to=parse_local_us_iso(seg["local_to_us"]),
                slope=seg["slope"],
                intercept_ms=seg["intercept_us"] / 1000.0,
                rms_residual_ms=seg["rms_residual_us"] / 1000.0,
                max_residual_ms=seg["max_residual_us"] / 1000.0,
            )
        )
    return schemas.CalibrationOut(
        id=row.id,
        station_id=row.station_id,
        boot_id=row.boot_id,
        version_hash=row.version_hash,
        anchor_count=row.anchor_count,
        anchor_ids=row.anchor_ids or [],
        valid=row.valid,
        invalid_reason=row.invalid_reason,
        rms_residual_ms=(row.rms_residual_us / 1000.0 if row.rms_residual_us is not None else None),
        max_residual_ms=(row.max_residual_us / 1000.0 if row.max_residual_us is not None else None),
        uncertainty_ms=(row.uncertainty_us / 1000.0 if row.uncertainty_us is not None else None),
        anchor_local_span_s=(
            (row.local_anchor_max_us - row.local_anchor_min_us) / 1e6
            if row.local_anchor_min_us is not None and row.local_anchor_max_us is not None
            else None
        ),
        segments=segments,
        created_at=iso(row.created_at_us),
    )


# --------------------------------------------------------------------------- #
# Exclusions
# --------------------------------------------------------------------------- #
@router.post(
    "/stations/{station_id}/exclusions",
    response_model=schemas.ExclusionOut,
    status_code=201,
    tags=["exclusions"],
)
def create_station_exclusion(
    station_id: str, body: schemas.StationExclusionCreate, db: Session = Depends(get_db)
):
    if not db.get(Station, station_id):
        raise HTTPException(404, "station not found")
    row = StationExclusion(
        station_id=station_id,
        reason=body.reason,
        from_ref_us=to_us(body.from_ref_at) if body.from_ref_at else None,
        to_ref_us=to_us(body.to_ref_at) if body.to_ref_at else None,
        created_by=body.created_by,
        active=True,
        created_at_us=now_us(),
    )
    db.add(row)
    db.add(
        AuditLog(
            actor=body.created_by,
            action="station_exclusion.create",
            target_type="station",
            target_id=station_id,
            detail={"reason": body.reason},
            created_at_us=now_us(),
        )
    )
    db.commit()
    return _exclusion_out(row)


def _exclusion_out(row: StationExclusion) -> schemas.ExclusionOut:
    return schemas.ExclusionOut(
        id=row.id,
        station_id=row.station_id,
        reason=row.reason,
        from_ref_at=iso(row.from_ref_us),
        to_ref_at=iso(row.to_ref_us),
        active=row.active,
        created_by=row.created_by,
        created_at=iso(row.created_at_us),
    )


@router.get("/exclusions", response_model=list[schemas.ExclusionOut], tags=["exclusions"], summary="List exclusions")
def list_exclusions(active_only: bool = True, db: Session = Depends(get_db)):
    stmt = select(StationExclusion).order_by(StationExclusion.id.desc())
    if active_only:
        stmt = stmt.where(StationExclusion.active.is_(True))
    return [_exclusion_out(r) for r in db.execute(stmt).scalars()]


@router.delete("/exclusions/{exclusion_id}", response_model=schemas.ExclusionOut, tags=["exclusions"], summary="Deactivate an exclusion")
def deactivate_exclusion(exclusion_id: int, db: Session = Depends(get_db)):
    row = db.get(StationExclusion, exclusion_id)
    if not row:
        raise HTTPException(404, "exclusion not found")
    row.active = False
    db.add(
        AuditLog(
            actor="investigator",
            action="station_exclusion.deactivate",
            target_type="station_exclusion",
            target_id=str(row.id),
            detail={"station_id": row.station_id},
            created_at_us=now_us(),
        )
    )
    db.commit()
    return _exclusion_out(row)


@router.post("/events/{event_id}/exclusions", response_model=schemas.ExclusionOut, tags=["exclusions"], summary="Record an investigator exclusion for one event")
def exclude_station_for_event(
    event_id: int, body: schemas.EventExclusionCreate, db: Session = Depends(get_db)
):
    """Record an investigator's explicit exclusion of a station for one event.

    The exclusion is stored with its reason and station, and the event's
    finalization block is updated *only if* that station was the reason the
    event was still provisional.  A judgement about one station never silently
    finalises an event that has other unsettled stations.
    """
    event = db.get(Event, event_id)
    if not event:
        raise HTTPException(404, "event not found")
    if not db.get(Station, body.station_id):
        raise HTTPException(404, "station not found")
    if event.status == "superseded":
        raise HTTPException(
            409,
            "event is superseded; record the exclusion on its replacement event instead",
        )
    exc = StationExclusion(
        station_id=body.station_id,
        reason=body.reason,
        from_ref_us=event.window_start_us,
        to_ref_us=event.window_end_us,
        created_by=body.created_by,
        active=True,
        created_at_us=now_us(),
    )
    db.add(exc)
    db.flush()
    db.add(
        EventExclusion(
            event_id=event.id,
            station_id=body.station_id,
            exclusion_id=exc.id,
            reason=body.reason,
            created_at_us=now_us(),
        )
    )
    fin = dict(event.finalization_json or {})
    stations = [
        p for p in fin.get("stations", []) if p.get("station_id") != body.station_id
    ]
    existing = next(
        (p for p in fin.get("stations", []) if p.get("station_id") == body.station_id),
        None,
    )
    was_unsettled = body.station_id in (fin.get("unsettled_stations") or [])
    settled_entry = {
        "station_id": body.station_id,
        "mode": "excluded_by_investigator",
        "settled": True,
        "reason": body.reason,
        "exclusion_id": exc.id,
        "progress_through": existing.get("progress_through") if existing else None,
        "window_end": iso(event.window_end_us),
        "recorded_at": iso(now_us()),
    }
    stations.append(settled_entry)
    fin["stations"] = sorted(stations, key=lambda p: p["station_id"])
    fin["unsettled_stations"] = [
        s for s in (fin.get("unsettled_stations") or []) if s != body.station_id
    ]
    if was_unsettled:
        fin["eligible"] = not fin["unsettled_stations"]
        if fin["eligible"] and event.status == "provisional":
            event.status = "final"
            fin["status"] = "final"
            fin["resolved_by"] = "investigator_exclusion"
        elif not fin["eligible"]:
            fin.setdefault("resolution_notes", []).append(
                f"{body.station_id} excluded by investigator; still waiting on "
                + ", ".join(fin["unsettled_stations"])
            )
    else:
        fin.setdefault("resolution_notes", []).append(
            f"{body.station_id} was excluded by investigator but had already settled"
        )
    event.finalization_json = fin
    event.updated_at_us = now_us()
    db.add(
        AuditLog(
            actor=body.created_by,
            action="event.exclusion.record",
            target_type="event",
            target_id=str(event.id),
            detail={"station_id": body.station_id, "reason": body.reason},
            created_at_us=now_us(),
        )
    )
    db.commit()
    return _exclusion_out(exc)


# --------------------------------------------------------------------------- #
# Runs
# --------------------------------------------------------------------------- #
def _run_out(run: ProcessingRun) -> schemas.RunOut:
    return schemas.RunOut(
        id=run.id,
        batch_id=run.batch_id,
        mode=run.mode,
        status=run.status,
        phase=run.phase,
        cutoff=iso(run.cutoff_us),
        created_at=iso(run.created_at_us),
        started_at=iso(run.started_at_us),
        finished_at=iso(run.finished_at_us),
        calibration_hash=run.calibration_hash,
        summary=run.summary_json,
        error=run.error,
        resume_of_run_id=run.resume_of_run_id,
        revision_of_run_id=run.revision_of_run_id,
    )


@router.post("/runs", response_model=schemas.RunOut, status_code=201, tags=["runs"], summary="Start a processing run")
def create_processing_run(body: schemas.RunRequest, db: Session = Depends(get_db)):
    if body.mode == "revision":
        latest = db.execute(
            select(ProcessingRun)
            .where(ProcessingRun.status.in_(["done", "awaiting_review", "completed"]))
            .order_by(ProcessingRun.id.desc())
        ).scalars().first()
        if latest is None:
            raise HTTPException(400, "no completed run exists to revise")
        run = pipeline.create_run(
            db,
            mode="revision",
            created_by=body.created_by,
            revision_of_run_id=latest.id,
        )
    else:
        run = pipeline.create_run(db, mode="full", created_by=body.created_by)
    db.commit()
    run_id = run.id
    if body.wait:
        return _execute_now(run_id, body)
    # Asynchronous execution: the run row is committed with status=pending and
    # a worker thread owns its own database session, so a client can interrupt
    # or poll it while it is still processing.
    import threading

    def _worker() -> None:
        from .db import SessionLocal

        worker_db = SessionLocal()
        try:
            worker_run = worker_db.get(ProcessingRun, run_id)
            pipeline.execute_run(
                worker_db,
                worker_run,
                class_labels=body.class_labels,
                debug_stop_after_phase=body.debug_stop_after_phase,
            )
            worker_db.commit()
        except Exception:
            worker_db.rollback()
        finally:
            worker_db.close()

    threading.Thread(target=_worker, name=f"run-{run_id}", daemon=True).start()
    return _run_out(db.get(ProcessingRun, run_id))


def _execute_now(run_id: int, body: schemas.RunRequest) -> schemas.RunOut:
    from .db import SessionLocal

    db = SessionLocal()
    try:
        run = db.get(ProcessingRun, run_id)
        try:
            pipeline.execute_run(
                db,
                run,
                class_labels=body.class_labels,
                debug_stop_after_phase=body.debug_stop_after_phase,
            )
            db.commit()
        except Exception as exc:
            db.rollback()
            run = db.get(ProcessingRun, run_id)
            raise HTTPException(500, f"run failed: {run.error if run else exc}")
        return _run_out(db.get(ProcessingRun, run_id))
    finally:
        db.close()


@router.get("/runs", response_model=list[schemas.RunOut], tags=["runs"], summary="List processing runs")
def list_runs(limit: int = Query(20, ge=1, le=200), db: Session = Depends(get_db)):
    rows = list(
        db.execute(select(ProcessingRun).order_by(ProcessingRun.id.desc()).limit(limit)).scalars()
    )
    return [_run_out(r) for r in rows]


@router.get("/runs/{run_id}", response_model=schemas.RunOut, tags=["runs"], summary="Fetch a run")
def get_run(run_id: int, db: Session = Depends(get_db)):
    run = db.get(ProcessingRun, run_id)
    if not run:
        raise HTTPException(404, "run not found")
    return _run_out(run)


@router.get("/runs/{run_id}/options", response_model=schemas.RunOptions, tags=["runs"], summary="Show competing decompositions when the solver tied")
def run_options(run_id: int, db: Session = Depends(get_db)):
    run = db.get(ProcessingRun, run_id)
    if not run:
        raise HTTPException(404, "run not found")
    payload = run.run_options_json or {}
    options = payload.get("options", [])
    return schemas.RunOptions(
        run_id=run.id,
        status=run.status,
        options=[
            schemas.RunOption(
                option_id=o["option_id"],
                explained_observations=o["explained_observations"],
                total_residual_us=o["total_residual_us"],
                events=_option_events_preview(db, run, o["candidate_ids"]),
            )
            for o in options
        ],
        proven_optimal=payload.get("proven_optimal", False),
        notes=payload.get("notes", []),
    )


def _option_events_preview(db: Session, run: ProcessingRun, candidate_ids: list[str]) -> list[dict]:
    """Describe what each option would publish, before it is selected.

    Candidate ids are deterministic functions of the frozen input set, and the
    solver's per-candidate summary is stored in the solve checkpoint, so an
    investigator can compare options without materialising duplicates.
    """
    cp = run.checkpoint_json or {}
    candidates = (cp.get("candidates") or {}).get("candidates") or []
    finalization = (cp.get("solve") or {}).get("finalization") or {}
    by_id = {c["candidate_id"]: c for c in candidates}
    out = []
    for cid in candidate_ids:
        cand = by_id.get(cid, {})
        state = finalization.get(cid, {})
        out.append(
            {
                "candidate_id": cid,
                "station_ids": cand.get("station_ids", []),
                "observation_ids": cand.get("observation_ids", []),
                "region_count": cand.get("region_count"),
                "residual_rms_ms": (cand.get("residual_rms_us") or 0) / 1000.0,
                "emission": iso(cand["emission_us"]) if cand.get("emission_us") else None,
                "finalization_status": state.get("status", "unknown"),
                "unsettled_stations": state.get("unsettled_stations", []),
            }
        )
    return out


@router.post("/runs/{run_id}/select", response_model=schemas.RunOut, tags=["runs"], summary="Pick one decomposition for a tied run")
def select_option(run_id: int, body: schemas.SelectOptionRequest, db: Session = Depends(get_db)):
    run = db.get(ProcessingRun, run_id)
    if not run:
        raise HTTPException(404, "run not found")
    if run.status != "awaiting_selection":
        raise HTTPException(409, f"run status is {run.status}, not awaiting_selection")
    payload = run.run_options_json or {}
    valid = {o["option_id"] for o in payload.get("options", [])}
    if body.option_id not in valid:
        raise HTTPException(400, f"unknown option {body.option_id}; choose one of {sorted(valid)}")
    run.selected_option_id = body.option_id
    run.status = "running"
    run.summary_json = {**(run.summary_json or {}), "selected_by": "investigator"}
    db.add(
        AuditLog(
            actor="investigator",
            action="run.option_selected",
            target_type="run",
            target_id=str(run.id),
            detail={"option_id": body.option_id},
            created_at_us=now_us(),
        )
    )
    db.commit()
    try:
        pipeline.execute_run(db, run)
        db.commit()
    except Exception as exc:
        db.rollback()
        raise HTTPException(500, f"run failed after selection: {exc}")
    return _run_out(db.get(ProcessingRun, run.id))


@router.post(
    "/runs/{run_id}/resume",
    response_model=schemas.RunOut,
    tags=["runs"],
    summary="Resume a paused or failed run from its cutoff",
)
def resume_run(run_id: int, db: Session = Depends(get_db)):
    original = db.get(ProcessingRun, run_id)
    if not original:
        raise HTTPException(404, "run not found")
    if original.status not in {"paused", "failed"}:
        raise HTTPException(409, f"run status is {original.status}; only paused or failed runs resume")
    run = pipeline.create_run(
        db,
        mode=original.mode,
        cutoff_us=original.cutoff_us,
        created_by=original.created_by,
        resume_of_run_id=original.id,
        revision_of_run_id=original.revision_of_run_id,
    )
    # Same cutoff and same frozen calibration snapshot: copy the checkpoint
    # instead of recomputing it against newer uploads.
    run.calibration_json = original.calibration_json
    run.calibration_hash = original.calibration_hash
    run.checkpoint_json = original.checkpoint_json
    run.params_json = original.params_json
    db.commit()
    try:
        pipeline.execute_run(db, run)
        db.commit()
    except Exception:
        db.rollback()
        raise HTTPException(500, f"resume failed: {db.get(ProcessingRun, run.id).error}")
    return _run_out(db.get(ProcessingRun, run.id))


# --------------------------------------------------------------------------- #
# Events
# --------------------------------------------------------------------------- #
def _region_cells_from_json(regions: list) -> list[schemas.RegionOut]:
    out = []
    for r in regions:
        cells = r.get("cells", [])
        out.append(
            schemas.RegionOut(
                region_id=r["region_id"],
                bounds=r["bounds"],
                area_m2=r["area_m2"],
                cell_count=r.get("cell_count", len(cells)),
                representative=r["representative"],
                cells=cells,
                cells_truncated=r.get("cells_truncated", False),
            )
        )
    return out


def _event_out(event: Event) -> schemas.EventOut:
    return schemas.EventOut(
        id=event.id,
        run_id=event.run_id,
        batch_id=event.batch_id,
        class_label=event.class_label,
        status=event.status,
        revision=event.revision,
        reason=event.reason,
        position={
            "x": event.x,
            "y": event.y,
            "bbox": {
                "x_min": event.x_min,
                "x_max": event.x_max,
                "y_min": event.y_min,
                "y_max": event.y_max,
            },
            "grid_resolution_m": event.position_precision_m,
            "certain": event.region_count == 1
            and max(event.x_max - event.x_min, event.y_max - event.y_min)
            <= settings.definite_region_max_extent_m,
        },
        emission={
            "at": iso(event.emission_us),
            "min": iso(event.emission_min_us),
            "max": iso(event.emission_max_us),
        },
        residual={"rms_ms": event.residual_rms_us / 1000.0, "max_ms": event.residual_max_us / 1000.0},
        station_count=event.station_count,
        observation_count=event.observation_count,
        region_count=event.region_count,
        regions=_region_cells_from_json(event.regions_json),
        finalization=event.finalization_json,
        calibration_hash=event.calibration_hash,
        supersedes_events=event.supersedes_events or [],
        superseded_by_event_id=event.superseded_by_event_id,
        window={"start": iso(event.window_start_us), "end": iso(event.window_end_us)},
        created_at=iso(event.created_at_us),
        updated_at=iso(event.updated_at_us),
    )


@router.get("/events", response_model=schemas.EventList, tags=["events"], summary="List events with position, time and candidate regions")
def list_events(
    status: str | None = None,
    class_label: str | None = None,
    run_id: int | None = None,
    include_superseded: bool = False,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    stmt = select(Event)
    if status:
        stmt = stmt.where(Event.status == status)
    elif not include_superseded:
        stmt = stmt.where(Event.status != "superseded")
    if class_label:
        stmt = stmt.where(Event.class_label == class_label)
    if run_id is not None:
        stmt = stmt.where(Event.run_id == run_id)
    total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    rows = list(
        db.execute(stmt.order_by(Event.emission_us, Event.id).offset(offset).limit(limit)).scalars()
    )
    return schemas.EventList(total=total, events=[_event_out(e) for e in rows])


@router.get("/events/{event_id}", response_model=schemas.EventDetail, tags=["events"], summary="Event detail with observations, exclusions and revisions")
def get_event(event_id: int, db: Session = Depends(get_db)):
    event = db.get(Event, event_id)
    if not event:
        raise HTTPException(404, "event not found")
    base = _event_out(event).model_dump()
    links = list(
        db.execute(
            select(EventObservation).where(EventObservation.event_id == event.id)
        ).scalars()
    )
    observations = []
    for link in links:
        obs = db.get(Observation, link.observation_id)
        observations.append(
            schemas.EventObservationOut(
                observation_id=link.observation_id,
                station_id=link.station_id,
                boot_id=obs.boot_id,
                source_seq=obs.source_seq,
                class_label=obs.class_label,
                local_capture_at=parse_local_us_iso(obs.local_capture_us),
                corrected_arrival=iso(link.corrected_arrival_us),
                uncertainty_ms=link.uncertainty_us / 1000.0,
                residual_ms=link.residual_us / 1000.0,
                calibration_model_id=link.calibration_model_id,
                anchor_ids=link.anchor_ids or [],
            )
        )
    exclusions = list(
        db.execute(select(EventExclusion).where(EventExclusion.event_id == event.id)).scalars()
    )
    revision_links = list(
        db.execute(
            select(EventRevisionLink).where(
                (EventRevisionLink.new_event_id == event.id)
                | (EventRevisionLink.old_event_id == event.id)
            )
        ).scalars()
    )
    revisions = [
        {
            "new_event_id": r.new_event_id,
            "old_event_id": r.old_event_id,
            "reason": r.reason,
            "at": iso(r.created_at_us),
        }
        for r in revision_links
    ]
    return schemas.EventDetail(
        **base,
        observations=observations,
        exclusion_reasons=[
            {
                "station_id": e.station_id,
                "reason": e.reason,
                "exclusion_id": e.exclusion_id,
                "recorded_at": iso(e.created_at_us),
            }
            for e in exclusions
        ],
        revisions=revisions,
    )


@router.get("/events/{event_id}/anchors", tags=["events"], summary="Calibration anchors behind an event's observations")
def event_anchors(event_id: int, db: Session = Depends(get_db)):
    """Calibration anchors behind every observation used by an event."""
    event = db.get(Event, event_id)
    if not event:
        raise HTTPException(404, "event not found")
    links = list(
        db.execute(select(EventObservation).where(EventObservation.event_id == event_id)).scalars()
    )
    out = []
    for link in links:
        model = db.get(CalibrationModel, link.calibration_model_id) if link.calibration_model_id else None
        out.append(
            {
                "station_id": link.station_id,
                "observation_id": link.observation_id,
                "calibration_model_id": link.calibration_model_id,
                "version_hash": model.version_hash if model else None,
                "segments": model.segments if model else None,
                "anchor_ids": link.anchor_ids,
                "anchor_count": len(link.anchor_ids),
            }
        )
    return {"event_id": event_id, "calibration_hash": event.calibration_hash, "observations": out}


@router.get("/events/{event_id}/observations", response_model=schemas.ObservationList, tags=["events"], summary="Raw observations used by an event")
def event_observations(event_id: int, db: Session = Depends(get_db)):
    if not db.get(Event, event_id):
        raise HTTPException(404, "event not found")
    rows = list(
        db.execute(
            select(Observation)
            .join(EventObservation, EventObservation.observation_id == Observation.id)
            .where(EventObservation.event_id == event_id)
            .order_by(Observation.id)
        ).scalars()
    )
    return schemas.ObservationList(
        total=len(rows), observations=[_observation_out(o) for o in rows]
    )


# --------------------------------------------------------------------------- #
# Comparability analysis
# --------------------------------------------------------------------------- #
def _region_out(row: AnalysisRegion) -> schemas.AnalysisRegionOut:
    return schemas.AnalysisRegionOut(
        id=row.id,
        name=row.name,
        grid_cell_size_m=row.grid_cell_size_m,
        cell_count=len(row.cells_json),
        cells=[list(c) for c in row.cells_json],
        notes=row.notes,
        created_at=iso(row.created_at_us),
    )


@router.post(
    "/analysis/regions",
    response_model=schemas.AnalysisRegionOut,
    status_code=201,
    tags=["analysis"],
    summary="Publish an immutable analysis region (a named set of grid cells)",
)
def create_analysis_region(body: schemas.AnalysisRegionCreate, db: Session = Depends(get_db)):
    if db.get(AnalysisRegion, body.id):
        raise HTTPException(409, f"analysis region {body.id} already exists")
    if body.cells is not None:
        cells = sorted({(int(ix), int(iy)) for ix, iy in body.cells})
    else:
        box = body.bbox or {}
        try:
            ix0, ix1 = int(box["ix_min"]), int(box["ix_max"])
            iy0, iy1 = int(box["iy_min"]), int(box["iy_max"])
        except (KeyError, TypeError, ValueError):
            raise HTTPException(400, "bbox must carry ix_min, ix_max, iy_min, iy_max")
        if ix1 < ix0 or iy1 < iy0:
            raise HTTPException(400, "bbox max must be >= min")
        cells = [(ix, iy) for ix in range(ix0, ix1 + 1) for iy in range(iy0, iy1 + 1)]
    row = AnalysisRegion(
        id=body.id,
        name=body.name,
        grid_cell_size_m=body.grid_cell_size_m or settings.analysis_grid_cell_m,
        cells_json=[list(c) for c in cells],
        notes=body.notes,
        created_at_us=now_us(),
    )
    db.add(row)
    db.commit()
    return _region_out(row)


@router.get("/analysis/regions", response_model=list[schemas.AnalysisRegionOut], tags=["analysis"], summary="List analysis regions")
def list_analysis_regions(db: Session = Depends(get_db)):
    rows = list(db.execute(select(AnalysisRegion).order_by(AnalysisRegion.id)).scalars())
    return [_region_out(r) for r in rows]


@router.get("/analysis/regions/{region_id}", response_model=schemas.AnalysisRegionOut, tags=["analysis"], summary="Fetch one analysis region")
def get_analysis_region(region_id: str, db: Session = Depends(get_db)):
    row = db.get(AnalysisRegion, region_id)
    if not row:
        raise HTTPException(404, "analysis region not found")
    return _region_out(row)


@router.post(
    "/analysis/habitat-maps",
    response_model=schemas.HabitatMapOut,
    status_code=201,
    tags=["analysis"],
    summary="Publish the pre-published habitat class of each grid cell",
)
def create_habitat_map(body: schemas.HabitatMapCreate, db: Session = Depends(get_db)):
    if db.get(HabitatMap, body.id):
        raise HTTPException(409, f"habitat map {body.id} already exists")
    row = HabitatMap(
        id=body.id,
        name=body.name,
        grid_cell_size_m=body.grid_cell_size_m or settings.analysis_grid_cell_m,
        cells_json=[{"ix": c.ix, "iy": c.iy, "habitat": c.habitat} for c in body.cells],
        created_at_us=now_us(),
    )
    db.add(row)
    db.commit()
    return _habitat_out(row)


def _habitat_out(row: HabitatMap) -> schemas.HabitatMapOut:
    return schemas.HabitatMapOut(
        id=row.id,
        name=row.name,
        grid_cell_size_m=row.grid_cell_size_m,
        cell_count=len(row.cells_json),
        cells=row.cells_json,
        created_at=iso(row.created_at_us),
    )


@router.get("/analysis/habitat-maps", response_model=list[schemas.HabitatMapOut], tags=["analysis"], summary="List habitat maps")
def list_habitat_maps(db: Session = Depends(get_db)):
    rows = list(db.execute(select(HabitatMap).order_by(HabitatMap.id)).scalars())
    return [_habitat_out(r) for r in rows]


@router.post(
    "/analysis/reference-compositions",
    response_model=schemas.ReferenceCompositionOut,
    status_code=201,
    tags=["analysis"],
    summary="Publish a habitat x period-class reference composition for standardisation",
)
def create_reference_composition(body: schemas.ReferenceCompositionCreate, db: Session = Depends(get_db)):
    if db.get(ReferenceComposition, body.id):
        raise HTTPException(409, f"reference composition {body.id} already exists")
    row = ReferenceComposition(
        id=body.id,
        name=body.name,
        entries_json=[
            {"habitat": e.habitat, "period_class": e.period_class, "weight": e.weight}
            for e in body.entries
        ],
        notes=body.notes,
        created_at_us=now_us(),
    )
    db.add(row)
    db.commit()
    return _reference_out(row)


def _reference_out(row: ReferenceComposition) -> schemas.ReferenceCompositionOut:
    return schemas.ReferenceCompositionOut(
        id=row.id,
        name=row.name,
        entries=row.entries_json,
        notes=row.notes,
        created_at=iso(row.created_at_us),
    )


@router.get("/analysis/reference-compositions", response_model=list[schemas.ReferenceCompositionOut], tags=["analysis"], summary="List reference compositions")
def list_reference_compositions(db: Session = Depends(get_db)):
    rows = list(
        db.execute(select(ReferenceComposition).order_by(ReferenceComposition.id)).scalars()
    )
    return [_reference_out(r) for r in rows]


def _report_config_from(body: schemas.AnalysisReportCreate) -> analysis.ReportConfig:
    return analysis.ReportConfig(
        region_a_id=body.region_a_id,
        region_b_id=body.region_b_id,
        habitat_map_id=body.habitat_map_id,
        reference_composition_id=body.reference_composition_id,
        class_label=body.class_label,
        window_start_us=to_us(body.window.start),
        window_end_us=to_us(body.window.end),
        grid_cell_m=body.grid_cell_size_m or settings.analysis_grid_cell_m,
        bin_seconds=body.bin_seconds or settings.analysis_bin_seconds,
        day_start_hour=(
            body.day_start_hour
            if body.day_start_hour is not None
            else settings.analysis_day_start_hour
        ),
        night_start_hour=(
            body.night_start_hour
            if body.night_start_hour is not None
            else settings.analysis_night_start_hour
        ),
        methods=tuple(body.methods) if body.methods else ("common_coverage", "standardized"),
    )


def _generate_report_row(
    db: Session, cfg: analysis.ReportConfig, as_of_us: int, created_by: str | None
) -> AnalysisReport:
    """Compute a report and persist it frozen at its knowledge cutoff."""
    try:
        result = analysis.generate_report(db, cfg, as_of_us, settings)
    except KeyError as exc:
        raise HTTPException(404, f"unknown analysis definition: {exc}")
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    row = AnalysisReport(
        as_of_us=as_of_us,
        config_json=cfg.resolved(),
        inputs_hash=analysis.inputs_hash(cfg),
        result_json=result,
        report_hash=analysis.report_hash(result),
        status="done",
        created_by=created_by,
        created_at_us=now_us(),
    )
    db.add(row)
    db.flush()
    return row


def _report_summary(row: AnalysisReport) -> schemas.AnalysisReportSummary:
    return schemas.AnalysisReportSummary(
        id=row.id,
        as_of=iso(row.as_of_us),
        report_hash=row.report_hash,
        inputs_hash=row.inputs_hash,
        status=row.status,
        supersedes_report_id=row.supersedes_report_id,
        created_by=row.created_by,
        created_at=iso(row.created_at_us),
        config=row.config_json,
    )


def _report_out(row: AnalysisReport) -> schemas.AnalysisReportOut:
    return schemas.AnalysisReportOut(
        id=row.id,
        as_of=iso(row.as_of_us),
        report_hash=row.report_hash,
        inputs_hash=row.inputs_hash,
        status=row.status,
        supersedes_report_id=row.supersedes_report_id,
        created_by=row.created_by,
        created_at=iso(row.created_at_us),
        result=row.result_json,
    )


@router.post(
    "/analysis/reports",
    response_model=schemas.AnalysisReportOut,
    status_code=201,
    tags=["analysis"],
    summary="Generate a versioned comparability report frozen at a knowledge cutoff",
)
def create_analysis_report(body: schemas.AnalysisReportCreate, db: Session = Depends(get_db)):
    if body.region_a_id == body.region_b_id:
        raise HTTPException(400, "region_a_id and region_b_id must differ")
    cfg = _report_config_from(body)
    as_of_us = to_us(body.as_of) if body.as_of else now_us()
    row = _generate_report_row(db, cfg, as_of_us, body.created_by)
    db.add(
        AuditLog(
            actor=body.created_by or "analyst",
            action="analysis_report.create",
            target_type="analysis_report",
            target_id=str(row.id),
            detail={"as_of": iso(as_of_us), "report_hash": row.report_hash},
            created_at_us=now_us(),
        )
    )
    db.commit()
    return _report_out(row)


@router.get("/analysis/reports", response_model=list[schemas.AnalysisReportSummary], tags=["analysis"], summary="List comparability reports")
def list_analysis_reports(limit: int = Query(20, ge=1, le=200), db: Session = Depends(get_db)):
    rows = list(
        db.execute(
            select(AnalysisReport).order_by(AnalysisReport.id.desc()).limit(limit)
        ).scalars()
    )
    return [_report_summary(r) for r in rows]


@router.get("/analysis/reports/compare", tags=["analysis"], summary="Diff two report versions and explain what changed")
def compare_analysis_reports(
    baseline_id: int = Query(...),
    current_id: int = Query(...),
    db: Session = Depends(get_db),
):
    """Version difference between two cutoffs.

    Late shutdown or calibration information produces a new report at a new
    cutoff; this endpoint shows, stratum by stratum, which coverage facts and
    rates changed, and lists the evidence that arrived between the two
    cutoffs.  The baseline report itself stays reproducible.
    """
    baseline = db.get(AnalysisReport, baseline_id)
    current = db.get(AnalysisReport, current_id)
    if not baseline or not current:
        raise HTTPException(404, "report not found")
    older, newer = (
        (baseline, current) if baseline.as_of_us <= current.as_of_us else (current, baseline)
    )
    base_strata = {
        (s["habitat"], s["period_class"]): s
        for s in baseline.result_json.get("strata", [])
    }
    curr_strata = {
        (s["habitat"], s["period_class"]): s
        for s in current.result_json.get("strata", [])
    }
    changes = []
    for key in sorted(set(base_strata) | set(curr_strata)):
        before = base_strata.get(key)
        after = curr_strata.get(key)
        for side in ("a", "b"):
            b_reg = (before or {}).get("regions", {}).get(side, {})
            c_reg = (after or {}).get("regions", {}).get(side, {})
            b_status = b_reg.get("status", "absent")
            c_status = c_reg.get("status", "absent")
            b_eff = b_reg.get("effort_cell_hours", 0.0)
            c_eff = c_reg.get("effort_cell_hours", 0.0)
            if b_status == c_status and abs(b_eff - c_eff) < 1e-9 and (
                b_reg.get("contributions") == c_reg.get("contributions")
            ):
                continue
            if b_status != c_status:
                kind = (
                    "coverage_lost"
                    if c_status.startswith("no_coverage")
                    else "coverage_gained"
                )
            elif abs(b_eff - c_eff) > 1e-9:
                kind = "effort_changed"
            else:
                kind = "contributions_changed"
            changes.append(
                {
                    "habitat": key[0],
                    "period_class": key[1],
                    "region": (before or after)["regions"][side]["region_id"],
                    "change": kind,
                    "before": {
                        "status": b_status,
                        "effort_cell_hours": b_eff,
                        "contributions": b_reg.get("contributions"),
                        "rate_per_cell_hour": b_reg.get("rate_per_cell_hour"),
                    },
                    "after": {
                        "status": c_status,
                        "effort_cell_hours": c_eff,
                        "contributions": c_reg.get("contributions"),
                        "rate_per_cell_hour": c_reg.get("rate_per_cell_hour"),
                    },
                }
            )
    lo, hi = older.as_of_us, newer.as_of_us
    late_progress = list(
        db.execute(
            select(StationProgress).where(
                StationProgress.received_at_us > lo, StationProgress.received_at_us <= hi
            )
        ).scalars()
    )
    late_calibrations = list(
        db.execute(
            select(CalibrationModel).where(
                CalibrationModel.created_at_us > lo, CalibrationModel.created_at_us <= hi
            )
        ).scalars()
    )
    late_events = list(
        db.execute(
            select(Event).where(Event.created_at_us > lo, Event.created_at_us <= hi)
        ).scalars()
    )
    gap_reports = [p for p in late_progress if p.gaps_json]
    return {
        "baseline_id": baseline.id,
        "current_id": current.id,
        "baseline_as_of": iso(baseline.as_of_us),
        "current_as_of": iso(current.as_of_us),
        "same_inputs": baseline.inputs_hash == current.inputs_hash,
        "same_result": baseline.report_hash == current.report_hash,
        "stratum_changes": changes,
        "evidence_arrived_between_cutoffs": {
            "progress_reports": len(late_progress),
            "progress_reports_declaring_gaps": len(gap_reports),
            "gap_declarations": [
                {
                    "station_id": p.station_id,
                    "boot_id": p.boot_id,
                    "received_at": iso(p.received_at_us),
                    "gaps": p.gaps_json,
                }
                for p in gap_reports
            ],
            "calibration_models": len(late_calibrations),
            "events_created": len(late_events),
        },
        "note": (
            "the baseline report remains reproducible at its own cutoff; "
            "differences reflect evidence that arrived after it"
        ),
    }


@router.get("/analysis/reports/{report_id}", response_model=schemas.AnalysisReportOut, tags=["analysis"], summary="Fetch a full comparability report")
def get_analysis_report(report_id: int, db: Session = Depends(get_db)):
    row = db.get(AnalysisReport, report_id)
    if not row:
        raise HTTPException(404, "analysis report not found")
    return _report_out(row)


@router.post(
    "/analysis/reports/{report_id}/reproduce",
    response_model=schemas.ReproduceResponse,
    tags=["analysis"],
    summary="Re-derive the report at its original cutoff and compare hashes",
)
def reproduce_analysis_report(report_id: int, db: Session = Depends(get_db)):
    """Re-derivation check: an old report must be exactly reproducible.

    The computation only reads evidence received at or before the report's
    cutoff, so late shutdown or calibration information cannot change what the
    old report said — it can only produce a *new* report at a new cutoff.
    """
    row = db.get(AnalysisReport, report_id)
    if not row:
        raise HTTPException(404, "analysis report not found")
    cfg = _report_config_from(schemas.AnalysisReportCreate(**_config_to_create(row.config_json)))
    recomputed = analysis.generate_report(db, cfg, row.as_of_us, settings)
    recomputed_hash = analysis.report_hash(recomputed)
    reproducible = recomputed_hash == row.report_hash
    return schemas.ReproduceResponse(
        report_id=row.id,
        reproducible=reproducible,
        report_hash=row.report_hash,
        recomputed_hash=recomputed_hash,
        as_of=iso(row.as_of_us),
        note=(
            "report re-derived identically from its cutoff"
            if reproducible
            else "MISMATCH: the stored report cannot be re-derived from its cutoff"
        ),
    )


def _config_to_create(config: dict) -> dict:
    """Translate a stored resolved config back into create-request shape."""
    return {
        "region_a_id": config["region_a_id"],
        "region_b_id": config["region_b_id"],
        "habitat_map_id": config["habitat_map_id"],
        "reference_composition_id": config.get("reference_composition_id"),
        "class_label": config.get("class_label"),
        "window": config["window"],
        "grid_cell_size_m": config.get("grid_cell_m"),
        "bin_seconds": config.get("bin_seconds"),
        "day_start_hour": config.get("period_classes", {}).get("day_start_hour"),
        "night_start_hour": config.get("period_classes", {}).get("night_start_hour"),
        "methods": config.get("methods"),
    }


@router.post(
    "/analysis/reports/{report_id}/refresh",
    response_model=schemas.AnalysisReportOut,
    status_code=201,
    tags=["analysis"],
    summary="Issue the same report at a new cutoff (late evidence creates a new version)",
)
def refresh_analysis_report(
    report_id: int,
    body: schemas.RefreshReportRequest | None = None,
    db: Session = Depends(get_db),
):
    row = db.get(AnalysisReport, report_id)
    if not row:
        raise HTTPException(404, "analysis report not found")
    as_of_us = to_us(body.as_of) if body and body.as_of else now_us()
    if as_of_us <= row.as_of_us:
        raise HTTPException(400, "the new cutoff must be later than the report's cutoff")
    cfg = _report_config_from(schemas.AnalysisReportCreate(**_config_to_create(row.config_json)))
    new_row = _generate_report_row(db, cfg, as_of_us, (body.created_by if body else None) or row.created_by)
    new_row.supersedes_report_id = row.id
    db.add(
        AuditLog(
            actor=new_row.created_by or "analyst",
            action="analysis_report.refresh",
            target_type="analysis_report",
            target_id=str(new_row.id),
            detail={"supersedes_report_id": row.id, "as_of": iso(as_of_us)},
            created_at_us=now_us(),
        )
    )
    db.commit()
    return _report_out(new_row)


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #
@router.get("/stats", response_model=schemas.StatsSummary, tags=["system"], summary="Service-wide counts")
def stats(db: Session = Depends(get_db)):
    runs = dict(
        db.execute(select(ProcessingRun.status, func.count(ProcessingRun.id)).group_by(ProcessingRun.status)).all()
    )
    events = dict(
        db.execute(select(Event.status, func.count(Event.id)).group_by(Event.status)).all()
    )
    latest = db.execute(select(ProcessingRun).order_by(ProcessingRun.id.desc())).scalars().first()
    return schemas.StatsSummary(
        stations=db.scalar(select(func.count(Station.id))) or 0,
        observations=db.scalar(select(func.count(Observation.id))) or 0,
        uncalibrated_observations=db.scalar(
            select(func.count(Observation.id)).where(Observation.calibration_status == "uncalibrated")
        )
        or 0,
        conflicts=db.scalar(select(func.count(ObservationConflict.id))) or 0,
        runs=runs,
        events=events,
        latest_run=_run_out(latest) if latest else None,
    )
