"""Observation ingestion: idempotency, conflict detection, progress reporting.

Identity of a record is ``(station_id, boot_id, source_seq)``.  Re-uploading an
identical record is a no-op counted as a duplicate.  Uploading a *different*
record under the same identity is recorded in ``observation_conflicts`` and the
original observation is left untouched: a retransmission never overwrites
evidence.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field as dataclass_field

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import (
    CalibrationSourceRecord,
    Observation,
    ObservationConflict,
    Station,
    StationBoot,
    StationProgress,
    UploadBatch,
)
from .schemas import ObservationBatch, ProgressRequest, ProgressUpload
from .timeutil import iso, now_us, parse_local_us, to_us


def canonical_payload(record) -> dict:
    return {
        "source_seq": record.source_seq,
        "kind": record.kind,
        "class_label": record.class_label,
        "local_capture_at": record.local_capture_at.strip(),
        "features": [round(float(v), 9) for v in record.features],
        "uncertainty_ms": round(float(record.uncertainty_ms), 6),
        "calibration": (
            {
                "source_id": record.calibration.source_id,
                "emission_reference_at": record.calibration.emission_reference_at.strip(),
            }
            if record.calibration
            else None
        ),
        "detector_note": record.detector_note,
    }


def content_hash(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass
class IngestionResult:
    batch_id: int | None
    replayed: bool
    accepted_ids: list[int]
    duplicates: list[int]
    conflicts: list[int]
    rejected: list[dict]
    progress_recorded: bool
    replay_of_batch_id: int | None = None
    original_observation_ids: list[int] = dataclass_field(default_factory=list)


def _touch_boot(db: Session, station_id: str, boot_id: str, local_us: int, t: int) -> None:
    boot = db.execute(
        select(StationBoot).where(
            StationBoot.station_id == station_id, StationBoot.boot_id == boot_id
        )
    ).scalar_one_or_none()
    if boot is None:
        db.add(
            StationBoot(
                station_id=station_id,
                boot_id=boot_id,
                first_seen_us=local_us,
                last_seen_us=local_us,
            )
        )
        # Flush so the next row in the same batch sees this boot instead of
        # attempting a duplicate insert.
        db.flush()
    else:
        boot.first_seen_us = min(boot.first_seen_us, local_us)
        boot.last_seen_us = max(boot.last_seen_us, local_us)


def ingest_batch(db: Session, payload: ObservationBatch) -> IngestionResult:
    station_id = payload.station_id
    station = db.get(Station, station_id)
    if station is None:
        raise KeyError(station_id)
    if not station.active:
        raise ValueError(f"station {station_id} is inactive")

    incoming_fingerprint = content_hash(
        {"records": [canonical_payload(r) for r in payload.records]}
    )
    existing_batch = db.execute(
        select(UploadBatch).where(
            UploadBatch.station_id == station_id, UploadBatch.batch_key == payload.batch_key
        )
    ).scalar_one_or_none()
    if (
        existing_batch is not None
        and existing_batch.content_hash is not None
        and existing_batch.content_hash == incoming_fingerprint
    ):
        # Whole-batch replay of identical content: write nothing and report
        # "this request created zero records", separately from the ids the
        # original request created.
        ids = [
            row[0]
            for row in db.execute(
                select(Observation.id).where(Observation.upload_batch_id == existing_batch.id)
            )
        ]
        return IngestionResult(
            batch_id=existing_batch.id,
            replayed=True,
            accepted_ids=[],
            duplicates=sorted({r.source_seq for r in payload.records}),
            conflicts=[],
            rejected=[],
            progress_recorded=False,
            replay_of_batch_id=existing_batch.id,
            original_observation_ids=sorted(ids),
        )

    received = now_us()
    uploaded_at_us = to_us(payload.uploaded_at) if payload.uploaded_at else received
    if existing_batch is not None:
        # Same batch_key, different content: the station reused an identifier.
        # Treat the records on their own merits (dedupe/conflict per sequence)
        # and flag the identifier reuse in the batch notes rather than silently
        # discarding an entire upload.
        batch = existing_batch
        batch.notes = (
            (batch.notes + "; " if batch.notes else "")
            + f"batch_key reused with different content at {iso(received)}"
        )
    else:
        batch = UploadBatch(
            station_id=station_id,
            batch_key=payload.batch_key,
            received_at_us=received,
            record_count=len(payload.records),
            content_hash=incoming_fingerprint,
            notes=payload.notes,
        )
    if payload.progress is not None:
        batch.declared_complete_through_local_us = parse_local_us(
            payload.progress.complete_through_local_at
        )
        batch.complete = payload.progress.complete
        batch.gaps_json = payload.progress.gaps or []
    if existing_batch is None:
        db.add(batch)
        db.flush()

    accepted: list[int] = []
    duplicates: list[int] = []
    conflicts: list[int] = []
    rejected: list[dict] = []

    seen_in_batch: dict[int, tuple[str, dict]] = {}
    for index, record in enumerate(payload.records):
        canonical = canonical_payload(record)
        digest = content_hash(canonical)
        seq = record.source_seq
        if seq in seen_in_batch:
            prev_digest, _prev_payload = seen_in_batch[seq]
            rejected.append(
                {
                    "index": index,
                    "source_seq": seq,
                    "reason": "duplicate_source_seq_in_batch",
                    "detail": "later occurrence ignored; the first occurrence was ingested on its own merits",
                }
            )
            if prev_digest == digest:
                duplicates.append(seq)
            else:
                conflicts.append(seq)
                db.add(
                    ObservationConflict(
                        station_id=station_id,
                        boot_id=payload.boot_id,
                        source_seq=seq,
                        existing_observation_id=None,
                        content_hash=digest,
                        payload=canonical,
                        received_at_us=received,
                        resolved=False,
                        resolution_note="same source_seq with different content inside one upload batch",
                    )
                )
            continue
        seen_in_batch[seq] = (digest, canonical)

        existing = db.execute(
            select(Observation).where(
                Observation.station_id == station_id,
                Observation.boot_id == payload.boot_id,
                Observation.source_seq == seq,
            )
        ).scalar_one_or_none()
        if existing is not None:
            if existing.content_hash == digest:
                duplicates.append(seq)
            else:
                conflicts.append(seq)
                db.add(
                    ObservationConflict(
                        station_id=station_id,
                        boot_id=payload.boot_id,
                        source_seq=seq,
                        existing_observation_id=existing.id,
                        content_hash=digest,
                        payload=canonical,
                        received_at_us=received,
                        resolved=False,
                        resolution_note="same source_seq with different content; original kept",
                    )
                )
            continue
        if record.kind == "calibration":
            cal = db.get(CalibrationSourceRecord, record.calibration.source_id)
            if cal is None:
                rejected.append(
                    {
                        "index": index,
                        "source_seq": seq,
                        "reason": "unknown_calibration_source",
                        "detail": f"calibration source {record.calibration.source_id!r} is not configured",
                    }
                )
                continue
            emission_us = to_us(record.calibration.emission_reference_at)
            obs = Observation(
                station_id=station_id,
                boot_id=payload.boot_id,
                source_seq=seq,
                kind="calibration",
                class_label=record.class_label,
                local_capture_us=parse_local_us(record.local_capture_at),
                uploaded_at_us=uploaded_at_us,
                ingest_at_us=received,
                features=[float(v) for v in record.features],
                uncertainty_us=int(round(record.uncertainty_ms * 1000)),
                content_hash=digest,
                upload_batch_id=batch.id,
                calibration_source_id=record.calibration.source_id,
                reference_emission_us=emission_us,
                calibration_status="anchor_candidate",
                calibration_reason=None,
                created_at_us=received,
            )
        else:
            obs = Observation(
                station_id=station_id,
                boot_id=payload.boot_id,
                source_seq=seq,
                kind="sound",
                class_label=record.class_label,
                local_capture_us=parse_local_us(record.local_capture_at),
                uploaded_at_us=uploaded_at_us,
                ingest_at_us=received,
                features=[float(v) for v in record.features],
                uncertainty_us=int(round(record.uncertainty_ms * 1000)),
                content_hash=digest,
                upload_batch_id=batch.id,
                calibration_status="pending",
                created_at_us=received,
            )
        db.add(obs)
        db.flush()
        accepted.append(obs.id)

    batch.accepted_count = (batch.accepted_count or 0) + len(accepted)
    batch.duplicate_count = (batch.duplicate_count or 0) + len(duplicates)
    batch.conflict_count = (batch.conflict_count or 0) + len(conflicts)

    progress_recorded = False
    if payload.progress is not None:
        progress_recorded = record_progress(
            db, station_id, payload.boot_id, payload.progress, batch=batch, received=received
        )

    for record in payload.records:
        if record.kind == "sound":
            _touch_boot(db, station_id, payload.boot_id, parse_local_us(record.local_capture_at), received)

    return IngestionResult(
        batch_id=batch.id,
        replayed=False,
        accepted_ids=accepted,
        duplicates=duplicates,
        conflicts=conflicts,
        rejected=rejected,
        progress_recorded=progress_recorded,
    )


def record_progress(
    db: Session,
    station_id: str,
    boot_id: str,
    progress: ProgressUpload,
    *,
    batch: UploadBatch | None,
    received: int,
) -> bool:
    declared = parse_local_us(progress.complete_through_local_at)
    row = StationProgress(
        station_id=station_id,
        boot_id=boot_id,
        upload_batch_id=batch.id if batch else None,
        declared_local_us=declared,
        complete=progress.complete,
        gaps_json=progress.gaps or [],
        received_at_us=received,
        created_at_us=received,
    )
    db.add(row)
    if batch is not None:
        batch.declared_complete_through_local_us = declared
        batch.complete = progress.complete
        batch.gaps_json = progress.gaps or []
    return True


def ingest_progress(db: Session, station_id: str, body: ProgressRequest) -> int:
    station = db.get(Station, station_id)
    if station is None:
        raise KeyError(station_id)
    received = now_us()
    record_progress(db, station_id, body.boot_id, body.progress, batch=None, received=received)
    db.flush()
    return 1


def parse_local_us_iso(local_us: int) -> str:
    """Render a station-local counter as an ISO string *without* a zone."""
    from datetime import datetime, timedelta

    dt = datetime(1970, 1, 1) + timedelta(microseconds=local_us)
    return dt.isoformat(timespec="microseconds")
