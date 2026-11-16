"""Database engine and session plumbing."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import settings


class Base(DeclarativeBase):
    pass


engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


def init_db() -> None:
    # Import models for their side effect: register tables on Base.metadata.
    from . import models as _models  # noqa: F401

    Base.metadata.create_all(engine)
    _ensure_columns()


def _ensure_columns() -> None:
    """Add columns introduced after a database was first created.

    This project ships without a migration tool; the service is self-contained
    and the only schema evolution so far is additive, so a guarded
    ``ADD COLUMN IF NOT EXISTS`` is enough and keeps existing data.
    """
    from sqlalchemy import text

    with engine.begin() as conn:
        conn.execute(
            text("ALTER TABLE upload_batches ADD COLUMN IF NOT EXISTS content_hash VARCHAR(64)")
        )


@contextmanager
def session_scope() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
