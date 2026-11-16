"""FastAPI application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from . import __version__
from .api import router
from .config import settings
from .db import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Wetland Multi-Station Acoustic Observation Service",
    version=__version__,
    description=(
        "Groups records of the same sound heard at several wetland stations into one "
        "event, using per-boot clock calibration, a coarse-to-fine grid search over the "
        "configured observation area, and an exclusive set-packing solver. On top of the "
        "event stream it serves comparability reports: activity rates of two regions "
        "restricted to their comparable sample, with coverage facts, incomparable strata "
        "and versioned knowledge cutoffs — never a population estimate. "
        "Results are exploratory field-survey output, not an alerting service."
    ),
    lifespan=lifespan,
)


@app.middleware("http")
async def _auth_middleware(request, call_next):
    # Dependency-free auth so the interactive docs still load; the OpenAPI
    # document itself is public and the /health probe is open.
    path = request.url.path
    if path.startswith(("/docs", "/openapi.json", "/redoc", "/health", "/admin/reset")):
        return await call_next(request)
    if settings.auth_disabled:
        return await call_next(request)
    if request.headers.get("x-api-key") != settings.api_key:
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=401, content={"detail": "missing or invalid X-API-Key"})
    return await call_next(request)


app.include_router(router)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    schema.setdefault("components", {}).setdefault("securitySchemes", {})["ApiKeyAuth"] = {
        "type": "apiKey",
        "in": "header",
        "name": "X-API-Key",
    }
    public = {
        "/health",
        "/docs",
        "/openapi.json",
        "/redoc",
        "/admin/reset",
    }
    for path, ops in schema.get("paths", {}).items():
        if path in public:
            continue
        for op in ops.values():
            if isinstance(op, dict):
                op["security"] = [{"ApiKeyAuth": []}]
    schema["info"]["x-results-are-exploratory"] = True
    app.openapi_schema = schema
    return schema


app.openapi = custom_openapi
