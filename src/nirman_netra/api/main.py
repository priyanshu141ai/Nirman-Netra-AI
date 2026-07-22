"""Minimal HTTP health surface."""

import logging
import re
from time import perf_counter

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import RequestResponseEndpoint

from nirman_netra.config import Settings, load_settings
from nirman_netra.logging import configure_logging, log_context
from nirman_netra.utils import new_correlation_id

_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def _safe_header_id(value: str | None) -> str:
    return value if value and _SAFE_ID.fullmatch(value) else new_correlation_id()


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create the API without database or network calls."""

    active_settings = settings or load_settings()
    configure_logging(active_settings.log_level)
    logger = logging.getLogger(__name__)
    app = FastAPI(title="NirmanNetra AI", version="0.1.0")
    app.state.settings = active_settings

    @app.middleware("http")
    async def request_context(request: Request, call_next: RequestResponseEndpoint) -> Response:
        correlation_id = _safe_header_id(request.headers.get("x-correlation-id"))
        request_id = _safe_header_id(request.headers.get("x-request-id"))
        started = perf_counter()
        with log_context(correlation_id=correlation_id, request_id=request_id):
            response = await call_next(request)
            response.headers["x-correlation-id"] = correlation_id
            response.headers["x-request-id"] = request_id
            logger.info(
                "request.completed",
                extra={
                    "metadata": {
                        "method": request.method,
                        "status_code": response.status_code,
                        "duration_ms": round((perf_counter() - started) * 1000, 2),
                    }
                },
            )
            return response

    @app.get("/health/live", tags=["health"])
    async def liveness() -> dict[str, str]:
        return {"status": "live"}

    @app.get("/health/ready", tags=["health"])
    async def readiness() -> dict[str, str]:
        return {"status": "ready", "environment": active_settings.app_env}

    return app


app = create_app()
