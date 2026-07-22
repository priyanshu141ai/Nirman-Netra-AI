"""Structured logging with request-scoped identifiers and safe metadata."""

import json
import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import cast

from nirman_netra.utils import utc_now

_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)
_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
_job_id: ContextVar[str | None] = ContextVar("job_id", default=None)
_SENSITIVE_PARTS = frozenset({"address", "citizen", "email", "image", "name", "phone", "token"})


def safe_metadata(metadata: Mapping[str, object]) -> dict[str, bool | float | int | str | None]:
    """Keep only scalar, non-sensitive metadata suitable for normal logs."""

    safe: dict[str, bool | float | int | str | None] = {}
    for key, value in metadata.items():
        normalized = key.lower()
        if any(part in normalized for part in _SENSITIVE_PARTS):
            continue
        if value is None or isinstance(value, bool | float | int):
            safe[key] = value
        elif isinstance(value, str):
            safe[key] = value[:256]
    return safe


@contextmanager
def log_context(
    *, correlation_id: str | None = None, request_id: str | None = None, job_id: str | None = None
) -> Iterator[None]:
    """Bind identifiers for logs emitted inside the context."""

    tokens = (
        (_correlation_id, _correlation_id.set(correlation_id)),
        (_request_id, _request_id.set(request_id)),
        (_job_id, _job_id.set(job_id)),
    )
    try:
        yield
    finally:
        for variable, token in reversed(tokens):
            variable.reset(token)


class JsonFormatter(logging.Formatter):
    """Serialize stable, privacy-safe log fields as JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": utc_now().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": _correlation_id.get(),
            "request_id": _request_id.get(),
            "job_id": _job_id.get(),
        }
        metadata = getattr(record, "metadata", None)
        if isinstance(metadata, Mapping):
            payload["metadata"] = safe_metadata(cast(Mapping[str, object], metadata))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, separators=(",", ":"), default=str)


def configure_logging(level: str) -> None:
    """Configure the process root logger for structured output."""

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
