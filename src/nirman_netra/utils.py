"""Small deterministic and time-safe utilities."""

from datetime import UTC, datetime
from hashlib import sha256
from uuid import NAMESPACE_URL, uuid4, uuid5


def utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp."""

    return datetime.now(UTC)


def deterministic_id(namespace: str, *parts: str) -> str:
    """Return the same UUID for the same namespace and ordered values."""

    value = ":".join((namespace, *parts))
    return str(uuid5(NAMESPACE_URL, value))


def content_hash(content: bytes) -> str:
    """Return a SHA-256 checksum for immutable content."""

    return sha256(content).hexdigest()


def new_correlation_id() -> str:
    """Create an opaque correlation identifier."""

    return str(uuid4())
