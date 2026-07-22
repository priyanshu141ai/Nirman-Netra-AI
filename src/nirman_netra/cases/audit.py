"""Hash-chained append-only audit event construction."""

from datetime import datetime

from nirman_netra.cases.contracts import (
    ActorIdentity,
    AuditAction,
    AuditEvent,
    AuditTrail,
    CaseStatus,
    audit_event_hash,
)
from nirman_netra.exceptions import AuditIntegrityError
from nirman_netra.utils import deterministic_id


def append_audit_event(
    trail: AuditTrail,
    *,
    actor: ActorIdentity,
    timestamp: datetime,
    action: AuditAction,
    case_id: str,
    previous_state: CaseStatus | None,
    new_state: CaseStatus,
    correlation_id: str,
) -> AuditTrail:
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise AuditIntegrityError("audit timestamp must be timezone-aware")
    event_id = deterministic_id(
        "case-audit",
        case_id,
        action.value,
        actor.actor_id,
        timestamp.isoformat(),
        correlation_id,
        trail.head_hash or "root",
    )
    unsigned = AuditEvent(
        event_id=event_id,
        actor_id=actor.actor_id,
        actor_role=actor.role,
        timestamp=timestamp,
        action=action,
        case_id=case_id,
        previous_state=previous_state,
        new_state=new_state,
        correlation_id=correlation_id,
        previous_event_hash=trail.head_hash,
        event_hash="0" * 64,
    )
    event = unsigned.model_copy(update={"event_hash": audit_event_hash(unsigned)})
    return AuditTrail(events=(*trail.events, event), head_hash=event.event_hash)
