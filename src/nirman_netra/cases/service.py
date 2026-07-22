"""Role-gated, immutable inspector case workflow services."""

from collections.abc import Mapping
from datetime import datetime

from pydantic import JsonValue

from nirman_netra.cases.audit import append_audit_event
from nirman_netra.cases.contracts import (
    ActorIdentity,
    ActorRole,
    AssignmentAction,
    AssignmentEvent,
    AuditAction,
    AuditTrail,
    CaseStatus,
    EvidenceAsset,
    EvidenceIntegrityReport,
    EvidenceTimeline,
    InspectionCase,
    InspectorAssignment,
    InspectorNote,
    TimelineEventType,
)
from nirman_netra.cases.evidence import (
    append_timeline_event,
    verify_chain_of_custody,
    verify_evidence_content,
)
from nirman_netra.data.contracts import Inspector
from nirman_netra.exceptions import (
    AuthorizationError,
    CaseTransitionError,
    EvidenceIntegrityError,
)
from nirman_netra.risk.contracts import ComplaintGroup, PermitMatch, RiskAssessment
from nirman_netra.utils import deterministic_id

_SUPERVISOR_ROLES = {ActorRole.SUPERVISOR, ActorRole.ADMINISTRATOR}
_HUMAN_ROLES = {ActorRole.INSPECTOR, ActorRole.SUPERVISOR, ActorRole.ADMINISTRATOR}
_ALLOWED_TRANSITIONS: dict[CaseStatus, set[CaseStatus]] = {
    CaseStatus.CREATED: {CaseStatus.TRIAGED},
    CaseStatus.TRIAGED: {
        CaseStatus.DISMISSED,
        CaseStatus.EVIDENCE_INSUFFICIENT,
    },
    CaseStatus.ASSIGNED: {
        CaseStatus.UNDER_REVIEW,
        CaseStatus.REINSPECTION_REQUIRED,
        CaseStatus.EVIDENCE_INSUFFICIENT,
        CaseStatus.DISMISSED,
    },
    CaseStatus.UNDER_REVIEW: {
        CaseStatus.REINSPECTION_REQUIRED,
        CaseStatus.EVIDENCE_INSUFFICIENT,
        CaseStatus.DISMISSED,
        CaseStatus.CONFIRMED_FOR_ADMIN_REVIEW,
    },
    CaseStatus.REINSPECTION_REQUIRED: {
        CaseStatus.UNDER_REVIEW,
        CaseStatus.EVIDENCE_INSUFFICIENT,
        CaseStatus.DISMISSED,
    },
    CaseStatus.EVIDENCE_INSUFFICIENT: {
        CaseStatus.REINSPECTION_REQUIRED,
        CaseStatus.UNDER_REVIEW,
        CaseStatus.DISMISSED,
    },
    CaseStatus.DISMISSED: {CaseStatus.CLOSED},
    CaseStatus.CONFIRMED_FOR_ADMIN_REVIEW: {CaseStatus.CLOSED},
    CaseStatus.CLOSED: set(),
}


def _require_role(actor: ActorIdentity, roles: set[ActorRole], action: str) -> None:
    if actor.role not in roles:
        raise AuthorizationError(f"{actor.role.value} cannot {action}")


def _require_case_actor(case: InspectionCase, actor: ActorIdentity, action: str) -> None:
    _require_role(actor, _HUMAN_ROLES, action)
    if actor.role == ActorRole.INSPECTOR and (
        case.assigned_inspector is None or case.assigned_inspector.inspector_id != actor.actor_id
    ):
        raise AuthorizationError("inspector is not assigned to this case")


def _validate_timestamp(case: InspectionCase, timestamp: datetime) -> None:
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise CaseTransitionError("case action timestamp must be timezone-aware")
    if timestamp < case.updated_at:
        raise CaseTransitionError("case action cannot predate the current case snapshot")


def _update_case(case: InspectionCase, **updates: object) -> InspectionCase:
    payload = case.model_dump()
    payload.update(updates)
    return InspectionCase.model_validate(payload)


def _record_action(
    case: InspectionCase,
    *,
    actor: ActorIdentity,
    timestamp: datetime,
    correlation_id: str,
    audit_action: AuditAction,
    timeline_type: TimelineEventType,
    new_status: CaseStatus,
    details: dict[str, JsonValue],
    evidence_asset_id: str | None = None,
    updates: Mapping[str, object] | None = None,
) -> InspectionCase:
    _validate_timestamp(case, timestamp)
    timeline = append_timeline_event(
        case.evidence_timeline,
        case_id=case.case_id,
        event_type=timeline_type,
        actor=actor,
        timestamp=timestamp,
        evidence_asset_id=evidence_asset_id,
        details=details,
    )
    audit = append_audit_event(
        case.audit_trail,
        actor=actor,
        timestamp=timestamp,
        action=audit_action,
        case_id=case.case_id,
        previous_state=case.status,
        new_state=new_status,
        correlation_id=correlation_id,
    )
    values: dict[str, object] = {
        "status": new_status,
        "updated_at": timestamp,
        "evidence_timeline": timeline,
        "audit_trail": audit,
    }
    if updates:
        values.update(updates)
    return _update_case(case, **values)


def create_case_from_risk(
    assessment: RiskAssessment,
    *,
    parcel_id: str | None,
    image_pair_id: str,
    change_result_id: str,
    permit_result: PermitMatch,
    complaint_groups: tuple[ComplaintGroup, ...],
    quality_warnings: tuple[str, ...],
    actor: ActorIdentity,
    timestamp: datetime,
    correlation_id: str,
) -> InspectionCase:
    _require_role(
        actor,
        {ActorRole.SYSTEM_PROCESSOR, ActorRole.SUPERVISOR, ActorRole.ADMINISTRATOR},
        "create a risk case",
    )
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise CaseTransitionError("case creation timestamp must be timezone-aware")
    if not assessment.requires_human_review or assessment.legal_verdict is not None:
        raise CaseTransitionError("risk result is not a human-review candidate")
    timeline = append_timeline_event(
        EvidenceTimeline(),
        case_id=assessment.case_id,
        event_type=TimelineEventType.CASE_CREATED,
        actor=actor,
        timestamp=timestamp,
        details={"risk_score": assessment.score},
    )
    audit = append_audit_event(
        AuditTrail(),
        actor=actor,
        timestamp=timestamp,
        action=AuditAction.CASE_CREATED,
        case_id=assessment.case_id,
        previous_state=None,
        new_state=CaseStatus.CREATED,
        correlation_id=correlation_id,
    )
    return InspectionCase(
        case_id=assessment.case_id,
        parcel_id=parcel_id,
        image_pair_id=image_pair_id,
        change_result_id=change_result_id,
        risk_score=assessment.score,
        risk_factors=assessment.factors,
        permit_result=permit_result,
        complaint_groups=complaint_groups,
        quality_warnings=quality_warnings,
        assigned_inspector=None,
        status=CaseStatus.CREATED,
        created_at=timestamp,
        updated_at=timestamp,
        evidence_timeline=timeline,
        audit_trail=audit,
    )


def transition_case(
    case: InspectionCase,
    target: CaseStatus,
    *,
    actor: ActorIdentity,
    timestamp: datetime,
    correlation_id: str,
    reason: str,
) -> InspectionCase:
    if not reason.strip():
        raise CaseTransitionError("case transition reason is required")
    if target == CaseStatus.ASSIGNED:
        raise CaseTransitionError("ASSIGNED state must be entered through inspector assignment")
    if target not in _ALLOWED_TRANSITIONS[case.status]:
        raise CaseTransitionError(f"invalid case transition: {case.status.value} -> {target.value}")
    if target == CaseStatus.UNDER_REVIEW and case.assigned_inspector is None:
        raise CaseTransitionError("UNDER_REVIEW requires an assigned inspector")
    if target == CaseStatus.TRIAGED:
        _require_role(actor, _SUPERVISOR_ROLES, "triage a case")
    elif target == CaseStatus.CONFIRMED_FOR_ADMIN_REVIEW:
        _require_role(actor, _SUPERVISOR_ROLES, "confirm a case for administrative review")
    elif target == CaseStatus.CLOSED:
        _require_role(actor, {ActorRole.ADMINISTRATOR}, "close a case")
    else:
        _require_case_actor(case, actor, "perform this case transition")
    return _record_action(
        case,
        actor=actor,
        timestamp=timestamp,
        correlation_id=correlation_id,
        audit_action=AuditAction.STATE_TRANSITION,
        timeline_type=TimelineEventType.STATE_CHANGED,
        new_status=target,
        details={"reason": reason, "previous_state": case.status.value, "new_state": target.value},
    )


def assign_case(
    case: InspectionCase,
    inspector: Inspector,
    *,
    actor: ActorIdentity,
    timestamp: datetime,
    correlation_id: str,
    reason: str,
) -> InspectionCase:
    _require_role(actor, _SUPERVISOR_ROLES, "assign an inspector")
    if not inspector.active:
        raise AuthorizationError("inactive inspector cannot be assigned")
    if case.assigned_inspector is not None:
        raise CaseTransitionError("case is already assigned; use reassignment")
    if case.status not in {
        CaseStatus.TRIAGED,
        CaseStatus.REINSPECTION_REQUIRED,
        CaseStatus.EVIDENCE_INSUFFICIENT,
    }:
        raise CaseTransitionError(f"case cannot be assigned from {case.status.value}")
    assignment = InspectorAssignment(
        inspector_id=inspector.inspector_id,
        assigned_at=timestamp,
        reason=reason,
        assigned_by=actor.actor_id,
    )
    event = AssignmentEvent(
        action=AssignmentAction.ASSIGN,
        inspector_id=inspector.inspector_id,
        previous_inspector_id=None,
        timestamp=timestamp,
        reason=reason,
        actor_id=actor.actor_id,
    )
    return _record_action(
        case,
        actor=actor,
        timestamp=timestamp,
        correlation_id=correlation_id,
        audit_action=AuditAction.ASSIGNED,
        timeline_type=TimelineEventType.ASSIGNMENT_CHANGED,
        new_status=CaseStatus.ASSIGNED,
        details={"action": "assign", "inspector_id": inspector.inspector_id, "reason": reason},
        updates={
            "assigned_inspector": assignment,
            "assignment_history": (*case.assignment_history, event),
        },
    )


def reassign_case(
    case: InspectionCase,
    inspector: Inspector,
    *,
    actor: ActorIdentity,
    timestamp: datetime,
    correlation_id: str,
    reason: str,
) -> InspectionCase:
    _require_role(actor, _SUPERVISOR_ROLES, "reassign an inspector")
    if not inspector.active:
        raise AuthorizationError("inactive inspector cannot be assigned")
    if case.assigned_inspector is None:
        raise CaseTransitionError("case has no inspector to reassign")
    if case.status in {
        CaseStatus.DISMISSED,
        CaseStatus.CONFIRMED_FOR_ADMIN_REVIEW,
        CaseStatus.CLOSED,
    }:
        raise CaseTransitionError(f"case cannot be reassigned from {case.status.value}")
    previous = case.assigned_inspector.inspector_id
    assignment = InspectorAssignment(
        inspector_id=inspector.inspector_id,
        assigned_at=timestamp,
        reason=reason,
        assigned_by=actor.actor_id,
    )
    event = AssignmentEvent(
        action=AssignmentAction.REASSIGN,
        inspector_id=inspector.inspector_id,
        previous_inspector_id=previous,
        timestamp=timestamp,
        reason=reason,
        actor_id=actor.actor_id,
    )
    return _record_action(
        case,
        actor=actor,
        timestamp=timestamp,
        correlation_id=correlation_id,
        audit_action=AuditAction.REASSIGNED,
        timeline_type=TimelineEventType.ASSIGNMENT_CHANGED,
        new_status=case.status,
        details={
            "action": "reassign",
            "previous_inspector_id": previous,
            "inspector_id": inspector.inspector_id,
            "reason": reason,
        },
        updates={
            "assigned_inspector": assignment,
            "assignment_history": (*case.assignment_history, event),
        },
    )


def unassign_case(
    case: InspectionCase,
    *,
    actor: ActorIdentity,
    timestamp: datetime,
    correlation_id: str,
    reason: str,
) -> InspectionCase:
    _require_role(actor, _SUPERVISOR_ROLES, "unassign an inspector")
    if case.assigned_inspector is None:
        raise CaseTransitionError("case is not assigned")
    if case.status in {
        CaseStatus.DISMISSED,
        CaseStatus.CONFIRMED_FOR_ADMIN_REVIEW,
        CaseStatus.CLOSED,
    }:
        raise CaseTransitionError(f"case cannot be unassigned from {case.status.value}")
    previous = case.assigned_inspector.inspector_id
    event = AssignmentEvent(
        action=AssignmentAction.UNASSIGN,
        inspector_id=None,
        previous_inspector_id=previous,
        timestamp=timestamp,
        reason=reason,
        actor_id=actor.actor_id,
    )
    target = (
        CaseStatus.TRIAGED
        if case.status in {CaseStatus.ASSIGNED, CaseStatus.UNDER_REVIEW}
        else case.status
    )
    return _record_action(
        case,
        actor=actor,
        timestamp=timestamp,
        correlation_id=correlation_id,
        audit_action=AuditAction.UNASSIGNED,
        timeline_type=TimelineEventType.ASSIGNMENT_CHANGED,
        new_status=target,
        details={"action": "unassign", "previous_inspector_id": previous, "reason": reason},
        updates={
            "assigned_inspector": None,
            "assignment_history": (*case.assignment_history, event),
        },
    )


def add_inspector_note(
    case: InspectionCase,
    text: str,
    *,
    actor: ActorIdentity,
    timestamp: datetime,
    correlation_id: str,
) -> InspectionCase:
    if case.status == CaseStatus.CLOSED:
        raise CaseTransitionError("cannot add a note to a closed case")
    _require_case_actor(case, actor, "add an inspector note")
    note = InspectorNote(
        note_id=deterministic_id(
            "case-note", case.case_id, actor.actor_id, timestamp.isoformat(), text
        ),
        actor_id=actor.actor_id,
        timestamp=timestamp,
        text=text,
    )
    return _record_action(
        case,
        actor=actor,
        timestamp=timestamp,
        correlation_id=correlation_id,
        audit_action=AuditAction.NOTE_ADDED,
        timeline_type=TimelineEventType.NOTE_ADDED,
        new_status=case.status,
        details={"note_id": note.note_id},
        updates={"inspector_notes": (*case.inspector_notes, note)},
    )


def attach_new_evidence(
    case: InspectionCase,
    asset: EvidenceAsset,
    content: bytes,
    *,
    actor: ActorIdentity,
    timestamp: datetime,
    correlation_id: str,
) -> InspectionCase:
    if case.status == CaseStatus.CLOSED:
        raise CaseTransitionError("cannot attach evidence to a closed case")
    if actor.role == ActorRole.INSPECTOR:
        _require_case_actor(case, actor, "attach evidence")
    if asset.case_id != case.case_id:
        raise EvidenceIntegrityError("evidence asset belongs to another case")
    verify_evidence_content(asset, content)
    assets = (*case.evidence_assets, asset)
    verify_chain_of_custody(assets)
    return _record_action(
        case,
        actor=actor,
        timestamp=timestamp,
        correlation_id=correlation_id,
        audit_action=AuditAction.EVIDENCE_ATTACHED,
        timeline_type=TimelineEventType.EVIDENCE_ATTACHED,
        new_status=case.status,
        evidence_asset_id=asset.asset_id,
        details={"asset_id": asset.asset_id, "content_sha256": asset.content_sha256},
        updates={"evidence_assets": assets},
    )


def verify_case_evidence(
    case: InspectionCase,
    content_by_asset_id: Mapping[str, bytes],
    *,
    actor: ActorIdentity,
    timestamp: datetime,
    correlation_id: str,
) -> tuple[InspectionCase, EvidenceIntegrityReport]:
    if actor.role == ActorRole.INSPECTOR:
        _require_case_actor(case, actor, "verify evidence")
    report = verify_chain_of_custody(case.evidence_assets, content_by_asset_id)
    updated = _record_action(
        case,
        actor=actor,
        timestamp=timestamp,
        correlation_id=correlation_id,
        audit_action=AuditAction.EVIDENCE_VERIFIED,
        timeline_type=TimelineEventType.EVIDENCE_VERIFIED,
        new_status=case.status,
        details={"verified_asset_ids": list(report.verified_asset_ids)},
    )
    return updated, report


def accept_for_further_review(
    case: InspectionCase,
    *,
    actor: ActorIdentity,
    timestamp: datetime,
    correlation_id: str,
    reason: str,
) -> InspectionCase:
    return transition_case(
        case,
        CaseStatus.CONFIRMED_FOR_ADMIN_REVIEW,
        actor=actor,
        timestamp=timestamp,
        correlation_id=correlation_id,
        reason=reason,
    )


def dismiss_case(
    case: InspectionCase,
    *,
    actor: ActorIdentity,
    timestamp: datetime,
    correlation_id: str,
    reason: str,
) -> InspectionCase:
    return transition_case(
        case,
        CaseStatus.DISMISSED,
        actor=actor,
        timestamp=timestamp,
        correlation_id=correlation_id,
        reason=reason,
    )


def request_reinspection(
    case: InspectionCase,
    *,
    actor: ActorIdentity,
    timestamp: datetime,
    correlation_id: str,
    reason: str,
) -> InspectionCase:
    return transition_case(
        case,
        CaseStatus.REINSPECTION_REQUIRED,
        actor=actor,
        timestamp=timestamp,
        correlation_id=correlation_id,
        reason=reason,
    )


def mark_evidence_insufficient(
    case: InspectionCase,
    *,
    actor: ActorIdentity,
    timestamp: datetime,
    correlation_id: str,
    reason: str,
) -> InspectionCase:
    return transition_case(
        case,
        CaseStatus.EVIDENCE_INSUFFICIENT,
        actor=actor,
        timestamp=timestamp,
        correlation_id=correlation_id,
        reason=reason,
    )


def start_review(
    case: InspectionCase,
    *,
    actor: ActorIdentity,
    timestamp: datetime,
    correlation_id: str,
    reason: str,
) -> InspectionCase:
    return transition_case(
        case,
        CaseStatus.UNDER_REVIEW,
        actor=actor,
        timestamp=timestamp,
        correlation_id=correlation_id,
        reason=reason,
    )


def close_case(
    case: InspectionCase,
    *,
    actor: ActorIdentity,
    timestamp: datetime,
    correlation_id: str,
    reason: str,
) -> InspectionCase:
    return transition_case(
        case,
        CaseStatus.CLOSED,
        actor=actor,
        timestamp=timestamp,
        correlation_id=correlation_id,
        reason=reason,
    )
