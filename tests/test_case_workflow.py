from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from nirman_netra.cases.contracts import (
    ActorIdentity,
    ActorRole,
    AuditAction,
    CaseStatus,
    EvidenceAsset,
    EvidenceAssetKind,
    InspectionCase,
)
from nirman_netra.cases.evidence import (
    verify_chain_of_custody,
    verify_evidence_content,
)
from nirman_netra.cases.service import (
    accept_for_further_review,
    add_inspector_note,
    assign_case,
    attach_new_evidence,
    close_case,
    create_case_from_risk,
    dismiss_case,
    reassign_case,
    request_reinspection,
    start_review,
    transition_case,
    unassign_case,
)
from nirman_netra.data.contracts import Inspector
from nirman_netra.exceptions import (
    AuthorizationError,
    CaseTransitionError,
    EvidenceIntegrityError,
)
from nirman_netra.persistence.cases import export_case_package
from nirman_netra.risk.contracts import (
    ComplaintGroup,
    PermitMatch,
    PermitMatchStatus,
    RiskAssessment,
    RiskFactor,
    RiskLevel,
)
from nirman_netra.utils import content_hash

BASE_TIME = datetime(2025, 2, 1, 10, tzinfo=UTC)
SYSTEM = ActorIdentity(actor_id="processor-001", role=ActorRole.SYSTEM_PROCESSOR)
SUPERVISOR = ActorIdentity(actor_id="supervisor-001", role=ActorRole.SUPERVISOR)
ADMINISTRATOR = ActorIdentity(actor_id="administrator-001", role=ActorRole.ADMINISTRATOR)
INSPECTOR_ACTOR = ActorIdentity(actor_id="inspector-001", role=ActorRole.INSPECTOR)
SECOND_INSPECTOR_ACTOR = ActorIdentity(actor_id="inspector-002", role=ActorRole.INSPECTOR)
CORRELATION_ID = "correlation-001"


def _inspector(identifier: str) -> Inspector:
    return Inspector(inspector_id=identifier, team_code="inspection-team", active=True)


def _risk() -> RiskAssessment:
    return RiskAssessment(
        case_id="case-001",
        rule_set_id="rules-2025",
        score=68,
        level=RiskLevel.HIGH,
        factors=(
            RiskFactor(
                code="UNAPPROVED_ADDED_AREA",
                points=35,
                explanation="observed addition needs human review",
            ),
            RiskFactor(
                code="ACTIVE_PERMIT_NOT_FOUND",
                points=20,
                explanation="no matching permit at capture",
            ),
        ),
    )


def _permit_result() -> PermitMatch:
    return PermitMatch(
        parcel_id="parcel-001",
        capture_date=date(2025, 1, 15),
        requested_plan_version=1,
        status=PermitMatchStatus.NO_MATCH,
        matched_permit=None,
        candidate_permit_ids=(),
    )


def _complaint_group() -> ComplaintGroup:
    return ComplaintGroup(
        group_id="complaint-group-001",
        representative_complaint_id="complaint-001",
        complaint_ids=("complaint-001", "complaint-002"),
        duplicate_signals=("MEDIA_HASH", "SAME_PARCEL"),
    )


def _created_case() -> InspectionCase:
    return create_case_from_risk(
        _risk(),
        parcel_id="parcel-001",
        image_pair_id="pair-001",
        change_result_id="change-001",
        permit_result=_permit_result(),
        complaint_groups=(_complaint_group(),),
        quality_warnings=("MEDIUM_REGISTRATION",),
        actor=SYSTEM,
        timestamp=BASE_TIME,
        correlation_id=CORRELATION_ID,
    )


def _triaged_case() -> InspectionCase:
    return transition_case(
        _created_case(),
        CaseStatus.TRIAGED,
        actor=SUPERVISOR,
        timestamp=BASE_TIME + timedelta(minutes=1),
        correlation_id=CORRELATION_ID,
        reason="risk candidate accepted for triage",
    )


def _assigned_case() -> InspectionCase:
    return assign_case(
        _triaged_case(),
        _inspector("inspector-001"),
        actor=SUPERVISOR,
        timestamp=BASE_TIME + timedelta(minutes=2),
        correlation_id=CORRELATION_ID,
        reason="geographic workload allocation",
    )


def _under_review_case() -> InspectionCase:
    return start_review(
        _assigned_case(),
        actor=INSPECTOR_ACTOR,
        timestamp=BASE_TIME + timedelta(minutes=3),
        correlation_id=CORRELATION_ID,
        reason="evidence review started",
    )


def _original_asset(content: bytes = b"original-evidence") -> EvidenceAsset:
    return EvidenceAsset(
        asset_id="evidence-original-001",
        case_id="case-001",
        kind=EvidenceAssetKind.ORIGINAL,
        original_asset_sha256=content_hash(content),
        source="municipal-imagery-catalogue",
        capture_time=BASE_TIME - timedelta(days=10),
        ingestion_time=BASE_TIME - timedelta(days=9),
        transformation_parameters={},
        storage_uri="memory://evidence/original-001.tif",
    )


def _derived_asset(source: EvidenceAsset, content: bytes = b"derived-evidence") -> EvidenceAsset:
    return EvidenceAsset(
        asset_id="evidence-derived-001",
        case_id="case-001",
        kind=EvidenceAssetKind.DERIVED,
        original_asset_sha256=source.content_sha256,
        derived_asset_sha256=content_hash(content),
        source_asset_id=source.asset_id,
        source="registration-pipeline",
        capture_time=source.capture_time,
        ingestion_time=BASE_TIME,
        processing_job_id="job-001",
        transformation_parameters={"transform": [1, 0, 0, 0, 1, 0]},
        model_version="registration-1.0.0",
        storage_uri="memory://evidence/derived-001.tif",
    )


def test_valid_case_creation_from_risk_result() -> None:
    case = _created_case()

    assert case.status == CaseStatus.CREATED
    assert case.parcel_id == "parcel-001"
    assert case.image_pair_id == "pair-001"
    assert case.change_result_id == "change-001"
    assert case.risk_score == 68
    assert len(case.risk_factors) == 2
    assert case.permit_result.status == PermitMatchStatus.NO_MATCH
    assert len(case.complaint_groups) == 1
    assert case.assigned_inspector is None
    assert case.legal_verdict is None
    assert case.automatic_penalty is None


def test_invalid_transition_fails_clearly() -> None:
    with pytest.raises(CaseTransitionError, match="invalid case transition"):
        transition_case(
            _created_case(),
            CaseStatus.UNDER_REVIEW,
            actor=SUPERVISOR,
            timestamp=BASE_TIME + timedelta(minutes=1),
            correlation_id=CORRELATION_ID,
            reason="invalid shortcut",
        )


def test_assignment_reassignment_and_unassignment() -> None:
    assigned = _assigned_case()
    reassigned = reassign_case(
        assigned,
        _inspector("inspector-002"),
        actor=SUPERVISOR,
        timestamp=BASE_TIME + timedelta(minutes=3),
        correlation_id=CORRELATION_ID,
        reason="workload balancing",
    )
    unassigned = unassign_case(
        reassigned,
        actor=SUPERVISOR,
        timestamp=BASE_TIME + timedelta(minutes=4),
        correlation_id=CORRELATION_ID,
        reason="awaiting specialist availability",
    )

    assert assigned.assigned_inspector is not None
    assert assigned.assigned_inspector.inspector_id == "inspector-001"
    assert reassigned.assigned_inspector is not None
    assert reassigned.assigned_inspector.inspector_id == SECOND_INSPECTOR_ACTOR.actor_id
    assert reassigned.assignment_history[-1].reason == "workload balancing"
    assert unassigned.assigned_inspector is None
    assert unassigned.status == CaseStatus.TRIAGED
    assert len(unassigned.assignment_history) == 3


def test_evidence_hash_match_and_source_link() -> None:
    original_content = b"original-evidence"
    derived_content = b"derived-evidence"
    original = _original_asset(original_content)
    derived = _derived_asset(original, derived_content)

    verify_evidence_content(original, original_content)
    report = verify_chain_of_custody(
        (original, derived),
        {
            original.asset_id: original_content,
            derived.asset_id: derived_content,
        },
    )

    assert report.verified_asset_ids == (original.asset_id, derived.asset_id)
    assert report.verified_source_links == ((original.asset_id, derived.asset_id),)


def test_evidence_hash_mismatch_is_tamper_error() -> None:
    with pytest.raises(EvidenceIntegrityError, match="hash mismatch"):
        verify_evidence_content(_original_asset(), b"tampered-content")


def test_missing_source_asset_is_detected() -> None:
    original = _original_asset()
    derived = _derived_asset(original)

    with pytest.raises(EvidenceIntegrityError, match="missing source asset"):
        verify_chain_of_custody((derived,))


def test_duplicate_evidence_is_rejected() -> None:
    first = _original_asset()
    duplicate = first.model_copy(update={"asset_id": "evidence-original-002"})

    with pytest.raises(EvidenceIntegrityError, match="duplicate evidence content"):
        verify_chain_of_custody((first, duplicate))


def test_evidence_timeline_is_append_only() -> None:
    original_case = _created_case()
    asset = _original_asset()

    updated = attach_new_evidence(
        original_case,
        asset,
        b"original-evidence",
        actor=SYSTEM,
        timestamp=BASE_TIME + timedelta(minutes=1),
        correlation_id=CORRELATION_ID,
    )

    assert len(original_case.evidence_timeline.events) == 1
    assert len(updated.evidence_timeline.events) == 2
    assert updated.evidence_timeline.events[-1].evidence_asset_id == asset.asset_id
    assert updated.evidence_timeline.head_hash != original_case.evidence_timeline.head_hash


def test_reinspection_flow_requires_human_transition() -> None:
    under_review = _under_review_case()
    requested = request_reinspection(
        under_review,
        actor=INSPECTOR_ACTOR,
        timestamp=BASE_TIME + timedelta(minutes=4),
        correlation_id=CORRELATION_ID,
        reason="new oblique capture required",
    )
    with_evidence = attach_new_evidence(
        requested,
        _original_asset(),
        b"original-evidence",
        actor=INSPECTOR_ACTOR,
        timestamp=BASE_TIME + timedelta(minutes=5),
        correlation_id=CORRELATION_ID,
    )
    resumed = start_review(
        with_evidence,
        actor=INSPECTOR_ACTOR,
        timestamp=BASE_TIME + timedelta(minutes=6),
        correlation_id=CORRELATION_ID,
        reason="reinspection evidence attached and manually reopened",
    )

    assert requested.status == CaseStatus.REINSPECTION_REQUIRED
    assert with_evidence.status == CaseStatus.REINSPECTION_REQUIRED
    assert resumed.status == CaseStatus.UNDER_REVIEW


def test_dismissal_flow() -> None:
    dismissed = dismiss_case(
        _under_review_case(),
        actor=INSPECTOR_ACTOR,
        timestamp=BASE_TIME + timedelta(minutes=4),
        correlation_id=CORRELATION_ID,
        reason="registered evidence shows no persistent footprint change",
    )

    closed = close_case(
        dismissed,
        actor=ADMINISTRATOR,
        timestamp=BASE_TIME + timedelta(minutes=5),
        correlation_id=CORRELATION_ID,
        reason="administrative closure after human dismissal",
    )

    assert dismissed.status == CaseStatus.DISMISSED
    assert closed.status == CaseStatus.CLOSED
    assert closed.legal_verdict is None


def test_human_approval_is_required_for_admin_review_state() -> None:
    case = _under_review_case()

    with pytest.raises(AuthorizationError):
        accept_for_further_review(
            case,
            actor=SYSTEM,
            timestamp=BASE_TIME + timedelta(minutes=4),
            correlation_id=CORRELATION_ID,
            reason="processor cannot approve",
        )
    with pytest.raises(AuthorizationError):
        accept_for_further_review(
            case,
            actor=INSPECTOR_ACTOR,
            timestamp=BASE_TIME + timedelta(minutes=4),
            correlation_id=CORRELATION_ID,
            reason="inspector escalation needs supervisor approval",
        )
    approved = accept_for_further_review(
        case,
        actor=SUPERVISOR,
        timestamp=BASE_TIME + timedelta(minutes=4),
        correlation_id=CORRELATION_ID,
        reason="supervisor accepts evidence for administrative review",
    )

    assert approved.status == CaseStatus.CONFIRMED_FOR_ADMIN_REVIEW
    assert approved.legal_verdict is None


def test_note_and_state_actions_generate_audit_events() -> None:
    case = _under_review_case()
    noted = add_inspector_note(
        case,
        "Observed footprint requires comparison with the approved plan version.",
        actor=INSPECTOR_ACTOR,
        timestamp=BASE_TIME + timedelta(minutes=4),
        correlation_id="correlation-note",
    )
    event = noted.audit_trail.events[-1]

    assert event.action == AuditAction.NOTE_ADDED
    assert event.actor_id == INSPECTOR_ACTOR.actor_id
    assert event.previous_state == CaseStatus.UNDER_REVIEW
    assert event.new_state == CaseStatus.UNDER_REVIEW
    assert event.correlation_id == "correlation-note"
    assert len(noted.inspector_notes) == 1


def test_case_export_is_reproducible(tmp_path: Path) -> None:
    case = attach_new_evidence(
        _created_case(),
        _original_asset(),
        b"original-evidence",
        actor=SYSTEM,
        timestamp=BASE_TIME + timedelta(minutes=1),
        correlation_id=CORRELATION_ID,
    )

    first = export_case_package(case, tmp_path / "first")
    second = export_case_package(case, tmp_path / "second")

    assert first.checksum == second.checksum
    assert (tmp_path / "first" / "case-package.json").read_bytes() == (
        tmp_path / "second" / "case-package.json"
    ).read_bytes()
