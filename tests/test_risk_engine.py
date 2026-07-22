from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
from shapely.geometry import LineString, Point, box, mapping
from shapely.geometry.base import BaseGeometry

from nirman_netra.data.contracts import (
    ApprovedFootprint,
    Complaint,
    Parcel,
    Permit,
    PublicBoundary,
)
from nirman_netra.domain import CoordinateReference, GeometryReference
from nirman_netra.persistence.risk import persist_manual_review_queue
from nirman_netra.risk.complaints import group_duplicate_complaints
from nirman_netra.risk.contracts import (
    ApprovedPlanComparison,
    ComplaintDeduplicationConfig,
    MunicipalRuleSet,
    ParcelIntersectionResult,
    PermitMatch,
    PermitMatchStatus,
    RiskEvidence,
    RiskLevel,
)
from nirman_netra.risk.engine import build_manual_review_queue, score_inspection_risk
from nirman_netra.risk.intersections import (
    compare_approved_plan,
    intersect_change_with_parcels,
)
from nirman_netra.risk.permits import match_permit_at_capture, select_effective_rule_set

CRS = CoordinateReference(value="EPSG:32643")
CAPTURE_TIME = datetime(2025, 1, 15, 12, tzinfo=UTC)


def _geometry(identifier: str, value: BaseGeometry) -> GeometryReference:
    return GeometryReference(
        geometry_id=identifier,
        geometry=dict(mapping(value)),
        crs=CRS,
    )


def _parcel(identifier: str, minimum_x: float, maximum_x: float) -> Parcel:
    return Parcel(
        parcel_id=identifier,
        ward_id="ward-001",
        setback_m=5,
        geometry=dict(mapping(box(minimum_x, 0, maximum_x, 100))),
        crs=CRS,
    )


def _approved() -> ApprovedFootprint:
    return ApprovedFootprint(
        footprint_id="footprint-001",
        building_id="building-001",
        parcel_id="parcel-001",
        plan_version=1,
        geometry=dict(mapping(box(20, 20, 50, 50))),
        crs=CRS,
    )


def _permit(
    identifier: str,
    valid_from: date,
    valid_until: date,
    *,
    plan_version: int = 1,
) -> Permit:
    return Permit(
        permit_id=identifier,
        parcel_id="parcel-001",
        footprint_id="footprint-001",
        plan_version=plan_version,
        permit_type="extension",
        valid_from=valid_from,
        valid_until=valid_until,
        status="active" if valid_until >= CAPTURE_TIME.date() else "expired",
        applicable_conditions=("retain-approved-setback",),
    )


def _rules(
    identifier: str = "rules-2025",
    effective_from: date = date(2025, 1, 1),
    effective_until: date | None = None,
) -> MunicipalRuleSet:
    return MunicipalRuleSet(
        rule_set_id=identifier,
        municipality_id="municipality-001",
        zone="residential-a",
        effective_from=effective_from,
        effective_until=effective_until,
        source_reference=f"synthetic://rules/{identifier}",
        maximum_site_coverage=0.6,
        setback_metres=5,
    )


def _plan_comparison(outside: bool = False) -> ApprovedPlanComparison:
    old = _geometry("old", box(20, 20, 40 if not outside else 50, 50))
    new = _geometry("new", box(20, 20, 50 if not outside else 60, 50))
    return compare_approved_plan(
        _approved(),
        old,
        new,
        _parcel("parcel-001", 0, 100),
        setback_metres=5,
    )


def _valid_permit_match() -> PermitMatch:
    permit = _permit("permit-active", date(2024, 12, 1), date(2025, 3, 1))
    return match_permit_at_capture(
        (permit,),
        parcel_id="parcel-001",
        capture_time=CAPTURE_TIME,
        approved_plan_version=1,
    )


def _parcel_match() -> ParcelIntersectionResult:
    return intersect_change_with_parcels(
        _geometry("change", box(30, 30, 40, 40)),
        (_parcel("parcel-001", 0, 100),),
    )


def _evidence(**updates: object) -> RiskEvidence:
    values: dict[str, object] = {
        "case_id": "case-001",
        "change_confidence": 0.9,
        "registration_quality": 0.95,
        "changed_area_square_metres": 100,
        "imagery_capture_time": CAPTURE_TIME,
        "evaluation_time": CAPTURE_TIME + timedelta(days=5),
        "quality_warnings": (),
        "corroborating_complaint_groups": 0,
    }
    values.update(updates)
    return RiskEvidence.model_validate(values)


def _complaint(
    identifier: str,
    x: float,
    description: str,
    media_hash: str,
    *,
    received_at: datetime = CAPTURE_TIME,
) -> Complaint:
    return Complaint(
        complaint_id=identifier,
        parcel_id="parcel-001",
        coordinate=dict(mapping(Point(x, 20))),
        crs=CRS,
        category="construction_change",
        received_at=received_at,
        description=description,
        media_sha256=media_hash,
    )


def test_change_fully_inside_approved_footprint() -> None:
    result = _plan_comparison()

    assert result.newly_added_area_square_metres > 0
    assert result.added_area_outside_approval_square_metres == 0
    assert result.removed_approved_area_square_metres == 0


def test_change_outside_approved_footprint_and_public_boundary() -> None:
    boundary = PublicBoundary(
        boundary_id="road-edge-001",
        boundary_type="road_edge",
        geometry=dict(mapping(LineString([(60, 0), (60, 100)]))),
        crs=CRS,
    )
    result = compare_approved_plan(
        _approved(),
        _geometry("old", box(20, 20, 50, 50)),
        _geometry("new", box(20, 20, 60, 50)),
        _parcel("parcel-001", 0, 100),
        setback_metres=5,
        public_boundaries=(boundary,),
    )

    assert result.added_area_outside_approval_square_metres == pytest.approx(300)
    assert result.public_boundary_intersection_ids == ("road-edge-001",)


def test_expired_permit_at_capture() -> None:
    result = match_permit_at_capture(
        (_permit("permit-expired", date(2024, 1, 1), date(2024, 12, 31)),),
        parcel_id="parcel-001",
        capture_time=CAPTURE_TIME,
        approved_plan_version=1,
    )

    assert result.status == PermitMatchStatus.EXPIRED_AT_CAPTURE
    assert not result.valid_at_capture


def test_active_permit_at_capture_returns_conditions() -> None:
    result = _valid_permit_match()

    assert result.valid_at_capture
    assert result.matched_permit is not None
    assert result.matched_permit.permit_type == "extension"
    assert result.matched_permit.applicable_conditions == ("retain-approved-setback",)


def test_wrong_plan_version() -> None:
    permit = _permit(
        "permit-plan-two",
        date(2024, 12, 1),
        date(2025, 3, 1),
        plan_version=2,
    )

    result = match_permit_at_capture(
        (permit,),
        parcel_id="parcel-001",
        capture_time=CAPTURE_TIME,
        approved_plan_version=1,
    )

    assert result.status == PermitMatchStatus.WRONG_PLAN_VERSION


def test_multiple_parcel_ambiguity() -> None:
    result = intersect_change_with_parcels(
        _geometry("crossing", box(90, 20, 110, 40)),
        (
            _parcel("parcel-001", 0, 100),
            _parcel("parcel-002", 100, 200),
        ),
    )

    assert result.multiple_parcel_ambiguity
    assert result.boundary_crossing
    assert len(result.overlaps) == 2


def test_no_parcel_match_status() -> None:
    result = intersect_change_with_parcels(
        _geometry("outside", box(300, 20, 310, 30)),
        (_parcel("parcel-001", 0, 100),),
    )

    assert result.no_parcel_match
    assert result.matching_parcel_id is None


def test_duplicate_complaints_are_grouped_and_preserved() -> None:
    complaints = (
        _complaint("complaint-001", 20, "Possible new extension", "a" * 64),
        _complaint(
            "complaint-002",
            21,
            "possible new extension",
            "a" * 64,
            received_at=CAPTURE_TIME + timedelta(hours=2),
        ),
    )

    result = group_duplicate_complaints(complaints, ComplaintDeduplicationConfig())

    assert result.original_complaints == complaints
    assert len(result.groups) == 1
    assert result.groups[0].is_duplicate_group
    assert "MEDIA_HASH" in result.groups[0].duplicate_signals


def test_distinct_nearby_complaints_remain_separate() -> None:
    complaints = (
        _complaint("complaint-001", 20, "wall paint concern", "a" * 64),
        _complaint("complaint-002", 21, "temporary material storage", "b" * 64),
    )

    result = group_duplicate_complaints(complaints, ComplaintDeduplicationConfig())

    assert len(result.groups) == 2
    assert all(not group.is_duplicate_group for group in result.groups)


def test_risk_score_has_separate_explanation_factors() -> None:
    expired = match_permit_at_capture(
        (_permit("permit-expired", date(2024, 1, 1), date(2024, 12, 31)),),
        parcel_id="parcel-001",
        capture_time=CAPTURE_TIME,
        approved_plan_version=1,
    )
    ambiguous = intersect_change_with_parcels(
        _geometry("crossing", box(90, 20, 110, 40)),
        (_parcel("parcel-001", 0, 100), _parcel("parcel-002", 100, 200)),
    )

    assessment = score_inspection_risk(
        _evidence(corroborating_complaint_groups=1),
        ambiguous,
        _plan_comparison(outside=True),
        expired,
        _rules(),
    )
    factors = {factor.code: factor.points for factor in assessment.factors}

    assert factors["UNAPPROVED_ADDED_AREA"] == 35
    assert factors["ACTIVE_PERMIT_NOT_FOUND"] == 20
    assert factors["PARCEL_BOUNDARY_REVIEW"] == 15
    assert assessment.level == RiskLevel.CRITICAL_REVIEW


def test_low_quality_evidence_reduces_risk_score() -> None:
    high_quality = score_inspection_risk(
        _evidence(change_confidence=0.5, changed_area_square_metres=10),
        _parcel_match(),
        _plan_comparison(),
        _valid_permit_match(),
        _rules(),
    )
    low_quality = score_inspection_risk(
        _evidence(
            change_confidence=0.5,
            changed_area_square_metres=10,
            registration_quality=0.6,
            quality_warnings=("BLUR",),
        ),
        _parcel_match(),
        _plan_comparison(),
        _valid_permit_match(),
        _rules(),
    )

    assert low_quality.score < high_quality.score
    assert any(factor.points < 0 for factor in low_quality.factors)


def test_rule_set_effective_date_selection() -> None:
    old = _rules("rules-2024", date(2024, 1, 1), date(2024, 12, 31))
    current = _rules("rules-2025", date(2025, 1, 1))

    selected = select_effective_rule_set(
        (old, current),
        municipality_id="municipality-001",
        zone="residential-a",
        capture_time=datetime(2024, 6, 1, tzinfo=UTC),
    )

    assert selected.rule_set_id == "rules-2024"


def test_no_automatic_legal_verdict_and_queue_is_persisted(tmp_path: Path) -> None:
    assessment = score_inspection_risk(
        _evidence(),
        _parcel_match(),
        _plan_comparison(),
        _valid_permit_match(),
        _rules(),
    )
    queue = build_manual_review_queue(
        "rules-2025",
        ((assessment, "parcel-001"),),
        created_at=CAPTURE_TIME + timedelta(days=5),
    )
    artifact = persist_manual_review_queue(queue, tmp_path)

    assert assessment.requires_human_review
    assert assessment.legal_verdict is None
    assert assessment.automatic_action is None
    assert queue.items[0].legal_verdict is None
    assert queue.items[0].status == "pending_human_review"
    assert artifact.item_count == 1
    assert (tmp_path / "manual-review-queue.json").is_file()
