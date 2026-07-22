"""Explainable inspection-priority scoring without legal conclusions."""

from collections.abc import Sequence
from datetime import datetime

from nirman_netra.exceptions import ConfigurationError, DatasetValidationError
from nirman_netra.risk.contracts import (
    ApprovedPlanComparison,
    ManualReviewQueue,
    ManualReviewQueueItem,
    MunicipalRuleSet,
    ParcelIntersectionResult,
    PermitMatch,
    PermitMatchStatus,
    RiskAssessment,
    RiskEvidence,
    RiskFactor,
    RiskLevel,
)
from nirman_netra.utils import content_hash


def _factor(code: str, points: float, explanation: str) -> RiskFactor:
    return RiskFactor(code=code, points=round(points, 2), explanation=explanation)


def _risk_level(score: float, rules: MunicipalRuleSet) -> RiskLevel:
    thresholds = rules.thresholds
    if score >= thresholds.critical_review_minimum:
        return RiskLevel.CRITICAL_REVIEW
    if score >= thresholds.high_minimum:
        return RiskLevel.HIGH
    if score >= thresholds.medium_minimum:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def score_inspection_risk(
    evidence: RiskEvidence,
    parcel_match: ParcelIntersectionResult,
    plan_comparison: ApprovedPlanComparison,
    permit_match: PermitMatch,
    rules: MunicipalRuleSet,
) -> RiskAssessment:
    """Return separately auditable factors and a bounded review-priority score."""

    capture_date = evidence.imagery_capture_time.date()
    if not (
        rules.effective_from <= capture_date
        and (rules.effective_until is None or capture_date <= rules.effective_until)
    ):
        raise ConfigurationError("risk rule set was not effective at imagery capture time")
    if permit_match.capture_date != capture_date:
        raise DatasetValidationError("permit match date differs from imagery capture date")
    if permit_match.requested_plan_version != plan_comparison.approved_plan_version:
        raise DatasetValidationError("permit and approved-plan versions are inconsistent")
    weights = rules.weights
    factors: list[RiskFactor] = []
    if evidence.change_confidence:
        factors.append(
            _factor(
                "CHANGE_CONFIDENCE",
                weights.change_confidence * evidence.change_confidence,
                f"change confidence {evidence.change_confidence:.2f}",
            )
        )
    if evidence.changed_area_square_metres:
        area_fraction = min(
            1.0,
            evidence.changed_area_square_metres / rules.changed_area_reference_square_metres,
        )
        factors.append(
            _factor(
                "CHANGED_AREA",
                weights.changed_area * area_fraction,
                f"changed area {evidence.changed_area_square_metres:.2f} square metres",
            )
        )
    if plan_comparison.added_area_outside_approval_square_metres > 1e-9:
        factors.append(
            _factor(
                "UNAPPROVED_ADDED_AREA",
                weights.unapproved_added_area,
                "observed added footprint extends outside the approved plan",
            )
        )
    if permit_match.status == PermitMatchStatus.WRONG_PLAN_VERSION:
        factors.append(
            _factor(
                "WRONG_PLAN_VERSION",
                weights.wrong_plan_version,
                "permit valid at capture references another approved plan version",
            )
        )
    elif not permit_match.valid_at_capture:
        factors.append(
            _factor(
                "ACTIVE_PERMIT_NOT_FOUND",
                weights.permit_not_found,
                f"no matching permit valid at capture: {permit_match.status.value}",
            )
        )
    if (
        parcel_match.boundary_crossing
        or parcel_match.multiple_parcel_ambiguity
        or parcel_match.no_parcel_match
    ):
        factors.append(
            _factor(
                "PARCEL_BOUNDARY_REVIEW",
                weights.parcel_boundary_crossing,
                "change crosses, ambiguously overlaps, or does not match parcel boundaries",
            )
        )
    if evidence.corroborating_complaint_groups:
        factors.append(
            _factor(
                "COMPLAINT_CORROBORATION",
                weights.complaint_corroboration,
                f"{evidence.corroborating_complaint_groups} independent complaint group(s)",
            )
        )
    if (
        rules.maximum_site_coverage is not None
        and plan_comparison.site_coverage_ratio > rules.maximum_site_coverage
    ):
        factors.append(
            _factor(
                "SITE_COVERAGE_EXCEEDED",
                weights.coverage_exceeded,
                "observed site coverage exceeds the selected municipal rule set",
            )
        )
    if plan_comparison.setback_zone_intersection_area_square_metres > 1e-9:
        factors.append(
            _factor(
                "SETBACK_ZONE_INTERSECTION",
                weights.setback_intersection,
                "observed footprint intersects the configured setback zone",
            )
        )
    if plan_comparison.public_boundary_intersection_ids:
        factors.append(
            _factor(
                "PUBLIC_BOUNDARY_INTERSECTION",
                weights.public_boundary_intersection,
                "observed footprint intersects a configured public boundary",
            )
        )
    if evidence.registration_quality < 0.8:
        factors.append(
            _factor(
                "REGISTRATION_QUALITY_PENALTY",
                -weights.medium_registration_penalty,
                f"registration quality {evidence.registration_quality:.2f} reduces confidence",
            )
        )
    if evidence.quality_warnings:
        factors.append(
            _factor(
                "IMAGE_QUALITY_PENALTY",
                -weights.quality_warning_penalty * min(3, len(evidence.quality_warnings)),
                f"{len(evidence.quality_warnings)} image-quality warning(s)",
            )
        )
    age_days = (evidence.evaluation_time - evidence.imagery_capture_time).days
    if age_days > rules.evidence_freshness_days:
        factors.append(
            _factor(
                "STALE_EVIDENCE_PENALTY",
                -weights.stale_evidence_penalty,
                f"evidence age {age_days} days exceeds configured freshness",
            )
        )
    score = round(max(0.0, min(100.0, sum(item.points for item in factors))), 2)
    return RiskAssessment(
        case_id=evidence.case_id,
        rule_set_id=rules.rule_set_id,
        score=score,
        level=_risk_level(score, rules),
        factors=tuple(factors),
    )


def build_manual_review_queue(
    rule_set_id: str,
    assessments: Sequence[tuple[RiskAssessment, str | None]],
    *,
    created_at: datetime,
) -> ManualReviewQueue:
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise DatasetValidationError("manual-review queue timestamp must be timezone-aware")
    items = [
        ManualReviewQueueItem(
            queue_item_id=f"review-{content_hash(f'{assessment.case_id}:{rule_set_id}'.encode())[:20]}",
            case_id=assessment.case_id,
            parcel_id=parcel_id,
            risk_score=assessment.score,
            risk_level=assessment.level,
            factor_codes=tuple(factor.code for factor in assessment.factors),
            created_at=created_at,
        )
        for assessment, parcel_id in assessments
    ]
    items.sort(key=lambda item: (-item.risk_score, item.case_id))
    return ManualReviewQueue(rule_set_id=rule_set_id, items=tuple(items))
