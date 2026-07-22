"""Typed parcel, permit, complaint, rule, and risk outputs."""

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nirman_netra.data.contracts import Complaint, Permit
from nirman_netra.domain import CoordinateReference


class ContractModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ParcelOverlap(ContractModel):
    parcel_id: str = Field(min_length=1)
    intersection_area_square_metres: float = Field(gt=0)
    percentage_of_change_inside: float = Field(gt=0, le=100)


class ParcelIntersectionResult(ContractModel):
    change_geometry_id: str = Field(min_length=1)
    matching_parcel_id: str | None
    intersection_area_square_metres: float = Field(ge=0)
    percentage_inside_parcel: float = Field(ge=0, le=100)
    boundary_crossing: bool
    multiple_parcel_ambiguity: bool
    no_parcel_match: bool
    overlaps: tuple[ParcelOverlap, ...]
    analysis_crs: CoordinateReference

    @model_validator(mode="after")
    def validate_match_state(self) -> "ParcelIntersectionResult":
        if self.no_parcel_match != (self.matching_parcel_id is None):
            raise ValueError("parcel match status is inconsistent")
        if self.multiple_parcel_ambiguity != (len(self.overlaps) > 1):
            raise ValueError("multiple-parcel status is inconsistent")
        return self


class ApprovedPlanComparison(ContractModel):
    approved_plan_version: int = Field(gt=0)
    approved_footprint_area_square_metres: float = Field(gt=0)
    observed_old_area_square_metres: float = Field(ge=0)
    observed_new_area_square_metres: float = Field(ge=0)
    newly_added_area_square_metres: float = Field(ge=0)
    added_area_outside_approval_square_metres: float = Field(ge=0)
    removed_approved_area_square_metres: float = Field(ge=0)
    setback_zone_intersection_area_square_metres: float = Field(ge=0)
    public_boundary_intersection_ids: tuple[str, ...]
    site_coverage_ratio: float = Field(ge=0)
    comparison_crs: CoordinateReference


class PermitMatchStatus(StrEnum):
    VALID_MATCH = "valid_match"
    EXPIRED_AT_CAPTURE = "expired_at_capture"
    NOT_YET_VALID = "not_yet_valid"
    WRONG_PLAN_VERSION = "wrong_plan_version"
    NO_MATCH = "no_match"


class PermitMatch(ContractModel):
    parcel_id: str = Field(min_length=1)
    capture_date: date
    requested_plan_version: int = Field(gt=0)
    status: PermitMatchStatus
    matched_permit: Permit | None
    candidate_permit_ids: tuple[str, ...]

    @property
    def valid_at_capture(self) -> bool:
        return self.status == PermitMatchStatus.VALID_MATCH

    @model_validator(mode="after")
    def validate_match(self) -> "PermitMatch":
        if (self.matched_permit is not None) != self.valid_at_capture:
            raise ValueError("matched permit must be present only for a valid match")
        return self


class ComplaintDeduplicationConfig(ContractModel):
    spatial_distance_metres: float = Field(default=25, gt=0)
    time_window_days: int = Field(default=7, gt=0)
    text_similarity_threshold: float = Field(default=0.82, ge=0, le=1)
    perceptual_hash_max_distance: int = Field(default=6, ge=0)


class ComplaintGroup(ContractModel):
    group_id: str = Field(min_length=1)
    representative_complaint_id: str = Field(min_length=1)
    complaint_ids: tuple[str, ...] = Field(min_length=1)
    duplicate_signals: tuple[str, ...]

    @property
    def is_duplicate_group(self) -> bool:
        return len(self.complaint_ids) > 1


class ComplaintDeduplicationResult(ContractModel):
    original_complaints: tuple[Complaint, ...]
    groups: tuple[ComplaintGroup, ...]


class RiskThresholds(ContractModel):
    medium_minimum: float = Field(default=25, ge=0, le=100)
    high_minimum: float = Field(default=50, ge=0, le=100)
    critical_review_minimum: float = Field(default=75, ge=0, le=100)

    @model_validator(mode="after")
    def validate_order(self) -> "RiskThresholds":
        if not self.medium_minimum < self.high_minimum < self.critical_review_minimum:
            raise ValueError("risk thresholds must be strictly increasing")
        return self


class RiskWeights(ContractModel):
    change_confidence: float = Field(default=20, ge=0)
    changed_area: float = Field(default=10, ge=0)
    unapproved_added_area: float = Field(default=35, ge=0)
    permit_not_found: float = Field(default=20, ge=0)
    wrong_plan_version: float = Field(default=15, ge=0)
    parcel_boundary_crossing: float = Field(default=15, ge=0)
    complaint_corroboration: float = Field(default=10, ge=0)
    coverage_exceeded: float = Field(default=15, ge=0)
    setback_intersection: float = Field(default=10, ge=0)
    public_boundary_intersection: float = Field(default=10, ge=0)
    medium_registration_penalty: float = Field(default=10, ge=0)
    quality_warning_penalty: float = Field(default=5, ge=0)
    stale_evidence_penalty: float = Field(default=8, ge=0)


class MunicipalRuleSet(ContractModel):
    rule_set_id: str = Field(min_length=1)
    rule_version: str = "1.0.0"
    risk_threshold_version: str = "1.0.0"
    municipality_id: str = Field(min_length=1)
    zone: str = Field(min_length=1)
    effective_from: date
    effective_until: date | None = None
    source_reference: str = Field(min_length=1)
    maximum_far: float | None = Field(default=None, gt=0)
    maximum_site_coverage: float | None = Field(default=None, gt=0, le=1)
    setback_metres: float = Field(ge=0)
    changed_area_reference_square_metres: float = Field(default=100, gt=0)
    evidence_freshness_days: int = Field(default=180, gt=0)
    thresholds: RiskThresholds = Field(default_factory=RiskThresholds)
    weights: RiskWeights = Field(default_factory=RiskWeights)

    @model_validator(mode="after")
    def validate_dates(self) -> "MunicipalRuleSet":
        if self.effective_until is not None and self.effective_from > self.effective_until:
            raise ValueError("rule-set effective dates are invalid")
        return self


class RiskLevel(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL_REVIEW = "CRITICAL_REVIEW"


class RiskFactor(ContractModel):
    code: str = Field(min_length=1)
    points: float
    explanation: str = Field(min_length=1)


class RiskAssessment(ContractModel):
    case_id: str = Field(min_length=1)
    rule_set_id: str = Field(min_length=1)
    score: float = Field(ge=0, le=100)
    level: RiskLevel
    factors: tuple[RiskFactor, ...]
    requires_human_review: bool = True
    legal_verdict: None = None
    automatic_action: None = None


class RiskEvidence(ContractModel):
    case_id: str = Field(min_length=1)
    change_confidence: float = Field(ge=0, le=1)
    registration_quality: float = Field(ge=0, le=1)
    changed_area_square_metres: float = Field(ge=0)
    imagery_capture_time: datetime
    evaluation_time: datetime
    quality_warnings: tuple[str, ...] = ()
    corroborating_complaint_groups: int = Field(default=0, ge=0)

    @field_validator("imagery_capture_time", "evaluation_time")
    @classmethod
    def timestamps_must_be_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("risk evidence timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_timeline(self) -> "RiskEvidence":
        if self.evaluation_time < self.imagery_capture_time:
            raise ValueError("risk evaluation cannot predate imagery capture")
        return self


class ManualReviewQueueItem(ContractModel):
    queue_item_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    parcel_id: str | None
    risk_score: float = Field(ge=0, le=100)
    risk_level: RiskLevel
    factor_codes: tuple[str, ...]
    status: str = "pending_human_review"
    created_at: datetime
    legal_verdict: None = None

    @field_validator("created_at")
    @classmethod
    def created_at_must_be_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("queue timestamp must be timezone-aware")
        return value


class ManualReviewQueue(ContractModel):
    rule_set_id: str = Field(min_length=1)
    items: tuple[ManualReviewQueueItem, ...]


class QueueArtifact(ContractModel):
    storage_uri: str = Field(min_length=1)
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    item_count: int = Field(ge=0)
