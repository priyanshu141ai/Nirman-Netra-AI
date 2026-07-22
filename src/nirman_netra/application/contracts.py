"""Strict records for integration, jobs, readiness, and demo outputs."""

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nirman_netra.cases.contracts import CaseStatus, InspectionCase
from nirman_netra.change_detection.contracts import ChangePolygon
from nirman_netra.data.contracts import (
    ApprovedFootprint,
    Complaint,
    Inspector,
    Parcel,
    Permit,
    PublicBoundary,
)
from nirman_netra.domain import GeometryReference
from nirman_netra.imagery.contracts import QualityReport, QualityStatus, RasterMetadata
from nirman_netra.risk.contracts import MunicipalRuleSet, RiskLevel
from nirman_netra.segmentation.contracts import InputSchema, Normalization


class ContractModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class JobState(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class JobType(StrEnum):
    PROCESS_IMAGE_PAIR = "PROCESS_IMAGE_PAIR"


class ProcessingJob(ContractModel):
    job_id: str = Field(min_length=1)
    job_type: JobType
    source_asset_ids: tuple[str, str]
    image_pair_id: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    rule_set_version: str = Field(min_length=1)
    status: JobState
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    failure_code: str | None = None
    safe_failure_message: str | None = None
    correlation_id: str = Field(min_length=1)
    output_result_id: str | None = None
    idempotency_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    retry_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_timestamps(self) -> "ProcessingJob":
        for value in (self.created_at, self.started_at, self.completed_at):
            if value is not None and (value.tzinfo is None or value.utcoffset() is None):
                raise ValueError("job timestamps must be timezone-aware")
        if self.status == JobState.SUCCEEDED and self.output_result_id is None:
            raise ValueError("successful job requires an output result")
        if self.status == JobState.FAILED and self.failure_code is None:
            raise ValueError("failed job requires a failure code")
        return self


class AssetRecord(ContractModel):
    asset_id: str = Field(min_length=1)
    municipality_id: str = Field(min_length=1)
    object_key: str = Field(min_length=1)
    storage_uri: str = Field(min_length=1)
    metadata: RasterMetadata
    quality: QualityReport
    ingested_at: datetime

    @field_validator("ingested_at")
    @classmethod
    def ingested_at_must_be_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("asset ingestion timestamp must be timezone-aware")
        return value


class ImagePairRecord(ContractModel):
    pair_id: str = Field(min_length=1)
    old_asset_id: str = Field(min_length=1)
    new_asset_id: str = Field(min_length=1)
    source_hashes: tuple[str, str]
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def created_at_must_be_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("pair timestamp must be timezone-aware")
        return value


class ModelRequirement(ContractModel):
    model_name: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    artifact_schema_version: str = Field(min_length=1)
    training_dataset_version: str = Field(min_length=1)
    feature_schema_version: str | None = None
    input_schema: InputSchema
    normalization: Normalization
    class_mapping: dict[int, str]
    required_crs: tuple[str, ...]
    supported_resolution_m: tuple[float, float]

    @model_validator(mode="after")
    def validate_resolution(self) -> "ModelRequirement":
        if (
            self.supported_resolution_m[0] <= 0
            or self.supported_resolution_m[0] > self.supported_resolution_m[1]
        ):
            raise ValueError("required model resolution range is invalid")
        return self


class RuleSetRequirement(ContractModel):
    rule_set_id: str = Field(min_length=1)
    municipality_id: str = Field(min_length=1)
    zone: str = Field(min_length=1)
    rule_version: str = Field(min_length=1)
    risk_threshold_version: str = Field(min_length=1)


class ProcessPairRequest(ContractModel):
    image_pair_id: str = Field(min_length=1)
    model_requirement: ModelRequirement
    rule_requirement: RuleSetRequirement
    processing_configuration_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    correlation_id: str = Field(min_length=1)


class ChangeResultRecord(ContractModel):
    result_id: str = Field(min_length=1)
    job_id: str = Field(min_length=1)
    source_asset_ids: tuple[str, str]
    image_pair_id: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    artifact_schema_version: str = Field(min_length=1)
    rule_set_id: str = Field(min_length=1)
    rule_set_version: str = Field(min_length=1)
    source_crs: str = Field(min_length=1)
    output_crs: str = Field(min_length=1)
    processed_at: datetime
    image_quality_status: QualityStatus
    registration_quality_status: QualityStatus
    registration_score: float = Field(ge=0, le=1)
    warnings: tuple[str, ...]
    polygons: tuple[ChangePolygon, ...]
    changed_area_square_metres: float = Field(ge=0)
    risk_level: RiskLevel
    human_review_status: CaseStatus | None = None
    case_id: str | None = None
    change_mask_storage_uri: str = Field(min_length=1)
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("processed_at")
    @classmethod
    def processed_at_must_be_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("processing timestamp must be timezone-aware")
        return value


class DataQualitySnapshot(ContractModel):
    created_at: datetime
    total_assets: int = Field(ge=0)
    low_quality_assets: int = Field(ge=0)
    registration_failures: int = Field(ge=0)
    model_checksum_failures: int = Field(ge=0)
    high_risk_cases: int = Field(ge=0)


class ReadinessComponent(ContractModel):
    name: str
    ready: bool
    code: str | None = None


class ReadinessReport(ContractModel):
    ready: bool
    components: tuple[ReadinessComponent, ...]


class DemoScenarioResult(ContractModel):
    scenario: Literal[
        "approved_change",
        "permit_mismatch",
        "duplicate_complaints",
        "low_registration_quality",
        "evidence_integrity_failure",
    ]
    case_id: str
    case_status: CaseStatus
    risk_level: RiskLevel
    warnings: tuple[str, ...] = ()


class DemoReport(ContractModel):
    demo_version: str
    deterministic_seed: int
    scenarios: tuple[DemoScenarioResult, ...]
    legal_verdict: None = None


class CasePage(ContractModel):
    items: tuple[InspectionCase, ...]
    total: int = Field(ge=0)
    offset: int = Field(ge=0)
    limit: int = Field(gt=0)


class RegisteredRuleSet(ContractModel):
    rule_set: MunicipalRuleSet


class MunicipalContext(ContractModel):
    municipality_id: str = Field(min_length=1)
    zone: str = Field(min_length=1)
    parcels: tuple[Parcel, ...]
    approved_footprints: tuple[ApprovedFootprint, ...]
    permits: tuple[Permit, ...]
    complaints: tuple[Complaint, ...]
    public_boundaries: tuple[PublicBoundary, ...]
    inspectors: tuple[Inspector, ...]
    rule_sets: tuple[MunicipalRuleSet, ...]


class PipelineEvidence(ContractModel):
    old_observed_footprint: GeometryReference
    new_observed_footprint: GeometryReference
    change_polygons: tuple[ChangePolygon, ...]
    changed_area_square_metres: float = Field(ge=0)
    change_confidence: float = Field(ge=0, le=1)
    segmentation_confidence: float = Field(ge=0, le=1)
    source_crs: str = Field(min_length=1)
    output_crs: str = Field(min_length=1)
    image_quality_status: QualityStatus
    registration_quality_status: QualityStatus
    registration_score: float = Field(ge=0, le=1)
    warnings: tuple[str, ...] = ()
    change_mask_storage_uri: str = Field(min_length=1)
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
