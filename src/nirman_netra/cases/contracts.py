"""Immutable case, evidence, identity, timeline, and audit contracts."""

import json
from datetime import datetime
from enum import StrEnum
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from nirman_netra.risk.contracts import ComplaintGroup, PermitMatch, RiskFactor
from nirman_netra.utils import content_hash


class ContractModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CaseStatus(StrEnum):
    CREATED = "CREATED"
    TRIAGED = "TRIAGED"
    ASSIGNED = "ASSIGNED"
    UNDER_REVIEW = "UNDER_REVIEW"
    REINSPECTION_REQUIRED = "REINSPECTION_REQUIRED"
    EVIDENCE_INSUFFICIENT = "EVIDENCE_INSUFFICIENT"
    DISMISSED = "DISMISSED"
    CONFIRMED_FOR_ADMIN_REVIEW = "CONFIRMED_FOR_ADMIN_REVIEW"
    CLOSED = "CLOSED"


class ActorRole(StrEnum):
    SYSTEM_PROCESSOR = "system_processor"
    INSPECTOR = "inspector"
    SUPERVISOR = "supervisor"
    ADMINISTRATOR = "administrator"


class ActorIdentity(ContractModel):
    actor_id: str = Field(min_length=1)
    role: ActorRole


class AssignmentAction(StrEnum):
    ASSIGN = "assign"
    REASSIGN = "reassign"
    UNASSIGN = "unassign"


class InspectorAssignment(ContractModel):
    inspector_id: str = Field(min_length=1)
    assigned_at: datetime
    reason: str = Field(min_length=1, max_length=1_000)
    assigned_by: str = Field(min_length=1)

    @field_validator("assigned_at")
    @classmethod
    def assigned_at_must_be_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("assignment timestamp must be timezone-aware")
        return value


class AssignmentEvent(ContractModel):
    action: AssignmentAction
    inspector_id: str | None
    previous_inspector_id: str | None
    timestamp: datetime
    reason: str = Field(min_length=1, max_length=1_000)
    actor_id: str = Field(min_length=1)

    @field_validator("timestamp")
    @classmethod
    def timestamp_must_be_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("assignment event timestamp must be timezone-aware")
        return value


class EvidenceAssetKind(StrEnum):
    ORIGINAL = "original"
    DERIVED = "derived"


def _contains_sensitive_key(value: JsonValue) -> bool:
    if isinstance(value, dict):
        for key, nested in value.items():
            normalised = key.casefold().replace("-", "_")
            if any(
                term in normalised
                for term in (
                    "password",
                    "secret",
                    "token",
                    "credential",
                    "api_key",
                    "access_key",
                    "private_key",
                    "authorization",
                )
            ):
                return True
            if _contains_sensitive_key(nested):
                return True
    if isinstance(value, list):
        return any(_contains_sensitive_key(item) for item in value)
    return False


class EvidenceAsset(ContractModel):
    asset_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    kind: EvidenceAssetKind
    original_asset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    derived_asset_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source_asset_id: str | None = None
    source: str = Field(min_length=1)
    capture_time: datetime
    ingestion_time: datetime
    processing_job_id: str | None = None
    transformation_parameters: dict[str, JsonValue] = Field(default_factory=dict)
    model_version: str | None = None
    storage_uri: str = Field(min_length=1)

    @property
    def content_sha256(self) -> str:
        return self.derived_asset_sha256 or self.original_asset_sha256

    @model_validator(mode="after")
    def validate_asset(self) -> "EvidenceAsset":
        for value in (self.capture_time, self.ingestion_time):
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError("evidence timestamps must be timezone-aware")
        if self.capture_time > self.ingestion_time:
            raise ValueError("evidence capture time must not exceed ingestion time")
        if self.kind == EvidenceAssetKind.ORIGINAL:
            if self.derived_asset_sha256 is not None or self.source_asset_id is not None:
                raise ValueError("original evidence cannot declare a derived hash or source asset")
        elif (
            self.derived_asset_sha256 is None
            or self.source_asset_id is None
            or self.processing_job_id is None
            or self.model_version is None
        ):
            raise ValueError("derived evidence requires hash, source, processing job, and model")
        if _contains_sensitive_key(self.transformation_parameters):
            raise ValueError("evidence transformation parameters contain sensitive metadata")
        parsed_uri = urlsplit(self.storage_uri)
        if parsed_uri.query or parsed_uri.username or parsed_uri.password:
            raise ValueError("evidence storage URI must not contain credentials or query secrets")
        return self


class TimelineEventType(StrEnum):
    CASE_CREATED = "CASE_CREATED"
    STATE_CHANGED = "STATE_CHANGED"
    ASSIGNMENT_CHANGED = "ASSIGNMENT_CHANGED"
    NOTE_ADDED = "NOTE_ADDED"
    EVIDENCE_ATTACHED = "EVIDENCE_ATTACHED"
    EVIDENCE_VERIFIED = "EVIDENCE_VERIFIED"


class TimelineEvent(ContractModel):
    event_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    event_type: TimelineEventType
    actor_id: str = Field(min_length=1)
    timestamp: datetime
    evidence_asset_id: str | None = None
    details: dict[str, JsonValue] = Field(default_factory=dict)
    previous_event_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    event_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("timestamp")
    @classmethod
    def timestamp_must_be_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timeline timestamp must be timezone-aware")
        return value


def timeline_event_hash(event: TimelineEvent) -> str:
    payload = json.dumps(
        event.model_dump(mode="json", exclude={"event_hash"}),
        sort_keys=True,
        separators=(",", ":"),
    )
    return content_hash(payload.encode())


class EvidenceTimeline(ContractModel):
    events: tuple[TimelineEvent, ...] = ()
    head_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_chain(self) -> "EvidenceTimeline":
        previous: str | None = None
        for event in self.events:
            if (
                event.previous_event_hash != previous
                or timeline_event_hash(event) != event.event_hash
            ):
                raise ValueError("evidence timeline hash chain is invalid")
            previous = event.event_hash
        if self.head_hash != previous:
            raise ValueError("evidence timeline head hash is invalid")
        return self


class AuditAction(StrEnum):
    CASE_CREATED = "CASE_CREATED"
    STATE_TRANSITION = "STATE_TRANSITION"
    ASSIGNED = "ASSIGNED"
    REASSIGNED = "REASSIGNED"
    UNASSIGNED = "UNASSIGNED"
    NOTE_ADDED = "NOTE_ADDED"
    EVIDENCE_ATTACHED = "EVIDENCE_ATTACHED"
    EVIDENCE_VERIFIED = "EVIDENCE_VERIFIED"


class AuditEvent(ContractModel):
    event_id: str = Field(min_length=1)
    actor_id: str = Field(min_length=1)
    actor_role: ActorRole
    timestamp: datetime
    action: AuditAction
    case_id: str = Field(min_length=1)
    previous_state: CaseStatus | None
    new_state: CaseStatus
    correlation_id: str = Field(min_length=1)
    previous_event_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    event_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("timestamp")
    @classmethod
    def timestamp_must_be_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("audit timestamp must be timezone-aware")
        return value


def audit_event_hash(event: AuditEvent) -> str:
    payload = json.dumps(
        event.model_dump(mode="json", exclude={"event_hash"}),
        sort_keys=True,
        separators=(",", ":"),
    )
    return content_hash(payload.encode())


class AuditTrail(ContractModel):
    events: tuple[AuditEvent, ...] = ()
    head_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_chain(self) -> "AuditTrail":
        previous: str | None = None
        for event in self.events:
            if event.previous_event_hash != previous or audit_event_hash(event) != event.event_hash:
                raise ValueError("audit event hash chain is invalid")
            previous = event.event_hash
        if self.head_hash != previous:
            raise ValueError("audit trail head hash is invalid")
        return self


class InspectorNote(ContractModel):
    note_id: str = Field(min_length=1)
    actor_id: str = Field(min_length=1)
    timestamp: datetime
    text: str = Field(min_length=1, max_length=5_000)

    @field_validator("timestamp")
    @classmethod
    def timestamp_must_be_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("note timestamp must be timezone-aware")
        return value


class InspectionCase(ContractModel):
    case_id: str = Field(min_length=1)
    parcel_id: str | None
    image_pair_id: str = Field(min_length=1)
    change_result_id: str = Field(min_length=1)
    risk_score: float = Field(ge=0, le=100)
    risk_factors: tuple[RiskFactor, ...]
    permit_result: PermitMatch
    complaint_groups: tuple[ComplaintGroup, ...]
    quality_warnings: tuple[str, ...]
    assigned_inspector: InspectorAssignment | None
    status: CaseStatus
    created_at: datetime
    updated_at: datetime
    evidence_assets: tuple[EvidenceAsset, ...] = ()
    evidence_timeline: EvidenceTimeline
    inspector_notes: tuple[InspectorNote, ...] = ()
    assignment_history: tuple[AssignmentEvent, ...] = ()
    audit_trail: AuditTrail
    legal_verdict: None = None
    automatic_penalty: None = None

    @model_validator(mode="after")
    def validate_case(self) -> "InspectionCase":
        for value in (self.created_at, self.updated_at):
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError("case timestamps must be timezone-aware")
        if self.updated_at < self.created_at:
            raise ValueError("case update time cannot predate creation")
        if (
            self.status in {CaseStatus.ASSIGNED, CaseStatus.UNDER_REVIEW}
            and self.assigned_inspector is None
        ):
            raise ValueError("assigned and review states require an inspector")
        if (
            self.status in {CaseStatus.CREATED, CaseStatus.TRIAGED}
            and self.assigned_inspector is not None
        ):
            raise ValueError("created and triaged cases cannot retain an assignment")
        if any(asset.case_id != self.case_id for asset in self.evidence_assets):
            raise ValueError("evidence asset belongs to another case")
        return self


class EvidenceIntegrityReport(ContractModel):
    verified_asset_ids: tuple[str, ...]
    verified_source_links: tuple[tuple[str, str], ...]


class CaseExportArtifact(ContractModel):
    case_id: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)
    storage_uri: str = Field(min_length=1)
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
