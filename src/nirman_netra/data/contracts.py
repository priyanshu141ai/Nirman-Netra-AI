"""Strict contracts for synthetic municipal and geospatial records."""

import json
from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nirman_netra.domain import BoundingBox, CoordinateReference
from nirman_netra.exceptions import GeometryValidationError
from nirman_netra.geospatial import validate_geometry
from nirman_netra.utils import content_hash

EntityId = Annotated[str, Field(min_length=3, max_length=96, pattern=r"^[a-z0-9][a-z0-9._-]+$")]
Geometry = dict[str, Any]
ValidationStage = Literal["contract", "relationship", "deduplication"]


class Scenario(StrEnum):
    APPROVED_EXTENSION = "approved_extension"
    UNAPPROVED_FOOTPRINT_EXPANSION = "unapproved_footprint_expansion"
    POSSIBLE_NEW_FLOOR = "possible_new_floor"
    PARTIAL_DEMOLITION = "partial_demolition"
    PARCEL_ENCROACHMENT = "parcel_encroachment"
    DUPLICATE_COMPLAINTS = "duplicate_complaints"
    LOW_QUALITY_IMAGERY_METADATA = "low_quality_imagery_metadata"
    EXPIRED_PERMIT = "expired_permit"
    VALID_ACTIVE_PERMIT = "valid_active_permit"


class ContractModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SpatialContract(ContractModel):
    geometry: Geometry
    crs: CoordinateReference

    @field_validator("geometry")
    @classmethod
    def geometry_must_be_valid(cls, value: Geometry) -> Geometry:
        try:
            validate_geometry(value)
        except GeometryValidationError as exc:
            raise ValueError(str(exc)) from exc
        return value


class Municipality(SpatialContract):
    municipality_id: EntityId
    code: EntityId


class Ward(SpatialContract):
    ward_id: EntityId
    municipality_id: EntityId


class Parcel(SpatialContract):
    parcel_id: EntityId
    ward_id: EntityId
    setback_m: float = Field(ge=0)


class ApprovedBuilding(ContractModel):
    building_id: EntityId
    parcel_id: EntityId
    footprint_id: EntityId
    approved_floor_count: int = Field(gt=0)


class ApprovedFootprint(SpatialContract):
    footprint_id: EntityId
    building_id: EntityId
    parcel_id: EntityId
    plan_version: int = Field(gt=0)


class Permit(ContractModel):
    permit_id: EntityId
    parcel_id: EntityId
    footprint_id: EntityId
    plan_version: int = Field(gt=0)
    valid_from: date
    valid_until: date
    status: Literal["active", "expired"]

    @model_validator(mode="after")
    def validate_dates(self) -> "Permit":
        if self.valid_from > self.valid_until:
            raise ValueError("permit valid_from must not exceed valid_until")
        return self


class ImageryAsset(ContractModel):
    asset_id: EntityId
    source: str = Field(min_length=1)
    capture_time: datetime
    ingestion_time: datetime
    crs: CoordinateReference
    resolution_m: float = Field(gt=0)
    bounding_box: BoundingBox
    storage_uri: str = Field(min_length=1)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    licence_reference: str = Field(min_length=1)
    quality_status: Literal["accepted", "low_quality", "rejected"]

    @model_validator(mode="after")
    def validate_metadata(self) -> "ImageryAsset":
        for value in (self.capture_time, self.ingestion_time):
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError("imagery timestamps must be timezone-aware")
        if self.capture_time > self.ingestion_time:
            raise ValueError("capture_time must not exceed ingestion_time")
        if self.bounding_box.crs != self.crs:
            raise ValueError("imagery bounding-box CRS must match asset CRS")
        return self


class Complaint(ContractModel):
    complaint_id: EntityId
    parcel_id: EntityId
    coordinate: Geometry
    crs: CoordinateReference
    category: Literal["construction_change", "possible_encroachment"]
    received_at: datetime

    @model_validator(mode="after")
    def validate_coordinate(self) -> "Complaint":
        if self.received_at.tzinfo is None or self.received_at.utcoffset() is None:
            raise ValueError("received_at must be timezone-aware")
        try:
            geometry = validate_geometry(self.coordinate)
        except GeometryValidationError as exc:
            raise ValueError(str(exc)) from exc
        if geometry.geom_type != "Point":
            raise ValueError("complaint coordinate must be a Point")
        return self


class Inspector(ContractModel):
    inspector_id: EntityId
    team_code: EntityId
    active: bool = True


class PublicBoundary(SpatialContract):
    boundary_id: EntityId
    boundary_type: Literal["road_edge", "municipal_boundary"]


class InspectionCaseSeed(ContractModel):
    case_id: EntityId
    scenario: Scenario
    parcel_id: EntityId
    building_id: EntityId
    permit_id: EntityId
    imagery_asset_ids: tuple[EntityId, ...] = Field(min_length=1)
    complaint_ids: tuple[EntityId, ...]
    inspector_id: EntityId
    observed_footprint: Geometry
    observed_floor_count: int = Field(gt=0)
    status: Literal["seeded", "review_pending", "closed"] = "seeded"

    @field_validator("observed_footprint")
    @classmethod
    def observed_geometry_must_be_valid(cls, value: Geometry) -> Geometry:
        try:
            validate_geometry(value)
        except GeometryValidationError as exc:
            raise ValueError(str(exc)) from exc
        return value


class GenerationConfig(ContractModel):
    scenario: Scenario
    random_seed: int
    processing_crs: CoordinateReference
    generator_version: str = "2.0.0"
    reference_time: datetime = datetime(2025, 1, 15, 12, tzinfo=UTC)
    origin_x: float = 500_000
    origin_y: float = 2_000_000
    municipality_size_m: float = Field(default=2_000, gt=0)
    ward_count: int = Field(default=2, ge=2, le=10)
    parcels_per_ward: int = Field(default=2, ge=1, le=20)
    parcel_margin_m: float = Field(default=20, ge=0)
    setback_m: float = Field(default=5, ge=0)

    @field_validator("reference_time")
    @classmethod
    def reference_time_must_be_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("reference_time must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_spatial_parameters(self) -> "GenerationConfig":
        smallest_span = min(
            self.municipality_size_m / self.ward_count,
            self.municipality_size_m / self.parcels_per_ward,
        )
        if 2 * (self.parcel_margin_m + self.setback_m) >= smallest_span:
            raise ValueError("margins and setback leave no usable parcel area")
        return self

    @property
    def configuration_hash(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return content_hash(payload.encode())


class SyntheticDataset(ContractModel):
    municipalities: tuple[Municipality, ...]
    wards: tuple[Ward, ...]
    parcels: tuple[Parcel, ...]
    approved_buildings: tuple[ApprovedBuilding, ...]
    approved_footprints: tuple[ApprovedFootprint, ...]
    permits: tuple[Permit, ...]
    imagery_assets: tuple[ImageryAsset, ...]
    complaints: tuple[Complaint, ...]
    inspectors: tuple[Inspector, ...]
    inspection_cases: tuple[InspectionCaseSeed, ...]
    public_boundaries: tuple[PublicBoundary, ...]

    @property
    def record_count(self) -> int:
        return sum(
            len(records)
            for records in (
                self.municipalities,
                self.wards,
                self.parcels,
                self.approved_buildings,
                self.approved_footprints,
                self.permits,
                self.imagery_assets,
                self.complaints,
                self.inspectors,
                self.inspection_cases,
                self.public_boundaries,
            )
        )


class QuarantineRecord(ContractModel):
    original_record_reference: str
    reason_code: str
    validation_stage: ValidationStage
    timestamp: datetime
    source_batch_id: EntityId

    @field_validator("timestamp")
    @classmethod
    def timestamp_must_be_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("quarantine timestamp must be timezone-aware")
        return value


class DatasetManifest(ContractModel):
    dataset_version: str
    generator_version: str
    scenario: Scenario
    random_seed: int
    configuration_hash: str
    entity_counts: dict[str, int]
    quarantine_counts: dict[str, int]
    output_paths: dict[str, str]
    checksums: dict[str, str]
