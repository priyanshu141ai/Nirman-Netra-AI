"""Typed raster, quality, alignment, and registration contracts."""

import math
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pyproj import CRS
from pyproj.exceptions import CRSError

from nirman_netra.domain import BoundingBox, CoordinateReference


class QualityStatus(StrEnum):
    PASS = "PASS"
    PASS_WITH_WARNING = "PASS_WITH_WARNING"
    REQUIRES_MANUAL_ALIGNMENT = "REQUIRES_MANUAL_ALIGNMENT"
    REJECTED = "REJECTED"


class RasterMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    asset_id: str = Field(min_length=1)
    source_uri: str = Field(min_length=1)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    band_count: int = Field(gt=0)
    dtype: str = Field(min_length=1)
    channel_order: tuple[str, ...]
    crs: CoordinateReference
    epsg_code: int | None
    transform: tuple[float, float, float, float, float, float]
    bounds: BoundingBox
    resolution: tuple[float, float]
    nodata: float | None
    captured_at: datetime | None = None
    capture_timestamp_raw: str | None = None
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="before")
    @classmethod
    def populate_epsg_code(cls, value: Any) -> Any:
        if not isinstance(value, dict) or "epsg_code" in value:
            return value
        raw_crs = value.get("crs")
        crs_value = raw_crs.get("value") if isinstance(raw_crs, dict) else raw_crs
        if isinstance(crs_value, str):
            try:
                epsg_code = CRS.from_user_input(crs_value).to_epsg()
            except CRSError:
                return value
            return {**value, "epsg_code": epsg_code}
        return value

    @model_validator(mode="after")
    def validate_metadata(self) -> "RasterMetadata":
        if self.bounds.crs != self.crs:
            raise ValueError("bounds CRS must match raster CRS")
        if self.epsg_code != CRS.from_user_input(self.crs.value).to_epsg():
            raise ValueError("EPSG code must match raster CRS when available")
        if len(self.channel_order) != self.band_count:
            raise ValueError("channel_order length must match band_count")
        if not all(math.isfinite(value) for value in self.transform):
            raise ValueError("transform values must be finite")
        if self.transform[0] * self.transform[4] - self.transform[1] * self.transform[3] == 0:
            raise ValueError("transform must be invertible")
        if any(not math.isfinite(value) or value <= 0 for value in self.resolution):
            raise ValueError("resolution must contain positive finite values")
        if self.captured_at is not None and (
            self.captured_at.tzinfo is None or self.captured_at.utcoffset() is None
        ):
            raise ValueError("captured_at must be timezone-aware")
        return self


class CRSTransformation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    asset_id: str = Field(min_length=1)
    source_crs: CoordinateReference
    source_epsg_code: int | None
    target_crs: CoordinateReference
    target_epsg_code: int | None
    operation: Literal["identity", "reprojection"]

    @model_validator(mode="after")
    def validate_transformation(self) -> "CRSTransformation":
        source = CRS.from_user_input(self.source_crs.value)
        target = CRS.from_user_input(self.target_crs.value)
        if self.source_epsg_code != source.to_epsg() or self.target_epsg_code != target.to_epsg():
            raise ValueError("transformation EPSG metadata is inconsistent")
        if not target.is_projected:
            raise ValueError("registration transformation target must be projected")
        expected = "identity" if source.equals(target) else "reprojection"
        if self.operation != expected:
            raise ValueError("CRS transformation operation is inconsistent")
        return self


class CommonGrid(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    crs: CoordinateReference
    epsg_code: int | None
    transform: tuple[float, float, float, float, float, float]
    bounds: BoundingBox
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    resolution: tuple[float, float]

    @model_validator(mode="after")
    def validate_grid(self) -> "CommonGrid":
        if self.bounds.crs != self.crs:
            raise ValueError("common-grid bounds CRS must match grid CRS")
        parsed = CRS.from_user_input(self.crs.value)
        if not parsed.is_projected:
            raise ValueError("common grid must use a projected CRS")
        if self.epsg_code != parsed.to_epsg():
            raise ValueError("common-grid EPSG metadata is inconsistent")
        if self.transform[0] * self.transform[4] - self.transform[1] * self.transform[3] == 0:
            raise ValueError("common-grid transform must be invertible")
        if any(not math.isfinite(value) or value <= 0 for value in self.resolution):
            raise ValueError("common-grid resolution must be positive and finite")
        return self


class QualityIssue(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str
    severity: Literal["warning", "rejected", "manual"]
    message: str


class QualityMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    blur_variance: float
    dark_pixel_ratio: float
    bright_pixel_ratio: float
    valid_pixel_ratio: float


class QualityReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    asset_id: str
    status: QualityStatus
    metrics: QualityMetrics
    issues: tuple[QualityIssue, ...] = ()


class PairQualityReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: QualityStatus
    geographic_overlap_ratio: float = Field(ge=0, le=1)
    resolution_ratio: float = Field(ge=1)
    issues: tuple[QualityIssue, ...] = ()


class RegistrationMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    inlier_count: int = Field(ge=0)
    inlier_ratio: float = Field(ge=0, le=1)
    reprojection_error: float = Field(ge=0)
    overlap_ratio: float = Field(ge=0, le=1)
    transform_plausible: bool
    registration_quality_score: float = Field(ge=0, le=1)


class RegistrationArtifact(BaseModel):
    model_config = ConfigDict(frozen=True)

    pair_id: str
    before_asset_id: str
    after_asset_id: str
    transform_matrix: tuple[tuple[float, ...], ...]
    metrics: RegistrationMetrics
    status: QualityStatus
    registered_output_uri: str
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    warnings: tuple[str, ...] = ()
    before_metadata: RasterMetadata
    after_metadata: RasterMetadata
    common_grid: CommonGrid
    crs_transformations: tuple[CRSTransformation, ...]
    visual_refinement_applied: bool
    visual_refinement_method: Literal["none", "features", "features_ecc"]
    reliable_for_change_detection: bool

    @model_validator(mode="after")
    def validate_reliability(self) -> "RegistrationArtifact":
        reliable_statuses = {QualityStatus.PASS, QualityStatus.PASS_WITH_WARNING}
        if self.reliable_for_change_detection != (self.status in reliable_statuses):
            raise ValueError("registration reliability must match its explicit status")
        if self.visual_refinement_applied != (self.visual_refinement_method != "none"):
            raise ValueError("visual refinement method is inconsistent")
        return self


class RegistrationReliabilityReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    pair_id: str | None = None
    before_asset_id: str | None = None
    after_asset_id: str | None = None
    status: QualityStatus
    reliable_for_change_detection: bool
    failure_stage: Literal["ingestion", "geospatial_alignment", "visual_refinement"] | None = None
    reason_codes: tuple[str, ...]
    artifact: RegistrationArtifact | None = None

    @model_validator(mode="after")
    def validate_decision(self) -> "RegistrationReliabilityReport":
        reliable_statuses = {QualityStatus.PASS, QualityStatus.PASS_WITH_WARNING}
        if self.reliable_for_change_detection != (self.status in reliable_statuses):
            raise ValueError("reliability decision must match its explicit status")
        if self.reliable_for_change_detection and self.artifact is None:
            raise ValueError("reliable registration requires a persisted artifact")
        if self.artifact is not None and self.artifact.status != self.status:
            raise ValueError("artifact and reliability-report statuses must match")
        return self


class ManualControlPoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    before_x: float
    before_y: float
    after_x: float
    after_y: float

    @field_validator("before_x", "before_y", "after_x", "after_y")
    @classmethod
    def coordinates_must_be_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("control-point coordinates must be finite")
        return value


class ManualAlignmentRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    pair_id: str
    estimator: Literal["affine", "homography"]
    points: tuple[ManualControlPoint, ...]

    @model_validator(mode="after")
    def validate_point_count(self) -> "ManualAlignmentRequest":
        minimum = 3 if self.estimator == "affine" else 4
        if len(self.points) < minimum:
            raise ValueError(f"{self.estimator} alignment requires at least {minimum} points")
        return self
