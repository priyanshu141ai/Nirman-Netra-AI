"""Typed raster, quality, alignment, and registration contracts."""

import math
from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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
    transform: tuple[float, float, float, float, float, float]
    bounds: BoundingBox
    resolution: tuple[float, float]
    nodata: float | None
    captured_at: datetime | None = None
    capture_timestamp_raw: str | None = None
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_metadata(self) -> "RasterMetadata":
        if self.bounds.crs != self.crs:
            raise ValueError("bounds CRS must match raster CRS")
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
