"""Validated, transport-neutral geospatial domain models."""

import math
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pyproj import CRS
from pyproj.exceptions import CRSError


class CoordinateReference(BaseModel):
    """Canonical CRS reference without assuming a project default."""

    model_config = ConfigDict(frozen=True)
    value: str = Field(min_length=1)

    @field_validator("value")
    @classmethod
    def canonicalize(cls, value: str) -> str:
        try:
            return CRS.from_user_input(value).to_string()
        except CRSError as exc:
            raise ValueError(f"invalid CRS: {value}") from exc

    @property
    def is_projected(self) -> bool:
        return CRS.from_user_input(self.value).is_projected


class BoundingBox(BaseModel):
    """Axis-ordered bounds tied to their source CRS."""

    model_config = ConfigDict(frozen=True)
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    crs: CoordinateReference

    @model_validator(mode="after")
    def validate_order(self) -> "BoundingBox":
        coordinates = (self.min_x, self.min_y, self.max_x, self.max_y)
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError("bounding-box coordinates must be finite")
        if self.min_x >= self.max_x or self.min_y >= self.max_y:
            raise ValueError("expected min_x < max_x and min_y < max_y")
        return self


class RasterAssetMetadata(BaseModel):
    """Traceable raster metadata; pixel content is stored separately."""

    model_config = ConfigDict(frozen=True)
    asset_id: str = Field(min_length=1)
    source_uri: str = Field(min_length=1)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    band_count: int = Field(gt=0)
    dtype: str = Field(min_length=1)
    crs: CoordinateReference
    bounds: BoundingBox
    captured_at: datetime
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_metadata(self) -> "RasterAssetMetadata":
        if self.bounds.crs != self.crs:
            raise ValueError("raster bounds CRS must match raster CRS")
        if self.captured_at.tzinfo is None or self.captured_at.utcoffset() is None:
            raise ValueError("captured_at must be timezone-aware")
        return self


class ParcelReference(BaseModel):
    """Non-personal reference to a parcel in an external source system."""

    model_config = ConfigDict(frozen=True)
    parcel_id: str = Field(min_length=1)
    source_system: str = Field(min_length=1)


class GeometryReference(BaseModel):
    """Traceable GeoJSON-like geometry and its CRS."""

    model_config = ConfigDict(frozen=True)
    geometry_id: str = Field(min_length=1)
    geometry: dict[str, Any]
    crs: CoordinateReference
    source_asset_id: str | None = None
    repair_record: str | None = None
