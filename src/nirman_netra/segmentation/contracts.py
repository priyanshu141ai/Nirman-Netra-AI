"""Typed segmentation dataset, training, evaluation, and artifact contracts."""

import math
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nirman_netra.domain import BoundingBox, CoordinateReference


class SplitAssignment(StrEnum):
    UNASSIGNED = "unassigned"
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class LabelQualityStatus(StrEnum):
    VERIFIED = "verified"
    REVIEWED = "reviewed"
    WEAK = "weak"
    REJECTED = "rejected"


class ContractModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SegmentationTileMetadata(ContractModel):
    tile_id: str = Field(min_length=1)
    source_asset_id: str = Field(min_length=1)
    geographic_bounds: BoundingBox
    crs: CoordinateReference
    region_group: str = Field(min_length=1)
    split: SplitAssignment
    label_quality_status: LabelQualityStatus
    tile_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_crs(self) -> "SegmentationTileMetadata":
        if self.geographic_bounds.crs != self.crs:
            raise ValueError("tile bounds CRS must match tile CRS")
        return self


class GeographicSplitConfig(ContractModel):
    random_seed: int = 17
    train_ratio: float = Field(default=0.7, gt=0, lt=1)
    validation_ratio: float = Field(default=0.15, gt=0, lt=1)
    proximity_distance: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def validate_ratios(self) -> "GeographicSplitConfig":
        if self.train_ratio + self.validation_ratio >= 1:
            raise ValueError("train and validation ratios must leave a test split")
        return self


class DatasetValidationReport(ContractModel):
    sample_count: int = Field(ge=0)
    split_counts: dict[str, int]
    empty_mask_count: int = Field(ge=0)
    label_coverage_ratio: float = Field(ge=0, le=1)
    class_pixel_counts: dict[int, int]


class AugmentationConfig(ContractModel):
    horizontal_flip_probability: float = Field(default=0.5, ge=0, le=1)
    vertical_flip_probability: float = Field(default=0.5, ge=0, le=1)
    maximum_rotation_degrees: float = Field(default=8.0, ge=0, le=30)
    brightness_delta: float = Field(default=0.08, ge=0, le=0.5)
    contrast_delta: float = Field(default=0.1, ge=0, le=0.5)
    scale_minimum: float = Field(default=0.95, gt=0)
    scale_maximum: float = Field(default=1.05, gt=0)

    @model_validator(mode="after")
    def validate_scale(self) -> "AugmentationConfig":
        if self.scale_minimum > self.scale_maximum:
            raise ValueError("scale_minimum must not exceed scale_maximum")
        return self


class TrainingConfig(ContractModel):
    random_seed: int = 23
    device: Literal["cpu"] = "cpu"
    batch_size: int = Field(default=4, gt=0)
    learning_rate: float = Field(default=0.1, gt=0)
    epochs: int = Field(default=30, gt=0)
    early_stopping_patience: int = Field(default=5, gt=0)
    early_stopping_min_delta: float = Field(default=1e-5, ge=0)
    checkpoint_path: Path
    deterministic: bool = True
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)


class EvaluationConfig(ContractModel):
    threshold: float = Field(default=0.5, gt=0, lt=1)
    boundary_tolerance_pixels: int = Field(default=1, ge=0)
    object_overlap_threshold: float = Field(default=0.25, gt=0, le=1)
    small_building_max_pixels: int = Field(default=64, gt=0)


class SegmentationMetrics(ContractModel):
    iou: float = Field(ge=0, le=1)
    dice_f1: float = Field(ge=0, le=1)
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    boundary_f1: float = Field(ge=0, le=1)
    object_recall: float = Field(ge=0, le=1)
    small_building_recall: float = Field(ge=0, le=1)


class InputSchema(ContractModel):
    height: int = Field(gt=0)
    width: int = Field(gt=0)
    channels: int = Field(gt=0)
    channel_order: tuple[str, ...]
    dtype: Literal["uint8"] = "uint8"

    @model_validator(mode="after")
    def validate_channel_order(self) -> "InputSchema":
        if len(self.channel_order) != self.channels:
            raise ValueError("channel order must match input channels")
        return self


class Normalization(ContractModel):
    mean: tuple[float, ...]
    std: tuple[float, ...]

    @model_validator(mode="after")
    def validate_values(self) -> "Normalization":
        if len(self.mean) != len(self.std) or not self.mean:
            raise ValueError("normalization mean and std must have equal non-zero length")
        if not all(math.isfinite(value) for value in (*self.mean, *self.std)):
            raise ValueError("normalization values must be finite")
        if any(value <= 0 for value in self.std):
            raise ValueError("normalization std values must be positive")
        return self


class ModelArtifactMetadata(ContractModel):
    artifact_schema_version: str = "1.0.0"
    model_name: str
    model_version: str
    training_dataset_version: str
    feature_schema_version: str | None = None
    runtime: Literal["onnx"]
    input_schema: InputSchema
    normalization: Normalization
    class_mapping: dict[int, str]
    required_crs: tuple[str, ...] = ()
    supported_resolution_m: tuple[float, float] | None = None
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    onnx_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_metrics: SegmentationMetrics

    @model_validator(mode="after")
    def validate_resolution_range(self) -> "ModelArtifactMetadata":
        if self.supported_resolution_m is not None:
            minimum, maximum = self.supported_resolution_m
            if minimum <= 0 or minimum > maximum:
                raise ValueError("supported resolution range is invalid")
        return self


class PolygonExtractionConfig(ContractModel):
    minimum_pixel_area: float = Field(default=4.0, gt=0)
    simplification_tolerance: float = Field(default=0.0, ge=0)


class ExtractedPolygon(ContractModel):
    geometry: dict[str, object]
    crs: CoordinateReference
    source_pixel_area: float = Field(gt=0)
    was_repaired: bool
    repair_method: str | None = None

    @model_validator(mode="after")
    def validate_repair_record(self) -> "ExtractedPolygon":
        if self.was_repaired != (self.repair_method is not None):
            raise ValueError("geometry repairs must be recorded")
        return self


class TrainingHistory(ContractModel):
    train_loss: tuple[float, ...]
    validation_loss: tuple[float, ...]
    epochs_completed: int = Field(gt=0)
