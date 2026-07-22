"""Typed change-detection dataset, evaluation, and artifact contracts."""

from enum import IntEnum, StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nirman_netra.domain import BoundingBox, CoordinateReference
from nirman_netra.imagery.contracts import RegistrationMetrics
from nirman_netra.segmentation.contracts import LabelQualityStatus, SplitAssignment


class ContractModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ChangeLabel(IntEnum):
    NO_CHANGE = 0
    ADDED_BUILDING = 1
    REMOVED_BUILDING = 2
    FOOTPRINT_EXPANSION = 3
    PARTIAL_DEMOLITION = 4
    UNCERTAIN_CHANGE = 5


CHANGE_LABEL_NAMES: dict[ChangeLabel, str] = {
    ChangeLabel.NO_CHANGE: "no_change",
    ChangeLabel.ADDED_BUILDING: "added_building",
    ChangeLabel.REMOVED_BUILDING: "removed_building",
    ChangeLabel.FOOTPRINT_EXPANSION: "footprint_expansion",
    ChangeLabel.PARTIAL_DEMOLITION: "partial_demolition",
    ChangeLabel.UNCERTAIN_CHANGE: "uncertain_change",
}


class RegistrationTier(StrEnum):
    HIGH = "high_quality"
    MEDIUM = "medium_quality"
    DIFFICULT = "difficult_or_off_nadir"


class ChangePairMetadata(ContractModel):
    pair_id: str = Field(min_length=1)
    old_asset_id: str = Field(min_length=1)
    new_asset_id: str = Field(min_length=1)
    geographic_bounds: BoundingBox
    crs: CoordinateReference
    region_group: str = Field(min_length=1)
    split: SplitAssignment
    label_quality_status: LabelQualityStatus
    registration_metrics: RegistrationMetrics
    image_quality_warnings: tuple[str, ...] = ()
    off_nadir: bool = False
    pixel_area_square_metres: float = Field(gt=0)
    pair_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_metadata(self) -> "ChangePairMetadata":
        if self.geographic_bounds.crs != self.crs:
            raise ValueError("pair bounds CRS must match pair CRS")
        if self.old_asset_id == self.new_asset_id:
            raise ValueError("old and new source asset IDs must differ")
        return self


class ChangeDatasetReport(ContractModel):
    sample_count: int = Field(gt=0)
    split_counts: dict[str, int]
    class_pixel_counts: dict[int, int]
    empty_change_count: int = Field(ge=0)


class ImageDifferenceConfig(ContractModel):
    mode: Literal["pixel", "structural"] = "pixel"
    threshold: float = Field(default=0.15, gt=0, lt=1)
    registration_threshold_penalty: float = Field(default=0.1, ge=0, le=0.5)
    morphology_kernel_size: int = Field(default=3, ge=1)
    minimum_region_pixels: int = Field(default=8, gt=0)
    minimum_registration_score: float = Field(default=0.25, ge=0, le=1)

    @model_validator(mode="after")
    def validate_kernel(self) -> "ImageDifferenceConfig":
        if self.morphology_kernel_size % 2 == 0:
            raise ValueError("morphology kernel size must be odd")
        return self


class MaskDifferenceConfig(ContractModel):
    minimum_region_pixels: int = Field(default=4, gt=0)
    adjacency_pixels: int = Field(default=1, ge=0)
    minimum_registration_score: float = Field(default=0.25, ge=0, le=1)


class ChangedAreaStatistics(ContractModel):
    added_pixels: int = Field(ge=0)
    removed_pixels: int = Field(ge=0)
    common_pixels: int = Field(ge=0)
    added_area_square_metres: float = Field(ge=0)
    removed_area_square_metres: float = Field(ge=0)


class ChangeEvaluationConfig(ContractModel):
    object_overlap_threshold: float = Field(default=0.25, gt=0, le=1)
    high_registration_score: float = Field(default=0.8, gt=0, le=1)
    medium_registration_score: float = Field(default=0.55, gt=0, le=1)

    @model_validator(mode="after")
    def validate_tiers(self) -> "ChangeEvaluationConfig":
        if self.medium_registration_score >= self.high_registration_score:
            raise ValueError("medium registration threshold must be below high")
        return self


class ChangeMetrics(ContractModel):
    change_iou: float = Field(ge=0, le=1)
    change_f1: float = Field(ge=0, le=1)
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    false_positive_changed_area_square_metres: float = Field(ge=0)
    object_level_change_recall: float = Field(ge=0, le=1)
    false_alarms_per_square_kilometre: float = Field(ge=0)
    added_area_error_square_metres: float = Field(ge=0)
    removed_area_error_square_metres: float = Field(ge=0)


class RegistrationAwareEvaluation(ContractModel):
    overall: ChangeMetrics
    by_registration_tier: dict[RegistrationTier, ChangeMetrics]


class MethodComparison(ContractModel):
    baseline_image_difference: RegistrationAwareEvaluation
    baseline_mask_difference: RegistrationAwareEvaluation
    siamese_candidate: RegistrationAwareEvaluation


class SiameseTrainingConfig(ContractModel):
    random_seed: int = 37
    learning_rate: float = Field(default=0.2, gt=0)
    epochs: int = Field(default=40, gt=0)
    threshold: float = Field(default=0.5, gt=0, lt=1)
    minimum_registration_score: float = Field(default=0.25, ge=0, le=1)
    deterministic: bool = True


class ConfidenceConfig(ContractModel):
    warning_penalty: float = Field(default=0.08, ge=0, le=1)
    minimum_region_pixels: int = Field(default=8, gt=0)
    full_size_confidence_pixels: int = Field(default=64, gt=0)

    @model_validator(mode="after")
    def validate_sizes(self) -> "ConfidenceConfig":
        if self.minimum_region_pixels > self.full_size_confidence_pixels:
            raise ValueError("minimum region size must not exceed full-confidence size")
        return self


ChangePolygonLabel = Literal[
    "added_building",
    "removed_building",
    "footprint_expansion",
    "partial_demolition",
]


class ChangePolygon(ContractModel):
    label: ChangePolygonLabel
    geometry: dict[str, object]
    crs: CoordinateReference
    source_pair_id: str = Field(min_length=1)
    changed_area_square_metres: float = Field(gt=0)
    was_repaired: bool
    repair_method: str | None = None

    @model_validator(mode="after")
    def validate_repair_record(self) -> "ChangePolygon":
        if self.was_repaired != (self.repair_method is not None):
            raise ValueError("geometry repairs must be recorded")
        return self


class ChangeDetectionArtifact(ContractModel):
    pair_id: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    change_mask_uri: str = Field(min_length=1)
    mask_shape: tuple[int, int]
    change_polygons: tuple[ChangePolygon, ...]
    confidence: float = Field(ge=0, le=1)
    registration_score: float = Field(ge=0, le=1)
    warnings: tuple[str, ...] = ()
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_shape(self) -> "ChangeDetectionArtifact":
        if len(self.mask_shape) != 2 or any(value <= 0 for value in self.mask_shape):
            raise ValueError("change mask shape must contain two positive dimensions")
        return self
