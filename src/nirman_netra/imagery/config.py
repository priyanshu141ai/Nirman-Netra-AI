"""Configurable image-quality and registration thresholds."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ImageryPipelineConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    minimum_dimension: int = Field(default=64, ge=8)
    allowed_band_counts: tuple[int, ...] = (1, 3, 4)
    minimum_blur_variance: float = Field(default=20.0, ge=0)
    dark_pixel_value: int = Field(default=12, ge=0, le=255)
    bright_pixel_value: int = Field(default=243, ge=0, le=255)
    extreme_exposure_ratio: float = Field(default=0.9, ge=0, le=1)
    minimum_valid_pixel_ratio: float = Field(default=0.8, ge=0, le=1)
    minimum_geographic_overlap: float = Field(default=0.2, gt=0, le=1)
    maximum_resolution_ratio: float = Field(default=4.0, ge=1)
    resampling: Literal["nearest", "bilinear", "cubic"] = "bilinear"
    estimator: Literal["affine", "homography"] = "affine"
    maximum_features: int = Field(default=1_000, ge=50)
    normalization_low_percentile: float = Field(default=2.0, ge=0, le=100)
    normalization_high_percentile: float = Field(default=98.0, ge=0, le=100)
    feature_ratio_test: float = Field(default=0.78, gt=0, lt=1)
    ransac_reprojection_threshold: float = Field(default=3.0, gt=0)
    minimum_inliers: int = Field(default=8, ge=3)
    minimum_inlier_ratio: float = Field(default=0.2, gt=0, le=1)
    maximum_reprojection_error: float = Field(default=4.0, gt=0)
    maximum_rotation_degrees: float = Field(default=15.0, gt=0, le=90)
    maximum_scale_deviation: float = Field(default=0.3, gt=0)
    maximum_translation_fraction: float = Field(default=0.35, gt=0)
    maximum_perspective_term: float = Field(default=0.001, gt=0)
    minimum_registration_score: float = Field(default=0.5, ge=0, le=1)
    enable_ecc: bool = False
    ecc_iterations: int = Field(default=50, ge=1)
    ecc_epsilon: float = Field(default=1e-5, gt=0)
    cv_random_seed: int = 17

    @model_validator(mode="after")
    def validate_percentiles(self) -> "ImageryPipelineConfig":
        if self.normalization_low_percentile >= self.normalization_high_percentile:
            raise ValueError("normalization percentiles must be ordered")
        return self
