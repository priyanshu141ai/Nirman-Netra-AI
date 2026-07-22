"""Deterministic single-raster and pair-quality checks."""

from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray
from pyproj import Transformer
from rasterio.warp import calculate_default_transform
from shapely.geometry import box
from shapely.ops import transform

from nirman_netra.imagery.config import ImageryPipelineConfig
from nirman_netra.imagery.contracts import (
    PairQualityReport,
    QualityIssue,
    QualityMetrics,
    QualityReport,
    QualityStatus,
)
from nirman_netra.imagery.raster import IngestedRaster


def grayscale_uint8(raster: IngestedRaster) -> NDArray[np.uint8]:
    """Convert known raster channel order to a stable 8-bit grayscale view."""

    pixels = raster.pixels
    if np.issubdtype(pixels.dtype, np.integer):
        maximum = float(np.iinfo(pixels.dtype).max)  # type: ignore[type-var]
        normalized = np.clip(pixels.astype(np.float32) * (255.0 / maximum), 0, 255)
    else:
        normalized = np.nan_to_num(pixels.astype(np.float32), nan=0.0)
        if float(np.max(normalized, initial=0)) <= 1.0:
            normalized *= 255.0
        normalized = np.clip(normalized, 0, 255)
    if raster.metadata.band_count == 1:
        return cast(NDArray[np.uint8], normalized[0].astype(np.uint8))
    rgb = np.moveaxis(normalized[:3], 0, 2).astype(np.uint8)
    return cast(NDArray[np.uint8], cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))


def _status(issues: list[QualityIssue]) -> QualityStatus:
    severities = {issue.severity for issue in issues}
    if "rejected" in severities:
        return QualityStatus.REJECTED
    if "manual" in severities:
        return QualityStatus.REQUIRES_MANUAL_ALIGNMENT
    if "warning" in severities:
        return QualityStatus.PASS_WITH_WARNING
    return QualityStatus.PASS


def assess_raster(raster: IngestedRaster, config: ImageryPipelineConfig) -> QualityReport:
    """Assess readable pixels without altering the ingested raster."""

    issues: list[QualityIssue] = []
    metadata = raster.metadata
    if min(metadata.width, metadata.height) < config.minimum_dimension:
        issues.append(
            QualityIssue(
                code="MINIMUM_DIMENSIONS",
                severity="rejected",
                message="raster dimensions are below the configured minimum",
            )
        )
    if metadata.band_count not in config.allowed_band_counts:
        issues.append(
            QualityIssue(
                code="INCOMPATIBLE_BANDS",
                severity="rejected",
                message="raster band count is unsupported",
            )
        )
    if metadata.band_count >= 3 and metadata.channel_order[:3] != ("red", "green", "blue"):
        issues.append(
            QualityIssue(
                code="INCOMPATIBLE_CHANNEL_ORDER",
                severity="rejected",
                message="first three bands must be ordered red, green, blue",
            )
        )

    gray = grayscale_uint8(raster)
    valid = raster.valid_mask
    valid_ratio = float(np.count_nonzero(valid) / valid.size)
    if valid_ratio < config.minimum_valid_pixel_ratio:
        issues.append(
            QualityIssue(
                code="LOW_VALID_PIXEL_RATIO",
                severity="rejected",
                message="valid-pixel ratio is below the configured minimum",
            )
        )
    valid_values = gray[valid]
    if valid_values.size:
        blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F)[valid].var())
        dark_ratio = float(np.mean(valid_values <= config.dark_pixel_value))
        bright_ratio = float(np.mean(valid_values >= config.bright_pixel_value))
    else:
        blur_variance, dark_ratio, bright_ratio = 0.0, 1.0, 0.0
    if blur_variance < config.minimum_blur_variance:
        issues.append(
            QualityIssue(code="EXCESSIVE_BLUR", severity="rejected", message="image is too blurred")
        )
    if dark_ratio >= config.extreme_exposure_ratio:
        issues.append(
            QualityIssue(
                code="EXTREME_DARKNESS",
                severity="warning",
                message="most valid pixels are very dark",
            )
        )
    if bright_ratio >= config.extreme_exposure_ratio:
        issues.append(
            QualityIssue(
                code="EXTREME_OVEREXPOSURE",
                severity="warning",
                message="most valid pixels are overexposed",
            )
        )
    return QualityReport(
        asset_id=metadata.asset_id,
        status=_status(issues),
        metrics=QualityMetrics(
            blur_variance=blur_variance,
            dark_pixel_ratio=dark_ratio,
            bright_pixel_ratio=bright_ratio,
            valid_pixel_ratio=valid_ratio,
        ),
        issues=tuple(issues),
    )


def _bounds_in_before_crs(
    before: IngestedRaster, after: IngestedRaster
) -> tuple[float, float, float, float]:
    bounds = after.metadata.bounds
    transformer = Transformer.from_crs(
        after.metadata.crs.value, before.metadata.crs.value, always_xy=True
    )
    projected = transform(
        transformer.transform,
        box(bounds.min_x, bounds.min_y, bounds.max_x, bounds.max_y),
    )
    return (
        float(projected.bounds[0]),
        float(projected.bounds[1]),
        float(projected.bounds[2]),
        float(projected.bounds[3]),
    )


def assess_pair(
    before: IngestedRaster, after: IngestedRaster, config: ImageryPipelineConfig
) -> PairQualityReport:
    """Assess geographic overlap, resolution, and band compatibility."""

    issues: list[QualityIssue] = []
    before_bounds = before.metadata.bounds
    before_shape = box(
        before_bounds.min_x, before_bounds.min_y, before_bounds.max_x, before_bounds.max_y
    )
    after_left, after_bottom, after_right, after_top = _bounds_in_before_crs(before, after)
    after_shape = box(after_left, after_bottom, after_right, after_top)
    intersection = before_shape.intersection(after_shape)
    overlap = (
        0.0
        if intersection.is_empty
        else float(intersection.area / min(before_shape.area, after_shape.area))
    )
    if overlap < config.minimum_geographic_overlap:
        issues.append(
            QualityIssue(
                code="INSUFFICIENT_GEOGRAPHIC_OVERLAP",
                severity="rejected",
                message="raster footprints have insufficient overlap",
            )
        )
    after_resolution = after.metadata.resolution
    if before.metadata.crs != after.metadata.crs:
        after_bounds = after.metadata.bounds
        normalized_transform, _, _ = calculate_default_transform(
            after.metadata.crs.value,
            before.metadata.crs.value,
            after.metadata.width,
            after.metadata.height,
            after_bounds.min_x,
            after_bounds.min_y,
            after_bounds.max_x,
            after_bounds.max_y,
        )
        after_resolution = (abs(normalized_transform.a), abs(normalized_transform.e))
    resolution_values = (*before.metadata.resolution, *after_resolution)
    resolution_ratio = max(resolution_values) / min(resolution_values)
    if resolution_ratio > config.maximum_resolution_ratio:
        issues.append(
            QualityIssue(
                code="LARGE_RESOLUTION_MISMATCH",
                severity="manual",
                message="pixel resolutions differ beyond the configured ratio",
            )
        )
    if before.metadata.band_count != after.metadata.band_count:
        issues.append(
            QualityIssue(
                code="INCOMPATIBLE_BANDS",
                severity="rejected",
                message="before and after band counts differ",
            )
        )
    return PairQualityReport(
        status=_status(issues),
        geographic_overlap_ratio=overlap,
        resolution_ratio=resolution_ratio,
        issues=tuple(issues),
    )
