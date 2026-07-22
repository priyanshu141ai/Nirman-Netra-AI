"""Registered-image and building-mask change baselines."""

from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from nirman_netra.change_detection.contracts import (
    ChangedAreaStatistics,
    ChangeLabel,
    ImageDifferenceConfig,
    MaskDifferenceConfig,
)
from nirman_netra.exceptions import DatasetValidationError, RegistrationError
from nirman_netra.imagery.contracts import RegistrationMetrics


@dataclass(frozen=True)
class ImageDifferenceResult:
    difference_score: NDArray[np.float32]
    change_mask: NDArray[np.uint8]
    label_mask: NDArray[np.uint8]
    model_score: float
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class MaskDifferenceResult:
    added_mask: NDArray[np.uint8]
    removed_mask: NDArray[np.uint8]
    common_footprint: NDArray[np.uint8]
    label_mask: NDArray[np.uint8]
    statistics: ChangedAreaStatistics


def validate_registration_dependency(metrics: RegistrationMetrics, minimum_score: float) -> None:
    if not metrics.transform_plausible or metrics.overlap_ratio <= 0:
        raise RegistrationError("change detection requires a valid registered overlap")
    if metrics.registration_quality_score < minimum_score:
        raise RegistrationError("registration quality is too low for change inference")


def _remove_small_regions(mask: NDArray[np.uint8], minimum_pixels: int) -> NDArray[np.uint8]:
    count, labels, statistics, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    output = np.zeros_like(mask, dtype=np.uint8)
    for label in range(1, count):
        if int(statistics[label, cv2.CC_STAT_AREA]) >= minimum_pixels:
            output[labels == label] = 1
    return output


def _grayscale(image: NDArray[np.uint8]) -> NDArray[np.float32]:
    return cast(NDArray[np.float32], image.astype(np.float32).mean(axis=2) / 255.0)


def _difference_score(
    old_image: NDArray[np.uint8], new_image: NDArray[np.uint8], mode: str
) -> NDArray[np.float32]:
    old = _grayscale(old_image)
    new = _grayscale(new_image)
    if mode == "pixel":
        return cast(NDArray[np.float32], np.abs(new - old).astype(np.float32))
    old_mean = cv2.GaussianBlur(old, (5, 5), 1.0)
    new_mean = cv2.GaussianBlur(new, (5, 5), 1.0)
    old_variance = cv2.GaussianBlur(old * old, (5, 5), 1.0) - old_mean**2
    new_variance = cv2.GaussianBlur(new * new, (5, 5), 1.0) - new_mean**2
    covariance = cv2.GaussianBlur(old * new, (5, 5), 1.0) - old_mean * new_mean
    c1, c2 = 0.01**2, 0.03**2
    similarity = ((2 * old_mean * new_mean + c1) * (2 * covariance + c2)) / (
        (old_mean**2 + new_mean**2 + c1) * (old_variance + new_variance + c2)
    )
    return cast(NDArray[np.float32], np.clip((1 - similarity) / 2, 0, 1).astype(np.float32))


def detect_image_difference(
    old_image: NDArray[np.uint8],
    new_image: NDArray[np.uint8],
    registration: RegistrationMetrics,
    config: ImageDifferenceConfig,
) -> ImageDifferenceResult:
    """Detect uncertain changed regions without assigning a legal interpretation."""

    validate_registration_dependency(registration, config.minimum_registration_score)
    if (
        old_image.dtype != np.uint8
        or new_image.dtype != np.uint8
        or old_image.ndim != 3
        or old_image.shape != new_image.shape
    ):
        raise DatasetValidationError("registered images must be matching HWC uint8 arrays")
    score = _difference_score(old_image, new_image, config.mode)
    threshold = min(
        1.0,
        config.threshold
        + config.registration_threshold_penalty * (1 - registration.registration_quality_score),
    )
    mask = (score >= threshold).astype(np.uint8)
    if config.morphology_kernel_size > 1:
        kernel = np.ones(
            (config.morphology_kernel_size, config.morphology_kernel_size), dtype=np.uint8
        )
        mask = cast(
            NDArray[np.uint8], cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel).astype(np.uint8)
        )
        mask = cast(
            NDArray[np.uint8], cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel).astype(np.uint8)
        )
    mask = _remove_small_regions(mask, config.minimum_region_pixels)
    labels = np.where(mask, int(ChangeLabel.UNCERTAIN_CHANGE), 0).astype(np.uint8)
    changed_scores = score[mask > 0]
    model_score = float(changed_scores.mean()) if changed_scores.size else float(1 - score.mean())
    warnings = (
        ("REGISTRATION_QUALITY_REDUCED",) if registration.registration_quality_score < 0.55 else ()
    )
    return ImageDifferenceResult(
        difference_score=score,
        change_mask=mask,
        label_mask=labels,
        model_score=model_score,
        warnings=warnings,
    )


def _classify_components(
    mask: NDArray[np.uint8],
    common: NDArray[np.uint8],
    standalone_label: ChangeLabel,
    connected_label: ChangeLabel,
    adjacency_pixels: int,
) -> NDArray[np.uint8]:
    count, components = cv2.connectedComponents(mask, connectivity=8)
    output = np.zeros_like(mask, dtype=np.uint8)
    if adjacency_pixels:
        size = 2 * adjacency_pixels + 1
        near_common = cv2.dilate(common, np.ones((size, size), dtype=np.uint8))
    else:
        near_common = common
    for component_id in range(1, count):
        component = components == component_id
        label = connected_label if np.any(near_common[component]) else standalone_label
        output[component] = int(label)
    return output


def detect_mask_difference(
    old_mask: NDArray[np.uint8],
    new_mask: NDArray[np.uint8],
    registration: RegistrationMetrics,
    pixel_area_square_metres: float,
    config: MaskDifferenceConfig,
) -> MaskDifferenceResult:
    validate_registration_dependency(registration, config.minimum_registration_score)
    if (
        old_mask.ndim != 2
        or old_mask.shape != new_mask.shape
        or not set(int(value) for value in np.unique(old_mask)) <= {0, 1}
        or not set(int(value) for value in np.unique(new_mask)) <= {0, 1}
    ):
        raise DatasetValidationError("old and new building masks must be aligned and binary")
    old = (old_mask > 0).astype(np.uint8)
    new = (new_mask > 0).astype(np.uint8)
    added = _remove_small_regions(new & (1 - old), config.minimum_region_pixels)
    removed = _remove_small_regions(old & (1 - new), config.minimum_region_pixels)
    common = old & new
    added_labels = _classify_components(
        added,
        common,
        ChangeLabel.ADDED_BUILDING,
        ChangeLabel.FOOTPRINT_EXPANSION,
        config.adjacency_pixels,
    )
    removed_labels = _classify_components(
        removed,
        common,
        ChangeLabel.REMOVED_BUILDING,
        ChangeLabel.PARTIAL_DEMOLITION,
        config.adjacency_pixels,
    )
    label_mask = np.where(added > 0, added_labels, removed_labels).astype(np.uint8)
    added_pixels = int(np.count_nonzero(added))
    removed_pixels = int(np.count_nonzero(removed))
    return MaskDifferenceResult(
        added_mask=added,
        removed_mask=removed,
        common_footprint=common,
        label_mask=label_mask,
        statistics=ChangedAreaStatistics(
            added_pixels=added_pixels,
            removed_pixels=removed_pixels,
            common_pixels=int(np.count_nonzero(common)),
            added_area_square_metres=added_pixels * pixel_area_square_metres,
            removed_area_square_metres=removed_pixels * pixel_area_square_metres,
        ),
    )
