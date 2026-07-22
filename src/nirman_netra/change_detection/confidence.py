"""Explicit composition of change-inference confidence inputs."""

import math

from nirman_netra.change_detection.contracts import ConfidenceConfig
from nirman_netra.exceptions import InferenceValidationError


def compose_change_confidence(
    *,
    model_score: float,
    registration_quality: float,
    image_quality_warnings: tuple[str, ...],
    changed_region_pixels: int,
    segmentation_confidence: float | None,
    config: ConfidenceConfig,
) -> float:
    values = (model_score, registration_quality)
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise InferenceValidationError("model and registration scores must be within zero and one")
    if segmentation_confidence is not None and (
        not math.isfinite(segmentation_confidence) or not 0 <= segmentation_confidence <= 1
    ):
        raise InferenceValidationError("segmentation confidence must be within zero and one")
    if changed_region_pixels < 0:
        raise InferenceValidationError("changed-region size cannot be negative")
    if changed_region_pixels == 0:
        size_factor = 1.0
    elif changed_region_pixels < config.minimum_region_pixels:
        size_factor = 0.35
    else:
        span = config.full_size_confidence_pixels - config.minimum_region_pixels
        size_factor = min(
            1.0,
            0.6 + 0.4 * (changed_region_pixels - config.minimum_region_pixels) / max(1, span),
        )
    warning_factor = max(0.0, 1 - len(image_quality_warnings) * config.warning_penalty)
    registration_factor = 0.2 + 0.8 * registration_quality
    segmentation_factor = (
        1.0 if segmentation_confidence is None else 0.5 + 0.5 * segmentation_confidence
    )
    return float(
        max(
            0.0,
            min(
                1.0,
                model_score
                * registration_factor
                * warning_factor
                * size_factor
                * segmentation_factor,
            ),
        )
    )
