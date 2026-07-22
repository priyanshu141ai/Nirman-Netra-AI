"""Registration-aware change metrics and method comparison."""

from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from nirman_netra.change_detection.baselines import (
    detect_image_difference,
    detect_mask_difference,
)
from nirman_netra.change_detection.contracts import (
    ChangeEvaluationConfig,
    ChangeLabel,
    ChangeMetrics,
    ImageDifferenceConfig,
    MaskDifferenceConfig,
    MethodComparison,
    RegistrationAwareEvaluation,
    RegistrationTier,
)
from nirman_netra.change_detection.dataset import ChangeSample, validate_change_dataset
from nirman_netra.change_detection.model import SiameseLinearChangeModel
from nirman_netra.exceptions import DatasetValidationError
from nirman_netra.segmentation.contracts import GeographicSplitConfig, SplitAssignment


def _safe_ratio(numerator: float, denominator: float, empty_value: float = 1.0) -> float:
    return empty_value if denominator == 0 else numerator / denominator


def _registration_tier(sample: ChangeSample, config: ChangeEvaluationConfig) -> RegistrationTier:
    score = sample.metadata.registration_metrics.registration_quality_score
    if sample.metadata.off_nadir or score < config.medium_registration_score:
        return RegistrationTier.DIFFICULT
    if score < config.high_registration_score:
        return RegistrationTier.MEDIUM
    return RegistrationTier.HIGH


def _component_counts(
    prediction: NDArray[np.uint8],
    target: NDArray[np.uint8],
    overlap_threshold: float,
) -> tuple[int, int, int]:
    target_count, target_labels, target_stats, _ = cv2.connectedComponentsWithStats(
        target, connectivity=8
    )
    detected = total = 0
    for label in range(1, target_count):
        component = target_labels == label
        area = int(target_stats[label, cv2.CC_STAT_AREA])
        detected += int(np.count_nonzero(prediction[component]) / area >= overlap_threshold)
        total += 1
    predicted_count, predicted_labels = cv2.connectedComponents(prediction, connectivity=8)
    false_alarms = sum(
        int(not np.any(target[predicted_labels == label])) for label in range(1, predicted_count)
    )
    return detected, total, false_alarms


def evaluate_change_masks(
    predictions: Sequence[NDArray[np.uint8]],
    samples: Sequence[ChangeSample],
    config: ChangeEvaluationConfig,
) -> ChangeMetrics:
    if not predictions or len(predictions) != len(samples):
        raise DatasetValidationError("change predictions and samples must be non-empty and aligned")
    true_positive = false_positive = false_negative = 0
    detected = objects = false_alarms = 0
    false_positive_area = total_area = 0.0
    added_error = removed_error = 0.0
    added_labels = {int(ChangeLabel.ADDED_BUILDING), int(ChangeLabel.FOOTPRINT_EXPANSION)}
    removed_labels = {int(ChangeLabel.REMOVED_BUILDING), int(ChangeLabel.PARTIAL_DEMOLITION)}
    valid_labels = {int(label) for label in ChangeLabel}
    for prediction, sample in zip(predictions, samples, strict=True):
        if prediction.shape != sample.change_mask.shape:
            raise DatasetValidationError("predicted and target change-mask shapes differ")
        if not set(int(value) for value in np.unique(prediction)) <= valid_labels:
            raise DatasetValidationError("prediction contains unsupported change labels")
        predicted_change = (prediction > 0).astype(np.uint8)
        target_change = (sample.change_mask > 0).astype(np.uint8)
        true_positive += int(np.count_nonzero(predicted_change & target_change))
        sample_false_positive = int(np.count_nonzero(predicted_change & (1 - target_change)))
        false_positive += sample_false_positive
        false_negative += int(np.count_nonzero((1 - predicted_change) & target_change))
        values = _component_counts(predicted_change, target_change, config.object_overlap_threshold)
        detected += values[0]
        objects += values[1]
        false_alarms += values[2]
        pixel_area = sample.metadata.pixel_area_square_metres
        false_positive_area += sample_false_positive * pixel_area
        total_area += sample.change_mask.size * pixel_area
        predicted_added = int(np.count_nonzero(np.isin(prediction, tuple(added_labels))))
        target_added = int(np.count_nonzero(np.isin(sample.change_mask, tuple(added_labels))))
        predicted_removed = int(np.count_nonzero(np.isin(prediction, tuple(removed_labels))))
        target_removed = int(np.count_nonzero(np.isin(sample.change_mask, tuple(removed_labels))))
        added_error += abs(predicted_added - target_added) * pixel_area
        removed_error += abs(predicted_removed - target_removed) * pixel_area
    precision = _safe_ratio(true_positive, true_positive + false_positive)
    recall = _safe_ratio(true_positive, true_positive + false_negative)
    return ChangeMetrics(
        change_iou=_safe_ratio(true_positive, true_positive + false_positive + false_negative),
        change_f1=_safe_ratio(
            2 * true_positive, 2 * true_positive + false_positive + false_negative
        ),
        precision=precision,
        recall=recall,
        false_positive_changed_area_square_metres=false_positive_area,
        object_level_change_recall=_safe_ratio(detected, objects),
        false_alarms_per_square_kilometre=_safe_ratio(
            false_alarms, total_area / 1_000_000, empty_value=0.0
        ),
        added_area_error_square_metres=added_error,
        removed_area_error_square_metres=removed_error,
    )


def evaluate_registration_aware(
    predictions: Sequence[NDArray[np.uint8]],
    samples: Sequence[ChangeSample],
    config: ChangeEvaluationConfig,
) -> RegistrationAwareEvaluation:
    overall = evaluate_change_masks(predictions, samples, config)
    tiers: dict[RegistrationTier, ChangeMetrics] = {}
    for tier in RegistrationTier:
        indices = [
            index
            for index, sample in enumerate(samples)
            if _registration_tier(sample, config) == tier
        ]
        if indices:
            tiers[tier] = evaluate_change_masks(
                [predictions[index] for index in indices],
                [samples[index] for index in indices],
                config,
            )
    return RegistrationAwareEvaluation(overall=overall, by_registration_tier=tiers)


def compare_change_methods(
    samples: tuple[ChangeSample, ...],
    candidate: SiameseLinearChangeModel,
    split: SplitAssignment,
    split_config: GeographicSplitConfig,
    image_config: ImageDifferenceConfig,
    mask_config: MaskDifferenceConfig,
    evaluation_config: ChangeEvaluationConfig,
) -> MethodComparison:
    """Evaluate both baselines and the sole candidate on identical registered pairs."""

    if split not in {SplitAssignment.VALIDATION, SplitAssignment.TEST}:
        raise DatasetValidationError("change evaluation is limited to validation or test splits")
    validate_change_dataset(samples, split_config)
    selected = tuple(sample for sample in samples if sample.metadata.split == split)
    if not selected:
        raise DatasetValidationError(f"change evaluation split is empty: {split.value}")
    image_predictions: list[NDArray[np.uint8]] = []
    mask_predictions: list[NDArray[np.uint8]] = []
    candidate_predictions: list[NDArray[np.uint8]] = []
    for sample in selected:
        image_predictions.append(
            detect_image_difference(
                sample.old_image,
                sample.new_image,
                sample.metadata.registration_metrics,
                image_config,
            ).label_mask
        )
        mask_predictions.append(
            detect_mask_difference(
                sample.old_building_mask,
                sample.new_building_mask,
                sample.metadata.registration_metrics,
                sample.metadata.pixel_area_square_metres,
                mask_config,
            ).label_mask
        )
        candidate_predictions.append(candidate.predict_labels(sample))
    return MethodComparison(
        baseline_image_difference=evaluate_registration_aware(
            image_predictions, selected, evaluation_config
        ),
        baseline_mask_difference=evaluate_registration_aware(
            mask_predictions, selected, evaluation_config
        ),
        siamese_candidate=evaluate_registration_aware(
            candidate_predictions, selected, evaluation_config
        ),
    )
