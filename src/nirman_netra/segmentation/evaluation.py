"""Pixel, boundary, object, and small-building segmentation metrics."""

from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from nirman_netra.exceptions import DatasetValidationError
from nirman_netra.segmentation.contracts import (
    EvaluationConfig,
    GeographicSplitConfig,
    SegmentationMetrics,
    SplitAssignment,
)
from nirman_netra.segmentation.dataset import SegmentationSample, validate_dataset
from nirman_netra.segmentation.model import LinearSegmentationModel


def _safe_ratio(numerator: int, denominator: int, empty_value: float = 1.0) -> float:
    return empty_value if denominator == 0 else numerator / denominator


def _boundary_counts(
    prediction: NDArray[np.uint8], target: NDArray[np.uint8], tolerance: int
) -> tuple[int, int, int, int]:
    kernel = np.ones((3, 3), dtype=np.uint8)
    predicted_boundary = prediction - np.asarray(cv2.erode(prediction, kernel), dtype=np.uint8)
    target_boundary = target - np.asarray(cv2.erode(target, kernel), dtype=np.uint8)
    if tolerance:
        tolerance_kernel = np.ones((2 * tolerance + 1, 2 * tolerance + 1), dtype=np.uint8)
        target_near = np.asarray(cv2.dilate(target_boundary, tolerance_kernel), dtype=np.uint8)
        predicted_near = np.asarray(
            cv2.dilate(predicted_boundary, tolerance_kernel), dtype=np.uint8
        )
    else:
        target_near, predicted_near = target_boundary, predicted_boundary
    precision_hits = int(np.count_nonzero(predicted_boundary & target_near))
    recall_hits = int(np.count_nonzero(target_boundary & predicted_near))
    return (
        precision_hits,
        int(np.count_nonzero(predicted_boundary)),
        recall_hits,
        int(np.count_nonzero(target_boundary)),
    )


def _object_counts(
    prediction: NDArray[np.uint8], target: NDArray[np.uint8], config: EvaluationConfig
) -> tuple[int, int, int, int]:
    object_count, labels, stats, _ = cv2.connectedComponentsWithStats(target, connectivity=8)
    detected = total = small_detected = small_total = 0
    for label in range(1, object_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        component = labels == label
        overlap = int(np.count_nonzero(prediction[component])) / area
        found = overlap >= config.object_overlap_threshold
        total += 1
        detected += int(found)
        if area <= config.small_building_max_pixels:
            small_total += 1
            small_detected += int(found)
    return detected, total, small_detected, small_total


def evaluate_binary_masks(
    predictions: Sequence[NDArray[np.uint8]],
    targets: Sequence[NDArray[np.uint8]],
    config: EvaluationConfig,
) -> SegmentationMetrics:
    """Evaluate binary masks without optimizing only pixel accuracy."""

    if len(predictions) != len(targets) or not predictions:
        raise DatasetValidationError("predictions and targets must be non-empty and aligned")
    true_positive = false_positive = false_negative = 0
    boundary_precision_hits = boundary_predicted = 0
    boundary_recall_hits = boundary_target = 0
    detected = objects = small_detected = small_objects = 0
    for prediction, target in zip(predictions, targets, strict=True):
        if prediction.shape != target.shape:
            raise DatasetValidationError("prediction and target shapes differ")
        predicted = (prediction > 0).astype(np.uint8)
        expected = (target > 0).astype(np.uint8)
        true_positive += int(np.count_nonzero(predicted & expected))
        false_positive += int(np.count_nonzero(predicted & (1 - expected)))
        false_negative += int(np.count_nonzero((1 - predicted) & expected))
        values = _boundary_counts(predicted, expected, config.boundary_tolerance_pixels)
        boundary_precision_hits += values[0]
        boundary_predicted += values[1]
        boundary_recall_hits += values[2]
        boundary_target += values[3]
        values = _object_counts(predicted, expected, config)
        detected += values[0]
        objects += values[1]
        small_detected += values[2]
        small_objects += values[3]
    precision = _safe_ratio(true_positive, true_positive + false_positive)
    recall = _safe_ratio(true_positive, true_positive + false_negative)
    iou = _safe_ratio(true_positive, true_positive + false_positive + false_negative)
    dice = _safe_ratio(2 * true_positive, 2 * true_positive + false_positive + false_negative)
    boundary_precision = _safe_ratio(boundary_precision_hits, boundary_predicted)
    boundary_recall = _safe_ratio(boundary_recall_hits, boundary_target)
    boundary_f1 = (
        1.0
        if boundary_precision + boundary_recall == 0
        else 2 * boundary_precision * boundary_recall / (boundary_precision + boundary_recall)
    )
    return SegmentationMetrics(
        iou=iou,
        dice_f1=dice,
        precision=precision,
        recall=recall,
        boundary_f1=boundary_f1,
        object_recall=_safe_ratio(detected, objects),
        small_building_recall=_safe_ratio(small_detected, small_objects),
    )


def evaluate_model(
    model: LinearSegmentationModel,
    samples: tuple[SegmentationSample, ...],
    split: SplitAssignment,
    split_config: GeographicSplitConfig,
    config: EvaluationConfig,
) -> SegmentationMetrics:
    """Evaluate only a declared validation or test geographic split."""

    if split not in {SplitAssignment.VALIDATION, SplitAssignment.TEST}:
        raise DatasetValidationError("evaluation is limited to validation or test splits")
    validate_dataset(samples, split_config)
    selected = tuple(sample for sample in samples if sample.metadata.split == split)
    if not selected:
        raise DatasetValidationError(f"evaluation split is empty: {split.value}")
    return evaluate_binary_masks(
        [model.predict(sample.image, config.threshold) for sample in selected],
        [sample.mask for sample in selected],
        config,
    )
