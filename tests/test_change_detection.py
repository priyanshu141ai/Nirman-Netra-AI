from pathlib import Path

import numpy as np
import pytest
from affine import Affine

from nirman_netra.change_detection.artifact import (
    load_change_artifact,
    save_change_artifact,
)
from nirman_netra.change_detection.baselines import (
    detect_image_difference,
    detect_mask_difference,
)
from nirman_netra.change_detection.confidence import compose_change_confidence
from nirman_netra.change_detection.contracts import (
    ChangeEvaluationConfig,
    ChangeLabel,
    ChangePairMetadata,
    ConfidenceConfig,
    ImageDifferenceConfig,
    MaskDifferenceConfig,
    RegistrationTier,
    SiameseTrainingConfig,
)
from nirman_netra.change_detection.dataset import (
    ChangeSample,
    change_pair_hash,
    validate_change_dataset,
    validate_geographic_splits,
)
from nirman_netra.change_detection.evaluation import (
    compare_change_methods,
    evaluate_registration_aware,
)
from nirman_netra.change_detection.model import train_siamese_candidate
from nirman_netra.change_detection.polygons import extract_change_polygons
from nirman_netra.domain import BoundingBox, CoordinateReference
from nirman_netra.exceptions import DatasetValidationError, RegistrationError
from nirman_netra.imagery.contracts import RegistrationMetrics
from nirman_netra.segmentation.contracts import (
    GeographicSplitConfig,
    LabelQualityStatus,
    PolygonExtractionConfig,
    SplitAssignment,
)

CRS = CoordinateReference(value="EPSG:32643")
SIZE = 24
SPLIT_CONFIG = GeographicSplitConfig(random_seed=19, proximity_distance=0)


def _registration(score: float) -> RegistrationMetrics:
    return RegistrationMetrics(
        inlier_count=20,
        inlier_ratio=max(0.1, score),
        reprojection_error=max(0.0, 1 - score),
        overlap_ratio=0.95,
        transform_plausible=True,
        registration_quality_score=score,
    )


def _sample(
    pair_id: str,
    split: SplitAssignment,
    x_offset: float,
    scenario: str,
    *,
    score: float = 0.9,
    tint: int = 0,
    off_nadir: bool = False,
) -> ChangeSample:
    old_mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    new_mask = np.zeros_like(old_mask)
    old_mask[3:11, 3:11] = 1
    new_mask[3:11, 3:11] = 1
    if scenario == "addition":
        new_mask[15:21, 15:21] = 1
    elif scenario == "removal":
        old_mask[15:21, 15:21] = 1
    elif scenario != "no_change":
        raise ValueError(f"unsupported test scenario: {scenario}")
    old_image = np.where(old_mask > 0, 220 + tint, 20 + tint).astype(np.uint8)[..., None]
    new_image = np.where(new_mask > 0, 220 + tint, 20 + tint).astype(np.uint8)[..., None]
    change_mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    change_mask[(new_mask > 0) & (old_mask == 0)] = int(ChangeLabel.ADDED_BUILDING)
    change_mask[(old_mask > 0) & (new_mask == 0)] = int(ChangeLabel.REMOVED_BUILDING)
    return ChangeSample(
        old_image=old_image,
        new_image=new_image,
        change_mask=change_mask,
        old_building_mask=old_mask,
        new_building_mask=new_mask,
        metadata=ChangePairMetadata(
            pair_id=pair_id,
            old_asset_id=f"old-{pair_id}",
            new_asset_id=f"new-{pair_id}",
            geographic_bounds=BoundingBox(
                min_x=x_offset,
                min_y=0,
                max_x=x_offset + SIZE,
                max_y=SIZE,
                crs=CRS,
            ),
            crs=CRS,
            region_group=f"region-{pair_id}",
            split=split,
            label_quality_status=LabelQualityStatus.VERIFIED,
            registration_metrics=_registration(score),
            off_nadir=off_nadir,
            pixel_area_square_metres=4,
            pair_sha256=change_pair_hash(old_image, new_image),
        ),
    )


def _dataset() -> tuple[ChangeSample, ...]:
    return (
        _sample("train-add", SplitAssignment.TRAIN, 0, "addition", tint=0),
        _sample("train-remove", SplitAssignment.TRAIN, 100, "removal", tint=1),
        _sample("validation", SplitAssignment.VALIDATION, 1_000, "no_change", tint=2),
        _sample("test", SplitAssignment.TEST, 2_000, "addition", tint=3),
    )


def _image_config(minimum_pixels: int = 4) -> ImageDifferenceConfig:
    return ImageDifferenceConfig(
        threshold=0.2,
        registration_threshold_penalty=0,
        morphology_kernel_size=1,
        minimum_region_pixels=minimum_pixels,
    )


def test_no_change_pair() -> None:
    sample = _sample("same", SplitAssignment.TEST, 0, "no_change")

    image_result = detect_image_difference(
        sample.old_image,
        sample.new_image,
        sample.metadata.registration_metrics,
        _image_config(),
    )
    mask_result = detect_mask_difference(
        sample.old_building_mask,
        sample.new_building_mask,
        sample.metadata.registration_metrics,
        4,
        MaskDifferenceConfig(),
    )

    assert not np.any(image_result.change_mask)
    assert not np.any(mask_result.label_mask)


def test_known_synthetic_addition() -> None:
    sample = _sample("addition", SplitAssignment.TEST, 0, "addition")

    image_result = detect_image_difference(
        sample.old_image,
        sample.new_image,
        sample.metadata.registration_metrics,
        _image_config(),
    )
    mask_result = detect_mask_difference(
        sample.old_building_mask,
        sample.new_building_mask,
        sample.metadata.registration_metrics,
        4,
        MaskDifferenceConfig(),
    )

    assert np.count_nonzero(image_result.change_mask) == 36
    assert mask_result.statistics.added_pixels == 36
    assert set(np.unique(mask_result.label_mask)) == {0, int(ChangeLabel.ADDED_BUILDING)}


def test_known_synthetic_removal() -> None:
    sample = _sample("removal", SplitAssignment.TEST, 0, "removal")

    image_result = detect_image_difference(
        sample.old_image,
        sample.new_image,
        sample.metadata.registration_metrics,
        _image_config(),
    )
    mask_result = detect_mask_difference(
        sample.old_building_mask,
        sample.new_building_mask,
        sample.metadata.registration_metrics,
        4,
        MaskDifferenceConfig(),
    )

    assert np.count_nonzero(image_result.change_mask) == 36
    assert mask_result.statistics.removed_area_square_metres == 144
    assert set(np.unique(mask_result.label_mask)) == {0, int(ChangeLabel.REMOVED_BUILDING)}


def test_registration_degradation_blocks_change_inference() -> None:
    sample = _sample("poor", SplitAssignment.TEST, 0, "addition", score=0.1)

    with pytest.raises(RegistrationError, match="quality is too low"):
        detect_image_difference(
            sample.old_image,
            sample.new_image,
            sample.metadata.registration_metrics,
            _image_config(),
        )


def test_tiny_artifact_is_removed() -> None:
    old = np.zeros((SIZE, SIZE, 1), dtype=np.uint8)
    new = old.copy()
    new[5, 5] = 255

    result = detect_image_difference(old, new, _registration(0.9), _image_config(2))

    assert not np.any(result.change_mask)


def test_polygon_area_calculation() -> None:
    labels = np.zeros((10, 10), dtype=np.uint8)
    labels[2:6, 3:8] = int(ChangeLabel.ADDED_BUILDING)

    polygons = extract_change_polygons(
        labels,
        Affine(2, 0, 100, 0, -2, 200),
        CRS,
        "pair-area",
        PolygonExtractionConfig(minimum_pixel_area=2),
    )

    assert len(polygons) == 1
    assert polygons[0].label == "added_building"
    assert polygons[0].changed_area_square_metres == pytest.approx(48)


def test_confidence_penalizes_poor_registration() -> None:
    high = compose_change_confidence(
        model_score=0.9,
        registration_quality=0.95,
        image_quality_warnings=(),
        changed_region_pixels=100,
        segmentation_confidence=0.9,
        config=ConfidenceConfig(),
    )
    poor = compose_change_confidence(
        model_score=0.9,
        registration_quality=0.3,
        image_quality_warnings=(),
        changed_region_pixels=100,
        segmentation_confidence=0.9,
        config=ConfidenceConfig(),
    )

    assert poor < high


def test_artifact_save_and_load(tmp_path: Path) -> None:
    labels = _sample("artifact", SplitAssignment.TEST, 0, "addition").change_mask
    polygons = extract_change_polygons(
        labels,
        Affine(2, 0, 100, 0, -2, 200),
        CRS,
        "artifact",
        PolygonExtractionConfig(minimum_pixel_area=2),
    )

    saved = save_change_artifact(
        tmp_path,
        pair_id="artifact",
        model_version="siamese-linear-1",
        change_mask=labels,
        change_polygons=polygons,
        confidence=0.8,
        registration_score=0.9,
    )
    loaded = load_change_artifact(tmp_path)

    assert loaded.metadata == saved
    assert np.array_equal(loaded.change_mask, labels)


def test_geographic_split_leakage_is_rejected() -> None:
    left = _sample("left", SplitAssignment.TRAIN, 0, "addition", tint=0)
    right = _sample("right", SplitAssignment.TEST, SIZE - 2, "removal", tint=1)

    with pytest.raises(DatasetValidationError, match="geographic split leakage"):
        validate_geographic_splits((left, right), SPLIT_CONFIG)
    with pytest.raises(DatasetValidationError, match="geographic split leakage"):
        validate_change_dataset((left, right), SPLIT_CONFIG)


def test_registration_aware_evaluation_reports_available_tiers() -> None:
    samples = (
        _sample("high", SplitAssignment.TEST, 0, "no_change", score=0.9, tint=0),
        _sample("medium", SplitAssignment.TEST, 100, "no_change", score=0.7, tint=1),
        _sample(
            "difficult",
            SplitAssignment.TEST,
            200,
            "no_change",
            score=0.4,
            tint=2,
            off_nadir=True,
        ),
    )
    predictions = [sample.change_mask.copy() for sample in samples]

    report = evaluate_registration_aware(predictions, samples, ChangeEvaluationConfig())

    assert set(report.by_registration_tier) == set(RegistrationTier)
    assert report.overall.change_iou == 1


def test_deterministic_candidate_and_method_comparison() -> None:
    samples = _dataset()
    config = SiameseTrainingConfig(random_seed=7, epochs=50, learning_rate=0.4)

    first = train_siamese_candidate(samples, SPLIT_CONFIG, config)
    second = train_siamese_candidate(samples, SPLIT_CONFIG, config)
    first_prediction = first.predict(samples[-1])
    second_prediction = second.predict(samples[-1])
    comparison = compare_change_methods(
        samples,
        first,
        SplitAssignment.TEST,
        SPLIT_CONFIG,
        _image_config(),
        MaskDifferenceConfig(),
        ChangeEvaluationConfig(),
    )

    assert np.array_equal(first.weights, second.weights)
    assert np.array_equal(first_prediction, second_prediction)
    assert comparison.baseline_image_difference.overall.change_iou == 1
    assert comparison.baseline_mask_difference.overall.change_iou == 1
    assert 0 <= comparison.siamese_candidate.overall.change_f1 <= 1
