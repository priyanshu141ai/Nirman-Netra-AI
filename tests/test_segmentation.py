import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from affine import Affine
from shapely.geometry import Polygon, shape

from nirman_netra.domain import BoundingBox, CoordinateReference
from nirman_netra.exceptions import DatasetValidationError
from nirman_netra.segmentation.artifact import (
    load_model_artifact,
    save_model_artifact,
)
from nirman_netra.segmentation.augmentations import augment_sample
from nirman_netra.segmentation.contracts import (
    AugmentationConfig,
    EvaluationConfig,
    GeographicSplitConfig,
    InputSchema,
    LabelQualityStatus,
    PolygonExtractionConfig,
    SegmentationMetrics,
    SegmentationTileMetadata,
    SplitAssignment,
    TrainingConfig,
)
from nirman_netra.segmentation.dataset import (
    SegmentationSample,
    assign_geographic_splits,
    image_tile_hash,
    validate_dataset,
    validate_geographic_splits,
)
from nirman_netra.segmentation.evaluation import evaluate_binary_masks, evaluate_model
from nirman_netra.segmentation.model import TrainingResult, train_baseline
from nirman_netra.segmentation.polygons import extract_building_polygons, repair_polygon

CRS = CoordinateReference(value="EPSG:32643")
SIZE = 16
SPLIT_CONFIG = GeographicSplitConfig(random_seed=11, proximity_distance=0)


def _sample(
    tile_id: str,
    split: SplitAssignment,
    x_offset: float,
    seed: int,
    *,
    empty: bool = False,
) -> SegmentationSample:
    rng = np.random.default_rng(seed)
    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    if not empty:
        mask[4:12, 5:11] = 1
    noise = rng.integers(0, 15, size=(SIZE, SIZE), dtype=np.uint8)
    image = np.where(mask > 0, 220 + noise, 20 + noise).astype(np.uint8)[..., np.newaxis]
    metadata = SegmentationTileMetadata(
        tile_id=tile_id,
        source_asset_id=f"asset-{tile_id}",
        geographic_bounds=BoundingBox(
            min_x=x_offset,
            min_y=0,
            max_x=x_offset + SIZE,
            max_y=SIZE,
            crs=CRS,
        ),
        crs=CRS,
        region_group=f"region-{tile_id}",
        split=split,
        label_quality_status=LabelQualityStatus.VERIFIED,
        tile_sha256=image_tile_hash(image),
    )
    return SegmentationSample(image=image, mask=mask, metadata=metadata)


def _training_samples() -> tuple[SegmentationSample, ...]:
    return (
        _sample("train-a", SplitAssignment.TRAIN, 0, 1),
        _sample("train-b", SplitAssignment.TRAIN, 100, 2),
        _sample("validation", SplitAssignment.VALIDATION, 1_000, 3),
        _sample("test", SplitAssignment.TEST, 2_000, 4),
    )


@pytest.fixture(scope="module")
def trained_artifact(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, TrainingResult, tuple[SegmentationSample, ...], SegmentationMetrics]:
    output = tmp_path_factory.mktemp("segmentation-artifact")
    samples = _training_samples()
    result = train_baseline(
        samples,
        SPLIT_CONFIG,
        TrainingConfig(
            random_seed=5,
            batch_size=2,
            learning_rate=0.4,
            epochs=40,
            early_stopping_patience=6,
            checkpoint_path=output / "checkpoint.npy",
            augmentation=AugmentationConfig(
                horizontal_flip_probability=0,
                vertical_flip_probability=0,
                maximum_rotation_degrees=0,
                brightness_delta=0,
                contrast_delta=0,
                scale_minimum=1,
                scale_maximum=1,
            ),
        ),
    )
    metrics = evaluate_model(
        result.model,
        samples,
        SplitAssignment.VALIDATION,
        SPLIT_CONFIG,
        EvaluationConfig(),
    )
    save_model_artifact(
        result.model,
        output,
        model_version="1.0.0",
        training_dataset_version="synthetic-1",
        input_schema=InputSchema(
            height=SIZE,
            width=SIZE,
            channels=1,
            channel_order=("gray",),
        ),
        evaluation_metrics=metrics,
    )
    return output, result, samples, metrics


def test_geographic_split_prevents_nearby_leakage() -> None:
    left = _sample("left", SplitAssignment.UNASSIGNED, 0, 10)
    right = _sample("right", SplitAssignment.UNASSIGNED, SIZE - 2, 11)

    assigned = assign_geographic_splits((left, right), SPLIT_CONFIG)

    assert assigned[0].metadata.split == assigned[1].metadata.split
    leaked = (
        assigned[0],
        SegmentationSample(
            image=assigned[1].image,
            mask=assigned[1].mask,
            metadata=assigned[1].metadata.model_copy(
                update={
                    "split": SplitAssignment.TEST
                    if assigned[0].metadata.split != SplitAssignment.TEST
                    else SplitAssignment.TRAIN
                }
            ),
        ),
    )
    with pytest.raises(DatasetValidationError, match="geographic split leakage"):
        validate_geographic_splits(leaked, SPLIT_CONFIG)


def test_image_mask_mismatch_is_rejected() -> None:
    sample = _sample("mismatch", SplitAssignment.TRAIN, 0, 12)
    mismatched = SegmentationSample(
        image=sample.image,
        mask=sample.mask[:-1],
        metadata=sample.metadata,
    )

    with pytest.raises(DatasetValidationError, match="dimensions differ"):
        validate_dataset((mismatched,), SPLIT_CONFIG)


def test_empty_mask_is_tracked() -> None:
    sample = _sample("empty", SplitAssignment.TRAIN, 0, 13, empty=True)

    report = validate_dataset((sample,), SPLIT_CONFIG)

    assert report.empty_mask_count == 1
    assert report.label_coverage_ratio == 0


def test_duplicate_tile_hash_cannot_cross_splits() -> None:
    sample = _sample("original", SplitAssignment.TRAIN, 0, 14)
    duplicate = SegmentationSample(
        image=sample.image.copy(),
        mask=sample.mask.copy(),
        metadata=sample.metadata.model_copy(
            update={
                "tile_id": "duplicate",
                "region_group": "different-region",
                "split": SplitAssignment.TEST,
                "geographic_bounds": BoundingBox(
                    min_x=1_000,
                    min_y=0,
                    max_x=1_000 + SIZE,
                    max_y=SIZE,
                    crs=CRS,
                ),
            }
        ),
    )

    with pytest.raises(DatasetValidationError, match="duplicate tile hash"):
        validate_dataset((sample, duplicate), SPLIT_CONFIG)


def test_augmentation_tracks_pixel_mapping() -> None:
    sample = _sample("augment", SplitAssignment.TRAIN, 0, 15)
    augmented = augment_sample(
        sample,
        AugmentationConfig(
            horizontal_flip_probability=1,
            vertical_flip_probability=0,
            maximum_rotation_degrees=0,
            brightness_delta=0,
            contrast_delta=0,
            scale_minimum=1,
            scale_maximum=1,
        ),
        np.random.default_rng(1),
    )

    assert np.array_equal(augmented.mask, np.fliplr(sample.mask))
    assert np.allclose(
        augmented.augmented_pixel_to_source_pixel,
        np.array([[-1, 0, SIZE - 1], [0, 1, 0], [0, 0, 1]]),
    )


def test_metric_correctness() -> None:
    prediction = np.array([[1, 0], [1, 0]], dtype=np.uint8)
    target = np.array([[1, 1], [0, 0]], dtype=np.uint8)

    metrics = evaluate_binary_masks([prediction], [target], EvaluationConfig())

    assert metrics.iou == pytest.approx(1 / 3)
    assert metrics.dice_f1 == pytest.approx(0.5)
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.object_recall == 1


def test_polygon_extraction_preserves_map_coordinates() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:6, 3:8] = 1

    polygons = extract_building_polygons(
        mask,
        Affine(2, 0, 100, 0, -2, 200),
        CRS,
        PolygonExtractionConfig(minimum_pixel_area=2),
    )

    assert len(polygons) == 1
    assert polygons[0].crs == CRS
    assert shape(polygons[0].geometry).bounds == pytest.approx((107, 189, 115, 195))


def test_invalid_polygon_repair_is_recorded() -> None:
    invalid = Polygon([(0, 0), (2, 2), (0, 2), (2, 0), (0, 0)])

    repaired, was_repaired, method = repair_polygon(invalid)

    assert repaired.is_valid
    assert was_repaired
    assert method == "make_valid"


def test_baseline_trains_on_small_fixture(
    trained_artifact: tuple[
        Path, TrainingResult, tuple[SegmentationSample, ...], SegmentationMetrics
    ],
) -> None:
    output, result, _, metrics = trained_artifact

    assert result.history.epochs_completed > 0
    assert (output / "checkpoint.npy").is_file()
    assert metrics.iou > 0.9


def test_artifact_save_load_and_onnx_compatibility(
    trained_artifact: tuple[
        Path, TrainingResult, tuple[SegmentationSample, ...], SegmentationMetrics
    ],
) -> None:
    output, result, samples, _ = trained_artifact
    inference = load_model_artifact(output)
    image = samples[-1].image

    onnx_probability = inference.predict_probabilities(image, ("gray",))
    numpy_probability = result.model.predict_probabilities(image)

    assert inference.metadata.model_version == "1.0.0"
    assert np.allclose(onnx_probability, numpy_probability, atol=1e-6)


def test_deterministic_inference_and_clean_process_load(
    trained_artifact: tuple[
        Path, TrainingResult, tuple[SegmentationSample, ...], SegmentationMetrics
    ],
) -> None:
    output, _, samples, _ = trained_artifact
    image = samples[-1].image
    first = load_model_artifact(output).predict(image, ("gray",))
    second = load_model_artifact(output).predict(image, ("gray",))

    assert np.array_equal(first, second)
    script = (
        "from pathlib import Path; import numpy as np; "
        "from nirman_netra.segmentation.artifact import load_model_artifact; "
        f"model=load_model_artifact(Path({str(output)!r})); "
        f"image=np.zeros(({SIZE},{SIZE},1),dtype=np.uint8); "
        "assert model.predict(image,('gray',)).shape==image.shape[:2]"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )

    assert completed.returncode == 0, completed.stderr
