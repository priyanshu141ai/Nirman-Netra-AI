"""Segmentation dataset interface, validation, and geographic split policy."""

from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from pyproj import Transformer
from shapely.geometry import box
from shapely.ops import transform

from nirman_netra.exceptions import DatasetValidationError
from nirman_netra.segmentation.contracts import (
    DatasetValidationReport,
    GeographicSplitConfig,
    LabelQualityStatus,
    SegmentationTileMetadata,
    SplitAssignment,
)


@dataclass(frozen=True)
class SegmentationSample:
    image: NDArray[np.uint8]
    mask: NDArray[np.uint8]
    metadata: SegmentationTileMetadata


class SegmentationDataset(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> SegmentationSample: ...


class InMemorySegmentationDataset:
    def __init__(self, samples: Sequence[SegmentationSample]) -> None:
        self._samples = tuple(samples)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> SegmentationSample:
        return self._samples[index]

    def __iter__(self) -> Iterator[SegmentationSample]:
        return iter(self._samples)


def image_tile_hash(image: NDArray[np.uint8]) -> str:
    """Hash image pixels, shape, and dtype for cross-split duplicate detection."""

    digest = sha256()
    digest.update(str(image.shape).encode())
    digest.update(image.dtype.str.encode())
    digest.update(image.tobytes(order="C"))
    return digest.hexdigest()


def _nearby(left: SegmentationSample, right: SegmentationSample, distance: float) -> bool:
    left_bounds = left.metadata.geographic_bounds
    right_bounds = right.metadata.geographic_bounds
    left_geometry = box(left_bounds.min_x, left_bounds.min_y, left_bounds.max_x, left_bounds.max_y)
    right_geometry = box(
        right_bounds.min_x,
        right_bounds.min_y,
        right_bounds.max_x,
        right_bounds.max_y,
    )
    if left.metadata.crs != right.metadata.crs:
        transformer = Transformer.from_crs(
            right.metadata.crs.value, left.metadata.crs.value, always_xy=True
        )
        right_geometry = transform(transformer.transform, right_geometry)
    if left_geometry.intersects(right_geometry):
        return True
    if distance and not left.metadata.crs.is_projected:
        raise DatasetValidationError("proximity distance requires a projected CRS")
    return left_geometry.distance(right_geometry) <= distance


def assign_geographic_splits(
    samples: Sequence[SegmentationSample], config: GeographicSplitConfig
) -> tuple[SegmentationSample, ...]:
    """Assign whole region/spatial components to a deterministic split."""

    parents = list(range(len(samples)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left_index, left in enumerate(samples):
        for right_index in range(left_index + 1, len(samples)):
            right = samples[right_index]
            if left.metadata.region_group == right.metadata.region_group or _nearby(
                left, right, config.proximity_distance
            ):
                union(left_index, right_index)

    component_splits: dict[int, SplitAssignment] = {}
    for index in range(len(samples)):
        root = find(index)
        if root in component_splits:
            continue
        key = min(
            samples[item].metadata.tile_id for item in range(len(samples)) if find(item) == root
        )
        value = int.from_bytes(sha256(f"{config.random_seed}:{key}".encode()).digest()[:8]) / 2**64
        if value < config.train_ratio:
            split = SplitAssignment.TRAIN
        elif value < config.train_ratio + config.validation_ratio:
            split = SplitAssignment.VALIDATION
        else:
            split = SplitAssignment.TEST
        component_splits[root] = split
    return tuple(
        SegmentationSample(
            image=sample.image,
            mask=sample.mask,
            metadata=sample.metadata.model_copy(update={"split": component_splits[find(index)]}),
        )
        for index, sample in enumerate(samples)
    )


def validate_geographic_splits(
    samples: Sequence[SegmentationSample], config: GeographicSplitConfig
) -> None:
    for left_index, left in enumerate(samples):
        if left.metadata.split == SplitAssignment.UNASSIGNED:
            raise DatasetValidationError(f"tile has no split assignment: {left.metadata.tile_id}")
        for right in samples[left_index + 1 :]:
            if left.metadata.split == right.metadata.split:
                continue
            if left.metadata.region_group == right.metadata.region_group or _nearby(
                left, right, config.proximity_distance
            ):
                raise DatasetValidationError(
                    f"geographic split leakage: {left.metadata.tile_id}, {right.metadata.tile_id}"
                )


def validate_dataset(
    samples: Sequence[SegmentationSample],
    split_config: GeographicSplitConfig,
    valid_classes: tuple[int, ...] = (0, 1),
) -> DatasetValidationReport:
    """Validate shape, masks, metadata, hashes, splits, and coverage statistics."""

    if not samples:
        raise DatasetValidationError("segmentation dataset is empty")
    seen_hashes: dict[str, SplitAssignment] = {}
    split_counts: Counter[str] = Counter()
    class_counts: Counter[int] = Counter()
    empty_masks = 0
    total_pixels = 0
    foreground_pixels = 0
    for sample in samples:
        if sample.image.ndim != 3:
            raise DatasetValidationError(f"image must be HWC: {sample.metadata.tile_id}")
        if sample.image.dtype != np.uint8 or not np.issubdtype(sample.mask.dtype, np.integer):
            raise DatasetValidationError(
                f"image or mask dtype is unsupported: {sample.metadata.tile_id}"
            )
        if image_tile_hash(sample.image) != sample.metadata.tile_sha256:
            raise DatasetValidationError(
                f"tile hash does not match pixels: {sample.metadata.tile_id}"
            )
        if sample.metadata.label_quality_status == LabelQualityStatus.REJECTED:
            raise DatasetValidationError(f"tile label is rejected: {sample.metadata.tile_id}")
        if sample.mask.ndim != 2 or sample.image.shape[:2] != sample.mask.shape:
            raise DatasetValidationError(f"image-mask dimensions differ: {sample.metadata.tile_id}")
        classes, counts = np.unique(sample.mask, return_counts=True)
        if any(int(value) not in valid_classes for value in classes):
            raise DatasetValidationError(
                f"mask contains unsupported classes: {sample.metadata.tile_id}"
            )
        for value, count in zip(classes, counts, strict=True):
            class_counts[int(value)] += int(count)
        if not np.any(sample.mask):
            empty_masks += 1
        total_pixels += sample.mask.size
        foreground_pixels += int(np.count_nonzero(sample.mask))
        previous_split = seen_hashes.get(sample.metadata.tile_sha256)
        if previous_split is not None and previous_split != sample.metadata.split:
            raise DatasetValidationError(
                f"duplicate tile hash crosses splits: {sample.metadata.tile_sha256}"
            )
        seen_hashes[sample.metadata.tile_sha256] = sample.metadata.split
        split_counts[sample.metadata.split.value] += 1
    validate_geographic_splits(samples, split_config)
    return DatasetValidationReport(
        sample_count=len(samples),
        split_counts=dict(sorted(split_counts.items())),
        empty_mask_count=empty_masks,
        label_coverage_ratio=foreground_pixels / total_pixels,
        class_pixel_counts=dict(sorted(class_counts.items())),
    )
