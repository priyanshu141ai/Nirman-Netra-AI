"""Bitemporal change dataset validation and geographic split policy."""

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

from nirman_netra.change_detection.contracts import (
    ChangeDatasetReport,
    ChangeLabel,
    ChangePairMetadata,
)
from nirman_netra.exceptions import DatasetValidationError
from nirman_netra.segmentation.contracts import (
    GeographicSplitConfig,
    LabelQualityStatus,
    SplitAssignment,
)


@dataclass(frozen=True)
class ChangeSample:
    old_image: NDArray[np.uint8]
    new_image: NDArray[np.uint8]
    change_mask: NDArray[np.uint8]
    old_building_mask: NDArray[np.uint8]
    new_building_mask: NDArray[np.uint8]
    metadata: ChangePairMetadata


class ChangeDataset(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> ChangeSample: ...


class InMemoryChangeDataset:
    def __init__(self, samples: Sequence[ChangeSample]) -> None:
        self._samples = tuple(samples)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> ChangeSample:
        return self._samples[index]

    def __iter__(self) -> Iterator[ChangeSample]:
        return iter(self._samples)


def change_pair_hash(old_image: NDArray[np.uint8], new_image: NDArray[np.uint8]) -> str:
    digest = sha256()
    for image in (old_image, new_image):
        digest.update(str(image.shape).encode())
        digest.update(image.dtype.str.encode())
        digest.update(image.tobytes(order="C"))
    return digest.hexdigest()


def _nearby(left: ChangeSample, right: ChangeSample, distance: float) -> bool:
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
        raise DatasetValidationError("change-pair proximity requires a projected CRS")
    return left_geometry.distance(right_geometry) <= distance


def assign_geographic_splits(
    samples: Sequence[ChangeSample], config: GeographicSplitConfig
) -> tuple[ChangeSample, ...]:
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
            samples[item].metadata.pair_id for item in range(len(samples)) if find(item) == root
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
        ChangeSample(
            old_image=sample.old_image,
            new_image=sample.new_image,
            change_mask=sample.change_mask,
            old_building_mask=sample.old_building_mask,
            new_building_mask=sample.new_building_mask,
            metadata=sample.metadata.model_copy(update={"split": component_splits[find(index)]}),
        )
        for index, sample in enumerate(samples)
    )


def validate_geographic_splits(
    samples: Sequence[ChangeSample], config: GeographicSplitConfig
) -> None:
    for left_index, left in enumerate(samples):
        if left.metadata.split == SplitAssignment.UNASSIGNED:
            raise DatasetValidationError(f"change pair has no split: {left.metadata.pair_id}")
        for right in samples[left_index + 1 :]:
            if left.metadata.split == right.metadata.split:
                continue
            if left.metadata.region_group == right.metadata.region_group or _nearby(
                left, right, config.proximity_distance
            ):
                raise DatasetValidationError(
                    f"geographic split leakage: {left.metadata.pair_id}, {right.metadata.pair_id}"
                )


def validate_change_dataset(
    samples: Sequence[ChangeSample], config: GeographicSplitConfig
) -> ChangeDatasetReport:
    if not samples:
        raise DatasetValidationError("change dataset is empty")
    valid_labels = {int(label) for label in ChangeLabel}
    seen_hashes: dict[str, SplitAssignment] = {}
    split_counts: Counter[str] = Counter()
    class_counts: Counter[int] = Counter()
    empty_change_count = 0
    for sample in samples:
        pair_id = sample.metadata.pair_id
        if (
            sample.old_image.dtype != np.uint8
            or sample.new_image.dtype != np.uint8
            or sample.old_image.ndim != 3
            or sample.old_image.shape != sample.new_image.shape
        ):
            raise DatasetValidationError(f"registered image pair is incompatible: {pair_id}")
        image_shape = sample.old_image.shape[:2]
        masks = (sample.change_mask, sample.old_building_mask, sample.new_building_mask)
        if any(mask.ndim != 2 or mask.shape != image_shape for mask in masks):
            raise DatasetValidationError(f"image and mask dimensions differ: {pair_id}")
        if any(not np.issubdtype(mask.dtype, np.integer) for mask in masks):
            raise DatasetValidationError(f"change masks must use integer classes: {pair_id}")
        if not set(int(value) for value in np.unique(sample.change_mask)) <= valid_labels:
            raise DatasetValidationError(f"change mask contains unsupported labels: {pair_id}")
        if any(not set(int(value) for value in np.unique(mask)) <= {0, 1} for mask in masks[1:]):
            raise DatasetValidationError(f"building masks must be binary: {pair_id}")
        if change_pair_hash(sample.old_image, sample.new_image) != sample.metadata.pair_sha256:
            raise DatasetValidationError(f"registered pair hash mismatch: {pair_id}")
        if sample.metadata.label_quality_status == LabelQualityStatus.REJECTED:
            raise DatasetValidationError(f"change label is rejected: {pair_id}")
        registration = sample.metadata.registration_metrics
        if not registration.transform_plausible or registration.overlap_ratio <= 0:
            raise DatasetValidationError(f"registration is invalid: {pair_id}")
        previous_split = seen_hashes.get(sample.metadata.pair_sha256)
        if previous_split is not None and previous_split != sample.metadata.split:
            raise DatasetValidationError(
                f"duplicate registered pair crosses splits: {sample.metadata.pair_sha256}"
            )
        seen_hashes[sample.metadata.pair_sha256] = sample.metadata.split
        split_counts[sample.metadata.split.value] += 1
        values, counts = np.unique(sample.change_mask, return_counts=True)
        for value, count in zip(values, counts, strict=True):
            class_counts[int(value)] += int(count)
        empty_change_count += int(not np.any(sample.change_mask))
    validate_geographic_splits(samples, config)
    return ChangeDatasetReport(
        sample_count=len(samples),
        split_counts=dict(sorted(split_counts.items())),
        class_pixel_counts=dict(sorted(class_counts.items())),
        empty_change_count=empty_change_count,
    )
