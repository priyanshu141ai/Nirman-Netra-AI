"""Deterministic change-result artifact persistence and strict loading."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray
from pydantic import ValidationError
from shapely.geometry import shape

from nirman_netra.change_detection.contracts import (
    ChangeDetectionArtifact,
    ChangeLabel,
    ChangePolygon,
)
from nirman_netra.exceptions import ArtifactValidationError, StorageError
from nirman_netra.utils import file_content_hash

MASK_FILENAME = "change-mask.npy"
METADATA_FILENAME = "change-artifact.json"


@dataclass(frozen=True)
class LoadedChangeArtifact:
    metadata: ChangeDetectionArtifact
    change_mask: NDArray[np.uint8]


def save_change_artifact(
    output_directory: Path,
    *,
    pair_id: str,
    model_version: str,
    change_mask: NDArray[np.uint8],
    change_polygons: tuple[ChangePolygon, ...],
    confidence: float,
    registration_score: float,
    warnings: tuple[str, ...] = (),
) -> ChangeDetectionArtifact:
    valid_labels = {int(label) for label in ChangeLabel}
    if (
        change_mask.dtype != np.uint8
        or change_mask.ndim != 2
        or not set(int(value) for value in np.unique(change_mask)) <= valid_labels
    ):
        raise ArtifactValidationError("change mask dtype, shape, or labels are invalid")
    if any(polygon.source_pair_id != pair_id for polygon in change_polygons):
        raise ArtifactValidationError("change polygon source pair does not match artifact")
    try:
        output_directory.mkdir(parents=True, exist_ok=True)
        mask_path = output_directory / MASK_FILENAME
        with mask_path.open("wb") as destination:
            np.save(destination, change_mask, allow_pickle=False)
        metadata = ChangeDetectionArtifact(
            pair_id=pair_id,
            model_version=model_version,
            change_mask_uri=MASK_FILENAME,
            mask_shape=cast(tuple[int, int], change_mask.shape),
            change_polygons=change_polygons,
            confidence=confidence,
            registration_score=registration_score,
            warnings=warnings,
            checksum=file_content_hash(mask_path),
        )
        serialized = (
            json.dumps(metadata.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
            + "\n"
        )
        (output_directory / METADATA_FILENAME).write_text(serialized, encoding="utf-8")
        return ChangeDetectionArtifact.model_validate_json(serialized)
    except OSError as exc:
        raise StorageError(f"failed to save change artifact: {output_directory}") from exc


def load_change_artifact(output_directory: Path) -> LoadedChangeArtifact:
    metadata_path = output_directory / METADATA_FILENAME
    try:
        metadata = ChangeDetectionArtifact.model_validate_json(
            metadata_path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError) as exc:
        raise ArtifactValidationError("change artifact metadata is missing or invalid") from exc
    if metadata.change_mask_uri != MASK_FILENAME:
        raise ArtifactValidationError("change artifact mask URI is unsupported")
    mask_path = output_directory / metadata.change_mask_uri
    try:
        if file_content_hash(mask_path) != metadata.checksum:
            raise ArtifactValidationError("change mask checksum mismatch")
        with mask_path.open("rb") as source:
            mask = np.load(source, allow_pickle=False)
    except ArtifactValidationError:
        raise
    except (OSError, ValueError) as exc:
        raise ArtifactValidationError("change mask is missing or unreadable") from exc
    valid_labels = {int(label) for label in ChangeLabel}
    if (
        mask.dtype != np.uint8
        or mask.shape != metadata.mask_shape
        or not set(int(value) for value in np.unique(mask)) <= valid_labels
    ):
        raise ArtifactValidationError("loaded change mask violates the artifact schema")
    for polygon in metadata.change_polygons:
        geometry = shape(polygon.geometry)
        if not geometry.is_valid or geometry.is_empty:
            raise ArtifactValidationError("artifact contains an invalid change polygon")
    return LoadedChangeArtifact(
        metadata=metadata,
        change_mask=cast(NDArray[np.uint8], mask),
    )
