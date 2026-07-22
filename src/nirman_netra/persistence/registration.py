"""Persist registered rasters and deterministic metric artifacts."""

import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.errors import RasterioIOError

from nirman_netra.exceptions import StorageError
from nirman_netra.imagery.alignment import AlignedRasterPair
from nirman_netra.imagery.contracts import QualityStatus, RegistrationArtifact
from nirman_netra.imagery.registration import RegistrationComputation
from nirman_netra.utils import file_content_hash


def persist_registration(
    computation: RegistrationComputation,
    pair: AlignedRasterPair,
    pair_id: str,
    status: QualityStatus,
    warnings: tuple[str, ...],
    output_directory: Path,
    source_paths: tuple[Path, Path],
) -> RegistrationArtifact:
    """Write a new GeoTIFF and JSON artifact without touching source rasters."""

    try:
        output_directory.mkdir(parents=True, exist_ok=True)
        output_path = (output_directory / f"{pair_id}.tif").resolve()
        resolved_sources = {path.resolve() for path in source_paths}
        if output_path in resolved_sources:
            raise StorageError("registered output must not overwrite a source raster")
        pixels = computation.registered_pixels.copy()
        pixels[:, ~computation.registered_valid_mask] = np.nan
        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            width=pixels.shape[2],
            height=pixels.shape[1],
            count=pixels.shape[0],
            dtype="float32",
            crs=pair.crs.value,
            transform=pair.transform,
            nodata=np.nan,
            compress="deflate",
        ) as destination:
            destination.write(pixels)
            destination.update_tags(
                pair_id=pair_id,
                before_asset_id=pair.before_metadata.asset_id,
                after_asset_id=pair.after_metadata.asset_id,
            )
        checksum = file_content_hash(output_path)
        artifact = RegistrationArtifact(
            pair_id=pair_id,
            before_asset_id=pair.before_metadata.asset_id,
            after_asset_id=pair.after_metadata.asset_id,
            transform_matrix=tuple(
                tuple(float(value) for value in row) for row in computation.transform_matrix
            ),
            metrics=computation.metrics,
            status=status,
            registered_output_uri=output_path.as_uri(),
            checksum=checksum,
            warnings=warnings,
            before_metadata=pair.before_metadata,
            after_metadata=pair.after_metadata,
            common_grid=pair.common_grid,
            crs_transformations=pair.crs_transformations,
            visual_refinement_applied=computation.visual_refinement_applied,
            visual_refinement_method=computation.visual_refinement_method,
            reliable_for_change_detection=status
            in {QualityStatus.PASS, QualityStatus.PASS_WITH_WARNING},
        )
        artifact_path = output_directory / f"{pair_id}.registration.json"
        artifact_path.write_text(
            json.dumps(
                artifact.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        return artifact
    except (OSError, RasterioIOError) as exc:
        raise StorageError(f"failed to persist registration in: {output_directory}") from exc
