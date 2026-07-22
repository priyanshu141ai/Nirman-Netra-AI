"""Adapter that composes the established imagery, model, and change modules."""

from pathlib import Path
from typing import Protocol, cast

import numpy as np
import rasterio
from affine import Affine
from numpy.typing import NDArray
from shapely.geometry import mapping, shape
from shapely.ops import unary_union

from nirman_netra.application.contracts import AssetRecord, ImagePairRecord, PipelineEvidence
from nirman_netra.application.storage import LocalObjectStorage
from nirman_netra.change_detection.artifact import save_change_artifact
from nirman_netra.change_detection.baselines import detect_mask_difference
from nirman_netra.change_detection.confidence import compose_change_confidence
from nirman_netra.change_detection.contracts import ConfidenceConfig, MaskDifferenceConfig
from nirman_netra.change_detection.polygons import extract_change_polygons
from nirman_netra.domain import CoordinateReference, GeometryReference
from nirman_netra.exceptions import ModelSchemaMismatchError, RegistrationQualityError
from nirman_netra.geospatial import pixel_area_square_metres
from nirman_netra.imagery.config import ImageryPipelineConfig
from nirman_netra.imagery.contracts import QualityStatus
from nirman_netra.imagery.pipeline import RegistrationPipeline
from nirman_netra.imagery.raster import LocalRasterIngestor
from nirman_netra.segmentation.artifact import SegmentationInference
from nirman_netra.segmentation.contracts import PolygonExtractionConfig
from nirman_netra.segmentation.polygons import extract_building_polygons


class PipelineBackend(Protocol):
    def process(
        self,
        pair: ImagePairRecord,
        old_asset: AssetRecord,
        new_asset: AssetRecord,
        model: SegmentationInference,
        output_directory: Path,
        output_uri: str,
    ) -> PipelineEvidence: ...


def _uint8_hwc(pixels: NDArray[np.generic]) -> NDArray[np.uint8]:
    values = np.nan_to_num(pixels, nan=0.0, posinf=255.0, neginf=0.0)
    if values.dtype != np.uint8:
        values = np.clip(values, 0, 255).astype(np.uint8)
    return cast(NDArray[np.uint8], np.moveaxis(values, 0, 2))


def _footprint(
    mask: NDArray[np.uint8], transform: Affine, crs: CoordinateReference, name: str
) -> GeometryReference:
    polygons = extract_building_polygons(
        mask,
        transform,
        crs,
        PolygonExtractionConfig(minimum_pixel_area=2),
    )
    if not polygons:
        raise RegistrationQualityError("segmentation produced no reviewable building footprint")
    merged = unary_union([shape(item.geometry) for item in polygons])
    return GeometryReference(geometry_id=name, geometry=dict(mapping(merged)), crs=crs)


class DefaultPipelineBackend:
    """Runs inference only from the explicitly validated model artifact."""

    def __init__(
        self,
        storage: LocalObjectStorage,
        imagery_config: ImageryPipelineConfig | None = None,
    ) -> None:
        self._storage = storage
        self._config = imagery_config or ImageryPipelineConfig()
        self._registration = RegistrationPipeline(self._config)
        self._ingestor = LocalRasterIngestor()

    def process(
        self,
        pair: ImagePairRecord,
        old_asset: AssetRecord,
        new_asset: AssetRecord,
        model: SegmentationInference,
        output_directory: Path,
        output_uri: str,
    ) -> PipelineEvidence:
        old_path = self._storage.original_path(old_asset.object_key)
        new_path = self._storage.original_path(new_asset.object_key)
        registration = self._registration.run(
            old_path,
            new_path,
            output_directory,
            before_asset_id=old_asset.asset_id,
            after_asset_id=new_asset.asset_id,
        )
        if registration.status == QualityStatus.REQUIRES_MANUAL_ALIGNMENT:
            raise RegistrationQualityError("registration requires controlled manual alignment")
        old_raster = self._ingestor.read(old_path, asset_id=old_asset.asset_id)
        registered_path = output_directory / f"{registration.pair_id}.tif"
        with rasterio.open(registered_path) as registered:
            new_pixels = cast(NDArray[np.generic], registered.read())
            transform = Affine(*registered.transform[:6])
            crs = CoordinateReference(value=str(registered.crs))
        old_image, new_image = _uint8_hwc(old_raster.pixels), _uint8_hwc(new_pixels)
        schema = model.metadata.input_schema
        expected = (schema.height, schema.width, schema.channels)
        if old_image.shape != expected or new_image.shape != expected:
            raise ModelSchemaMismatchError("registered images do not match model input shape")
        old_mask = model.predict(old_image, schema.channel_order)
        new_mask = model.predict(new_image, schema.channel_order)
        difference = detect_mask_difference(
            old_mask,
            new_mask,
            registration.metrics,
            pixel_area_square_metres(transform, crs),
            MaskDifferenceConfig(),
        )
        polygons = extract_change_polygons(
            difference.label_mask,
            transform,
            crs,
            pair.pair_id,
            PolygonExtractionConfig(minimum_pixel_area=2),
        )
        changed_pixels = int(np.count_nonzero(difference.label_mask))
        segmentation_confidence = float(
            np.mean(
                (
                    model.predict_probabilities(old_image, schema.channel_order),
                    model.predict_probabilities(new_image, schema.channel_order),
                )
            )
        )
        warnings = tuple(sorted(set(registration.warnings)))
        confidence = compose_change_confidence(
            model_score=segmentation_confidence,
            registration_quality=registration.metrics.registration_quality_score,
            image_quality_warnings=warnings,
            changed_region_pixels=changed_pixels,
            segmentation_confidence=segmentation_confidence,
            config=ConfidenceConfig(),
        )
        artifact = save_change_artifact(
            output_directory / "change",
            pair_id=pair.pair_id,
            model_version=model.metadata.model_version,
            change_mask=difference.label_mask,
            change_polygons=polygons,
            confidence=confidence,
            registration_score=registration.metrics.registration_quality_score,
            warnings=warnings,
        )
        return PipelineEvidence(
            old_observed_footprint=_footprint(old_mask, transform, crs, f"old-{pair.pair_id}"),
            new_observed_footprint=_footprint(new_mask, transform, crs, f"new-{pair.pair_id}"),
            change_polygons=polygons,
            changed_area_square_metres=sum(item.changed_area_square_metres for item in polygons),
            change_confidence=confidence,
            segmentation_confidence=segmentation_confidence,
            source_crs=old_asset.metadata.crs.value,
            output_crs=crs.value,
            image_quality_status=max(
                (old_asset.quality.status, new_asset.quality.status),
                key=lambda value: list(QualityStatus).index(value),
            ),
            registration_quality_status=registration.status,
            registration_score=registration.metrics.registration_quality_score,
            warnings=warnings,
            change_mask_storage_uri=f"{output_uri}/change/change-mask.npy",
            checksum=artifact.checksum,
        )
