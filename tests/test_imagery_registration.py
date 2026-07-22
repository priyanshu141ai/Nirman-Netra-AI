import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import pytest
import rasterio
from numpy.typing import NDArray
from rasterio.transform import Affine, from_origin

from nirman_netra.exceptions import (
    CRSMismatchError,
    InsufficientOverlapError,
    RegistrationQualityError,
    TransformValidationError,
)
from nirman_netra.imagery.alignment import align_pair
from nirman_netra.imagery.config import ImageryPipelineConfig
from nirman_netra.imagery.contracts import (
    ManualAlignmentRequest,
    ManualControlPoint,
    QualityStatus,
    RasterMetadata,
)
from nirman_netra.imagery.crs import select_target_crs
from nirman_netra.imagery.manual import estimate_manual_transform
from nirman_netra.imagery.pipeline import RegistrationPipeline
from nirman_netra.imagery.quality import assess_raster
from nirman_netra.imagery.raster import LocalRasterIngestor
from nirman_netra.imagery.registration import validate_transform
from nirman_netra.imagery.reliability import RegistrationReliabilityLab

SIZE = 192
CRS = "EPSG:32643"


def _texture() -> NDArray[np.uint8]:
    rng = np.random.default_rng(23)
    image = rng.integers(20, 236, size=(SIZE, SIZE), dtype=np.uint8)
    for offset in range(20, 180, 32):
        cv2.circle(image, (offset, SIZE - offset // 2), 7, 255, 2)
        cv2.rectangle(image, (offset - 6, offset - 6), (offset + 8, offset + 8), 5, 2)
    return image


def _write_raster(
    path: Path,
    image: NDArray[np.uint8],
    *,
    transform: Affine | None = None,
    crs: str | None = CRS,
    capture_time: str | None = "2025-01-10T12:00:00+00:00",
) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=image.shape[1],
        height=image.shape[0],
        count=1,
        dtype=image.dtype,
        crs=crs,
        transform=transform or from_origin(500_000, 2_000_000, 1, 1),
    ) as destination:
        destination.write(image, 1)
        if capture_time is not None:
            destination.update_tags(capture_time=capture_time)


def _pipeline() -> RegistrationPipeline:
    return RegistrationPipeline(ImageryPipelineConfig(minimum_blur_variance=5))


def test_raster_ingestion_reads_required_metadata(tmp_path: Path) -> None:
    path = tmp_path / "metadata.tif"
    _write_raster(path, _texture())

    raster = LocalRasterIngestor().read(path, asset_id="asset-metadata")

    assert (raster.metadata.width, raster.metadata.height) == (SIZE, SIZE)
    assert raster.metadata.band_count == 1
    assert raster.metadata.dtype == "uint8"
    assert raster.metadata.crs.value == CRS
    assert raster.metadata.epsg_code == 32643
    assert raster.metadata.transform == (1.0, 0.0, 500_000.0, 0.0, -1.0, 2_000_000.0)
    assert raster.metadata.resolution == (1.0, 1.0)
    assert raster.metadata.nodata is None
    assert raster.metadata.captured_at == datetime(2025, 1, 10, 12, tzinfo=UTC)
    assert raster.metadata.content_sha256 == sha256(path.read_bytes()).hexdigest()
    legacy_payload = raster.metadata.model_dump(mode="json", exclude={"epsg_code"})
    assert RasterMetadata.model_validate(legacy_payload).epsg_code == 32643


def test_shared_projected_crs_is_reused_before_configured_target(tmp_path: Path) -> None:
    before_path, after_path = tmp_path / "before.tif", tmp_path / "after.tif"
    _write_raster(before_path, _texture())
    _write_raster(after_path, _texture())
    ingestor = LocalRasterIngestor()

    target = select_target_crs(
        ingestor.read(before_path).metadata,
        ingestor.read(after_path).metadata,
        "EPSG:32644",
    )

    assert target.value == CRS


def test_different_crs_requires_configured_projected_target(tmp_path: Path) -> None:
    before_path, after_path = tmp_path / "before.tif", tmp_path / "after.tif"
    _write_raster(before_path, _texture(), crs="EPSG:32643")
    _write_raster(after_path, _texture(), crs="EPSG:32644")
    ingestor = LocalRasterIngestor()
    before, after = ingestor.read(before_path), ingestor.read(after_path)

    with pytest.raises(CRSMismatchError):
        select_target_crs(before.metadata, after.metadata, None)
    assert select_target_crs(before.metadata, after.metadata, CRS).value == CRS


def test_geographic_sources_use_projected_common_grid_and_record_reprojection(
    tmp_path: Path,
) -> None:
    before_path, after_path = tmp_path / "before.tif", tmp_path / "after.tif"
    geographic_transform = from_origin(75, 18, 0.00001, 0.00001)
    _write_raster(
        before_path, _texture(), crs="EPSG:4326", transform=geographic_transform
    )
    _write_raster(
        after_path, _texture(), crs="EPSG:4326", transform=geographic_transform
    )
    ingestor = LocalRasterIngestor()

    aligned = align_pair(
        ingestor.read(before_path),
        ingestor.read(after_path),
        ImageryPipelineConfig(minimum_blur_variance=5, target_crs=CRS),
    )

    assert aligned.crs.is_projected
    assert aligned.common_grid.epsg_code == 32643
    assert {item.operation for item in aligned.crs_transformations} == {"reprojection"}


def test_geographic_sources_without_target_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "geographic.tif"
    _write_raster(
        path,
        _texture(),
        crs="EPSG:4326",
        transform=from_origin(75, 18, 0.00001, 0.00001),
    )
    raster = LocalRasterIngestor().read(path)

    with pytest.raises(CRSMismatchError):
        align_pair(raster, raster, ImageryPipelineConfig(minimum_blur_variance=5))


def test_configured_registration_target_must_be_projected() -> None:
    with pytest.raises(ValueError, match="invalid"):
        ImageryPipelineConfig(target_crs="not-a-crs")
    with pytest.raises(ValueError, match="projected"):
        ImageryPipelineConfig(target_crs="EPSG:4326")


def test_identity_registration_is_deterministic_and_preserves_sources(tmp_path: Path) -> None:
    before_path, after_path = tmp_path / "before.tif", tmp_path / "after.tif"
    image = _texture()
    _write_raster(before_path, image)
    _write_raster(after_path, image)
    source_hashes = tuple(
        sha256(path.read_bytes()).hexdigest() for path in (before_path, after_path)
    )
    output = tmp_path / "registered"

    first = _pipeline().run(
        before_path,
        after_path,
        output,
        before_asset_id="before-identity",
        after_asset_id="after-identity",
    )
    second = _pipeline().run(
        before_path,
        after_path,
        output,
        before_asset_id="before-identity",
        after_asset_id="after-identity",
    )

    assert first.status in {QualityStatus.PASS, QualityStatus.PASS_WITH_WARNING}
    assert np.allclose(np.array(first.transform_matrix), np.eye(2, 3), atol=0.5)
    assert first.checksum == second.checksum
    assert source_hashes == tuple(
        sha256(path.read_bytes()).hexdigest() for path in (before_path, after_path)
    )
    persisted = json.loads((output / f"{first.pair_id}.registration.json").read_text())
    assert persisted["metrics"] == first.metrics.model_dump(mode="json")


def test_translation_registration(tmp_path: Path) -> None:
    before_path, after_path = tmp_path / "before.tif", tmp_path / "after.tif"
    image = _texture()
    movement = np.array([[1.0, 0.0, 6.0], [0.0, 1.0, -4.0]], dtype=np.float32)
    translated = cast(NDArray[np.uint8], cv2.warpAffine(image, movement, (SIZE, SIZE)))
    _write_raster(before_path, image)
    _write_raster(after_path, translated)

    artifact = _pipeline().run(before_path, after_path, tmp_path / "out")
    matrix = np.array(artifact.transform_matrix)

    assert matrix[0, 2] == pytest.approx(-6, abs=1.5)
    assert matrix[1, 2] == pytest.approx(4, abs=1.5)
    assert artifact.metrics.inlier_count >= 8


def test_rotation_within_supported_range(tmp_path: Path) -> None:
    before_path, after_path = tmp_path / "before.tif", tmp_path / "after.tif"
    image = _texture()
    rotated = cast(
        NDArray[np.uint8],
        cv2.warpAffine(image, cv2.getRotationMatrix2D((SIZE / 2, SIZE / 2), 5, 1), (SIZE, SIZE)),
    )
    _write_raster(before_path, image)
    _write_raster(after_path, rotated)

    artifact = _pipeline().run(before_path, after_path, tmp_path / "out")

    assert artifact.status != QualityStatus.REJECTED
    assert artifact.metrics.transform_plausible


def test_intensity_refinement_is_recorded(tmp_path: Path) -> None:
    before_path, after_path = tmp_path / "before.tif", tmp_path / "after.tif"
    image = _texture()
    _write_raster(before_path, image)
    _write_raster(after_path, image)
    pipeline = RegistrationPipeline(
        ImageryPipelineConfig(minimum_blur_variance=5, enable_ecc=True)
    )

    artifact = pipeline.run(before_path, after_path, tmp_path / "out")

    assert artifact.visual_refinement_applied
    assert artifact.visual_refinement_method == "features_ecc"


def test_insufficient_geographic_overlap_fails(tmp_path: Path) -> None:
    before_path, after_path = tmp_path / "before.tif", tmp_path / "after.tif"
    image = _texture()
    _write_raster(before_path, image)
    _write_raster(after_path, image, transform=from_origin(510_000, 2_000_000, 1, 1))

    with pytest.raises(InsufficientOverlapError):
        _pipeline().run(before_path, after_path, tmp_path / "out")


def test_missing_crs_fails(tmp_path: Path) -> None:
    path = tmp_path / "missing-crs.tif"
    _write_raster(path, _texture(), crs=None)

    with pytest.raises(CRSMismatchError):
        LocalRasterIngestor().read(path)


def test_blur_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "blur.tif"
    _write_raster(path, np.full((SIZE, SIZE), 128, dtype=np.uint8))
    raster = LocalRasterIngestor().read(path)

    report = assess_raster(raster, ImageryPipelineConfig(minimum_blur_variance=5))

    assert report.status == QualityStatus.REJECTED
    assert any(issue.code == "EXCESSIVE_BLUR" for issue in report.issues)
    with pytest.raises(RegistrationQualityError):
        _pipeline().run(path, path, tmp_path / "out")


def test_overexposure_produces_warning(tmp_path: Path) -> None:
    path = tmp_path / "bright.tif"
    image = np.full((SIZE, SIZE), 250, dtype=np.uint8)
    for offset in range(12, SIZE, 28):
        cv2.line(image, (0, offset), (SIZE - 1, offset), 20, 1)
    _write_raster(path, image)

    report = assess_raster(
        LocalRasterIngestor().read(path), ImageryPipelineConfig(minimum_blur_variance=5)
    )

    assert report.status == QualityStatus.PASS_WITH_WARNING
    assert any(issue.code == "EXTREME_OVEREXPOSURE" for issue in report.issues)


def test_implausible_transform_is_rejected() -> None:
    transform = np.array([[1.0, 0.0, 200.0], [0.0, 1.0, 0.0]], dtype=np.float64)

    with pytest.raises(TransformValidationError):
        validate_transform(transform, (SIZE, SIZE), ImageryPipelineConfig())


def test_manual_control_points_use_same_transform_policy() -> None:
    request = ManualAlignmentRequest(
        pair_id="manual-pair",
        estimator="affine",
        points=(
            ManualControlPoint(before_x=0, before_y=0, after_x=5, after_y=0),
            ManualControlPoint(before_x=50, before_y=0, after_x=55, after_y=0),
            ManualControlPoint(before_x=0, before_y=50, after_x=5, after_y=50),
        ),
    )

    matrix = estimate_manual_transform(request, (SIZE, SIZE), ImageryPipelineConfig())

    assert matrix[0, 2] == pytest.approx(-5)
    assert matrix[1, 2] == pytest.approx(0)


def test_geospatial_only_mode_records_skipped_visual_refinement(tmp_path: Path) -> None:
    before_path, after_path = tmp_path / "before.tif", tmp_path / "after.tif"
    image = _texture()
    _write_raster(before_path, image)
    _write_raster(after_path, image)
    pipeline = RegistrationPipeline(
        ImageryPipelineConfig(minimum_blur_variance=5, enable_visual_refinement=False)
    )

    artifact = pipeline.run(before_path, after_path, tmp_path / "out")

    assert artifact.status == QualityStatus.PASS_WITH_WARNING
    assert artifact.reliable_for_change_detection
    assert not artifact.visual_refinement_applied
    assert artifact.visual_refinement_method == "none"
    assert "VISUAL_REFINEMENT_SKIPPED" in artifact.warnings
    assert {item.operation for item in artifact.crs_transformations} == {"identity"}


def test_reliability_lab_rejects_missing_crs_without_no_change_claim(tmp_path: Path) -> None:
    before_path, after_path = tmp_path / "before.tif", tmp_path / "after.tif"
    _write_raster(before_path, _texture(), crs=None)
    _write_raster(after_path, _texture())

    report = RegistrationReliabilityLab().evaluate(
        before_path,
        after_path,
        tmp_path / "out",
        before_asset_id="before-missing-crs",
        after_asset_id="after-valid",
    )

    assert report.status == QualityStatus.REJECTED
    assert not report.reliable_for_change_detection
    assert report.failure_stage == "ingestion"
    assert report.reason_codes == ("MISSING_CRS",)
    assert report.artifact is None


def test_reliability_lab_returns_positive_decision_only_with_artifact(tmp_path: Path) -> None:
    before_path, after_path = tmp_path / "before.tif", tmp_path / "after.tif"
    image = _texture()
    _write_raster(before_path, image)
    _write_raster(after_path, image)

    report = RegistrationReliabilityLab(
        ImageryPipelineConfig(minimum_blur_variance=5)
    ).evaluate(before_path, after_path, tmp_path / "out")

    assert report.status in {QualityStatus.PASS, QualityStatus.PASS_WITH_WARNING}
    assert report.reliable_for_change_detection
    assert report.failure_stage is None
    assert report.artifact is not None


def test_reliability_lab_routes_visual_failure_to_manual_alignment(tmp_path: Path) -> None:
    before_path, after_path = tmp_path / "before.tif", tmp_path / "after.tif"
    featureless = np.full((SIZE, SIZE), 128, dtype=np.uint8)
    _write_raster(before_path, featureless)
    _write_raster(after_path, featureless)
    lab = RegistrationReliabilityLab(ImageryPipelineConfig(minimum_blur_variance=0))

    report = lab.evaluate(before_path, after_path, tmp_path / "out")

    assert report.status == QualityStatus.REQUIRES_MANUAL_ALIGNMENT
    assert not report.reliable_for_change_detection
    assert report.failure_stage == "visual_refinement"
    assert report.reason_codes == ("VISUAL_REGISTRATION_FAILED",)
