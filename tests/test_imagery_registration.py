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
from nirman_netra.imagery.config import ImageryPipelineConfig
from nirman_netra.imagery.contracts import (
    ManualAlignmentRequest,
    ManualControlPoint,
    QualityStatus,
)
from nirman_netra.imagery.manual import estimate_manual_transform
from nirman_netra.imagery.pipeline import RegistrationPipeline
from nirman_netra.imagery.quality import assess_raster
from nirman_netra.imagery.raster import LocalRasterIngestor
from nirman_netra.imagery.registration import validate_transform

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
    assert raster.metadata.transform == (1.0, 0.0, 500_000.0, 0.0, -1.0, 2_000_000.0)
    assert raster.metadata.resolution == (1.0, 1.0)
    assert raster.metadata.nodata is None
    assert raster.metadata.captured_at == datetime(2025, 1, 10, 12, tzinfo=UTC)


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
