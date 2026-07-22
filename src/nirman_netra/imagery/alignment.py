"""CRS-aware reprojection onto a shared geographic grid."""

from dataclasses import dataclass

import numpy as np
from affine import Affine
from numpy.typing import NDArray
from pyproj import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject

from nirman_netra.domain import BoundingBox, CoordinateReference
from nirman_netra.exceptions import InsufficientOverlapError, RegistrationQualityError
from nirman_netra.imagery.config import ImageryPipelineConfig
from nirman_netra.imagery.contracts import (
    CommonGrid,
    CRSTransformation,
    PairQualityReport,
    QualityStatus,
    RasterMetadata,
)
from nirman_netra.imagery.crs import (
    bounds_in_target,
    resolution_in_target,
    select_target_crs,
    transformation_record,
)
from nirman_netra.imagery.quality import assess_pair
from nirman_netra.imagery.raster import IngestedRaster


@dataclass(frozen=True)
class AlignedRasterPair:
    before_metadata: RasterMetadata
    after_metadata: RasterMetadata
    before_pixels: NDArray[np.float32]
    after_pixels: NDArray[np.float32]
    valid_mask: NDArray[np.bool_]
    transform: Affine
    crs: CoordinateReference
    pair_quality: PairQualityReport
    common_grid: CommonGrid
    crs_transformations: tuple[CRSTransformation, ...]


def _resampling(name: str) -> Resampling:
    return {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
    }[name]


def _reproject_raster(
    raster: IngestedRaster,
    shape: tuple[int, int],
    target_transform: Affine,
    target_crs: str,
    resampling: Resampling,
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    height, width = shape
    destination = np.full((raster.metadata.band_count, height, width), np.nan, dtype=np.float32)
    source_transform = Affine(*raster.metadata.transform)
    for index in range(raster.metadata.band_count):
        reproject(
            source=raster.pixels[index],
            destination=destination[index],
            src_transform=source_transform,
            src_crs=raster.metadata.crs.value,
            src_nodata=raster.metadata.nodata,
            dst_transform=target_transform,
            dst_crs=target_crs,
            dst_nodata=np.nan,
            resampling=resampling,
        )
    mask = np.zeros((height, width), dtype=np.uint8)
    reproject(
        source=raster.valid_mask.astype(np.uint8),
        destination=mask,
        src_transform=source_transform,
        src_crs=raster.metadata.crs.value,
        src_nodata=0,
        dst_transform=target_transform,
        dst_crs=target_crs,
        dst_nodata=0,
        resampling=Resampling.nearest,
    )
    valid = (mask > 0) & np.all(np.isfinite(destination), axis=0)
    return destination, valid


def align_pair(
    before: IngestedRaster,
    after: IngestedRaster,
    config: ImageryPipelineConfig,
) -> AlignedRasterPair:
    """Reproject a raster pair onto an explicit projected common grid."""

    pair_quality = assess_pair(before, after, config)
    if pair_quality.status == QualityStatus.REJECTED and any(
        issue.code == "INSUFFICIENT_GEOGRAPHIC_OVERLAP" for issue in pair_quality.issues
    ):
        raise InsufficientOverlapError(
            f"geographic overlap {pair_quality.geographic_overlap_ratio:.3f} is below "
            f"{config.minimum_geographic_overlap:.3f}"
        )
    if pair_quality.status == QualityStatus.REJECTED:
        codes = sorted(issue.code for issue in pair_quality.issues)
        raise RegistrationQualityError(f"raster pair rejected: {','.join(codes)}")

    target = select_target_crs(before.metadata, after.metadata, config.target_crs)
    before_bounds = bounds_in_target(before.metadata, target)
    after_bounds = bounds_in_target(after.metadata, target)
    left = max(before_bounds[0], after_bounds[0])
    bottom = max(before_bounds[1], after_bounds[1])
    right = min(before_bounds[2], after_bounds[2])
    top = min(before_bounds[3], after_bounds[3])
    if left >= right or bottom >= top:
        raise InsufficientOverlapError("raster footprints have no common geographic window")

    before_resolution = resolution_in_target(before.metadata, target)
    after_resolution = resolution_in_target(after.metadata, target)
    resolution_x = max(before_resolution[0], after_resolution[0])
    resolution_y = max(before_resolution[1], after_resolution[1])
    width = int(np.floor((right - left) / resolution_x))
    height = int(np.floor((top - bottom) / resolution_y))
    if width < 1 or height < 1:
        raise InsufficientOverlapError("common geographic window is smaller than one pixel")
    target_transform = from_origin(left, top, resolution_x, resolution_y)
    resampling = _resampling(config.resampling)
    before_pixels, before_valid = _reproject_raster(
        before, (height, width), target_transform, target.value, resampling
    )
    after_pixels, after_valid = _reproject_raster(
        after, (height, width), target_transform, target.value, resampling
    )
    valid_mask = before_valid & after_valid
    before_pixels.setflags(write=False)
    after_pixels.setflags(write=False)
    valid_mask.setflags(write=False)
    return AlignedRasterPair(
        before_metadata=before.metadata,
        after_metadata=after.metadata,
        before_pixels=before_pixels,
        after_pixels=after_pixels,
        valid_mask=valid_mask,
        transform=target_transform,
        crs=target,
        pair_quality=pair_quality,
        common_grid=CommonGrid(
            crs=target,
            epsg_code=CRS.from_user_input(target.value).to_epsg(),
            transform=(
                target_transform.a,
                target_transform.b,
                target_transform.c,
                target_transform.d,
                target_transform.e,
                target_transform.f,
            ),
            bounds=BoundingBox(
                min_x=left,
                min_y=top - height * resolution_y,
                max_x=left + width * resolution_x,
                max_y=top,
                crs=target,
            ),
            width=width,
            height=height,
            resolution=(resolution_x, resolution_y),
        ),
        crs_transformations=(
            transformation_record(before.metadata, target),
            transformation_record(after.metadata, target),
        ),
    )
