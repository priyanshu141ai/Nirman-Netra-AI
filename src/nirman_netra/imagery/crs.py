"""Explicit CRS selection and transformation records for raster registration."""

from typing import cast

from pyproj import CRS
from pyproj.exceptions import CRSError
from rasterio.warp import calculate_default_transform, transform_bounds

from nirman_netra.domain import CoordinateReference
from nirman_netra.exceptions import CRSMismatchError
from nirman_netra.imagery.contracts import CRSTransformation, RasterMetadata


def _parse(value: str) -> CRS:
    try:
        return CRS.from_user_input(value)
    except CRSError as exc:
        raise CRSMismatchError("raster CRS definition is invalid") from exc


def select_target_crs(
    before: RasterMetadata,
    after: RasterMetadata,
    configured_target: str | None,
) -> CoordinateReference:
    """Reuse a shared projected CRS; otherwise require an explicit projected target."""

    before_crs, after_crs = _parse(before.crs.value), _parse(after.crs.value)
    if before_crs.equals(after_crs) and before_crs.is_projected:
        return before.crs
    if configured_target is None:
        raise CRSMismatchError(
            "a configured projected target CRS is required for this raster pair"
        )
    target = _parse(configured_target)
    if not target.is_projected:
        raise CRSMismatchError("registration target CRS must be projected")
    return CoordinateReference(value=target.to_string())


def bounds_in_target(
    metadata: RasterMetadata, target: CoordinateReference
) -> tuple[float, float, float, float]:
    bounds = metadata.bounds
    source = _parse(metadata.crs.value)
    destination = _parse(target.value)
    if source.equals(destination):
        return bounds.min_x, bounds.min_y, bounds.max_x, bounds.max_y
    return cast(
        tuple[float, float, float, float],
        transform_bounds(
            source,
            destination,
            bounds.min_x,
            bounds.min_y,
            bounds.max_x,
            bounds.max_y,
            densify_pts=21,
        ),
    )


def resolution_in_target(
    metadata: RasterMetadata, target: CoordinateReference
) -> tuple[float, float]:
    source = _parse(metadata.crs.value)
    destination = _parse(target.value)
    if source.equals(destination):
        return metadata.resolution
    bounds = metadata.bounds
    transform, _, _ = calculate_default_transform(
        source,
        destination,
        metadata.width,
        metadata.height,
        bounds.min_x,
        bounds.min_y,
        bounds.max_x,
        bounds.max_y,
    )
    return abs(transform.a), abs(transform.e)


def transformation_record(
    metadata: RasterMetadata, target: CoordinateReference
) -> CRSTransformation:
    source = _parse(metadata.crs.value)
    destination = _parse(target.value)
    return CRSTransformation(
        asset_id=metadata.asset_id,
        source_crs=metadata.crs,
        source_epsg_code=metadata.epsg_code,
        target_crs=target,
        target_epsg_code=destination.to_epsg(),
        operation="identity" if source.equals(destination) else "reprojection",
    )
