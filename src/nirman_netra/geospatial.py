"""Safe geospatial validation and measurement primitives."""

from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError
from pyproj import CRS
from pyproj.exceptions import CRSError
from shapely import area, is_valid, is_valid_reason
from shapely.errors import GEOSException
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

from nirman_netra.domain import BoundingBox, CoordinateReference, GeometryReference
from nirman_netra.exceptions import CRSMismatchError, GeometryValidationError


def validate_crs(value: str | int | CoordinateReference) -> CoordinateReference:
    """Validate an EPSG code or any CRS string understood by PROJ."""

    if isinstance(value, CoordinateReference):
        return value
    if isinstance(value, bool):
        raise CRSMismatchError("boolean values are not CRS identifiers")
    candidate = f"EPSG:{value}" if isinstance(value, int) else value
    try:
        return CoordinateReference(value=CRS.from_user_input(candidate).to_string())
    except (CRSError, ValidationError) as exc:
        raise CRSMismatchError(f"invalid CRS: {value}") from exc


def validate_bounding_box(
    min_x: float, min_y: float, max_x: float, max_y: float, crs: str | int | CoordinateReference
) -> BoundingBox:
    """Validate coordinate order and bind bounds to an explicit CRS."""

    try:
        return BoundingBox(
            min_x=min_x,
            min_y=min_y,
            max_x=max_x,
            max_y=max_y,
            crs=validate_crs(crs),
        )
    except ValidationError as exc:
        raise GeometryValidationError(str(exc)) from exc


def bounding_box_overlap(left: BoundingBox, right: BoundingBox) -> BoundingBox | None:
    """Return the intersection of aligned bounds, or None when they do not overlap."""

    if left.crs != right.crs:
        raise CRSMismatchError("bounding boxes must use the same CRS")
    min_x = max(left.min_x, right.min_x)
    min_y = max(left.min_y, right.min_y)
    max_x = min(left.max_x, right.max_x)
    max_y = min(left.max_y, right.max_y)
    if min_x >= max_x or min_y >= max_y:
        return None
    return BoundingBox(min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y, crs=left.crs)


def validate_geometry(value: GeometryReference | Mapping[str, Any]) -> BaseGeometry:
    """Parse a geometry without silently repairing invalid input."""

    source = value.geometry if isinstance(value, GeometryReference) else value
    try:
        geometry = shape(dict(source))
    except (GEOSException, KeyError, TypeError, ValueError) as exc:
        raise GeometryValidationError("geometry cannot be parsed") from exc
    if geometry.is_empty:
        raise GeometryValidationError("geometry is empty")
    if not is_valid(geometry):
        raise GeometryValidationError(f"invalid geometry: {is_valid_reason(geometry)}")
    return geometry


def calculate_safe_area(
    value: GeometryReference | Mapping[str, Any], crs: str | int | CoordinateReference
) -> float:
    """Calculate area only when coordinates use a projected CRS."""

    reference = validate_crs(crs)
    if not reference.is_projected:
        raise CRSMismatchError("area requires a projected CRS")
    return float(area(validate_geometry(value)))
