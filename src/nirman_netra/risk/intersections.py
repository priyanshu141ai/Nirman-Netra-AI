"""CRS-safe parcel intersection and approved-plan comparison services."""

from collections.abc import Sequence

from pyproj import CRS, Transformer
from pyproj.exceptions import ProjError
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform

from nirman_netra.data.contracts import ApprovedFootprint, Parcel, PublicBoundary
from nirman_netra.domain import CoordinateReference, GeometryReference
from nirman_netra.exceptions import CRSMismatchError, GeometryValidationError
from nirman_netra.geospatial import validate_geometry
from nirman_netra.risk.contracts import (
    ApprovedPlanComparison,
    ParcelIntersectionResult,
    ParcelOverlap,
)


def _square_metre_factor(crs: CoordinateReference) -> float:
    parsed = CRS.from_user_input(crs.value)
    if not parsed.is_projected:
        raise CRSMismatchError("municipal area comparison requires a projected CRS")
    axes = parsed.axis_info
    if (
        len(axes) < 2
        or axes[0].unit_conversion_factor is None
        or axes[1].unit_conversion_factor is None
    ):
        raise CRSMismatchError("projected CRS units are unavailable")
    return float(axes[0].unit_conversion_factor * axes[1].unit_conversion_factor)


def _to_crs(
    geometry: BaseGeometry,
    source: CoordinateReference,
    target: CoordinateReference,
) -> BaseGeometry:
    if source == target:
        return geometry
    try:
        transformer = Transformer.from_crs(source.value, target.value, always_xy=True)
        converted = transform(transformer.transform, geometry)
    except ProjError as exc:
        raise CRSMismatchError(f"cannot transform {source.value} to {target.value}") from exc
    if converted.is_empty or not converted.is_valid:
        raise GeometryValidationError("CRS transformation produced invalid geometry")
    return converted


def intersect_change_with_parcels(
    change: GeometryReference,
    parcels: Sequence[Parcel],
) -> ParcelIntersectionResult:
    """Rank parcel intersections without silently choosing an ambiguous match."""

    change_geometry = validate_geometry(change)
    if change_geometry.geom_type not in {"Polygon", "MultiPolygon"}:
        raise GeometryValidationError("change evidence must be polygonal")
    factor = _square_metre_factor(change.crs)
    change_area = float(change_geometry.area * factor)
    if change_area <= 0:
        raise GeometryValidationError("change evidence has zero area")
    if len({parcel.parcel_id for parcel in parcels}) != len(parcels):
        raise GeometryValidationError("parcel identifiers must be unique")
    overlaps: list[ParcelOverlap] = []
    geometries: dict[str, BaseGeometry] = {}
    for parcel in parcels:
        parcel_geometry = _to_crs(validate_geometry(parcel.geometry), parcel.crs, change.crs)
        geometries[parcel.parcel_id] = parcel_geometry
        intersection_area = float(change_geometry.intersection(parcel_geometry).area * factor)
        if intersection_area > 1e-9:
            overlaps.append(
                ParcelOverlap(
                    parcel_id=parcel.parcel_id,
                    intersection_area_square_metres=intersection_area,
                    percentage_of_change_inside=min(100.0, 100 * intersection_area / change_area),
                )
            )
    overlaps.sort(key=lambda item: (-item.intersection_area_square_metres, item.parcel_id))
    if not overlaps:
        return ParcelIntersectionResult(
            change_geometry_id=change.geometry_id,
            matching_parcel_id=None,
            intersection_area_square_metres=0,
            percentage_inside_parcel=0,
            boundary_crossing=False,
            multiple_parcel_ambiguity=False,
            no_parcel_match=True,
            overlaps=(),
            analysis_crs=change.crs,
        )
    best = overlaps[0]
    matching_geometry = geometries[best.parcel_id]
    return ParcelIntersectionResult(
        change_geometry_id=change.geometry_id,
        matching_parcel_id=best.parcel_id,
        intersection_area_square_metres=best.intersection_area_square_metres,
        percentage_inside_parcel=best.percentage_of_change_inside,
        boundary_crossing=not matching_geometry.covers(change_geometry),
        multiple_parcel_ambiguity=len(overlaps) > 1,
        no_parcel_match=False,
        overlaps=tuple(overlaps),
        analysis_crs=change.crs,
    )


def compare_approved_plan(
    approved: ApprovedFootprint,
    observed_old: GeometryReference,
    observed_new: GeometryReference,
    parcel: Parcel,
    *,
    setback_metres: float,
    public_boundaries: Sequence[PublicBoundary] = (),
) -> ApprovedPlanComparison:
    """Compare horizontal footprints only; this does not infer legal or floor-count status."""

    if approved.parcel_id != parcel.parcel_id:
        raise GeometryValidationError("approved footprint and parcel relationship is inconsistent")
    if setback_metres < 0:
        raise GeometryValidationError("setback distance cannot be negative")
    comparison_crs = approved.crs
    factor = _square_metre_factor(comparison_crs)
    approved_geometry = validate_geometry(approved.geometry)
    old_geometry = _to_crs(validate_geometry(observed_old), observed_old.crs, comparison_crs)
    new_geometry = _to_crs(validate_geometry(observed_new), observed_new.crs, comparison_crs)
    parcel_geometry = _to_crs(validate_geometry(parcel.geometry), parcel.crs, comparison_crs)
    if any(
        geometry.geom_type not in {"Polygon", "MultiPolygon"}
        for geometry in (approved_geometry, old_geometry, new_geometry, parcel_geometry)
    ):
        raise GeometryValidationError("plan comparison geometries must be polygonal")
    newly_added = new_geometry.difference(old_geometry)
    added_outside_approval = newly_added.difference(approved_geometry)
    removed_approved = approved_geometry.intersection(old_geometry).difference(new_geometry)
    inner = parcel_geometry.buffer(-setback_metres) if setback_metres else parcel_geometry
    setback_zone = parcel_geometry if inner.is_empty else parcel_geometry.difference(inner)
    boundary_ids = tuple(
        sorted(
            boundary.boundary_id
            for boundary in public_boundaries
            if new_geometry.intersects(
                _to_crs(validate_geometry(boundary.geometry), boundary.crs, comparison_crs)
            )
        )
    )
    parcel_area = float(parcel_geometry.area)
    return ApprovedPlanComparison(
        approved_plan_version=approved.plan_version,
        approved_footprint_area_square_metres=float(approved_geometry.area * factor),
        observed_old_area_square_metres=float(old_geometry.area * factor),
        observed_new_area_square_metres=float(new_geometry.area * factor),
        newly_added_area_square_metres=float(newly_added.area * factor),
        added_area_outside_approval_square_metres=float(added_outside_approval.area * factor),
        removed_approved_area_square_metres=float(removed_approved.area * factor),
        setback_zone_intersection_area_square_metres=float(
            new_geometry.intersection(setback_zone).area * factor
        ),
        public_boundary_intersection_ids=boundary_ids,
        site_coverage_ratio=(
            float(new_geometry.intersection(parcel_geometry).area / parcel_area)
            if parcel_area
            else 0
        ),
        comparison_crs=comparison_crs,
    )
