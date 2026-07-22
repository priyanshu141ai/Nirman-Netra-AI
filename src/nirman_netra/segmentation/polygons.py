"""Building-mask contour extraction with traceable geometry repair."""

from typing import cast

import cv2
import numpy as np
from affine import Affine
from numpy.typing import NDArray
from shapely import GeometryCollection, MultiPolygon, Polygon, make_valid
from shapely.affinity import affine_transform
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry

from nirman_netra.domain import CoordinateReference
from nirman_netra.exceptions import GeometryValidationError
from nirman_netra.segmentation.contracts import ExtractedPolygon, PolygonExtractionConfig


def _polygonal_parts(geometry: BaseGeometry) -> tuple[Polygon, ...]:
    if isinstance(geometry, Polygon):
        return (geometry,)
    if isinstance(geometry, MultiPolygon):
        return tuple(geometry.geoms)
    if isinstance(geometry, GeometryCollection):
        return tuple(part for item in geometry.geoms for part in _polygonal_parts(item))
    return ()


def repair_polygon(geometry: Polygon) -> tuple[BaseGeometry, bool, str | None]:
    """Repair an invalid polygon and explicitly report the operation."""

    if geometry.is_valid:
        return geometry, False, None
    repaired = make_valid(geometry)
    parts = _polygonal_parts(repaired)
    if not parts:
        raise GeometryValidationError("polygon repair produced no polygonal geometry")
    result: BaseGeometry = parts[0] if len(parts) == 1 else MultiPolygon(parts)
    if not result.is_valid:
        raise GeometryValidationError("polygon remains invalid after make_valid")
    return result, True, "make_valid"


def extract_building_polygons(
    mask: NDArray[np.uint8],
    pixel_to_map: Affine,
    crs: CoordinateReference,
    config: PolygonExtractionConfig,
) -> tuple[ExtractedPolygon, ...]:
    """Extract external contours and transform coordinates into the source CRS."""

    if mask.ndim != 2:
        raise GeometryValidationError("segmentation mask must be two-dimensional")
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    extracted: list[ExtractedPolygon] = []
    for contour in contours:
        pixel_area = float(cv2.contourArea(contour))
        if pixel_area < config.minimum_pixel_area:
            continue
        coordinates = [(float(point[0][0]) + 0.5, float(point[0][1]) + 0.5) for point in contour]
        if len(coordinates) < 3:
            continue
        pixel_polygon = Polygon(coordinates)
        repaired_pixel, was_repaired, repair_method = repair_polygon(pixel_polygon)

        for pixel_part in _polygonal_parts(repaired_pixel):
            map_geometry: BaseGeometry = affine_transform(
                pixel_part,
                [
                    pixel_to_map.a,
                    pixel_to_map.b,
                    pixel_to_map.d,
                    pixel_to_map.e,
                    pixel_to_map.c,
                    pixel_to_map.f,
                ],
            )
            if config.simplification_tolerance:
                map_geometry = map_geometry.simplify(
                    config.simplification_tolerance, preserve_topology=True
                )
            if not isinstance(map_geometry, Polygon):
                raise GeometryValidationError("coordinate transformation changed polygon type")
            repaired_map, map_repaired, map_method = repair_polygon(map_geometry)
            part_was_repaired = was_repaired or map_repaired
            part_repair_method = repair_method or map_method
            extracted.append(
                ExtractedPolygon(
                    geometry=cast(dict[str, object], dict(mapping(repaired_map))),
                    crs=crs,
                    source_pixel_area=pixel_area,
                    was_repaired=part_was_repaired,
                    repair_method=part_repair_method,
                )
            )
    return tuple(extracted)
