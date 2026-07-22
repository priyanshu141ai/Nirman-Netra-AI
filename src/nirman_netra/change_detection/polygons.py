"""Traceable added/removed change polygons in source map coordinates."""

from typing import cast

import numpy as np
from affine import Affine
from numpy.typing import NDArray
from pyproj import CRS
from shapely.geometry import shape

from nirman_netra.change_detection.contracts import (
    CHANGE_LABEL_NAMES,
    ChangeLabel,
    ChangePolygon,
    ChangePolygonLabel,
)
from nirman_netra.domain import CoordinateReference
from nirman_netra.exceptions import GeometryValidationError
from nirman_netra.segmentation.contracts import PolygonExtractionConfig
from nirman_netra.segmentation.polygons import extract_building_polygons


def _square_metre_factor(crs: CoordinateReference) -> float:
    parsed = CRS.from_user_input(crs.value)
    if not parsed.is_projected:
        raise GeometryValidationError("change area requires a projected CRS")
    axes = parsed.axis_info
    if (
        len(axes) < 2
        or axes[0].unit_conversion_factor is None
        or axes[1].unit_conversion_factor is None
    ):
        raise GeometryValidationError("projected CRS axis units are unavailable")
    return float(axes[0].unit_conversion_factor * axes[1].unit_conversion_factor)


def extract_change_polygons(
    label_mask: NDArray[np.uint8],
    pixel_to_map: Affine,
    crs: CoordinateReference,
    source_pair_id: str,
    config: PolygonExtractionConfig,
) -> tuple[ChangePolygon, ...]:
    """Extract only horizontal footprint additions/removals; vertical change stays unconfirmed."""

    if label_mask.ndim != 2:
        raise GeometryValidationError("change label mask must be two-dimensional")
    square_metre_factor = _square_metre_factor(crs)
    output: list[ChangePolygon] = []
    polygon_labels = (
        ChangeLabel.ADDED_BUILDING,
        ChangeLabel.REMOVED_BUILDING,
        ChangeLabel.FOOTPRINT_EXPANSION,
        ChangeLabel.PARTIAL_DEMOLITION,
    )
    for label in polygon_labels:
        binary = (label_mask == int(label)).astype(np.uint8)
        for extracted in extract_building_polygons(binary, pixel_to_map, crs, config):
            geometry = shape(extracted.geometry)
            area = float(geometry.area * square_metre_factor)
            if area <= 0 or not geometry.is_valid:
                raise GeometryValidationError("change polygon is empty or invalid after repair")
            output.append(
                ChangePolygon(
                    label=cast(ChangePolygonLabel, CHANGE_LABEL_NAMES[label]),
                    geometry=extracted.geometry,
                    crs=crs,
                    source_pair_id=source_pair_id,
                    changed_area_square_metres=area,
                    was_repaired=extracted.was_repaired,
                    repair_method=extracted.repair_method,
                )
            )
    return tuple(output)
