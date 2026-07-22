import pytest

from nirman_netra.exceptions import CRSMismatchError, GeometryValidationError
from nirman_netra.geospatial import (
    bounding_box_overlap,
    validate_bounding_box,
    validate_crs,
    validate_geometry,
)


def test_valid_bounding_box() -> None:
    bounds = validate_bounding_box(100, 200, 300, 400, "EPSG:32643")

    assert (bounds.min_x, bounds.min_y, bounds.max_x, bounds.max_y) == (100, 200, 300, 400)


def test_reversed_coordinate_order_is_invalid() -> None:
    with pytest.raises(GeometryValidationError):
        validate_bounding_box(300, 200, 100, 400, "EPSG:32643")


def test_bounding_box_overlap() -> None:
    left = validate_bounding_box(0, 0, 10, 10, "EPSG:32643")
    right = validate_bounding_box(5, 4, 12, 8, "EPSG:32643")

    overlap = bounding_box_overlap(left, right)

    assert overlap is not None
    assert (overlap.min_x, overlap.min_y, overlap.max_x, overlap.max_y) == (5, 4, 10, 8)


def test_invalid_crs() -> None:
    with pytest.raises(CRSMismatchError):
        validate_crs("not-a-crs")


def test_invalid_geometry() -> None:
    bow_tie = {
        "type": "Polygon",
        "coordinates": [[(0, 0), (2, 2), (0, 2), (2, 0), (0, 0)]],
    }

    with pytest.raises(GeometryValidationError):
        validate_geometry(bow_tie)
