"""Controlled domain interface for reviewed manual control points."""

import cv2
import numpy as np
from numpy.typing import NDArray

from nirman_netra.exceptions import RegistrationError
from nirman_netra.imagery.config import ImageryPipelineConfig
from nirman_netra.imagery.contracts import ManualAlignmentRequest
from nirman_netra.imagery.registration import validate_transform


def estimate_manual_transform(
    request: ManualAlignmentRequest,
    image_shape: tuple[int, int],
    config: ImageryPipelineConfig,
) -> NDArray[np.float64]:
    """Estimate a reviewed after-to-before transform and apply the same plausibility policy."""

    source = np.array(
        [[point.after_x, point.after_y] for point in request.points], dtype=np.float64
    )
    destination = np.array(
        [[point.before_x, point.before_y] for point in request.points], dtype=np.float64
    )
    matrix: NDArray[np.float64]
    if request.estimator == "affine":
        design = np.column_stack((source, np.ones(len(source))))
        coefficients, _, rank, _ = np.linalg.lstsq(design, destination, rcond=None)
        if rank < 3:
            raise RegistrationError("manual affine control points are degenerate")
        matrix = np.asarray(coefficients.T, dtype=np.float64)
    else:
        estimated, _ = cv2.findHomography(source, destination, method=0)
        if estimated is None:
            raise RegistrationError("manual homography control points are degenerate")
        matrix = np.asarray(estimated, dtype=np.float64)
    validate_transform(matrix, image_shape, config)
    return matrix
