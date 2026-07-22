"""Deterministic feature-based visual registration baseline."""

import math
from dataclasses import dataclass
from typing import Literal, cast

import cv2
import numpy as np
from numpy.typing import NDArray
from shapely.geometry import Polygon, box

from nirman_netra.exceptions import RegistrationError, TransformValidationError
from nirman_netra.imagery.alignment import AlignedRasterPair
from nirman_netra.imagery.config import ImageryPipelineConfig
from nirman_netra.imagery.contracts import QualityStatus, RegistrationMetrics


@dataclass(frozen=True)
class RegistrationComputation:
    transform_matrix: NDArray[np.float64]
    registered_pixels: NDArray[np.float32]
    registered_valid_mask: NDArray[np.bool_]
    metrics: RegistrationMetrics
    status: QualityStatus
    warnings: tuple[str, ...]
    visual_refinement_applied: bool
    visual_refinement_method: Literal["none", "features", "features_ecc"]


def accept_geospatial_alignment(
    pair: AlignedRasterPair, config: ImageryPipelineConfig
) -> RegistrationComputation:
    """Use the common grid without local refinement when explicitly configured."""

    overlap = float(np.count_nonzero(pair.valid_mask) / pair.valid_mask.size)
    score = min(pair.pair_quality.geographic_overlap_ratio, overlap)
    reliable = score >= config.minimum_registration_score
    warnings = ["VISUAL_REFINEMENT_SKIPPED"]
    if not reliable:
        warnings.append("LOW_REGISTRATION_QUALITY")
    return RegistrationComputation(
        transform_matrix=np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float64),
        registered_pixels=pair.after_pixels,
        registered_valid_mask=pair.valid_mask,
        metrics=RegistrationMetrics(
            inlier_count=0,
            inlier_ratio=0,
            reprojection_error=0,
            overlap_ratio=overlap,
            transform_plausible=True,
            registration_quality_score=score,
        ),
        status=(
            QualityStatus.PASS_WITH_WARNING
            if reliable
            else QualityStatus.REQUIRES_MANUAL_ALIGNMENT
        ),
        warnings=tuple(warnings),
        visual_refinement_applied=False,
        visual_refinement_method="none",
    )


def _gray(
    pixels: NDArray[np.float32],
    valid: NDArray[np.bool_],
    config: ImageryPipelineConfig,
) -> NDArray[np.uint8]:
    if pixels.shape[0] == 1:
        gray = pixels[0]
    else:
        rgb = np.moveaxis(pixels[:3], 0, 2)
        gray = cast(NDArray[np.float32], cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    values = gray[valid & np.isfinite(gray)]
    if values.size == 0:
        return np.zeros(gray.shape, dtype=np.uint8)
    low, high = np.percentile(
        values,
        (config.normalization_low_percentile, config.normalization_high_percentile),
    )
    if high <= low:
        return np.zeros(gray.shape, dtype=np.uint8)
    return cast(
        NDArray[np.uint8],
        np.clip((gray - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8),
    )


def _homogeneous(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    if matrix.shape == (2, 3):
        return cast(NDArray[np.float64], np.vstack((matrix, np.array([0.0, 0.0, 1.0]))))
    if matrix[2, 2] == 0:
        raise TransformValidationError("homography normalization term cannot be zero")
    return cast(NDArray[np.float64], matrix / matrix[2, 2])


def validate_transform(
    matrix: NDArray[np.float64],
    image_shape: tuple[int, int],
    config: ImageryPipelineConfig,
) -> None:
    """Reject transforms beyond configured rotation, scale, translation, or perspective."""

    homogeneous = _homogeneous(matrix)
    if homogeneous.shape != (3, 3) or not np.isfinite(homogeneous).all():
        raise TransformValidationError("transform must be a finite 3x3 or 2x3 matrix")
    if (
        abs(homogeneous[2, 0]) > config.maximum_perspective_term
        or abs(homogeneous[2, 1]) > config.maximum_perspective_term
    ):
        raise TransformValidationError("perspective terms exceed the configured maximum")
    linear = homogeneous[:2, :2]
    if np.linalg.det(linear) <= 0:
        raise TransformValidationError("transform reflection or collapse is implausible")
    singular_values = np.linalg.svd(linear, compute_uv=False)
    if np.any(np.abs(singular_values - 1.0) > config.maximum_scale_deviation):
        raise TransformValidationError("transform scale is implausible")
    rotation = abs(math.degrees(math.atan2(linear[1, 0], linear[0, 0])))
    if rotation > config.maximum_rotation_degrees:
        raise TransformValidationError("transform rotation exceeds the configured maximum")
    height, width = image_shape
    translation_fraction = float(np.linalg.norm(homogeneous[:2, 2]) / math.hypot(width, height))
    if translation_fraction > config.maximum_translation_fraction:
        raise TransformValidationError("transform translation exceeds the configured maximum")


def _project_points(
    points: NDArray[np.float32], matrix: NDArray[np.float64]
) -> NDArray[np.float32]:
    shaped = points.reshape(-1, 1, 2)
    if matrix.shape == (2, 3):
        return cast(NDArray[np.float32], cv2.transform(shaped, matrix).reshape(-1, 2))
    return cast(NDArray[np.float32], cv2.perspectiveTransform(shaped, matrix).reshape(-1, 2))


def _visual_overlap(matrix: NDArray[np.float64], shape: tuple[int, int]) -> float:
    height, width = shape
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    transformed = _project_points(corners, matrix)
    polygon = Polygon(transformed)
    if not polygon.is_valid or polygon.is_empty:
        return 0.0
    intersection = polygon.intersection(box(0, 0, width, height))
    return float(max(0.0, min(1.0, intersection.area / (width * height))))


def _refine_ecc(
    before: NDArray[np.uint8],
    after: NDArray[np.uint8],
    matrix: NDArray[np.float64],
    config: ImageryPipelineConfig,
) -> NDArray[np.float64]:
    motion = cv2.MOTION_AFFINE if matrix.shape == (2, 3) else cv2.MOTION_HOMOGRAPHY
    forward = (
        cv2.invertAffineTransform(matrix).astype(np.float32)
        if matrix.shape == (2, 3)
        else np.linalg.inv(matrix).astype(np.float32)
    )
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        config.ecc_iterations,
        config.ecc_epsilon,
    )
    _, refined_forward = cv2.findTransformECC(before, after, forward, motion, criteria, None)
    refined = np.asarray(refined_forward, dtype=np.float64)
    return (
        cast(NDArray[np.float64], cv2.invertAffineTransform(refined).astype(np.float64))
        if matrix.shape == (2, 3)
        else cast(NDArray[np.float64], np.linalg.inv(refined).astype(np.float64))
    )


def _warp(
    pixels: NDArray[np.float32],
    valid: NDArray[np.bool_],
    matrix: NDArray[np.float64],
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    height, width = valid.shape
    registered = np.empty_like(pixels, dtype=np.float32)
    for index in range(pixels.shape[0]):
        if matrix.shape == (2, 3):
            registered[index] = cv2.warpAffine(
                pixels[index], matrix, (width, height), borderValue=float("nan")
            )
        else:
            registered[index] = cv2.warpPerspective(
                pixels[index], matrix, (width, height), borderValue=float("nan")
            )
    if matrix.shape == (2, 3):
        warped_mask = cv2.warpAffine(
            valid.astype(np.uint8), matrix, (width, height), flags=cv2.INTER_NEAREST
        )
    else:
        warped_mask = cv2.warpPerspective(
            valid.astype(np.uint8), matrix, (width, height), flags=cv2.INTER_NEAREST
        )
    registered_valid = (warped_mask > 0) & np.all(np.isfinite(registered), axis=0)
    registered.setflags(write=False)
    registered_valid.setflags(write=False)
    return registered, registered_valid


def register_aligned_pair(
    pair: AlignedRasterPair, config: ImageryPipelineConfig
) -> RegistrationComputation:
    """Estimate and validate an after-to-before transform using ORB and RANSAC."""

    before_gray = _gray(pair.before_pixels, pair.valid_mask, config)
    after_gray = _gray(pair.after_pixels, pair.valid_mask, config)
    mask = pair.valid_mask.astype(np.uint8) * 255
    cv2.setRNGSeed(config.cv_random_seed)
    detector = cv2.ORB.create(nfeatures=config.maximum_features)
    before_points, before_descriptors = detector.detectAndCompute(before_gray, mask)
    after_points, after_descriptors = detector.detectAndCompute(after_gray, mask)
    if before_descriptors is None or after_descriptors is None:
        raise RegistrationError("insufficient visual features; manual alignment required")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    candidates = matcher.knnMatch(after_descriptors, before_descriptors, k=2)
    matches = [
        first
        for candidate in candidates
        if len(candidate) == 2
        for first, second in [candidate]
        if first.distance < config.feature_ratio_test * second.distance
    ]
    if len(matches) < config.minimum_inliers:
        raise RegistrationError("insufficient feature matches; manual alignment required")
    source = np.asarray([after_points[item.queryIdx].pt for item in matches], dtype=np.float32)
    destination = np.asarray(
        [before_points[item.trainIdx].pt for item in matches], dtype=np.float32
    )
    if config.estimator == "affine":
        estimated, inlier_mask = cv2.estimateAffinePartial2D(
            source,
            destination,
            method=cv2.RANSAC,
            ransacReprojThreshold=config.ransac_reprojection_threshold,
        )
    else:
        estimated, inlier_mask = cv2.findHomography(
            source,
            destination,
            cv2.RANSAC,
            config.ransac_reprojection_threshold,
        )
    if estimated is None or inlier_mask is None:
        raise RegistrationError("transform estimation failed; manual alignment required")
    matrix = np.asarray(estimated, dtype=np.float64)
    warnings: list[str] = []
    ecc_applied = False
    if config.enable_ecc:
        try:
            matrix = _refine_ecc(before_gray, after_gray, matrix, config)
            ecc_applied = True
        except cv2.error:
            warnings.append("ECC_REFINEMENT_FAILED")

    validate_transform(matrix, pair.valid_mask.shape, config)
    inliers = np.asarray(inlier_mask).ravel().astype(bool)
    inlier_count = int(np.count_nonzero(inliers))
    if inlier_count == 0:
        raise RegistrationError("transform estimation produced no inliers")
    inlier_ratio = float(inlier_count / len(matches))
    projected = _project_points(source[inliers], matrix)
    reprojection_error = float(np.mean(np.linalg.norm(projected - destination[inliers], axis=1)))
    overlap = _visual_overlap(matrix, pair.valid_mask.shape)
    score = float(
        np.clip(
            0.25 * min(1.0, inlier_count / config.minimum_inliers)
            + 0.25 * min(1.0, inlier_ratio / config.minimum_inlier_ratio)
            + 0.25 * max(0.0, 1.0 - reprojection_error / config.maximum_reprojection_error)
            + 0.25 * overlap,
            0.0,
            1.0,
        )
    )
    reliable = (
        inlier_count >= config.minimum_inliers
        and inlier_ratio >= config.minimum_inlier_ratio
        and reprojection_error <= config.maximum_reprojection_error
        and score >= config.minimum_registration_score
    )
    status = QualityStatus.PASS if reliable else QualityStatus.REQUIRES_MANUAL_ALIGNMENT
    if not reliable:
        warnings.append("LOW_REGISTRATION_QUALITY")
    registered, registered_valid = _warp(pair.after_pixels, pair.valid_mask, matrix)
    return RegistrationComputation(
        transform_matrix=matrix,
        registered_pixels=registered,
        registered_valid_mask=registered_valid,
        metrics=RegistrationMetrics(
            inlier_count=inlier_count,
            inlier_ratio=inlier_ratio,
            reprojection_error=reprojection_error,
            overlap_ratio=overlap,
            transform_plausible=True,
            registration_quality_score=score,
        ),
        status=status,
        warnings=tuple(warnings),
        visual_refinement_applied=True,
        visual_refinement_method="features_ecc" if ecc_applied else "features",
    )
