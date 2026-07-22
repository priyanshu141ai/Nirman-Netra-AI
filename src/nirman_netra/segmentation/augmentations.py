"""Controlled paired image-mask augmentations with pixel transform tracking."""

from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from nirman_netra.segmentation.contracts import AugmentationConfig
from nirman_netra.segmentation.dataset import SegmentationSample


@dataclass(frozen=True)
class AugmentedSample:
    image: NDArray[np.uint8]
    mask: NDArray[np.uint8]
    augmented_pixel_to_source_pixel: NDArray[np.float64]


def augment_sample(
    sample: SegmentationSample,
    config: AugmentationConfig,
    rng: np.random.Generator,
) -> AugmentedSample:
    """Apply a paired affine augmentation and retain its inverse pixel mapping."""

    height, width = sample.mask.shape
    source_to_augmented = np.eye(3, dtype=np.float64)
    if rng.random() < config.horizontal_flip_probability:
        source_to_augmented = (
            np.array([[-1.0, 0.0, width - 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            @ source_to_augmented
        )
    if rng.random() < config.vertical_flip_probability:
        source_to_augmented = (
            np.array([[1.0, 0.0, 0.0], [0.0, -1.0, height - 1.0], [0.0, 0.0, 1.0]])
            @ source_to_augmented
        )
    angle = rng.uniform(-config.maximum_rotation_degrees, config.maximum_rotation_degrees)
    scale = rng.uniform(config.scale_minimum, config.scale_maximum)
    rotation = cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), angle, scale)
    rotation_homogeneous = np.vstack((rotation, np.array([0.0, 0.0, 1.0])))
    source_to_augmented = rotation_homogeneous @ source_to_augmented
    affine = source_to_augmented[:2].astype(np.float32)
    image = cast(
        NDArray[np.uint8],
        cv2.warpAffine(
            sample.image,
            affine,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        ),
    )
    mask = cast(
        NDArray[np.uint8],
        cv2.warpAffine(
            sample.mask,
            affine,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ),
    )
    brightness = rng.uniform(-config.brightness_delta, config.brightness_delta)
    contrast = rng.uniform(1 - config.contrast_delta, 1 + config.contrast_delta)
    adjusted = np.clip(
        ((image.astype(np.float32) / 255.0 - 0.5) * contrast + 0.5 + brightness) * 255.0,
        0,
        255,
    ).astype(np.uint8)
    return AugmentedSample(
        image=cast(NDArray[np.uint8], adjusted),
        mask=mask,
        augmented_pixel_to_source_pixel=np.linalg.inv(source_to_augmented).astype(np.float64),
    )
