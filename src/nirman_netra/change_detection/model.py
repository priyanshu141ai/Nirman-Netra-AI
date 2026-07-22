"""Single lightweight Siamese candidate for bitemporal change detection."""

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from nirman_netra.change_detection.contracts import ChangeLabel, SiameseTrainingConfig
from nirman_netra.change_detection.dataset import ChangeSample, validate_change_dataset
from nirman_netra.exceptions import RegistrationError, TrainingError
from nirman_netra.segmentation.contracts import GeographicSplitConfig, SplitAssignment


def _shared_difference_features(
    old_image: NDArray[np.uint8],
    new_image: NDArray[np.uint8],
    old_mask: NDArray[np.uint8],
    new_mask: NDArray[np.uint8],
) -> NDArray[np.float32]:
    if old_image.ndim != 3 or old_image.shape != new_image.shape:
        raise TrainingError("Siamese inputs must have matching HWC shapes")
    if old_mask.shape != old_image.shape[:2] or new_mask.shape != old_mask.shape:
        raise TrainingError("segmentation inputs must match the Siamese image shape")
    old_branch = old_image.astype(np.float32) / 255.0
    new_branch = new_image.astype(np.float32) / 255.0
    visual_difference = np.abs(new_branch - old_branch)
    segmentation_difference = np.abs(new_mask.astype(np.float32) - old_mask.astype(np.float32))[
        ..., np.newaxis
    ]
    return cast(
        NDArray[np.float32],
        np.concatenate((visual_difference, segmentation_difference), axis=2),
    )


@dataclass(frozen=True)
class SiameseLinearChangeModel:
    """Shared image encoder with an absolute-difference logistic head."""

    weights: NDArray[np.float32]
    bias: float
    threshold: float = 0.5
    minimum_registration_score: float = 0.25

    def predict_probabilities(
        self,
        old_image: NDArray[np.uint8],
        new_image: NDArray[np.uint8],
        old_mask: NDArray[np.uint8],
        new_mask: NDArray[np.uint8],
    ) -> NDArray[np.float32]:
        features = _shared_difference_features(old_image, new_image, old_mask, new_mask)
        if features.shape[2] != len(self.weights):
            raise TrainingError("Siamese feature channels do not match model weights")
        logits = features @ self.weights + self.bias
        return cast(
            NDArray[np.float32],
            (1 / (1 + np.exp(-np.clip(logits, -30, 30)))).astype(np.float32),
        )

    def predict(self, sample: ChangeSample) -> NDArray[np.uint8]:
        registration = sample.metadata.registration_metrics
        if (
            not registration.transform_plausible
            or registration.overlap_ratio <= 0
            or registration.registration_quality_score < self.minimum_registration_score
        ):
            raise RegistrationError("registration quality is too low for Siamese inference")
        probabilities = self.predict_probabilities(
            sample.old_image,
            sample.new_image,
            sample.old_building_mask,
            sample.new_building_mask,
        )
        return cast(NDArray[np.uint8], (probabilities >= self.threshold).astype(np.uint8))

    def predict_labels(self, sample: ChangeSample) -> NDArray[np.uint8]:
        return np.where(self.predict(sample), int(ChangeLabel.UNCERTAIN_CHANGE), 0).astype(np.uint8)


def train_siamese_candidate(
    samples: tuple[ChangeSample, ...],
    split_config: GeographicSplitConfig,
    config: SiameseTrainingConfig,
) -> SiameseLinearChangeModel:
    """Train only the declared geographic training split, deterministically when configured."""

    validate_change_dataset(samples, split_config)
    training = tuple(sample for sample in samples if sample.metadata.split == SplitAssignment.TRAIN)
    if not training:
        raise TrainingError("Siamese candidate requires a geographic training split")
    channels = training[0].old_image.shape[2]
    if any(sample.old_image.shape[2] != channels for sample in training):
        raise TrainingError("Siamese training images must have consistent channels")
    feature_batches = [
        _shared_difference_features(
            sample.old_image,
            sample.new_image,
            sample.old_building_mask,
            sample.new_building_mask,
        ).reshape(-1, channels + 1)
        for sample in training
    ]
    target_batches = [
        (sample.change_mask.reshape(-1) > 0).astype(np.float32) for sample in training
    ]
    features = np.concatenate(feature_batches)
    targets = np.concatenate(target_batches)
    if not np.any(targets) or np.all(targets):
        raise TrainingError("Siamese training requires changed and unchanged pixels")
    weights = np.zeros(channels + 1, dtype=np.float32)
    bias = 0.0
    positive_weight = float(np.count_nonzero(targets == 0) / np.count_nonzero(targets == 1))
    sample_weights = np.where(targets > 0, positive_weight, 1.0).astype(np.float32)
    rng = np.random.default_rng(config.random_seed if config.deterministic else None)
    for _epoch in range(config.epochs):
        order = rng.permutation(len(targets))
        epoch_features = features[order]
        epoch_targets = targets[order]
        epoch_weights = sample_weights[order]
        probabilities = 1 / (1 + np.exp(-np.clip(epoch_features @ weights + bias, -30, 30)))
        error = (probabilities - epoch_targets) * epoch_weights
        denominator = float(epoch_weights.sum())
        weights -= config.learning_rate * (epoch_features.T @ error / denominator)
        bias -= config.learning_rate * float(error.sum() / denominator)
    return SiameseLinearChangeModel(
        weights=weights,
        bias=bias,
        threshold=config.threshold,
        minimum_registration_score=config.minimum_registration_score,
    )
