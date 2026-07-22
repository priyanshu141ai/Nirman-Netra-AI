"""Deterministic NumPy per-pixel logistic segmentation baseline."""

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from nirman_netra.exceptions import TrainingError
from nirman_netra.segmentation.augmentations import augment_sample
from nirman_netra.segmentation.contracts import (
    GeographicSplitConfig,
    Normalization,
    SplitAssignment,
    TrainingConfig,
    TrainingHistory,
)
from nirman_netra.segmentation.dataset import SegmentationSample, validate_dataset


@dataclass(frozen=True)
class LinearSegmentationModel:
    weights: NDArray[np.float32]
    bias: float
    normalization: Normalization

    def predict_probabilities(self, image: NDArray[np.uint8]) -> NDArray[np.float32]:
        if image.ndim != 3 or image.shape[2] != len(self.weights):
            raise TrainingError("image channels do not match baseline weights")
        normalized = image.astype(np.float32) / 255.0
        mean = np.asarray(self.normalization.mean, dtype=np.float32)
        std = np.asarray(self.normalization.std, dtype=np.float32)
        logits = ((normalized - mean) / std) @ self.weights + self.bias
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        return cast(NDArray[np.float32], probabilities.astype(np.float32))

    def predict(self, image: NDArray[np.uint8], threshold: float = 0.5) -> NDArray[np.uint8]:
        return cast(
            NDArray[np.uint8], (self.predict_probabilities(image) >= threshold).astype(np.uint8)
        )


@dataclass(frozen=True)
class TrainingResult:
    model: LinearSegmentationModel
    history: TrainingHistory


def _normalization(samples: tuple[SegmentationSample, ...]) -> Normalization:
    pixels = np.concatenate(
        [
            sample.image.reshape(-1, sample.image.shape[2]).astype(np.float32) / 255.0
            for sample in samples
        ]
    )
    mean = pixels.mean(axis=0)
    std = pixels.std(axis=0)
    std[std < 1e-6] = 1.0
    return Normalization(
        mean=tuple(float(value) for value in mean),
        std=tuple(float(value) for value in std),
    )


def _loss(model: LinearSegmentationModel, samples: tuple[SegmentationSample, ...]) -> float:
    losses: list[float] = []
    for sample in samples:
        probability = np.clip(model.predict_probabilities(sample.image), 1e-7, 1 - 1e-7)
        target = sample.mask.astype(np.float32)
        losses.append(
            float(np.mean(-(target * np.log(probability) + (1 - target) * np.log(1 - probability))))
        )
    return float(np.mean(losses))


def _save_checkpoint(path: Path, weights: NDArray[np.float32], bias: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as destination:
        np.save(destination, np.append(weights, np.float32(bias)), allow_pickle=False)


def train_baseline(
    samples: tuple[SegmentationSample, ...],
    split_config: GeographicSplitConfig,
    config: TrainingConfig,
) -> TrainingResult:
    """Train one deterministic CPU logistic baseline with early stopping."""

    validate_dataset(samples, split_config)
    train_samples = tuple(
        sample for sample in samples if sample.metadata.split == SplitAssignment.TRAIN
    )
    validation_samples = tuple(
        sample for sample in samples if sample.metadata.split == SplitAssignment.VALIDATION
    )
    if not train_samples or not validation_samples:
        raise TrainingError("training and validation splits are both required")
    shape = train_samples[0].image.shape
    if any(sample.image.shape != shape for sample in samples):
        raise TrainingError("baseline requires a consistent input shape")
    normalization = _normalization(train_samples)
    channels = shape[2]
    weights = np.zeros(channels, dtype=np.float32)
    bias = 0.0
    rng = np.random.default_rng(config.random_seed if config.deterministic else None)
    best_loss = float("inf")
    patience = 0
    train_losses: list[float] = []
    validation_losses: list[float] = []

    for _epoch in range(config.epochs):
        order = rng.permutation(len(train_samples))
        for start in range(0, len(order), config.batch_size):
            batch = [
                augment_sample(train_samples[index], config.augmentation, rng)
                for index in order[start : start + config.batch_size]
            ]
            images = np.concatenate(
                [item.image.reshape(-1, channels).astype(np.float32) / 255.0 for item in batch]
            )
            targets = np.concatenate([item.mask.reshape(-1).astype(np.float32) for item in batch])
            mean = np.asarray(normalization.mean, dtype=np.float32)
            std = np.asarray(normalization.std, dtype=np.float32)
            features = (images - mean) / std
            probabilities = 1.0 / (1.0 + np.exp(-np.clip(features @ weights + bias, -30, 30)))
            error = probabilities - targets
            weights -= config.learning_rate * (features.T @ error / len(targets))
            bias -= config.learning_rate * float(np.mean(error))

        model = LinearSegmentationModel(
            weights=weights.copy(), bias=bias, normalization=normalization
        )
        train_loss = _loss(model, train_samples)
        validation_loss = _loss(model, validation_samples)
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)
        if validation_loss < best_loss - config.early_stopping_min_delta:
            best_loss = validation_loss
            patience = 0
            _save_checkpoint(config.checkpoint_path, weights, bias)
        else:
            patience += 1
            if patience >= config.early_stopping_patience:
                break

    if not config.checkpoint_path.is_file():
        raise TrainingError("training did not produce a checkpoint")
    with config.checkpoint_path.open("rb") as source:
        parameters = np.load(source, allow_pickle=False)
    final_model = LinearSegmentationModel(
        weights=cast(NDArray[np.float32], parameters[:-1].astype(np.float32)),
        bias=float(parameters[-1]),
        normalization=normalization,
    )
    return TrainingResult(
        model=final_model,
        history=TrainingHistory(
            train_loss=tuple(train_losses),
            validation_loss=tuple(validation_losses),
            epochs_completed=len(train_losses),
        ),
    )
