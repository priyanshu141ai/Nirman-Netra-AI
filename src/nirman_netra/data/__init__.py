"""Synthetic municipal data contracts and generation."""

from nirman_netra.data.contracts import GenerationConfig, Scenario, SyntheticDataset
from nirman_netra.data.generator import generate_dataset

__all__ = ["GenerationConfig", "Scenario", "SyntheticDataset", "generate_dataset"]
