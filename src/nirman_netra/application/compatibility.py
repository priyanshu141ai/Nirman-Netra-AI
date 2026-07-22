"""Strict requested model and municipal rule compatibility checks."""

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from nirman_netra.application.contracts import ModelRequirement, RuleSetRequirement
from nirman_netra.exceptions import (
    ArtifactValidationError,
    ModelChecksumMismatchError,
    ModelNotAvailableError,
    ModelSchemaMismatchError,
    RuleSetNotAvailableError,
)
from nirman_netra.risk.contracts import MunicipalRuleSet
from nirman_netra.segmentation.artifact import SegmentationInference, load_model_artifact


def load_compatible_model(
    artifact_directory: Path,
    requirement: ModelRequirement,
    *,
    source_crs: str,
    source_resolution_m: float,
) -> SegmentationInference:
    if not artifact_directory.is_dir():
        raise ModelNotAvailableError("requested model artifact is unavailable")
    try:
        inference = load_model_artifact(artifact_directory)
    except ArtifactValidationError as exc:
        if "checksum" in str(exc).casefold():
            raise ModelChecksumMismatchError("requested model checksum does not match") from exc
        raise ModelSchemaMismatchError("requested model artifact cannot be loaded") from exc
    metadata = inference.metadata
    expected = {
        "model_name": (metadata.model_name, requirement.model_name),
        "model_version": (metadata.model_version, requirement.model_version),
        "artifact_schema_version": (
            metadata.artifact_schema_version,
            requirement.artifact_schema_version,
        ),
        "training_dataset_version": (
            metadata.training_dataset_version,
            requirement.training_dataset_version,
        ),
        "feature_schema_version": (
            metadata.feature_schema_version,
            requirement.feature_schema_version,
        ),
        "input_schema": (metadata.input_schema, requirement.input_schema),
        "normalization": (metadata.normalization, requirement.normalization),
        "class_mapping": (metadata.class_mapping, requirement.class_mapping),
        "required_crs": (metadata.required_crs, requirement.required_crs),
        "supported_resolution_m": (
            metadata.supported_resolution_m,
            requirement.supported_resolution_m,
        ),
    }
    mismatches = tuple(name for name, values in expected.items() if values[0] != values[1])
    if mismatches:
        raise ModelSchemaMismatchError(
            f"model artifact compatibility mismatch: {','.join(mismatches)}"
        )
    if requirement.required_crs and source_crs not in requirement.required_crs:
        raise ModelSchemaMismatchError("source CRS is not supported by the requested model")
    minimum, maximum = requirement.supported_resolution_m
    if not minimum <= source_resolution_m <= maximum:
        raise ModelSchemaMismatchError("source resolution is not supported by the requested model")
    return inference


def require_compatible_rule_set(
    rule_sets: Sequence[MunicipalRuleSet],
    requirement: RuleSetRequirement,
    *,
    capture_time: datetime,
) -> MunicipalRuleSet:
    if capture_time.tzinfo is None or capture_time.utcoffset() is None:
        raise RuleSetNotAvailableError("rule selection capture time must be timezone-aware")
    matching = tuple(rule for rule in rule_sets if rule.rule_set_id == requirement.rule_set_id)
    if len(matching) != 1:
        raise RuleSetNotAvailableError("requested municipal rule set is unavailable")
    rule = matching[0]
    capture_date = capture_time.date()
    compatible = (
        rule.municipality_id == requirement.municipality_id
        and rule.zone == requirement.zone
        and rule.rule_version == requirement.rule_version
        and rule.risk_threshold_version == requirement.risk_threshold_version
        and rule.effective_from <= capture_date
        and (rule.effective_until is None or capture_date <= rule.effective_until)
    )
    if not compatible:
        raise RuleSetNotAvailableError("requested municipal rule set is incompatible")
    return rule
