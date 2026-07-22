from datetime import timedelta
from pathlib import Path

import pytest
from shapely import affinity
from shapely.geometry import mapping, shape

from nirman_netra.data.contracts import GenerationConfig, Scenario, SyntheticDataset
from nirman_netra.data.generator import build_synthetic_dataset, generate_dataset
from nirman_netra.data.validation import ValidationResult, validate_dataset
from nirman_netra.domain import CoordinateReference


def _config(scenario: Scenario = Scenario.VALID_ACTIVE_PERMIT, seed: int = 41) -> GenerationConfig:
    return GenerationConfig(
        scenario=scenario,
        random_seed=seed,
        processing_crs=CoordinateReference(value="EPSG:32643"),
    )


def _validate(dataset: SyntheticDataset, config: GenerationConfig) -> ValidationResult:
    return validate_dataset(dataset, "source-batch-001", config.reference_time)


def test_same_seed_produces_same_data_and_manifest(tmp_path: Path) -> None:
    config = _config()

    first = generate_dataset(config, tmp_path / "first")
    second = generate_dataset(config, tmp_path / "second")

    assert first == second
    assert first.checksums == second.checksums
    assert all((tmp_path / "first" / path).is_file() for path in first.output_paths.values())


def test_different_scenarios_change_observed_geometry() -> None:
    active = build_synthetic_dataset(_config())
    demolition = build_synthetic_dataset(_config(Scenario.PARTIAL_DEMOLITION))

    assert active.inspection_cases[0].observed_footprint != (
        demolition.inspection_cases[0].observed_footprint
    )


@pytest.mark.parametrize("scenario", list(Scenario))
def test_all_scenarios_generate_valid_contracts(scenario: Scenario) -> None:
    config = _config(scenario)
    result = _validate(build_synthetic_dataset(config), config)

    assert result.quarantine == ()


def test_invalid_geometry_is_quarantined() -> None:
    config = _config()
    dataset = build_synthetic_dataset(config)
    invalid = dataset.approved_footprints[0].model_copy(
        update={
            "geometry": {
                "type": "Polygon",
                "coordinates": [[(0, 0), (2, 2), (0, 2), (2, 0), (0, 0)]],
            }
        }
    )
    changed = dataset.model_copy(
        update={"approved_footprints": (invalid, *dataset.approved_footprints[1:])}
    )

    result = _validate(changed, config)

    quarantined = next(item for item in result.quarantine if item.reason_code == "INVALID_GEOMETRY")
    assert quarantined.original_record_reference.startswith("approved_footprints:")
    assert quarantined.validation_stage == "contract"
    assert quarantined.timestamp.tzinfo is not None
    assert quarantined.source_batch_id == "source-batch-001"


def test_building_outside_parcel_is_detected() -> None:
    config = _config()
    dataset = build_synthetic_dataset(config)
    footprint = dataset.approved_footprints[0]
    outside = affinity.translate(shape(footprint.geometry), xoff=config.municipality_size_m)
    changed_footprint = footprint.model_copy(update={"geometry": dict(mapping(outside))})
    changed = dataset.model_copy(
        update={"approved_footprints": (changed_footprint, *dataset.approved_footprints[1:])}
    )

    result = _validate(changed, config)

    assert any(item.reason_code == "BUILDING_OUTSIDE_PARCEL" for item in result.quarantine)


def test_invalid_permit_dates_are_detected() -> None:
    config = _config()
    dataset = build_synthetic_dataset(config)
    permit = dataset.permits[0]
    invalid = permit.model_copy(
        update={
            "valid_from": permit.valid_until + timedelta(days=1),
            "valid_until": permit.valid_from,
        }
    )
    changed = dataset.model_copy(update={"permits": (invalid, *dataset.permits[1:])})

    result = _validate(changed, config)

    assert any(item.reason_code == "INVALID_PERMIT_DATES" for item in result.quarantine)


def test_duplicate_asset_hash_is_detected() -> None:
    config = _config()
    dataset = build_synthetic_dataset(config)
    duplicate = dataset.imagery_assets[1].model_copy(
        update={"content_sha256": dataset.imagery_assets[0].content_sha256}
    )
    changed = dataset.model_copy(update={"imagery_assets": (dataset.imagery_assets[0], duplicate)})

    result = _validate(changed, config)

    assert any(item.reason_code == "DUPLICATE_ASSET_HASH" for item in result.quarantine)


def test_no_invalid_record_is_silently_dropped() -> None:
    config = _config()
    dataset = build_synthetic_dataset(config)
    permit = dataset.permits[0].model_copy(
        update={"valid_from": dataset.permits[0].valid_until + timedelta(days=1)}
    )
    changed = dataset.model_copy(update={"permits": (permit, *dataset.permits[1:])})

    result = _validate(changed, config)

    assert changed.record_count - result.dataset.record_count == len(result.quarantine)
    assert all(item.reason_code for item in result.quarantine)
