"""Deterministic Parquet, GeoJSON, and manifest persistence."""

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel

from nirman_netra.data.contracts import DatasetManifest, GenerationConfig, SyntheticDataset
from nirman_netra.data.validation import ValidationResult
from nirman_netra.exceptions import StorageError
from nirman_netra.utils import content_hash

_QUARANTINE_SCHEMA = pa.schema(
    [
        ("original_record_reference", pa.string()),
        ("reason_code", pa.string()),
        ("validation_stage", pa.string()),
        ("timestamp", pa.string()),
        ("source_batch_id", pa.string()),
    ]
)


def _tables(dataset: SyntheticDataset) -> dict[str, tuple[BaseModel, ...]]:
    return {
        "municipalities": dataset.municipalities,
        "wards": dataset.wards,
        "parcels": dataset.parcels,
        "approved_buildings": dataset.approved_buildings,
        "approved_footprints": dataset.approved_footprints,
        "permits": dataset.permits,
        "imagery_assets": dataset.imagery_assets,
        "complaints": dataset.complaints,
        "inspectors": dataset.inspectors,
        "inspection_cases": dataset.inspection_cases,
        "public_boundaries": dataset.public_boundaries,
    }


def _rows(records: tuple[BaseModel, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        row = record.model_dump(mode="json")
        rows.append(
            {
                key: json.dumps(value, sort_keys=True, separators=(",", ":"))
                if isinstance(value, dict | list)
                else value
                for key, value in row.items()
            }
        )
    return rows


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _geojson(dataset: SyntheticDataset) -> dict[str, object]:
    features: list[dict[str, object]] = []
    groups = (
        ("municipality", dataset.municipalities, "municipality_id"),
        ("ward", dataset.wards, "ward_id"),
        ("parcel", dataset.parcels, "parcel_id"),
        ("approved_footprint", dataset.approved_footprints, "footprint_id"),
        ("public_boundary", dataset.public_boundaries, "boundary_id"),
    )
    for record_type, records, identifier in groups:
        for record in records:
            features.append(
                {
                    "type": "Feature",
                    "id": getattr(record, identifier),
                    "geometry": record.geometry,
                    "properties": {
                        "record_type": record_type,
                        "crs": record.crs.value,
                    },
                }
            )
    return {"type": "FeatureCollection", "features": features}


def write_catalogue(
    result: ValidationResult, config: GenerationConfig, output_directory: Path
) -> DatasetManifest:
    """Write deterministic machine-readable outputs and their checksums."""

    try:
        output_directory.mkdir(parents=True, exist_ok=True)
        output_paths: dict[str, str] = {}
        written: list[Path] = []
        tables = _tables(result.dataset)
        for name, records in tables.items():
            path = output_directory / f"{name}.parquet"
            pq.write_table(  # type: ignore[no-untyped-call]
                pa.Table.from_pylist(_rows(records)), path, compression="zstd"
            )
            output_paths[name] = path.name
            written.append(path)

        quarantine_path = output_directory / "quarantine.parquet"
        pq.write_table(  # type: ignore[no-untyped-call]
            pa.Table.from_pylist(_rows(result.quarantine), schema=_QUARANTINE_SCHEMA),
            quarantine_path,
            compression="zstd",
        )
        output_paths["quarantine"] = quarantine_path.name
        written.append(quarantine_path)

        geojson_path = output_directory / "spatial.geojson"
        _write_json(geojson_path, _geojson(result.dataset))
        output_paths["spatial"] = geojson_path.name
        written.append(geojson_path)

        config_path = output_directory / "generation_config.json"
        _write_json(config_path, config.model_dump(mode="json"))
        output_paths["configuration"] = config_path.name
        written.append(config_path)

        output_paths["manifest"] = "dataset_manifest.json"
        checksums = {
            path.name: content_hash(path.read_bytes())
            for path in sorted(written, key=lambda item: item.name)
        }
        quarantine_counts = dict(
            sorted(Counter(record.reason_code for record in result.quarantine).items())
        )
        manifest = DatasetManifest(
            dataset_version="2.0.0",
            generator_version=config.generator_version,
            scenario=config.scenario,
            random_seed=config.random_seed,
            configuration_hash=config.configuration_hash,
            entity_counts={name: len(records) for name, records in tables.items()},
            quarantine_counts=quarantine_counts,
            output_paths=output_paths,
            checksums=checksums,
        )
        _write_json(output_directory / output_paths["manifest"], manifest.model_dump(mode="json"))
        return manifest
    except (OSError, pa.ArrowException) as exc:
        raise StorageError(f"failed to write synthetic catalogue: {output_directory}") from exc
