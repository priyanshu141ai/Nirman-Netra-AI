"""Deterministic synthetic municipal dataset generator."""

import random
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

from shapely import affinity
from shapely.geometry import LineString, box, mapping, shape
from shapely.geometry.base import BaseGeometry

from nirman_netra.data.contracts import (
    ApprovedBuilding,
    ApprovedFootprint,
    Complaint,
    DatasetManifest,
    GenerationConfig,
    ImageryAsset,
    InspectionCaseSeed,
    Inspector,
    Municipality,
    Parcel,
    Permit,
    PublicBoundary,
    Scenario,
    SyntheticDataset,
    Ward,
)
from nirman_netra.data.validation import validate_dataset
from nirman_netra.domain import BoundingBox
from nirman_netra.persistence.catalogue import write_catalogue
from nirman_netra.utils import content_hash, deterministic_id


def _id(kind: str, config: GenerationConfig, index: int = 0) -> str:
    return deterministic_id(f"nirman-netra:{kind}", config.configuration_hash, str(index))


def _geometry(value: BaseGeometry) -> dict[str, Any]:
    return cast(dict[str, Any], dict(mapping(value)))


def _bounds(value: BaseGeometry, config: GenerationConfig) -> BoundingBox:
    min_x, min_y, max_x, max_y = value.bounds
    return BoundingBox(
        min_x=min_x,
        min_y=min_y,
        max_x=max_x,
        max_y=max_y,
        crs=config.processing_crs,
    )


def build_synthetic_dataset(config: GenerationConfig) -> SyntheticDataset:
    """Build valid, in-memory records for one deterministic scenario."""

    rng = random.Random(config.random_seed)
    size = config.municipality_size_m
    municipality_shape = box(
        config.origin_x,
        config.origin_y,
        config.origin_x + size,
        config.origin_y + size,
    )
    municipality_id = _id("municipality", config)
    municipality = Municipality(
        municipality_id=municipality_id,
        code="synthetic-municipality",
        geometry=_geometry(municipality_shape),
        crs=config.processing_crs,
    )

    wards: list[Ward] = []
    parcels: list[Parcel] = []
    buildings: list[ApprovedBuilding] = []
    footprints: list[ApprovedFootprint] = []
    permits: list[Permit] = []
    ward_width = size / config.ward_count
    parcel_height = size / config.parcels_per_ward
    record_index = 0

    for ward_index in range(config.ward_count):
        ward_shape = box(
            config.origin_x + ward_index * ward_width,
            config.origin_y,
            config.origin_x + (ward_index + 1) * ward_width,
            config.origin_y + size,
        )
        ward_id = _id("ward", config, ward_index)
        wards.append(
            Ward(
                ward_id=ward_id,
                municipality_id=municipality_id,
                geometry=_geometry(ward_shape),
                crs=config.processing_crs,
            )
        )
        for parcel_index in range(config.parcels_per_ward):
            parcel_shape = box(
                ward_shape.bounds[0] + config.parcel_margin_m,
                config.origin_y + parcel_index * parcel_height + config.parcel_margin_m,
                ward_shape.bounds[2] - config.parcel_margin_m,
                config.origin_y + (parcel_index + 1) * parcel_height - config.parcel_margin_m,
            )
            parcel_id = _id("parcel", config, record_index)
            building_id = _id("building", config, record_index)
            footprint_id = _id("footprint", config, record_index)
            permit_id = _id("permit", config, record_index)
            parcels.append(
                Parcel(
                    parcel_id=parcel_id,
                    ward_id=ward_id,
                    setback_m=config.setback_m,
                    geometry=_geometry(parcel_shape),
                    crs=config.processing_crs,
                )
            )

            parcel_width = parcel_shape.bounds[2] - parcel_shape.bounds[0]
            usable_height = parcel_shape.bounds[3] - parcel_shape.bounds[1]
            centre_x = parcel_shape.centroid.x + rng.uniform(-0.04, 0.04) * parcel_width
            centre_y = parcel_shape.centroid.y + rng.uniform(-0.04, 0.04) * usable_height
            footprint_shape = box(
                centre_x - parcel_width * 0.18,
                centre_y - usable_height * 0.18,
                centre_x + parcel_width * 0.18,
                centre_y + usable_height * 0.18,
            )
            plan_version = (
                2 if record_index == 0 and config.scenario == Scenario.APPROVED_EXTENSION else 1
            )
            if plan_version == 2:
                footprint_shape = footprint_shape.buffer(min(4.0, config.setback_m / 2))
            footprints.append(
                ApprovedFootprint(
                    footprint_id=footprint_id,
                    building_id=building_id,
                    parcel_id=parcel_id,
                    plan_version=plan_version,
                    geometry=_geometry(footprint_shape),
                    crs=config.processing_crs,
                )
            )
            buildings.append(
                ApprovedBuilding(
                    building_id=building_id,
                    parcel_id=parcel_id,
                    footprint_id=footprint_id,
                    approved_floor_count=2,
                )
            )
            is_expired = record_index == 0 and config.scenario == Scenario.EXPIRED_PERMIT
            valid_from = (config.reference_time - timedelta(days=180 if is_expired else 30)).date()
            valid_until = (
                (config.reference_time - timedelta(days=30)).date()
                if is_expired
                else (config.reference_time + timedelta(days=90)).date()
            )
            permits.append(
                Permit(
                    permit_id=permit_id,
                    parcel_id=parcel_id,
                    footprint_id=footprint_id,
                    plan_version=plan_version,
                    permit_type="extension",
                    valid_from=valid_from,
                    valid_until=valid_until,
                    status="expired" if is_expired else "active",
                    applicable_conditions=("synthetic-plan-conformance",),
                )
            )
            record_index += 1

    target_parcel = parcels[0]
    target_building = buildings[0]
    target_footprint = footprints[0]
    target_permit = permits[0]
    approved_shape = shape(target_footprint.geometry)
    observed_shape: BaseGeometry = approved_shape
    observed_floors = target_building.approved_floor_count
    if config.scenario == Scenario.UNAPPROVED_FOOTPRINT_EXPANSION:
        observed_shape = approved_shape.buffer(8)
    elif config.scenario == Scenario.POSSIBLE_NEW_FLOOR:
        observed_floors += 1
    elif config.scenario == Scenario.PARTIAL_DEMOLITION:
        observed_shape = affinity.scale(approved_shape, xfact=0.65, yfact=0.65)
    elif config.scenario == Scenario.PARCEL_ENCROACHMENT:
        target_parcel_shape = shape(target_parcel.geometry)
        footprint_width = approved_shape.bounds[2] - approved_shape.bounds[0]
        observed_shape = affinity.translate(
            approved_shape,
            xoff=target_parcel_shape.bounds[2] - approved_shape.bounds[2] + footprint_width / 2,
        )

    imagery_assets = tuple(
        ImageryAsset(
            asset_id=_id("imagery", config, index),
            source="synthetic-catalogue",
            capture_time=config.reference_time - timedelta(days=60 - index * 59),
            ingestion_time=config.reference_time - timedelta(days=59 - index * 59),
            crs=config.processing_crs,
            resolution_m=(
                2.5
                if index == 1 and config.scenario == Scenario.LOW_QUALITY_IMAGERY_METADATA
                else 0.5
            ),
            bounding_box=_bounds(municipality_shape, config),
            storage_uri=f"synthetic://imagery/{_id('imagery', config, index)}.tif",
            content_sha256=content_hash(f"{config.configuration_hash}:imagery:{index}".encode()),
            licence_reference="synthetic-use-only",
            quality_status=(
                "low_quality"
                if index == 1 and config.scenario == Scenario.LOW_QUALITY_IMAGERY_METADATA
                else "accepted"
            ),
        )
        for index in range(2)
    )

    complaint_point = shape(target_parcel.geometry).centroid
    complaints = [
        Complaint(
            complaint_id=_id("complaint", config),
            parcel_id=target_parcel.parcel_id,
            coordinate=_geometry(complaint_point),
            crs=config.processing_crs,
            category="construction_change",
            received_at=config.reference_time - timedelta(days=2),
            description="possible exterior footprint change",
            media_sha256=content_hash(f"{config.configuration_hash}:complaint-media".encode()),
            perceptual_hash="0f0f0f0f0f0f0f0f",
        )
    ]
    if config.scenario == Scenario.DUPLICATE_COMPLAINTS:
        complaints.append(
            complaints[0].model_copy(update={"complaint_id": _id("complaint", config, 1)})
        )

    inspector = Inspector(inspector_id=_id("inspector", config), team_code="synthetic-team")
    case = InspectionCaseSeed(
        case_id=_id("case", config),
        scenario=config.scenario,
        parcel_id=target_parcel.parcel_id,
        building_id=target_building.building_id,
        permit_id=target_permit.permit_id,
        imagery_asset_ids=tuple(asset.asset_id for asset in imagery_assets),
        complaint_ids=tuple(complaint.complaint_id for complaint in complaints),
        inspector_id=inspector.inspector_id,
        observed_footprint=_geometry(observed_shape),
        observed_floor_count=observed_floors,
    )
    boundaries = (
        PublicBoundary(
            boundary_id=_id("boundary", config),
            boundary_type="municipal_boundary",
            geometry=_geometry(municipality_shape.boundary),
            crs=config.processing_crs,
        ),
        PublicBoundary(
            boundary_id=_id("boundary", config, 1),
            boundary_type="road_edge",
            geometry=_geometry(
                LineString(
                    [
                        (config.origin_x + size / 2, config.origin_y),
                        (config.origin_x + size / 2, config.origin_y + size),
                    ]
                )
            ),
            crs=config.processing_crs,
        ),
    )
    return SyntheticDataset(
        municipalities=(municipality,),
        wards=tuple(wards),
        parcels=tuple(parcels),
        approved_buildings=tuple(buildings),
        approved_footprints=tuple(footprints),
        permits=tuple(permits),
        imagery_assets=imagery_assets,
        complaints=tuple(complaints),
        inspectors=(inspector,),
        inspection_cases=(case,),
        public_boundaries=boundaries,
    )


def generate_dataset(config: GenerationConfig, output_directory: Path) -> DatasetManifest:
    """Idempotently validate and persist a deterministic dataset."""

    dataset = build_synthetic_dataset(config)
    batch_id = _id("batch", config)
    result = validate_dataset(dataset, batch_id, config.reference_time)
    return write_catalogue(result, config, output_directory)
