"""Cross-record validation and explicit quarantine workflow."""

from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import TypeVar

from shapely.geometry.base import BaseGeometry

from nirman_netra.data.contracts import (
    ApprovedFootprint,
    Municipality,
    Parcel,
    PublicBoundary,
    QuarantineRecord,
    SyntheticDataset,
    ValidationStage,
    Ward,
)
from nirman_netra.exceptions import GeometryValidationError
from nirman_netra.geospatial import validate_geometry

RecordT = TypeVar("RecordT")


@dataclass(frozen=True)
class ValidationResult:
    dataset: SyntheticDataset
    quarantine: tuple[QuarantineRecord, ...]


def validate_dataset(
    dataset: SyntheticDataset, source_batch_id: str, timestamp: datetime
) -> ValidationResult:
    """Validate records, quarantine every rejected record, and keep relationships consistent."""

    invalid: defaultdict[str, set[str]] = defaultdict(set)
    quarantines: dict[str, QuarantineRecord] = {}
    geometries: dict[tuple[str, str], BaseGeometry] = {}

    def reject(entity: str, record_id: str, reason: str, stage: ValidationStage) -> bool:
        reference = f"{entity}:{record_id}"
        if reference in quarantines:
            return False
        invalid[entity].add(record_id)
        quarantines[reference] = QuarantineRecord(
            original_record_reference=reference,
            reason_code=reason,
            validation_stage=stage,
            timestamp=timestamp,
            source_batch_id=source_batch_id,
        )
        return True

    spatial_groups: tuple[
        tuple[
            str, Iterable[Municipality | Ward | Parcel | ApprovedFootprint | PublicBoundary], str
        ],
        ...,
    ] = (
        ("municipalities", dataset.municipalities, "municipality_id"),
        ("wards", dataset.wards, "ward_id"),
        ("parcels", dataset.parcels, "parcel_id"),
        ("approved_footprints", dataset.approved_footprints, "footprint_id"),
        ("public_boundaries", dataset.public_boundaries, "boundary_id"),
    )
    for entity, records, id_field in spatial_groups:
        for record in records:
            record_id = str(getattr(record, id_field))
            try:
                geometries[(entity, record_id)] = validate_geometry(record.geometry)
            except GeometryValidationError:
                reject(entity, record_id, "INVALID_GEOMETRY", "contract")

    for complaint in dataset.complaints:
        try:
            geometry = validate_geometry(complaint.coordinate)
            if geometry.geom_type != "Point":
                raise GeometryValidationError("complaint coordinate must be a Point")
            geometries[("complaints", complaint.complaint_id)] = geometry
        except GeometryValidationError:
            reject("complaints", complaint.complaint_id, "INVALID_COMPLAINT_COORDINATE", "contract")

    for case in dataset.inspection_cases:
        try:
            geometries[("inspection_cases", case.case_id)] = validate_geometry(
                case.observed_footprint
            )
        except GeometryValidationError:
            reject("inspection_cases", case.case_id, "INVALID_GEOMETRY", "contract")

    reference_date = timestamp.date()
    for permit in dataset.permits:
        if permit.valid_from > permit.valid_until:
            reject("permits", permit.permit_id, "INVALID_PERMIT_DATES", "contract")
        elif (
            permit.status == "active"
            and not permit.valid_from <= reference_date <= permit.valid_until
        ) or (permit.status == "expired" and permit.valid_until >= reference_date):
            reject("permits", permit.permit_id, "PERMIT_STATUS_DATE_MISMATCH", "contract")

    seen_hashes: dict[str, str] = {}
    for asset in dataset.imagery_assets:
        if asset.capture_time > asset.ingestion_time:
            reject("imagery_assets", asset.asset_id, "INVALID_CAPTURE_TIME", "contract")
        if asset.bounding_box.crs != asset.crs:
            reject("imagery_assets", asset.asset_id, "CRS_MISMATCH", "contract")
        if asset.content_sha256 in seen_hashes:
            reject("imagery_assets", asset.asset_id, "DUPLICATE_ASSET_HASH", "deduplication")
        else:
            seen_hashes[asset.content_sha256] = asset.asset_id

    municipalities = {record.municipality_id: record for record in dataset.municipalities}
    wards = {record.ward_id: record for record in dataset.wards}
    parcels = {record.parcel_id: record for record in dataset.parcels}
    buildings = {record.building_id: record for record in dataset.approved_buildings}
    footprints = {record.footprint_id: record for record in dataset.approved_footprints}
    permits = {record.permit_id: record for record in dataset.permits}
    assets = {record.asset_id: record for record in dataset.imagery_assets}
    complaints = {record.complaint_id: record for record in dataset.complaints}
    inspectors = {record.inspector_id: record for record in dataset.inspectors}

    changed = True
    while changed:
        changed = False
        for ward in dataset.wards:
            municipality = municipalities.get(ward.municipality_id)
            if municipality is None or municipality.municipality_id in invalid["municipalities"]:
                changed |= reject("wards", ward.ward_id, "MISSING_MUNICIPALITY", "relationship")
            elif ward.crs != municipality.crs:
                changed |= reject("wards", ward.ward_id, "CRS_MISMATCH", "relationship")
            elif ward.ward_id not in invalid["wards"] and not geometries[
                ("municipalities", municipality.municipality_id)
            ].covers(geometries[("wards", ward.ward_id)]):
                changed |= reject("wards", ward.ward_id, "OUTSIDE_MUNICIPALITY", "relationship")

        for parcel in dataset.parcels:
            parent_ward = wards.get(parcel.ward_id)
            if parent_ward is None or parent_ward.ward_id in invalid["wards"]:
                changed |= reject("parcels", parcel.parcel_id, "MISSING_WARD", "relationship")
            elif parcel.crs != parent_ward.crs:
                changed |= reject("parcels", parcel.parcel_id, "CRS_MISMATCH", "relationship")
            elif parcel.parcel_id not in invalid["parcels"] and not geometries[
                ("wards", parent_ward.ward_id)
            ].covers(geometries[("parcels", parcel.parcel_id)]):
                changed |= reject("parcels", parcel.parcel_id, "OUTSIDE_WARD", "relationship")

        for footprint in dataset.approved_footprints:
            parent_parcel = parcels.get(footprint.parcel_id)
            building = buildings.get(footprint.building_id)
            if parent_parcel is None or parent_parcel.parcel_id in invalid["parcels"]:
                changed |= reject(
                    "approved_footprints", footprint.footprint_id, "MISSING_PARCEL", "relationship"
                )
            elif building is None or building.building_id in invalid["approved_buildings"]:
                changed |= reject(
                    "approved_footprints",
                    footprint.footprint_id,
                    "MISSING_BUILDING",
                    "relationship",
                )
            elif footprint.crs != parent_parcel.crs:
                changed |= reject(
                    "approved_footprints", footprint.footprint_id, "CRS_MISMATCH", "relationship"
                )
            elif footprint.footprint_id not in invalid["approved_footprints"]:
                allowed = geometries[("parcels", parent_parcel.parcel_id)].buffer(
                    -parent_parcel.setback_m
                )
                if allowed.is_empty or not allowed.covers(
                    geometries[("approved_footprints", footprint.footprint_id)]
                ):
                    changed |= reject(
                        "approved_footprints",
                        footprint.footprint_id,
                        "BUILDING_OUTSIDE_PARCEL",
                        "relationship",
                    )

        for building in dataset.approved_buildings:
            building_parcel = parcels.get(building.parcel_id)
            building_footprint = footprints.get(building.footprint_id)
            if building_parcel is None or building_parcel.parcel_id in invalid["parcels"]:
                changed |= reject(
                    "approved_buildings", building.building_id, "MISSING_PARCEL", "relationship"
                )
            elif (
                building_footprint is None
                or building_footprint.footprint_id in invalid["approved_footprints"]
                or building_footprint.building_id != building.building_id
                or building_footprint.parcel_id != building.parcel_id
            ):
                changed |= reject(
                    "approved_buildings",
                    building.building_id,
                    "INVALID_FOOTPRINT_LINK",
                    "relationship",
                )

        for permit in dataset.permits:
            permit_parcel = parcels.get(permit.parcel_id)
            permit_footprint = footprints.get(permit.footprint_id)
            if permit_parcel is None or permit_parcel.parcel_id in invalid["parcels"]:
                changed |= reject("permits", permit.permit_id, "MISSING_PARCEL", "relationship")
            elif (
                permit_footprint is None
                or permit_footprint.footprint_id in invalid["approved_footprints"]
                or permit.plan_version != permit_footprint.plan_version
            ):
                changed |= reject("permits", permit.permit_id, "INVALID_PLAN_LINK", "relationship")

        for complaint in dataset.complaints:
            complaint_parcel = parcels.get(complaint.parcel_id)
            if complaint_parcel is None or complaint_parcel.parcel_id in invalid["parcels"]:
                changed |= reject(
                    "complaints", complaint.complaint_id, "MISSING_PARCEL", "relationship"
                )
            elif complaint.crs != complaint_parcel.crs:
                changed |= reject(
                    "complaints", complaint.complaint_id, "CRS_MISMATCH", "relationship"
                )
            elif complaint.complaint_id not in invalid["complaints"] and not geometries[
                ("parcels", complaint_parcel.parcel_id)
            ].covers(geometries[("complaints", complaint.complaint_id)]):
                changed |= reject(
                    "complaints", complaint.complaint_id, "OUTSIDE_PARCEL", "relationship"
                )

        for case in dataset.inspection_cases:
            missing = (
                case.parcel_id not in parcels
                or case.parcel_id in invalid["parcels"]
                or case.building_id not in buildings
                or case.building_id in invalid["approved_buildings"]
                or case.permit_id not in permits
                or case.permit_id in invalid["permits"]
                or case.inspector_id not in inspectors
                or any(
                    asset_id not in assets or asset_id in invalid["imagery_assets"]
                    for asset_id in case.imagery_asset_ids
                )
                or any(
                    complaint_id not in complaints or complaint_id in invalid["complaints"]
                    for complaint_id in case.complaint_ids
                )
            )
            if missing:
                changed |= reject(
                    "inspection_cases", case.case_id, "INVALID_CASE_RELATIONSHIP", "relationship"
                )

    def keep(
        records: tuple[RecordT, ...], entity: str, identifier: Callable[[RecordT], str]
    ) -> tuple[RecordT, ...]:
        return tuple(record for record in records if identifier(record) not in invalid[entity])

    valid_dataset = SyntheticDataset(
        municipalities=keep(
            dataset.municipalities, "municipalities", lambda item: item.municipality_id
        ),
        wards=keep(dataset.wards, "wards", lambda item: item.ward_id),
        parcels=keep(dataset.parcels, "parcels", lambda item: item.parcel_id),
        approved_buildings=keep(
            dataset.approved_buildings, "approved_buildings", lambda item: item.building_id
        ),
        approved_footprints=keep(
            dataset.approved_footprints, "approved_footprints", lambda item: item.footprint_id
        ),
        permits=keep(dataset.permits, "permits", lambda item: item.permit_id),
        imagery_assets=keep(dataset.imagery_assets, "imagery_assets", lambda item: item.asset_id),
        complaints=keep(dataset.complaints, "complaints", lambda item: item.complaint_id),
        inspectors=keep(dataset.inspectors, "inspectors", lambda item: item.inspector_id),
        inspection_cases=keep(
            dataset.inspection_cases, "inspection_cases", lambda item: item.case_id
        ),
        public_boundaries=keep(
            dataset.public_boundaries, "public_boundaries", lambda item: item.boundary_id
        ),
    )
    return ValidationResult(
        dataset=valid_dataset,
        quarantine=tuple(quarantines[key] for key in sorted(quarantines)),
    )
