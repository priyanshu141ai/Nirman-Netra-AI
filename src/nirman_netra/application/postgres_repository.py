"""Durable PostgreSQL/PostGIS adapter for integrated application records."""

import json
from collections import defaultdict
from datetime import UTC, datetime, time
from typing import Any

from pydantic import ValidationError
from pyproj import CRS
from shapely.geometry import mapping, shape
from shapely.ops import unary_union
from sqlalchemy import Connection, Engine, create_engine, func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError

from nirman_netra.application.contracts import (
    AssetRecord,
    ChangeResultRecord,
    ImagePairRecord,
    MunicipalContext,
    ProcessingJob,
    ProcessPairRequest,
)
from nirman_netra.application.repository import InMemoryIntegrationRepository
from nirman_netra.cases.contracts import EvidenceAsset, InspectionCase
from nirman_netra.exceptions import CRSMismatchError, PersistenceError
from nirman_netra.persistence.schema import (
    approved_buildings,
    approved_plans,
    audit_events,
    change_results,
    complaint_groups,
    complaints,
    evidence_assets,
    evidence_events,
    image_pairs,
    imagery_assets,
    inspection_cases,
    municipalities,
    parcels,
    permits,
    processing_jobs,
    risk_results,
    wards,
)

JsonObject = dict[str, Any]


class PostgresIntegrationRepository(InMemoryIntegrationRepository):
    """Keep a validated memory index backed by transactional PostgreSQL records."""

    def __init__(self, database_url: str, *, geometry_srid: int) -> None:
        super().__init__()
        self._engine: Engine = create_engine(database_url, pool_pre_ping=True)
        self._geometry_srid = geometry_srid
        self._loaded = False

    def initialize(self) -> None:
        with self._lock:
            if self._loaded:
                return
            try:
                with self._engine.connect() as connection:
                    for municipality_id, payload in connection.execute(
                        select(municipalities.c.id, municipalities.c.payload)
                    ):
                        if payload is None:
                            raise PersistenceError(
                                f"municipality record has no durable payload: {municipality_id}"
                            )
                        context = MunicipalContext.model_validate(payload)
                        self.contexts[context.municipality_id] = context
                    for payload in connection.execute(select(imagery_assets.c.metadata)).scalars():
                        asset = AssetRecord.model_validate(payload)
                        self.assets[asset.asset_id] = asset
                    for pair_id, payload in connection.execute(
                        select(image_pairs.c.id, image_pairs.c.payload)
                    ):
                        if payload is None:
                            raise PersistenceError(
                                f"image pair record has no durable payload: {pair_id}"
                            )
                        pair = ImagePairRecord.model_validate(payload)
                        self.pairs[pair.pair_id] = pair
                    for payload in connection.execute(select(processing_jobs.c.payload)).scalars():
                        job_payload = payload.get("job", payload)
                        job = ProcessingJob.model_validate(job_payload)
                        self.jobs[job.job_id] = job
                        self.job_keys[job.idempotency_key] = job.job_id
                        request_payload = payload.get("request")
                        if request_payload is not None:
                            self.job_requests[job.job_id] = ProcessPairRequest.model_validate(
                                request_payload
                            )
                    for payload in connection.execute(select(change_results.c.payload)).scalars():
                        result = ChangeResultRecord.model_validate(payload)
                        self.results[result.result_id] = result
                    for payload in connection.execute(select(inspection_cases.c.payload)).scalars():
                        case = InspectionCase.model_validate(payload)
                        self.cases[case.case_id] = case
                self._loaded = True
            except PersistenceError:
                raise
            except (SQLAlchemyError, ValidationError, AttributeError, TypeError) as exc:
                raise PersistenceError("durable repository state could not be loaded") from exc

    def close(self) -> None:
        self._engine.dispose()

    def _require_srid(self, crs: str) -> None:
        if CRS.from_user_input(crs).to_epsg() != self._geometry_srid:
            raise CRSMismatchError(
                "record CRS does not match the configured database geometry SRID"
            )

    def _geometry(self, value: JsonObject, *, multi: bool = True) -> Any:
        expression = func.ST_SetSRID(
            func.ST_GeomFromGeoJSON(json.dumps(value)), self._geometry_srid
        )
        return func.ST_Multi(expression) if multi else expression

    @staticmethod
    def _upsert(connection: Connection, table: Any, values: JsonObject) -> None:
        statement = insert(table).values(**values)
        changes = {
            key: getattr(statement.excluded, key) for key in values if key != "id"
        }
        connection.execute(
            statement.on_conflict_do_update(index_elements=[table.c.id], set_=changes)
        )

    @staticmethod
    def _insert_immutable(
        connection: Connection,
        table: Any,
        values: JsonObject,
        payload: JsonObject,
        *,
        payload_column: Any | None = None,
    ) -> None:
        connection.execute(insert(table).values(**values).on_conflict_do_nothing())
        column = payload_column if payload_column is not None else table.c.payload
        stored = connection.execute(
            select(column).where(table.c.id == values["id"])
        ).scalar_one()
        if stored != payload:
            raise PersistenceError(f"immutable {table.name} record conflicts with stored data")

    def register_context(self, context: MunicipalContext) -> None:
        self.initialize()
        if not context.parcels:
            raise PersistenceError("municipal context requires at least one parcel")
        for item in (*context.parcels, *context.approved_footprints):
            self._require_srid(item.crs.value)
        for complaint in context.complaints:
            self._require_srid(complaint.crs.value)
        municipality_geometry = mapping(
            unary_union([shape(item.geometry) for item in context.parcels])
        )
        ward_geometries: dict[str, list[Any]] = defaultdict(list)
        for parcel in context.parcels:
            ward_geometries[parcel.ward_id].append(shape(parcel.geometry))
        footprints = {item.footprint_id: item for item in context.approved_footprints}
        try:
            with self._engine.begin() as connection:
                self._upsert(
                    connection,
                    municipalities,
                    {
                        "id": context.municipality_id,
                        "code": context.municipality_id,
                        "geometry": self._geometry(dict(municipality_geometry)),
                        "payload": context.model_dump(mode="json"),
                    },
                )
                for ward_id, geometries in ward_geometries.items():
                    self._upsert(
                        connection,
                        wards,
                        {
                            "id": ward_id,
                            "municipality_id": context.municipality_id,
                            "geometry": self._geometry(dict(mapping(unary_union(geometries)))),
                        },
                    )
                for parcel in context.parcels:
                    self._upsert(
                        connection,
                        parcels,
                        {
                            "id": parcel.parcel_id,
                            "ward_id": parcel.ward_id,
                            "geometry": self._geometry(parcel.geometry),
                        },
                    )
                for footprint in context.approved_footprints:
                    self._upsert(
                        connection,
                        approved_buildings,
                        {"id": footprint.building_id, "parcel_id": footprint.parcel_id},
                    )
                    self._upsert(
                        connection,
                        approved_plans,
                        {
                            "id": footprint.footprint_id,
                            "building_id": footprint.building_id,
                            "version": footprint.plan_version,
                            "geometry": self._geometry(footprint.geometry),
                        },
                    )
                for permit in context.permits:
                    if permit.footprint_id not in footprints:
                        raise PersistenceError("permit references an unknown approved footprint")
                    self._upsert(
                        connection,
                        permits,
                        {
                            "id": permit.permit_id,
                            "parcel_id": permit.parcel_id,
                            "approved_plan_id": permit.footprint_id,
                            "status": permit.status,
                            "valid_from": datetime.combine(
                                permit.valid_from, time.min, tzinfo=UTC
                            ),
                            "valid_until": datetime.combine(
                                permit.valid_until, time.max, tzinfo=UTC
                            ),
                        },
                    )
                for complaint in context.complaints:
                    self._upsert(
                        connection,
                        complaints,
                        {
                            "id": complaint.complaint_id,
                            "parcel_id": complaint.parcel_id,
                            "coordinate": self._geometry(complaint.coordinate, multi=False),
                            "payload": complaint.model_dump(mode="json"),
                        },
                    )
        except SQLAlchemyError as exc:
            raise PersistenceError("municipal context could not be persisted") from exc
        super().register_context(context)

    def save_asset(self, record: AssetRecord) -> AssetRecord:
        self.initialize()
        if record.metadata.captured_at is None:
            raise PersistenceError("persisted imagery requires a capture timestamp")
        self._require_srid(record.metadata.crs.value)
        bounds = record.metadata.bounds
        values: JsonObject = {
            "id": record.asset_id,
            "municipality_id": record.municipality_id,
            "storage_uri": record.storage_uri,
            "checksum": record.metadata.content_sha256,
            "captured_at": record.metadata.captured_at,
            "bounds": func.ST_MakeEnvelope(
                bounds.min_x,
                bounds.min_y,
                bounds.max_x,
                bounds.max_y,
                self._geometry_srid,
            ),
            "metadata": record.model_dump(mode="json"),
        }
        try:
            with self._engine.begin() as connection:
                self._insert_immutable(
                    connection,
                    imagery_assets,
                    values,
                    record.model_dump(mode="json"),
                    payload_column=imagery_assets.c.metadata,
                )
        except SQLAlchemyError as exc:
            raise PersistenceError("imagery metadata could not be persisted") from exc
        return super().save_asset(record)

    def save_pair(self, record: ImagePairRecord) -> ImagePairRecord:
        self.initialize()
        payload = record.model_dump(mode="json")
        try:
            with self._engine.begin() as connection:
                self._insert_immutable(
                    connection,
                    image_pairs,
                    {
                        "id": record.pair_id,
                        "old_asset_id": record.old_asset_id,
                        "new_asset_id": record.new_asset_id,
                        "payload": payload,
                    },
                    payload,
                )
        except SQLAlchemyError as exc:
            raise PersistenceError("image pair could not be persisted") from exc
        return super().save_pair(record)

    def save_job(self, job: ProcessingJob) -> ProcessingJob:
        self.initialize()
        existing_id = self.job_keys.get(job.idempotency_key)
        if existing_id is not None and existing_id != job.job_id:
            return self.jobs[existing_id]
        request = self.job_requests.get(job.job_id)
        payload: JsonObject = {"job": job.model_dump(mode="json")}
        if request is not None:
            payload["request"] = request.model_dump(mode="json")
        try:
            with self._engine.begin() as connection:
                self._upsert(
                    connection,
                    processing_jobs,
                    {
                        "id": job.job_id,
                        "image_pair_id": job.image_pair_id,
                        "idempotency_key": job.idempotency_key,
                        "status": job.status.value,
                        "correlation_id": job.correlation_id,
                        "payload": payload,
                    },
                )
        except SQLAlchemyError as exc:
            raise PersistenceError("processing job could not be persisted") from exc
        return super().save_job(job)

    def save_job_request(self, job_id: str, request: ProcessPairRequest) -> None:
        self.initialize()
        job = self.get_job(job_id)
        payload = {
            "job": job.model_dump(mode="json"),
            "request": request.model_dump(mode="json"),
        }
        try:
            with self._engine.begin() as connection:
                changed = connection.execute(
                    update(processing_jobs)
                    .where(processing_jobs.c.id == job_id)
                    .values(payload=payload)
                ).rowcount
                if changed != 1:
                    raise PersistenceError("processing job is missing")
        except SQLAlchemyError as exc:
            raise PersistenceError("processing request could not be persisted") from exc
        super().save_job_request(job_id, request)

    def _result_values(self, result: ChangeResultRecord) -> JsonObject:
        geometry: Any = None
        if result.polygons:
            merged = mapping(unary_union([shape(item.geometry) for item in result.polygons]))
            geometry = self._geometry(dict(merged))
        return {
            "id": result.result_id,
            "job_id": result.job_id,
            "storage_uri": result.change_mask_storage_uri,
            "checksum": result.checksum,
            "geometry": geometry,
            "payload": result.model_dump(mode="json"),
        }

    def save_result(self, result: ChangeResultRecord) -> ChangeResultRecord:
        self.initialize()
        self._require_srid(result.output_crs)
        payload = result.model_dump(mode="json")
        try:
            with self._engine.begin() as connection:
                self._insert_immutable(
                    connection, change_results, self._result_values(result), payload
                )
        except SQLAlchemyError as exc:
            raise PersistenceError("change result could not be persisted") from exc
        return super().save_result(result)

    def _persist_case_children(self, connection: Connection, case: InspectionCase) -> None:
        for group in case.complaint_groups:
            payload = group.model_dump(mode="json")
            self._upsert(
                connection, complaint_groups, {"id": group.group_id, "payload": payload}
            )
        pending: dict[str, EvidenceAsset] = {item.asset_id: item for item in case.evidence_assets}
        while pending:
            ready = [
                item
                for item in pending.values()
                if item.source_asset_id is None or item.source_asset_id not in pending
            ]
            if not ready:
                raise PersistenceError("evidence lineage contains a cycle")
            for item in ready:
                payload = item.model_dump(mode="json")
                self._insert_immutable(
                    connection,
                    evidence_assets,
                    {
                        "id": item.asset_id,
                        "case_id": case.case_id,
                        "source_asset_id": item.source_asset_id,
                        "storage_uri": item.storage_uri,
                        "checksum": item.content_sha256,
                        "payload": payload,
                    },
                    payload,
                )
                pending.pop(item.asset_id)
        for sequence, event in enumerate(case.evidence_timeline.events):
            payload = event.model_dump(mode="json")
            self._insert_immutable(
                connection,
                evidence_events,
                {
                    "id": event.event_id,
                    "case_id": case.case_id,
                    "sequence": sequence,
                    "event_hash": event.event_hash,
                    "payload": payload,
                },
                payload,
            )
        for sequence, audit_event in enumerate(case.audit_trail.events):
            payload = audit_event.model_dump(mode="json")
            self._insert_immutable(
                connection,
                audit_events,
                {
                    "id": audit_event.event_id,
                    "case_id": case.case_id,
                    "sequence": sequence,
                    "event_hash": audit_event.event_hash,
                    "payload": payload,
                },
                payload,
            )

    def commit_processing_bundle(
        self, result: ChangeResultRecord, case: InspectionCase
    ) -> tuple[ChangeResultRecord, InspectionCase]:
        self.initialize()
        self._require_srid(result.output_crs)
        existing_result = self.results.get(result.result_id)
        existing_case = self.cases.get(case.case_id)
        if existing_result is not None or existing_case is not None:
            if existing_result is None or existing_case is None:
                raise PersistenceError("partial processing bundle already exists")
            return existing_result, existing_case
        result_payload = result.model_dump(mode="json")
        case_payload = case.model_dump(mode="json")
        risk_id = f"risk-{case.case_id}"
        try:
            with self._engine.begin() as connection:
                self._insert_immutable(
                    connection, change_results, self._result_values(result), result_payload
                )
                self._insert_immutable(
                    connection,
                    risk_results,
                    {
                        "id": risk_id,
                        "change_result_id": result.result_id,
                        "score": case.risk_score,
                        "payload": {
                            "factors": [
                                item.model_dump(mode="json") for item in case.risk_factors
                            ]
                        },
                    },
                    {
                        "factors": [item.model_dump(mode="json") for item in case.risk_factors]
                    },
                )
                self._insert_immutable(
                    connection,
                    inspection_cases,
                    {
                        "id": case.case_id,
                        "risk_result_id": risk_id,
                        "status": case.status.value,
                        "payload": case_payload,
                    },
                    case_payload,
                )
                self._persist_case_children(connection, case)
        except SQLAlchemyError as exc:
            raise PersistenceError("processing bundle could not be persisted") from exc
        return super().commit_processing_bundle(result, case)

    def update_case(self, case: InspectionCase) -> InspectionCase:
        self.initialize()
        if case.case_id not in self.cases:
            return super().update_case(case)
        try:
            with self._engine.begin() as connection:
                changed = connection.execute(
                    update(inspection_cases)
                    .where(inspection_cases.c.id == case.case_id)
                    .values(status=case.status.value, payload=case.model_dump(mode="json"))
                ).rowcount
                if changed != 1:
                    raise PersistenceError("inspection case is missing")
                self._persist_case_children(connection, case)
        except SQLAlchemyError as exc:
            raise PersistenceError("inspection case could not be updated") from exc
        return super().update_case(case)
