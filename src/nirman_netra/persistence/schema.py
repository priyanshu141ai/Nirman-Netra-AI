"""PostgreSQL/PostGIS metadata schema; raster bytes remain in object storage."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
)
from sqlalchemy.engine import Dialect
from sqlalchemy.orm import Session
from sqlalchemy.types import UserDefinedType

from nirman_netra.application.contracts import ChangeResultRecord
from nirman_netra.cases.contracts import InspectionCase

metadata = MetaData()


class Geometry(UserDefinedType[Any]):
    cache_ok = True

    def __init__(self, geometry_type: str = "GEOMETRY", srid: int = 0) -> None:
        self.geometry_type = geometry_type
        self.srid = srid

    def get_col_spec(self, **_: object) -> str:
        return f"geometry({self.geometry_type},{self.srid})"

    def bind_processor(self, dialect: Dialect) -> None:
        return None


municipalities = Table(
    "municipalities",
    metadata,
    *[
        __import__("sqlalchemy").Column("id", String(96), primary_key=True),
        __import__("sqlalchemy").Column("code", String(96), nullable=False, unique=True),
        __import__("sqlalchemy").Column(
            "geometry", Geometry("MULTIPOLYGON", 32643), nullable=False
        ),
    ],
)
wards = Table(
    "wards",
    metadata,
    __import__("sqlalchemy").Column("id", String(96), primary_key=True),
    __import__("sqlalchemy").Column(
        "municipality_id", ForeignKey("municipalities.id"), nullable=False
    ),
    __import__("sqlalchemy").Column("geometry", Geometry("MULTIPOLYGON", 32643), nullable=False),
)
parcels = Table(
    "parcels",
    metadata,
    __import__("sqlalchemy").Column("id", String(96), primary_key=True),
    __import__("sqlalchemy").Column("ward_id", ForeignKey("wards.id"), nullable=False),
    __import__("sqlalchemy").Column("geometry", Geometry("MULTIPOLYGON", 32643), nullable=False),
)
approved_buildings = Table(
    "approved_buildings",
    metadata,
    __import__("sqlalchemy").Column("id", String(96), primary_key=True),
    __import__("sqlalchemy").Column("parcel_id", ForeignKey("parcels.id"), nullable=False),
)
approved_plans = Table(
    "approved_plans",
    metadata,
    __import__("sqlalchemy").Column("id", String(96), primary_key=True),
    __import__("sqlalchemy").Column(
        "building_id", ForeignKey("approved_buildings.id"), nullable=False
    ),
    __import__("sqlalchemy").Column("version", Integer, nullable=False),
    __import__("sqlalchemy").Column("geometry", Geometry("MULTIPOLYGON", 32643), nullable=False),
    UniqueConstraint("building_id", "version", name="uq_approved_plan_version"),
)
permits = Table(
    "permits",
    metadata,
    __import__("sqlalchemy").Column("id", String(96), primary_key=True),
    __import__("sqlalchemy").Column("parcel_id", ForeignKey("parcels.id"), nullable=False),
    __import__("sqlalchemy").Column(
        "approved_plan_id", ForeignKey("approved_plans.id"), nullable=False
    ),
    __import__("sqlalchemy").Column("status", String(32), nullable=False),
    __import__("sqlalchemy").Column("valid_from", DateTime(timezone=True), nullable=False),
    __import__("sqlalchemy").Column("valid_until", DateTime(timezone=True), nullable=False),
)
imagery_assets = Table(
    "imagery_assets",
    metadata,
    __import__("sqlalchemy").Column("id", String(96), primary_key=True),
    __import__("sqlalchemy").Column(
        "municipality_id", ForeignKey("municipalities.id"), nullable=False
    ),
    __import__("sqlalchemy").Column("storage_uri", Text, nullable=False),
    __import__("sqlalchemy").Column("checksum", String(64), nullable=False),
    __import__("sqlalchemy").Column("captured_at", DateTime(timezone=True), nullable=False),
    __import__("sqlalchemy").Column("bounds", Geometry("POLYGON", 32643), nullable=False),
    __import__("sqlalchemy").Column("metadata", JSON, nullable=False),
)
image_pairs = Table(
    "image_pairs",
    metadata,
    __import__("sqlalchemy").Column("id", String(96), primary_key=True),
    __import__("sqlalchemy").Column(
        "old_asset_id", ForeignKey("imagery_assets.id"), nullable=False
    ),
    __import__("sqlalchemy").Column(
        "new_asset_id", ForeignKey("imagery_assets.id"), nullable=False
    ),
    UniqueConstraint("old_asset_id", "new_asset_id", name="uq_image_pair_sources"),
)
processing_jobs = Table(
    "processing_jobs",
    metadata,
    __import__("sqlalchemy").Column("id", String(96), primary_key=True),
    __import__("sqlalchemy").Column("image_pair_id", ForeignKey("image_pairs.id"), nullable=False),
    __import__("sqlalchemy").Column("idempotency_key", String(64), nullable=False, unique=True),
    __import__("sqlalchemy").Column("status", String(32), nullable=False),
    __import__("sqlalchemy").Column("correlation_id", String(128), nullable=False),
    __import__("sqlalchemy").Column("payload", JSON, nullable=False),
)


def _artifact_table(name: str) -> Table:
    return Table(
        name,
        metadata,
        __import__("sqlalchemy").Column("id", String(96), primary_key=True),
        __import__("sqlalchemy").Column("job_id", ForeignKey("processing_jobs.id"), nullable=False),
        __import__("sqlalchemy").Column("storage_uri", Text, nullable=False),
        __import__("sqlalchemy").Column("checksum", String(64), nullable=False),
        __import__("sqlalchemy").Column("payload", JSON, nullable=False),
    )


registration_results = _artifact_table("registration_results")
segmentation_results = _artifact_table("segmentation_results")
change_results = Table(
    "change_results",
    metadata,
    __import__("sqlalchemy").Column("id", String(96), primary_key=True),
    __import__("sqlalchemy").Column(
        "job_id", ForeignKey("processing_jobs.id"), nullable=False, unique=True
    ),
    __import__("sqlalchemy").Column("storage_uri", Text, nullable=False),
    __import__("sqlalchemy").Column("checksum", String(64), nullable=False),
    __import__("sqlalchemy").Column("geometry", Geometry("MULTIPOLYGON", 32643)),
    __import__("sqlalchemy").Column("payload", JSON, nullable=False),
)
complaints = Table(
    "complaints",
    metadata,
    __import__("sqlalchemy").Column("id", String(96), primary_key=True),
    __import__("sqlalchemy").Column("parcel_id", ForeignKey("parcels.id"), nullable=False),
    __import__("sqlalchemy").Column("coordinate", Geometry("POINT", 32643), nullable=False),
    __import__("sqlalchemy").Column("payload", JSON, nullable=False),
)
complaint_groups = Table(
    "complaint_groups",
    metadata,
    __import__("sqlalchemy").Column("id", String(96), primary_key=True),
    __import__("sqlalchemy").Column("payload", JSON, nullable=False),
)
risk_results = Table(
    "risk_results",
    metadata,
    __import__("sqlalchemy").Column("id", String(96), primary_key=True),
    __import__("sqlalchemy").Column(
        "change_result_id", ForeignKey("change_results.id"), nullable=False, unique=True
    ),
    __import__("sqlalchemy").Column("score", Float, nullable=False),
    __import__("sqlalchemy").Column("payload", JSON, nullable=False),
)
inspection_cases = Table(
    "inspection_cases",
    metadata,
    __import__("sqlalchemy").Column("id", String(96), primary_key=True),
    __import__("sqlalchemy").Column(
        "risk_result_id", ForeignKey("risk_results.id"), nullable=False, unique=True
    ),
    __import__("sqlalchemy").Column("status", String(48), nullable=False),
    __import__("sqlalchemy").Column("payload", JSON, nullable=False),
)
evidence_assets = Table(
    "evidence_assets",
    metadata,
    __import__("sqlalchemy").Column("id", String(96), primary_key=True),
    __import__("sqlalchemy").Column("case_id", ForeignKey("inspection_cases.id"), nullable=False),
    __import__("sqlalchemy").Column("source_asset_id", ForeignKey("imagery_assets.id")),
    __import__("sqlalchemy").Column("storage_uri", Text, nullable=False),
    __import__("sqlalchemy").Column("checksum", String(64), nullable=False),
    __import__("sqlalchemy").Column("payload", JSON, nullable=False),
)


def _event_table(name: str) -> Table:
    return Table(
        name,
        metadata,
        __import__("sqlalchemy").Column("id", String(96), primary_key=True),
        __import__("sqlalchemy").Column(
            "case_id", ForeignKey("inspection_cases.id"), nullable=False
        ),
        __import__("sqlalchemy").Column("sequence", Integer, nullable=False),
        __import__("sqlalchemy").Column("event_hash", String(64), nullable=False),
        __import__("sqlalchemy").Column("payload", JSON, nullable=False),
        UniqueConstraint("case_id", "sequence", name=f"uq_{name}_sequence"),
    )


evidence_events = _event_table("evidence_events")
audit_events = _event_table("audit_events")

for table in (
    municipalities,
    wards,
    parcels,
    approved_plans,
    imagery_assets,
    complaints,
    change_results,
):
    geometry_column = next(column for column in table.c if isinstance(column.type, Geometry))
    Index(f"ix_{table.name}_{geometry_column.name}_gist", geometry_column, postgresql_using="gist")


@contextmanager
def transaction(session: Session) -> Iterator[Session]:
    with session.begin():
        yield session


def persist_processing_bundle(
    session: Session,
    result: ChangeResultRecord,
    case: InspectionCase,
) -> None:
    """Atomically store the result and human-review case metadata."""

    with transaction(session):
        session.execute(
            change_results.insert().values(
                id=result.result_id,
                job_id=result.job_id,
                storage_uri=result.change_mask_storage_uri,
                checksum=result.checksum,
                payload=result.model_dump(mode="json"),
            )
        )
        session.execute(
            risk_results.insert().values(
                id=f"risk-{case.case_id}",
                change_result_id=result.result_id,
                score=case.risk_score,
                payload={"factors": [item.model_dump(mode="json") for item in case.risk_factors]},
            )
        )
        session.execute(
            inspection_cases.insert().values(
                id=case.case_id,
                risk_result_id=f"risk-{case.case_id}",
                status=case.status.value,
                payload=case.model_dump(mode="json"),
            )
        )
