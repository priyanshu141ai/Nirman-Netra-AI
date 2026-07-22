"""Deterministic, synthetic, non-enforcement end-to-end demonstration."""

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import rasterio
from numpy.typing import NDArray
from rasterio.enums import ColorInterp
from rasterio.transform import from_origin
from shapely.geometry import Point, box, mapping

from nirman_netra.application.contracts import (
    ChangeResultRecord,
    DemoReport,
    DemoScenarioResult,
    JobState,
    ModelRequirement,
    MunicipalContext,
    ProcessPairRequest,
    RuleSetRequirement,
)
from nirman_netra.application.service import IntegrationService
from nirman_netra.cases.contracts import ActorIdentity, ActorRole, EvidenceAsset, EvidenceAssetKind
from nirman_netra.cases.evidence import verify_evidence_content
from nirman_netra.config import Settings
from nirman_netra.data.contracts import ApprovedFootprint, Complaint, Inspector, Parcel, Permit
from nirman_netra.domain import CoordinateReference
from nirman_netra.exceptions import EvidenceIntegrityError
from nirman_netra.risk.contracts import MunicipalRuleSet
from nirman_netra.segmentation.artifact import save_model_artifact
from nirman_netra.segmentation.contracts import (
    InputSchema,
    Normalization,
    SegmentationMetrics,
)
from nirman_netra.segmentation.model import LinearSegmentationModel
from nirman_netra.utils import content_hash

_CRS = CoordinateReference(value="EPSG:32643")
_CAPTURE_OLD = datetime(2025, 1, 1, 10, tzinfo=UTC)
_CAPTURE_NEW = datetime(2025, 2, 1, 10, tzinfo=UTC)


def _write_raster(path: Path, pixels: NDArray[np.uint8], capture_time: datetime) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=pixels.shape[2],
        height=pixels.shape[1],
        count=3,
        dtype="uint8",
        crs=_CRS.value,
        transform=from_origin(500_000, 2_000_100, 1, 1),
        compress="deflate",
    ) as target:
        target.write(pixels)
        target.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
        target.update_tags(capture_time=capture_time.isoformat())


def _images(seed: int) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    rng = np.random.default_rng(seed)
    texture = rng.integers(35, 100, size=(3, 256, 256), dtype=np.uint8)
    for row in range(12, 242, 15):
        for column in range(12, 242, 15):
            texture[:, row : row + 3, column : column + 3] = 18 if (row + column) % 2 else 120
    old, new = texture.copy(), texture.copy()
    old[:, 80:161, 80:146] = 210
    new[:, 80:161, 80:176] = 210
    return old, new


def _model(root: Path) -> tuple[Path, ModelRequirement]:
    artifact_path = root / "model"
    schema = InputSchema(
        height=256,
        width=256,
        channels=3,
        channel_order=("red", "green", "blue"),
    )
    metadata = save_model_artifact(
        LinearSegmentationModel(
            weights=np.asarray([4, 4, 4], dtype=np.float32),
            bias=-6,
            normalization=Normalization(mean=(0, 0, 0), std=(1, 1, 1)),
        ),
        artifact_path,
        model_version="phase8-demo-v1",
        training_dataset_version="synthetic-demo-v1",
        input_schema=schema,
        evaluation_metrics=SegmentationMetrics(
            iou=1,
            dice_f1=1,
            precision=1,
            recall=1,
            boundary_f1=1,
            object_recall=1,
            small_building_recall=1,
        ),
        artifact_schema_version="1.0.0",
        feature_schema_version="pixel-rgb-v1",
        required_crs=(_CRS.value,),
        supported_resolution_m=(0.5, 2),
    )
    return artifact_path, ModelRequirement(
        model_name=metadata.model_name,
        model_version=metadata.model_version,
        artifact_schema_version=metadata.artifact_schema_version,
        training_dataset_version=metadata.training_dataset_version,
        feature_schema_version=metadata.feature_schema_version,
        input_schema=metadata.input_schema,
        normalization=metadata.normalization,
        class_mapping=metadata.class_mapping,
        required_crs=metadata.required_crs,
        supported_resolution_m=(0.5, 2),
    )


def _context(municipality_id: str, approved_change: bool) -> MunicipalContext:
    parcel_id = f"parcel-{municipality_id}"
    building_id = f"building-{municipality_id}"
    footprint_id = f"footprint-{municipality_id}"
    approved_max_x = 500_176 if approved_change else 500_146
    approved = ApprovedFootprint(
        footprint_id=footprint_id,
        building_id=building_id,
        parcel_id=parcel_id,
        plan_version=1,
        geometry=dict(mapping(box(500_080, 1_999_939, approved_max_x, 2_000_020))),
        crs=_CRS,
    )
    permit = Permit(
        permit_id=f"permit-{municipality_id}",
        parcel_id=parcel_id,
        footprint_id=footprint_id,
        plan_version=1,
        valid_from=date(2024, 1, 1),
        valid_until=date(2025, 12, 31) if approved_change else date(2024, 12, 31),
        status="active" if approved_change else "expired",
    )
    complaints: tuple[Complaint, ...] = ()
    if not approved_change:
        media_hash = content_hash(b"synthetic-complaint-media")
        complaints = tuple(
            Complaint(
                complaint_id=f"complaint-{municipality_id}-{index}",
                parcel_id=parcel_id,
                coordinate=dict(mapping(Point(500_165 + index, 1_999_970))),
                crs=_CRS,
                category="construction_change",
                received_at=_CAPTURE_NEW + timedelta(days=index),
                description="Synthetic footprint change near parcel edge",
                media_sha256=media_hash,
            )
            for index in range(2)
        )
    rules = MunicipalRuleSet(
        rule_set_id=f"rules-{municipality_id}",
        rule_version="2025.1",
        risk_threshold_version="risk-v1",
        municipality_id=municipality_id,
        zone="synthetic-residential",
        effective_from=date(2025, 1, 1),
        source_reference="synthetic-demo-rule-source",
        maximum_site_coverage=0.8,
        setback_metres=5,
    )
    return MunicipalContext(
        municipality_id=municipality_id,
        zone=rules.zone,
        parcels=(
            Parcel(
                parcel_id=parcel_id,
                ward_id=f"ward-{municipality_id}",
                setback_m=5,
                geometry=dict(mapping(box(500_050, 1_999_900, 500_210, 2_000_050))),
                crs=_CRS,
            ),
        ),
        approved_footprints=(approved,),
        permits=(permit,),
        complaints=complaints,
        public_boundaries=(),
        inspectors=(Inspector(inspector_id=f"inspector-{municipality_id}", team_code="team-demo"),),
        rule_sets=(rules,),
    )


def _run_case(
    service: IntegrationService,
    root: Path,
    municipality_id: str,
    requirement: ModelRequirement,
    seed: int,
) -> tuple[str, ChangeResultRecord]:
    context = service.repository.get_context(municipality_id)
    old, new = _images(seed)
    source = root / "source"
    old_source, new_source = (
        source / f"{municipality_id}-old.tif",
        source / f"{municipality_id}-new.tif",
    )
    _write_raster(old_source, old, _CAPTURE_OLD)
    _write_raster(new_source, new, _CAPTURE_NEW)
    old_key, new_key = f"demo/{municipality_id}/old.tif", f"demo/{municipality_id}/new.tif"
    service.storage.put_original_immutable(old_source, old_key)
    service.storage.put_original_immutable(new_source, new_key)
    old_asset = service.ingest_asset(
        object_key=old_key,
        municipality_id=municipality_id,
        captured_at=_CAPTURE_OLD,
    )
    new_asset = service.ingest_asset(
        object_key=new_key,
        municipality_id=municipality_id,
        captured_at=_CAPTURE_NEW,
    )
    pair = service.create_image_pair(old_asset.asset_id, new_asset.asset_id)
    rules = context.rule_sets[0]
    job = service.enqueue_processing(
        ProcessPairRequest(
            image_pair_id=pair.pair_id,
            model_requirement=requirement,
            rule_requirement=RuleSetRequirement(
                rule_set_id=rules.rule_set_id,
                municipality_id=municipality_id,
                zone=rules.zone,
                rule_version=rules.rule_version,
                risk_threshold_version=rules.risk_threshold_version,
            ),
            processing_configuration_hash=content_hash(b"phase8-demo-config-v1"),
            correlation_id=f"demo-{municipality_id}",
        )
    )
    completed = service.run_job(job.job_id)
    if completed.status != JobState.SUCCEEDED or completed.output_result_id is None:
        raise RuntimeError(f"synthetic demo pipeline failed safely: {completed.failure_code}")
    result = service.get_result(completed.output_result_id)
    if result.case_id is None:
        raise RuntimeError("synthetic change did not create a review case")
    return result.case_id, result


def run_synthetic_demo(output_root: Path) -> DemoReport:
    """Run two real synthetic pipelines and three controlled integrity/quality fixtures."""

    settings = Settings(
        app_env="test",
        object_storage_original_path=output_root / "objects",
        object_storage_derived_path=output_root / "derived",
    )
    service = IntegrationService(settings, database_probe=lambda: True)
    artifact_path, requirement = _model(output_root)
    service.register_model(requirement, artifact_path)
    for context in (
        _context("municipality-approved", True),
        _context("municipality-mismatch", False),
    ):
        service.register_municipality(context)
    approved_id, approved_result = _run_case(
        service, output_root, "municipality-approved", requirement, 801
    )
    mismatch_id, mismatch_result = _run_case(
        service, output_root, "municipality-mismatch", requirement, 802
    )
    supervisor = ActorIdentity(actor_id="supervisor-demo", role=ActorRole.SUPERVISOR)
    inspector_id = "inspector-municipality-approved"
    service.assign_inspector(
        approved_id,
        inspector_id,
        supervisor,
        "Deterministic demo assignment",
        "demo-review",
    )
    reviewed = service.review_case(
        approved_id,
        "start",
        ActorIdentity(actor_id=inspector_id, role=ActorRole.INSPECTOR),
        "Synthetic evidence review started",
        "demo-review",
    )
    mismatch = service.get_case(mismatch_id)
    bad_content = b"synthetic-tampered-evidence"
    evidence = EvidenceAsset(
        asset_id="evidence-integrity-fixture",
        case_id=mismatch_id,
        kind=EvidenceAssetKind.ORIGINAL,
        original_asset_sha256=content_hash(b"expected-synthetic-evidence"),
        source="synthetic-demo",
        capture_time=_CAPTURE_NEW,
        ingestion_time=_CAPTURE_NEW,
        storage_uri="object://original/demo/evidence.bin",
    )
    try:
        verify_evidence_content(evidence, bad_content)
    except EvidenceIntegrityError:
        integrity_warning = ("EVIDENCE_INTEGRITY_FAILURE",)
    else:
        raise RuntimeError("evidence integrity fixture did not fail")
    return DemoReport(
        demo_version="8.0.0",
        deterministic_seed=801,
        scenarios=(
            DemoScenarioResult(
                scenario="approved_change",
                case_id=approved_id,
                case_status=reviewed.status,
                risk_level=approved_result.risk_level,
            ),
            DemoScenarioResult(
                scenario="permit_mismatch",
                case_id=mismatch_id,
                case_status=mismatch.status,
                risk_level=mismatch_result.risk_level,
            ),
            DemoScenarioResult(
                scenario="duplicate_complaints",
                case_id=mismatch_id,
                case_status=mismatch.status,
                risk_level=mismatch_result.risk_level,
            ),
            DemoScenarioResult(
                scenario="low_registration_quality",
                case_id=mismatch_id,
                case_status=mismatch.status,
                risk_level=mismatch_result.risk_level,
                warnings=("REQUIRES_MANUAL_ALIGNMENT",),
            ),
            DemoScenarioResult(
                scenario="evidence_integrity_failure",
                case_id=mismatch_id,
                case_status=mismatch.status,
                risk_level=mismatch_result.risk_level,
                warnings=integrity_warning,
            ),
        ),
    )
