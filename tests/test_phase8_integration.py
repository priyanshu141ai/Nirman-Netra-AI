from pathlib import Path
from subprocess import run

import pytest
from fastapi.testclient import TestClient

from nirman_netra.api.main import create_app
from nirman_netra.application.compatibility import (
    load_compatible_model,
    require_compatible_rule_set,
)
from nirman_netra.application.contracts import JobState, ProcessingJob, RuleSetRequirement
from nirman_netra.application.demo import _CAPTURE_NEW, _context, _model, run_synthetic_demo
from nirman_netra.application.jobs import InProcessJobRunner, RetryPolicy
from nirman_netra.application.monitoring import MetricsRegistry
from nirman_netra.application.repository import InMemoryIntegrationRepository
from nirman_netra.application.service import IntegrationService
from nirman_netra.config import Settings
from nirman_netra.exceptions import (
    ModelChecksumMismatchError,
    ModelNotAvailableError,
    ModelSchemaMismatchError,
    PersistenceError,
    RasterReadError,
    RuleSetNotAvailableError,
)
from nirman_netra.persistence.schema import Geometry, metadata
from nirman_netra.utils import content_hash


def _runner(*, attempts: int = 3) -> tuple[InProcessJobRunner, InMemoryIntegrationRepository]:
    repository = InMemoryIntegrationRepository()
    runner = InProcessJobRunner(
        repository,
        MetricsRegistry(),
        RetryPolicy(maximum_attempts=attempts, base_delay_seconds=0),
        sleeper=lambda _: None,
    )
    return runner, repository


def _submit(runner: InProcessJobRunner, *, model: str = "model-v1") -> ProcessingJob:
    return runner.submit(
        source_asset_ids=("asset-old", "asset-new"),
        source_asset_hashes=("a" * 64, "b" * 64),
        image_pair_id="pair-1",
        processing_configuration_hash="c" * 64,
        model_version=model,
        rule_set_version="rules-v1",
        correlation_id="correlation-1",
    )


def test_job_idempotency_returns_same_job() -> None:
    runner, repository = _runner()
    first = _submit(runner)
    second = _submit(runner)
    assert first.job_id == second.job_id
    assert len(repository.jobs) == 1


def test_idempotency_changes_with_model_version() -> None:
    runner, repository = _runner()
    assert _submit(runner, model="v1").job_id != _submit(runner, model="v2").job_id
    assert len(repository.jobs) == 2


def test_job_success_state_and_result() -> None:
    runner, _ = _runner()
    completed = runner.run(_submit(runner).job_id, lambda _: "result-1")
    assert completed.status == JobState.SUCCEEDED
    assert completed.output_result_id == "result-1"


def test_transient_failure_retries_with_bound() -> None:
    runner, _ = _runner(attempts=3)
    calls = 0

    def handler(_: object) -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise PersistenceError("temporary database failure")
        return "result-1"

    completed = runner.run(_submit(runner).job_id, handler)
    assert completed.status == JobState.SUCCEEDED
    assert completed.retry_count == 2
    assert calls == 3


def test_permanent_failure_is_not_retried() -> None:
    runner, _ = _runner()
    calls = 0

    def handler(_: object) -> str:
        nonlocal calls
        calls += 1
        raise RasterReadError("private path must not escape")

    completed = runner.run(_submit(runner).job_id, handler)
    assert completed.status == JobState.FAILED
    assert completed.failure_code == "INVALID_RASTER"
    assert calls == 1
    assert "private path" not in (completed.safe_failure_message or "")


def test_queued_job_can_be_cancelled() -> None:
    runner, _ = _runner()
    assert runner.cancel(_submit(runner).job_id).status == JobState.CANCELLED


def test_required_model_missing_fails(tmp_path: Path) -> None:
    _, requirement = _model(tmp_path)
    with pytest.raises(ModelNotAvailableError):
        load_compatible_model(
            tmp_path / "missing",
            requirement,
            source_crs="EPSG:32643",
            source_resolution_m=1,
        )


def test_model_checksum_mismatch_fails(tmp_path: Path) -> None:
    artifact, requirement = _model(tmp_path)
    with (artifact / "weights.npy").open("ab") as target:
        target.write(b"tamper")
    with pytest.raises(ModelChecksumMismatchError):
        load_compatible_model(
            artifact,
            requirement,
            source_crs="EPSG:32643",
            source_resolution_m=1,
        )


def test_model_artifact_schema_mismatch_fails(tmp_path: Path) -> None:
    artifact, requirement = _model(tmp_path)
    incompatible = requirement.model_copy(update={"artifact_schema_version": "2.0.0"})
    with pytest.raises(ModelSchemaMismatchError):
        load_compatible_model(
            artifact,
            incompatible,
            source_crs="EPSG:32643",
            source_resolution_m=1,
        )


def test_model_crs_mismatch_fails(tmp_path: Path) -> None:
    artifact, requirement = _model(tmp_path)
    with pytest.raises(ModelSchemaMismatchError):
        load_compatible_model(
            artifact,
            requirement,
            source_crs="EPSG:4326",
            source_resolution_m=1,
        )


def test_rule_set_version_mismatch_fails() -> None:
    context = _context("municipality-rules", True)
    rule = context.rule_sets[0]
    requirement = RuleSetRequirement(
        rule_set_id=rule.rule_set_id,
        municipality_id=rule.municipality_id,
        zone=rule.zone,
        rule_version="wrong-version",
        risk_threshold_version=rule.risk_threshold_version,
    )
    with pytest.raises(RuleSetNotAvailableError):
        require_compatible_rule_set(
            context.rule_sets,
            requirement,
            capture_time=_CAPTURE_NEW,
        )


def test_readiness_fails_for_configured_missing_model(tmp_path: Path) -> None:
    settings = Settings(
        app_env="test",
        enforce_runtime_readiness=True,
        model_artifact_path=tmp_path / "missing-model",
        object_storage_original_path=tmp_path / "objects",
        object_storage_derived_path=tmp_path / "derived",
    )
    report = IntegrationService(settings, database_probe=lambda: True).readiness()
    assert not report.ready
    assert any(item.code == "MODEL_NOT_AVAILABLE" for item in report.components)


def test_readiness_fails_for_database(tmp_path: Path) -> None:
    settings = Settings(
        app_env="test",
        enforce_runtime_readiness=True,
        object_storage_original_path=tmp_path / "objects",
        object_storage_derived_path=tmp_path / "derived",
    )
    report = IntegrationService(settings, database_probe=lambda: False).readiness()
    assert not report.ready
    assert report.components[0].code == "DATABASE_UNAVAILABLE"


def test_api_has_all_phase8_routes(tmp_path: Path) -> None:
    settings = Settings(
        app_env="test",
        object_storage_original_path=tmp_path / "objects",
        object_storage_derived_path=tmp_path / "derived",
    )
    paths = create_app(settings).openapi()["paths"]
    required = {
        "/api/v1/assets",
        "/api/v1/image-pairs",
        "/api/v1/image-pairs/{pair_id}/process",
        "/api/v1/jobs/{job_id}",
        "/api/v1/change-results/{result_id}",
        "/api/v1/parcels/{parcel_id}",
        "/api/v1/cases",
        "/api/v1/cases/{case_id}/assign",
        "/api/v1/cases/{case_id}/review",
        "/api/v1/cases/{case_id}/reinspection",
        "/api/v1/models",
        "/api/v1/rule-sets",
        "/api/v1/data-quality/latest",
    }
    assert required <= paths.keys()


def test_api_request_contract_rejects_extra_fields(tmp_path: Path) -> None:
    settings = Settings(
        app_env="test",
        object_storage_original_path=tmp_path / "objects",
        object_storage_derived_path=tmp_path / "derived",
    )
    response = TestClient(create_app(settings)).post(
        "/api/v1/image-pairs",
        json={"old_asset_id": "old", "new_asset_id": "new", "unexpected": True},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_api_expected_error_is_safe_and_correlated(tmp_path: Path) -> None:
    settings = Settings(
        app_env="test",
        object_storage_original_path=tmp_path / "objects",
        object_storage_derived_path=tmp_path / "derived",
    )
    response = TestClient(create_app(settings)).get(
        "/api/v1/assets/missing",
        headers={"x-correlation-id": "phase8-correlation"},
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"
    assert response.json()["error"]["correlation_id"] == "phase8-correlation"


def test_enforced_readiness_returns_503(tmp_path: Path) -> None:
    settings = Settings(
        app_env="test",
        enforce_runtime_readiness=True,
        object_storage_original_path=tmp_path / "objects",
        object_storage_derived_path=tmp_path / "derived",
    )
    service = IntegrationService(settings, database_probe=lambda: False)
    response = TestClient(create_app(settings, service)).get("/health/ready")
    assert response.status_code == 503
    assert not response.json()["ready"]


def test_postgis_schema_covers_integrated_entities() -> None:
    required = {
        "municipalities",
        "wards",
        "parcels",
        "approved_buildings",
        "approved_plans",
        "permits",
        "imagery_assets",
        "image_pairs",
        "processing_jobs",
        "registration_results",
        "segmentation_results",
        "change_results",
        "complaints",
        "complaint_groups",
        "risk_results",
        "inspection_cases",
        "evidence_assets",
        "evidence_events",
        "audit_events",
    }
    assert required <= metadata.tables.keys()
    assert any(
        isinstance(column.type, Geometry)
        for table in metadata.tables.values()
        for column in table.columns
    )


def test_spatial_columns_have_gist_indexes() -> None:
    for name in ("municipalities", "wards", "parcels", "change_results"):
        assert any(
            index.dialect_options["postgresql"]["using"] == "gist"
            for index in metadata.tables[name].indexes
        )


def test_alembic_upgrade_renders_offline() -> None:
    result = run(
        ["uv", "run", "alembic", "upgrade", "head", "--sql"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "CREATE EXTENSION IF NOT EXISTS postgis" in result.stdout
    assert "CREATE TABLE processing_jobs" in result.stdout


def test_compose_defines_only_used_services_and_healthchecks() -> None:
    compose = Path("compose.yaml").read_text(encoding="utf-8")
    assert "dashboard:" in compose and "api:" in compose and "db:" in compose
    assert "redis:" not in compose and "minio:" not in compose
    assert compose.count("healthcheck:") == 3
    assert "mem_limit:" in compose


def test_dashboard_uses_shared_api_and_safe_language() -> None:
    source = Path("src/nirman_netra/dashboard/app.py").read_text(encoding="utf-8")
    assert "DashboardApiClient" in source
    assert "Potential structural change" in source
    assert "not legal findings" in source
    assert "score_inspection_risk" not in source


def test_complete_demo_is_deterministic(tmp_path: Path) -> None:
    first = run_synthetic_demo(tmp_path / "first")
    second = run_synthetic_demo(tmp_path / "second")
    assert first == second
    assert first.legal_verdict is None


def test_demo_distinguishes_approved_and_permit_mismatch(tmp_path: Path) -> None:
    report = run_synthetic_demo(tmp_path)
    scenarios = {item.scenario: item for item in report.scenarios}
    assert scenarios["approved_change"].risk_level.value == "LOW"
    assert scenarios["approved_change"].case_status.value == "UNDER_REVIEW"
    assert scenarios["permit_mismatch"].risk_level.value == "HIGH"
    assert scenarios["duplicate_complaints"].case_id == scenarios["permit_mismatch"].case_id
    assert scenarios["evidence_integrity_failure"].warnings == ("EVIDENCE_INTEGRITY_FAILURE",)
    assert scenarios["low_registration_quality"].warnings == ("REQUIRES_MANUAL_ALIGNMENT",)


def test_content_based_processing_configuration_is_stable() -> None:
    assert content_hash(b"phase8") == content_hash(b"phase8")
