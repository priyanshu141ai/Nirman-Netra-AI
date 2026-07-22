"""Shared application orchestration for API, jobs, and dashboard clients."""

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError
from shapely.geometry import mapping, shape
from shapely.ops import unary_union
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from nirman_netra.application.backend import DefaultPipelineBackend, PipelineBackend
from nirman_netra.application.compatibility import (
    load_compatible_model,
    require_compatible_rule_set,
)
from nirman_netra.application.contracts import (
    AssetRecord,
    CasePage,
    ChangeResultRecord,
    DataQualitySnapshot,
    ImagePairRecord,
    ModelRequirement,
    MunicipalContext,
    PipelineEvidence,
    ProcessingJob,
    ProcessPairRequest,
    ReadinessComponent,
    ReadinessReport,
    RuleSetRequirement,
)
from nirman_netra.application.jobs import InProcessJobRunner, RetryPolicy
from nirman_netra.application.monitoring import MetricsRegistry
from nirman_netra.application.repository import InMemoryIntegrationRepository
from nirman_netra.application.storage import LocalObjectStorage
from nirman_netra.cases.contracts import (
    ActorIdentity,
    ActorRole,
    CaseStatus,
    EvidenceAsset,
    EvidenceAssetKind,
    InspectionCase,
)
from nirman_netra.cases.service import (
    accept_for_further_review,
    assign_case,
    attach_new_evidence,
    create_case_from_risk,
    dismiss_case,
    mark_evidence_insufficient,
    request_reinspection,
    start_review,
    transition_case,
    verify_case_evidence,
)
from nirman_netra.config import Settings
from nirman_netra.data.contracts import Parcel
from nirman_netra.domain import GeometryReference
from nirman_netra.exceptions import (
    CaseTransitionError,
    ConfigurationError,
    ModelCompatibilityError,
    ModelNotAvailableError,
    RecordNotFoundError,
    RuleSetNotAvailableError,
    UploadValidationError,
)
from nirman_netra.imagery.contracts import QualityStatus
from nirman_netra.imagery.quality import assess_raster
from nirman_netra.imagery.raster import LocalRasterIngestor
from nirman_netra.risk.complaints import group_duplicate_complaints
from nirman_netra.risk.contracts import (
    ComplaintDeduplicationConfig,
    MunicipalRuleSet,
    RiskEvidence,
    RiskLevel,
)
from nirman_netra.risk.engine import score_inspection_risk
from nirman_netra.risk.intersections import (
    compare_approved_plan,
    intersect_change_with_parcels,
)
from nirman_netra.risk.permits import match_permit_at_capture
from nirman_netra.segmentation.artifact import SegmentationInference
from nirman_netra.utils import deterministic_id, utc_now


class IntegrationService:
    def __init__(
        self,
        settings: Settings,
        *,
        repository: InMemoryIntegrationRepository | None = None,
        storage: LocalObjectStorage | None = None,
        metrics: MetricsRegistry | None = None,
        backend: PipelineBackend | None = None,
        database_probe: Callable[[], bool] | None = None,
    ) -> None:
        self.settings = settings
        if repository is not None:
            self.repository = repository
        elif settings.use_database_persistence:
            from nirman_netra.application.postgres_repository import (
                PostgresIntegrationRepository,
            )

            self.repository = PostgresIntegrationRepository(
                str(settings.database_url), geometry_srid=settings.database_geometry_srid
            )
        else:
            self.repository = InMemoryIntegrationRepository()
        self.storage = storage or LocalObjectStorage(
            settings.object_storage_original_path,
            settings.object_storage_derived_path,
        )
        self.metrics = metrics or MetricsRegistry()
        self.backend = backend or DefaultPipelineBackend(self.storage, settings.imagery)
        self.jobs = InProcessJobRunner(
            self.repository,
            self.metrics,
            RetryPolicy(settings.job_max_attempts, settings.job_backoff_seconds),
        )
        self._database_probe = database_probe or self._database_ready
        self._model_artifacts: dict[tuple[str, str], tuple[ModelRequirement, Path]] = {}
        self._initialized = False

    def initialize(self) -> None:
        if self._initialized:
            return
        self.repository.initialize()
        try:
            if self.settings.model_requirement_path is not None:
                if self.settings.model_artifact_path is None:
                    raise ConfigurationError("model requirement needs an artifact directory")
                requirement = ModelRequirement.model_validate_json(
                    self.settings.model_requirement_path.read_text(encoding="utf-8")
                )
                self.register_model(requirement, self.settings.model_artifact_path)
            if self.settings.municipal_context_path is not None:
                context = MunicipalContext.model_validate_json(
                    self.settings.municipal_context_path.read_text(encoding="utf-8")
                )
                self.register_municipality(context)
        except (OSError, ValidationError) as exc:
            raise ConfigurationError(
                "configured runtime resource is invalid or unavailable"
            ) from exc
        self._initialized = True

    def close(self) -> None:
        self.repository.close()

    def _database_ready(self) -> bool:
        if not self.settings.enforce_runtime_readiness:
            return True
        try:
            engine = create_engine(str(self.settings.database_url), pool_pre_ping=True)
            with engine.connect() as connection:
                result = connection.execute(text("SELECT PostGIS_Version()"))
                ready = result.scalar_one() is not None
            engine.dispose()
            return ready
        except (ModuleNotFoundError, SQLAlchemyError):
            return False

    def register_model(self, requirement: ModelRequirement, artifact_directory: Path) -> None:
        self._model_artifacts[(requirement.model_name, requirement.model_version)] = (
            requirement,
            artifact_directory,
        )

    def register_municipality(self, context: MunicipalContext) -> None:
        self.repository.register_context(context)

    def _models_ready(self) -> bool:
        if not self._model_artifacts:
            return (
                self.settings.model_artifact_path is None
                and self.settings.model_requirement_path is None
            )
        try:
            for requirement, path in self._model_artifacts.values():
                load_compatible_model(
                    path,
                    requirement,
                    source_crs=(
                        requirement.required_crs[0]
                        if requirement.required_crs
                        else (
                            self.settings.default_processing_crs
                            or f"EPSG:{self.settings.database_geometry_srid}"
                        )
                    ),
                    source_resolution_m=requirement.supported_resolution_m[0],
                )
        except (ModelCompatibilityError, ModelNotAvailableError):
            self.metrics.gauge("model_load_status", 0)
            return False
        self.metrics.gauge("model_load_status", 1)
        return True

    def ingest_asset(
        self,
        *,
        object_key: str,
        municipality_id: str,
        captured_at: datetime | None = None,
    ) -> AssetRecord:
        path = self.storage.original_path(object_key)
        if path.stat().st_size > self.settings.max_upload_bytes:
            raise UploadValidationError("raster exceeds configured upload-size limit")
        raster = LocalRasterIngestor().read(path, captured_at=captured_at)
        if raster.metadata.width * raster.metadata.height > self.settings.maximum_raster_pixels:
            raise UploadValidationError("raster exceeds configured decompression pixel limit")
        quality = assess_raster(raster, self.settings.imagery)
        if quality.status == QualityStatus.REJECTED:
            raise UploadValidationError("raster metadata or quality is rejected")
        asset_id = deterministic_id("asset", municipality_id, raster.metadata.content_sha256)
        metadata = raster.metadata.model_copy(
            update={"asset_id": asset_id, "source_uri": self.storage.original_uri(object_key)}
        )
        try:
            existing = self.repository.get_asset(asset_id)
        except RecordNotFoundError:
            existing = None
        if existing is not None:
            if (
                existing.municipality_id == municipality_id
                and existing.object_key == object_key
                and existing.storage_uri == self.storage.original_uri(object_key)
                and existing.metadata == metadata
                and existing.quality == quality.model_copy(update={"asset_id": asset_id})
            ):
                return existing
            raise UploadValidationError("asset hash already exists with different metadata")
        record = AssetRecord(
            asset_id=asset_id,
            municipality_id=municipality_id,
            object_key=object_key,
            storage_uri=self.storage.original_uri(object_key),
            metadata=metadata,
            quality=quality.model_copy(update={"asset_id": asset_id}),
            ingested_at=utc_now(),
        )
        self.metrics.increment("imagery_assets_ingested")
        if quality.status != QualityStatus.PASS:
            self.metrics.increment("low_quality_images")
        return self.repository.save_asset(record)

    def create_image_pair(self, old_asset_id: str, new_asset_id: str) -> ImagePairRecord:
        old, new = self.get_asset(old_asset_id), self.get_asset(new_asset_id)
        if old.municipality_id != new.municipality_id:
            raise UploadValidationError("image pair assets belong to different municipalities")
        if old.metadata.captured_at is None or new.metadata.captured_at is None:
            raise UploadValidationError("image pair requires both capture timestamps")
        if old.metadata.captured_at >= new.metadata.captured_at:
            raise UploadValidationError("historical capture must predate current capture")
        pair_id = deterministic_id(
            "image-pair",
            old.asset_id,
            new.asset_id,
            old.metadata.content_sha256,
            new.metadata.content_sha256,
        )
        try:
            existing = self.repository.get_pair(pair_id)
        except RecordNotFoundError:
            existing = None
        if existing is not None:
            if (
                existing.old_asset_id == old_asset_id
                and existing.new_asset_id == new_asset_id
                and existing.source_hashes == (
                    old.metadata.content_sha256,
                    new.metadata.content_sha256,
                )
            ):
                return existing
            raise UploadValidationError("image pair ID conflicts with stored source metadata")
        return self.repository.save_pair(
            ImagePairRecord(
                pair_id=pair_id,
                old_asset_id=old_asset_id,
                new_asset_id=new_asset_id,
                source_hashes=(old.metadata.content_sha256, new.metadata.content_sha256),
                created_at=utc_now(),
            )
        )

    def enqueue_processing(self, request: ProcessPairRequest) -> ProcessingJob:
        pair = self.get_image_pair(request.image_pair_id)
        old, new = self.get_asset(pair.old_asset_id), self.get_asset(pair.new_asset_id)
        self._load_model(request.model_requirement, old)
        self._load_rule(request.rule_requirement, new)
        job = self.jobs.submit(
            source_asset_ids=(old.asset_id, new.asset_id),
            source_asset_hashes=pair.source_hashes,
            image_pair_id=pair.pair_id,
            processing_configuration_hash=request.processing_configuration_hash,
            model_version=request.model_requirement.model_version,
            rule_set_version=request.rule_requirement.rule_version,
            correlation_id=request.correlation_id,
        )
        self.repository.save_job_request(job.job_id, request)
        return job

    def run_job(self, job_id: str) -> ProcessingJob:
        return self.jobs.run(job_id, self._process_job)

    def _load_model(
        self, requirement: ModelRequirement, asset: AssetRecord
    ) -> SegmentationInference:
        registered = self._model_artifacts.get((requirement.model_name, requirement.model_version))
        if registered is None or registered[0] != requirement:
            raise ModelNotAvailableError("requested approved model is unavailable")
        return load_compatible_model(
            registered[1],
            requirement,
            source_crs=asset.metadata.crs.value,
            source_resolution_m=max(asset.metadata.resolution),
        )

    def _load_rule(self, requirement: RuleSetRequirement, asset: AssetRecord) -> MunicipalRuleSet:
        context = self.repository.get_context(asset.municipality_id)
        if asset.metadata.captured_at is None:
            raise RuleSetNotAvailableError("capture time is required for municipal rules")
        return require_compatible_rule_set(
            context.rule_sets,
            requirement,
            capture_time=asset.metadata.captured_at,
        )

    def _process_job(self, job: ProcessingJob) -> str:
        request = self.repository.get_job_request(job.job_id)
        pair = self.get_image_pair(job.image_pair_id)
        old, new = self.get_asset(pair.old_asset_id), self.get_asset(pair.new_asset_id)
        model = self._load_model(request.model_requirement, old)
        rules = self._load_rule(request.rule_requirement, new)
        output_key = f"jobs/{job.job_id}"
        evidence = self.backend.process(
            pair,
            old,
            new,
            model,
            self.storage.derived_directory(output_key),
            self.storage.derived_uri(output_key),
        )
        result_id = deterministic_id("change-result", job.idempotency_key)
        if not evidence.change_polygons:
            result = self._result(job, request, evidence, result_id, RiskLevel.LOW, None)
            return self.repository.save_result(result).result_id
        context = self.repository.get_context(new.municipality_id)
        change_geometry = GeometryReference(
            geometry_id=f"change-{result_id}",
            geometry=dict(
                mapping(unary_union([shape(item.geometry) for item in evidence.change_polygons]))
            ),
            crs=evidence.change_polygons[0].crs,
            source_asset_id=new.asset_id,
        )
        parcel_match = intersect_change_with_parcels(change_geometry, context.parcels)
        parcel_id = parcel_match.matching_parcel_id
        if parcel_id is None:
            raise RecordNotFoundError("change result has no parcel match")
        parcel = next(item for item in context.parcels if item.parcel_id == parcel_id)
        approved = next(
            (item for item in context.approved_footprints if item.parcel_id == parcel_id),
            None,
        )
        if approved is None:
            raise RecordNotFoundError("approved footprint not found for matched parcel")
        plan = compare_approved_plan(
            approved,
            evidence.old_observed_footprint,
            evidence.new_observed_footprint,
            parcel,
            setback_metres=rules.setback_metres,
            public_boundaries=context.public_boundaries,
        )
        capture_time = new.metadata.captured_at
        if capture_time is None:
            raise RuleSetNotAvailableError("current image capture time is unavailable")
        permit = match_permit_at_capture(
            context.permits,
            parcel_id=parcel_id,
            capture_time=capture_time,
            approved_plan_version=approved.plan_version,
        )
        complaints = tuple(item for item in context.complaints if item.parcel_id == parcel_id)
        complaint_groups = group_duplicate_complaints(
            complaints,
            ComplaintDeduplicationConfig(),
        ).groups
        case_id = deterministic_id("inspection-case", job.idempotency_key)
        evaluation_time = max(utc_now(), capture_time)
        assessment = score_inspection_risk(
            RiskEvidence(
                case_id=case_id,
                change_confidence=evidence.change_confidence,
                registration_quality=evidence.registration_score,
                changed_area_square_metres=evidence.changed_area_square_metres,
                imagery_capture_time=capture_time,
                evaluation_time=evaluation_time,
                quality_warnings=evidence.warnings,
                corroborating_complaint_groups=len(complaint_groups),
            ),
            parcel_match,
            plan,
            permit,
            rules,
        )
        case = create_case_from_risk(
            assessment,
            parcel_id=parcel_id,
            image_pair_id=pair.pair_id,
            change_result_id=result_id,
            permit_result=permit,
            complaint_groups=complaint_groups,
            quality_warnings=evidence.warnings,
            actor=ActorIdentity(actor_id="system-processor", role=ActorRole.SYSTEM_PROCESSOR),
            timestamp=evaluation_time,
            correlation_id=job.correlation_id,
        )
        system_actor = ActorIdentity(actor_id="system-processor", role=ActorRole.SYSTEM_PROCESSOR)
        source_content = self.storage.original_path(new.object_key).read_bytes()
        source_evidence = EvidenceAsset(
            asset_id=f"evidence-source-{new.asset_id}",
            case_id=case_id,
            kind=EvidenceAssetKind.ORIGINAL,
            original_asset_sha256=new.metadata.content_sha256,
            source="immutable-ingested-raster",
            capture_time=capture_time,
            ingestion_time=new.ingested_at,
            storage_uri=new.storage_uri,
        )
        case = attach_new_evidence(
            case,
            source_evidence,
            source_content,
            actor=system_actor,
            timestamp=evaluation_time,
            correlation_id=job.correlation_id,
        )
        derived_content = (
            self.storage.derived_directory(output_key) / "change" / "change-mask.npy"
        ).read_bytes()
        derived_evidence = EvidenceAsset(
            asset_id=f"evidence-change-{result_id}",
            case_id=case_id,
            kind=EvidenceAssetKind.DERIVED,
            original_asset_sha256=new.metadata.content_sha256,
            derived_asset_sha256=evidence.checksum,
            source_asset_id=source_evidence.asset_id,
            source="bitemporal-change-pipeline",
            capture_time=capture_time,
            ingestion_time=evaluation_time,
            processing_job_id=job.job_id,
            transformation_parameters={
                "source_crs": evidence.source_crs,
                "output_crs": evidence.output_crs,
                "registration_score": evidence.registration_score,
            },
            model_version=request.model_requirement.model_version,
            storage_uri=evidence.change_mask_storage_uri,
        )
        case = attach_new_evidence(
            case,
            derived_evidence,
            derived_content,
            actor=system_actor,
            timestamp=evaluation_time,
            correlation_id=job.correlation_id,
        )
        result = self._result(job, request, evidence, result_id, assessment.level, case_id)
        self.repository.commit_processing_bundle(result, case)
        self.metrics.increment(f"change_result_{assessment.level.value.casefold()}")
        if assessment.level in {RiskLevel.HIGH, RiskLevel.CRITICAL_REVIEW}:
            self.metrics.increment("high_risk_cases_created")
        return result_id

    @staticmethod
    def _result(
        job: ProcessingJob,
        request: ProcessPairRequest,
        evidence: PipelineEvidence,
        result_id: str,
        risk_level: RiskLevel,
        case_id: str | None,
    ) -> ChangeResultRecord:
        return ChangeResultRecord(
            result_id=result_id,
            job_id=job.job_id,
            source_asset_ids=job.source_asset_ids,
            image_pair_id=job.image_pair_id,
            model_name=request.model_requirement.model_name,
            model_version=request.model_requirement.model_version,
            artifact_schema_version=request.model_requirement.artifact_schema_version,
            rule_set_id=request.rule_requirement.rule_set_id,
            rule_set_version=request.rule_requirement.rule_version,
            source_crs=evidence.source_crs,
            output_crs=evidence.output_crs,
            processed_at=utc_now(),
            image_quality_status=evidence.image_quality_status,
            registration_quality_status=evidence.registration_quality_status,
            registration_score=evidence.registration_score,
            warnings=evidence.warnings,
            polygons=evidence.change_polygons,
            changed_area_square_metres=evidence.changed_area_square_metres,
            risk_level=risk_level,
            human_review_status=CaseStatus.CREATED if case_id else None,
            case_id=case_id,
            change_mask_storage_uri=evidence.change_mask_storage_uri,
            checksum=evidence.checksum,
        )

    def assign_inspector(
        self,
        case_id: str,
        inspector_id: str,
        actor: ActorIdentity,
        reason: str,
        correlation_id: str,
    ) -> InspectionCase:
        case, now = self.get_case(case_id), utc_now()
        if case.status == CaseStatus.CREATED:
            case = transition_case(
                case,
                CaseStatus.TRIAGED,
                actor=actor,
                timestamp=now,
                correlation_id=correlation_id,
                reason="Supervisor triage before assignment",
            )
        context = self._context_for_case(case)
        inspector = next(
            (item for item in context.inspectors if item.inspector_id == inspector_id), None
        )
        if inspector is None:
            raise RecordNotFoundError(f"inspector not found: {inspector_id}")
        return self.repository.update_case(
            assign_case(
                case,
                inspector,
                actor=actor,
                timestamp=utc_now(),
                correlation_id=correlation_id,
                reason=reason,
            )
        )

    def review_case(
        self,
        case_id: str,
        action: str,
        actor: ActorIdentity,
        reason: str,
        correlation_id: str,
    ) -> InspectionCase:
        case = self.get_case(case_id)
        if case.evidence_assets:
            content = {
                asset.asset_id: self.storage.read(asset.storage_uri)
                for asset in case.evidence_assets
            }
            case, _ = verify_case_evidence(
                case,
                content,
                actor=actor,
                timestamp=utc_now(),
                correlation_id=correlation_id,
            )
            self.repository.update_case(case)
        handlers = {
            "start": start_review,
            "accept": accept_for_further_review,
            "dismiss": dismiss_case,
            "evidence_insufficient": mark_evidence_insufficient,
        }
        handler = handlers.get(action)
        if handler is None:
            raise CaseTransitionError(f"unsupported human review action: {action}")
        updated = handler(
            case,
            actor=actor,
            timestamp=utc_now(),
            correlation_id=correlation_id,
            reason=reason,
        )
        return self.repository.update_case(updated)

    def request_case_reinspection(
        self,
        case_id: str,
        actor: ActorIdentity,
        reason: str,
        correlation_id: str,
    ) -> InspectionCase:
        updated = request_reinspection(
            self.get_case(case_id),
            actor=actor,
            timestamp=utc_now(),
            correlation_id=correlation_id,
            reason=reason,
        )
        return self.repository.update_case(updated)

    def list_cases(
        self,
        *,
        offset: int = 0,
        limit: int = 25,
        municipality: str | None = None,
        ward: str | None = None,
        risk_level: RiskLevel | None = None,
        status: CaseStatus | None = None,
        inspector_id: str | None = None,
        created_from: datetime | None = None,
        created_until: datetime | None = None,
    ) -> CasePage:
        def matches(case: InspectionCase) -> bool:
            result = self.repository.get_result(case.change_result_id)
            assigned = case.assigned_inspector
            parcel = self.get_parcel(case.parcel_id) if case.parcel_id is not None else None
            return all(
                (
                    municipality is None
                    or self.get_asset(result.source_asset_ids[0]).municipality_id == municipality,
                    ward is None or (parcel is not None and parcel.ward_id == ward),
                    risk_level is None or result.risk_level == risk_level,
                    status is None or case.status == status,
                    inspector_id is None
                    or (assigned is not None and assigned.inspector_id == inspector_id),
                    created_from is None or case.created_at >= created_from,
                    created_until is None or case.created_at <= created_until,
                )
            )

        items = self.repository.list_cases(matches)
        return CasePage(
            items=items[offset : offset + limit], total=len(items), offset=offset, limit=limit
        )

    def _context_for_case(self, case: InspectionCase) -> MunicipalContext:
        result = self.repository.get_result(case.change_result_id)
        return self.repository.get_context(
            self.get_asset(result.source_asset_ids[0]).municipality_id
        )

    def get_asset(self, asset_id: str) -> AssetRecord:
        return self.repository.get_asset(asset_id)

    def get_image_pair(self, pair_id: str) -> ImagePairRecord:
        return self.repository.get_pair(pair_id)

    def get_job(self, job_id: str) -> ProcessingJob:
        return self.repository.get_job(job_id)

    def get_result(self, result_id: str) -> ChangeResultRecord:
        return self.repository.get_result(result_id)

    def get_case(self, case_id: str) -> InspectionCase:
        return self.repository.get_case(case_id)

    def get_parcel(self, parcel_id: str) -> Parcel:
        self.initialize()
        for context in self.repository.contexts.values():
            for parcel in context.parcels:
                if parcel.parcel_id == parcel_id:
                    return parcel
        raise RecordNotFoundError(f"parcel not found: {parcel_id}")

    def model_catalogue(self) -> tuple[ModelRequirement, ...]:
        self.initialize()
        return tuple(
            item[0]
            for item in sorted(
                self._model_artifacts.values(), key=lambda value: value[0].model_version
            )
        )

    def rule_set_catalogue(self) -> tuple[MunicipalRuleSet, ...]:
        self.initialize()
        return tuple(
            rule for context in self.repository.contexts.values() for rule in context.rule_sets
        )

    def data_quality(self) -> DataQualitySnapshot:
        self.initialize()
        return DataQualitySnapshot(
            created_at=utc_now(),
            total_assets=len(self.repository.assets),
            low_quality_assets=sum(
                asset.quality.status != QualityStatus.PASS
                for asset in self.repository.assets.values()
            ),
            registration_failures=int(self.metrics.value("processing_failure_registration_failed")),
            model_checksum_failures=int(
                self.metrics.value("processing_failure_model_checksum_mismatch")
            ),
            high_risk_cases=sum(case.risk_score >= 50 for case in self.repository.cases.values()),
        )

    def readiness(self) -> ReadinessReport:
        self.initialize()
        components = [
            ReadinessComponent(
                name="postgresql_postgis", ready=self._database_probe(), code="DATABASE_UNAVAILABLE"
            ),
            ReadinessComponent(
                name="object_storage", ready=self.storage.ready(), code="STORAGE_UNAVAILABLE"
            ),
            ReadinessComponent(
                name="worker_queue", ready=self.jobs.ready(), code="QUEUE_UNAVAILABLE"
            ),
            ReadinessComponent(
                name="approved_model",
                ready=self._models_ready(),
                code="MODEL_NOT_AVAILABLE",
            ),
            ReadinessComponent(
                name="municipal_rule_set",
                ready=not self.repository.contexts
                or all(context.rule_sets for context in self.repository.contexts.values()),
                code="RULE_SET_NOT_AVAILABLE",
            ),
        ]
        self.metrics.gauge("database_health", float(components[0].ready))
        self.metrics.gauge("object_storage_health", float(components[1].ready))
        self.metrics.gauge("worker_health", float(components[2].ready))
        return ReadinessReport(
            ready=all(item.ready for item in components),
            components=tuple(
                item.model_copy(update={"code": None}) if item.ready else item
                for item in components
            ),
        )
