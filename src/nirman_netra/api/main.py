"""Versioned thin HTTP surface over the shared application service."""

import logging
import re
from datetime import datetime
from time import perf_counter

from fastapi import BackgroundTasks, FastAPI, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import RequestResponseEndpoint

from nirman_netra.api.contracts import (
    ApiError,
    ApiErrorBody,
    AssetCreate,
    CaseAssign,
    CaseReinspection,
    CaseReview,
    ImagePairCreate,
    PairProcess,
)
from nirman_netra.application.contracts import (
    AssetRecord,
    CasePage,
    ChangeResultRecord,
    DataQualitySnapshot,
    ImagePairRecord,
    ModelRequirement,
    ProcessingJob,
    ProcessPairRequest,
)
from nirman_netra.application.service import IntegrationService
from nirman_netra.cases.contracts import ActorIdentity, CaseStatus, InspectionCase
from nirman_netra.config import Settings, load_settings
from nirman_netra.data.contracts import Parcel
from nirman_netra.exceptions import (
    ArtifactValidationError,
    CaseTransitionError,
    CRSMismatchError,
    EvidenceIntegrityError,
    GeometryValidationError,
    InsufficientOverlapError,
    ModelChecksumMismatchError,
    ModelNotAvailableError,
    ModelSchemaMismatchError,
    NirmanNetraError,
    RasterMetadataError,
    RasterReadError,
    RecordNotFoundError,
    RegistrationError,
    RegistrationQualityError,
    RuleSetNotAvailableError,
    StorageError,
    UploadValidationError,
)
from nirman_netra.logging import configure_logging, log_context
from nirman_netra.risk.contracts import MunicipalRuleSet, RiskLevel
from nirman_netra.utils import new_correlation_id

_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def _safe_header_id(value: str | None) -> str:
    return value if value and _SAFE_ID.fullmatch(value) else new_correlation_id()


def _api_error(request: Request, code: str, message: str, status: int) -> JSONResponse:
    payload = ApiError(
        error=ApiErrorBody(
            code=code,
            message=message,
            request_id=request.state.request_id,
            correlation_id=request.state.correlation_id,
        )
    )
    return JSONResponse(status_code=status, content=payload.model_dump(mode="json"))


def _expected_error(error: NirmanNetraError) -> tuple[str, str, int]:
    if isinstance(error, CRSMismatchError) and "missing" in str(error).casefold():
        return "MISSING_CRS", "Raster CRS is missing.", 422
    mappings: tuple[tuple[type[NirmanNetraError], str, str, int], ...] = (
        (RecordNotFoundError, "NOT_FOUND", "Requested record was not found.", 404),
        (ModelNotAvailableError, "MODEL_NOT_AVAILABLE", "Requested model is unavailable.", 409),
        (
            ModelChecksumMismatchError,
            "MODEL_CHECKSUM_MISMATCH",
            "Model checksum verification failed.",
            409,
        ),
        (ModelSchemaMismatchError, "MODEL_SCHEMA_MISMATCH", "Model artifact is incompatible.", 409),
        (ArtifactValidationError, "MODEL_SCHEMA_MISMATCH", "Model artifact is incompatible.", 409),
        (
            RuleSetNotAvailableError,
            "RULE_SET_NOT_AVAILABLE",
            "Requested rule set is unavailable.",
            409,
        ),
        (InsufficientOverlapError, "INSUFFICIENT_OVERLAP", "Image overlap is insufficient.", 422),
        (
            RegistrationQualityError,
            "REGISTRATION_QUALITY_TOO_LOW",
            "Registration requires review.",
            422,
        ),
        (RegistrationError, "REGISTRATION_FAILED", "Image registration failed.", 422),
        (CRSMismatchError, "CRS_MISMATCH", "Raster CRS is missing or incompatible.", 422),
        (RasterReadError, "INVALID_RASTER", "Raster content is unreadable.", 422),
        (RasterMetadataError, "INVALID_RASTER", "Raster metadata is invalid.", 422),
        (GeometryValidationError, "INVALID_GEOMETRY", "Geometry is invalid.", 422),
        (CaseTransitionError, "INVALID_CASE_TRANSITION", "Case transition is invalid.", 409),
        (EvidenceIntegrityError, "EVIDENCE_INTEGRITY_FAILURE", "Evidence integrity failed.", 409),
        (StorageError, "STORAGE_UNAVAILABLE", "Object storage is unavailable.", 503),
        (UploadValidationError, "INVALID_RASTER", "Asset metadata is invalid.", 422),
    )
    for error_type, code, message, status in mappings:
        if isinstance(error, error_type):
            return code, message, status
    return "OPERATION_FAILED", "The requested operation failed safely.", 400


def create_app(
    settings: Settings | None = None,
    service: IntegrationService | None = None,
) -> FastAPI:
    """Create the API without database or network calls."""

    active_settings = settings or load_settings()
    configure_logging(active_settings.log_level)
    logger = logging.getLogger(__name__)
    app = FastAPI(title="NirmanNetra AI", version="0.1.0")
    app.state.settings = active_settings
    active_service = service or IntegrationService(active_settings)
    app.state.service = active_service

    @app.exception_handler(NirmanNetraError)
    async def handle_domain_error(request: Request, error: NirmanNetraError) -> JSONResponse:
        return _api_error(request, *_expected_error(error))

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, _: RequestValidationError) -> JSONResponse:
        return _api_error(request, "VALIDATION_ERROR", "Request contract validation failed.", 422)

    @app.middleware("http")
    async def request_context(request: Request, call_next: RequestResponseEndpoint) -> Response:
        correlation_id = _safe_header_id(request.headers.get("x-correlation-id"))
        request_id = _safe_header_id(request.headers.get("x-request-id"))
        request.state.correlation_id = correlation_id
        request.state.request_id = request_id
        started = perf_counter()
        with log_context(correlation_id=correlation_id, request_id=request_id):
            response = await call_next(request)
            response.headers["x-correlation-id"] = correlation_id
            response.headers["x-request-id"] = request_id
            logger.info(
                "request.completed",
                extra={
                    "metadata": {
                        "method": request.method,
                        "status_code": response.status_code,
                        "duration_ms": round((perf_counter() - started) * 1000, 2),
                    }
                },
            )
            active_service.metrics.observe("api_request_latency_seconds", perf_counter() - started)
            return response

    @app.get("/health/live", tags=["health"])
    async def liveness() -> dict[str, str]:
        return {"status": "live"}

    @app.get("/health/ready", tags=["health"], response_model=None)
    async def readiness() -> Response | dict[str, str]:
        if active_settings.enforce_runtime_readiness:
            report = active_service.readiness()
            if not report.ready:
                return JSONResponse(status_code=503, content=report.model_dump(mode="json"))
        return {"status": "ready", "environment": active_settings.app_env}

    @app.post("/api/v1/assets", status_code=201)
    async def create_asset(body: AssetCreate) -> AssetRecord:
        return active_service.ingest_asset(**body.model_dump())

    @app.get("/api/v1/assets/{asset_id}")
    async def get_asset(asset_id: str) -> AssetRecord:
        return active_service.get_asset(asset_id)

    @app.post("/api/v1/image-pairs", status_code=201)
    async def create_pair(body: ImagePairCreate) -> ImagePairRecord:
        return active_service.create_image_pair(body.old_asset_id, body.new_asset_id)

    @app.get("/api/v1/image-pairs/{pair_id}")
    async def get_pair(pair_id: str) -> ImagePairRecord:
        return active_service.get_image_pair(pair_id)

    @app.post("/api/v1/image-pairs/{pair_id}/process", status_code=202)
    async def process_pair(
        pair_id: str, body: PairProcess, tasks: BackgroundTasks, request: Request
    ) -> ProcessingJob:
        process_request = ProcessPairRequest(
            image_pair_id=pair_id,
            correlation_id=request.state.correlation_id,
            **body.model_dump(),
        )
        job = active_service.enqueue_processing(process_request)
        if job.status.value == "QUEUED":
            tasks.add_task(active_service.run_job, job.job_id)
        return job

    @app.get("/api/v1/jobs/{job_id}")
    async def get_job(job_id: str) -> ProcessingJob:
        return active_service.get_job(job_id)

    @app.get("/api/v1/change-results/{result_id}")
    async def get_change_result(result_id: str) -> ChangeResultRecord:
        return active_service.get_result(result_id)

    @app.get("/api/v1/parcels/{parcel_id}")
    async def get_parcel(parcel_id: str) -> Parcel:
        return active_service.get_parcel(parcel_id)

    @app.get("/api/v1/cases")
    async def list_cases(
        offset: int = Query(0, ge=0),
        limit: int = Query(active_settings.default_page_size, ge=1, le=100),
        municipality: str | None = None,
        ward: str | None = None,
        risk_level: RiskLevel | None = None,
        status: CaseStatus | None = None,
        assigned_inspector: str | None = None,
        created_from: datetime | None = None,
        created_until: datetime | None = None,
    ) -> CasePage:
        return active_service.list_cases(
            offset=offset,
            limit=limit,
            municipality=municipality,
            ward=ward,
            risk_level=risk_level,
            status=status,
            inspector_id=assigned_inspector,
            created_from=created_from,
            created_until=created_until,
        )

    @app.get("/api/v1/cases/{case_id}")
    async def get_case(case_id: str) -> InspectionCase:
        return active_service.get_case(case_id)

    @app.post("/api/v1/cases/{case_id}/assign")
    async def assign(case_id: str, body: CaseAssign, request: Request) -> InspectionCase:
        return active_service.assign_inspector(
            case_id,
            body.inspector_id,
            ActorIdentity.model_validate(body.actor.model_dump()),
            body.reason,
            request.state.correlation_id,
        )

    @app.post("/api/v1/cases/{case_id}/review")
    async def review(case_id: str, body: CaseReview, request: Request) -> InspectionCase:
        return active_service.review_case(
            case_id,
            body.action,
            ActorIdentity.model_validate(body.actor.model_dump()),
            body.reason,
            request.state.correlation_id,
        )

    @app.post("/api/v1/cases/{case_id}/reinspection")
    async def reinspection(
        case_id: str, body: CaseReinspection, request: Request
    ) -> InspectionCase:
        return active_service.request_case_reinspection(
            case_id,
            ActorIdentity.model_validate(body.actor.model_dump()),
            body.reason,
            request.state.correlation_id,
        )

    @app.get("/api/v1/models")
    async def models() -> tuple[ModelRequirement, ...]:
        return active_service.model_catalogue()

    @app.get("/api/v1/rule-sets")
    async def rule_sets() -> tuple[MunicipalRuleSet, ...]:
        return active_service.rule_set_catalogue()

    @app.get("/api/v1/data-quality/latest")
    async def data_quality() -> DataQualitySnapshot:
        return active_service.data_quality()

    @app.get("/internal/metrics", include_in_schema=False)
    async def metrics() -> dict[str, object]:
        return active_service.metrics.snapshot()

    return app


app = create_app()
