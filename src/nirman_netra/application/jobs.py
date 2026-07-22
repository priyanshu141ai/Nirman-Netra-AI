"""Idempotent in-process background job abstraction with bounded retries."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock
from time import perf_counter, sleep

from nirman_netra.application.contracts import JobState, JobType, ProcessingJob
from nirman_netra.application.monitoring import MetricsRegistry
from nirman_netra.application.repository import InMemoryIntegrationRepository
from nirman_netra.exceptions import (
    ArtifactValidationError,
    CaseTransitionError,
    CRSMismatchError,
    EvidenceIntegrityError,
    GeometryValidationError,
    InsufficientOverlapError,
    JobStateError,
    ModelChecksumMismatchError,
    ModelNotAvailableError,
    ModelSchemaMismatchError,
    PersistenceError,
    QueueUnavailableError,
    RasterMetadataError,
    RasterReadError,
    RegistrationError,
    RegistrationQualityError,
    RuleSetNotAvailableError,
    StorageError,
)
from nirman_netra.utils import content_hash, deterministic_id, utc_now

JobHandler = Callable[[ProcessingJob], str]
Clock = Callable[[], object]


@dataclass(frozen=True)
class RetryPolicy:
    maximum_attempts: int = 3
    base_delay_seconds: float = 0.1

    def __post_init__(self) -> None:
        if self.maximum_attempts < 1 or self.base_delay_seconds < 0:
            raise ValueError("retry policy values are invalid")


_TRANSIENT_ERRORS = (PersistenceError, StorageError, QueueUnavailableError)


def _failure(error: Exception) -> tuple[str, str]:
    if isinstance(error, RasterReadError):
        return "INVALID_RASTER", "Raster content could not be read."
    if isinstance(error, RasterMetadataError):
        return "INVALID_RASTER", "Raster metadata is invalid."
    if isinstance(error, CRSMismatchError):
        code = "MISSING_CRS" if "missing" in str(error).casefold() else "CRS_MISMATCH"
        return code, "Raster coordinate reference is missing or incompatible."
    if isinstance(error, InsufficientOverlapError):
        return "INSUFFICIENT_OVERLAP", "Image pair has insufficient geographic overlap."
    if isinstance(error, RegistrationQualityError):
        return "REGISTRATION_QUALITY_TOO_LOW", "Registration quality requires manual review."
    if isinstance(error, RegistrationError):
        return "REGISTRATION_FAILED", "Image registration failed."
    if isinstance(error, GeometryValidationError):
        return "INVALID_GEOMETRY", "A required geometry is invalid."
    if isinstance(error, ModelNotAvailableError):
        return "MODEL_NOT_AVAILABLE", "The requested approved model is unavailable."
    if isinstance(error, ModelChecksumMismatchError):
        return "MODEL_CHECKSUM_MISMATCH", "The requested model checksum is invalid."
    if isinstance(error, ModelSchemaMismatchError | ArtifactValidationError):
        return "MODEL_SCHEMA_MISMATCH", "The requested model artifact is incompatible."
    if isinstance(error, RuleSetNotAvailableError):
        return "RULE_SET_NOT_AVAILABLE", "The requested municipal rule set is unavailable."
    if isinstance(error, EvidenceIntegrityError):
        return "EVIDENCE_INTEGRITY_FAILURE", "Evidence integrity verification failed."
    if isinstance(error, CaseTransitionError):
        return "INVALID_CASE_TRANSITION", "The requested case transition is invalid."
    if isinstance(error, _TRANSIENT_ERRORS):
        return "STORAGE_UNAVAILABLE", "A required persistence service is temporarily unavailable."
    return "PROCESSING_FAILED", "Processing failed without producing a trusted result."


def processing_idempotency_key(
    *,
    source_asset_hashes: tuple[str, str],
    image_pair_id: str,
    processing_configuration_hash: str,
    model_version: str,
    rule_set_version: str,
    operation_type: JobType,
) -> str:
    payload = json.dumps(
        {
            "source_asset_hashes": source_asset_hashes,
            "image_pair_id": image_pair_id,
            "processing_configuration_hash": processing_configuration_hash,
            "model_version": model_version,
            "rule_set_version": rule_set_version,
            "operation_type": operation_type.value,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return content_hash(payload.encode())


class InProcessJobRunner:
    def __init__(
        self,
        repository: InMemoryIntegrationRepository,
        metrics: MetricsRegistry,
        retry_policy: RetryPolicy | None = None,
        *,
        sleeper: Callable[[float], None] = sleep,
    ) -> None:
        self._repository = repository
        self._metrics = metrics
        self._retry_policy = retry_policy or RetryPolicy()
        self._sleeper = sleeper
        self._claim_lock = Lock()

    def submit(
        self,
        *,
        source_asset_ids: tuple[str, str],
        source_asset_hashes: tuple[str, str],
        image_pair_id: str,
        processing_configuration_hash: str,
        model_version: str,
        rule_set_version: str,
        correlation_id: str,
    ) -> ProcessingJob:
        key = processing_idempotency_key(
            source_asset_hashes=source_asset_hashes,
            image_pair_id=image_pair_id,
            processing_configuration_hash=processing_configuration_hash,
            model_version=model_version,
            rule_set_version=rule_set_version,
            operation_type=JobType.PROCESS_IMAGE_PAIR,
        )
        existing = self._repository.find_job_by_key(key)
        if existing is not None:
            return existing
        job = ProcessingJob(
            job_id=deterministic_id("processing-job", key),
            job_type=JobType.PROCESS_IMAGE_PAIR,
            source_asset_ids=source_asset_ids,
            image_pair_id=image_pair_id,
            model_version=model_version,
            rule_set_version=rule_set_version,
            status=JobState.QUEUED,
            created_at=utc_now(),
            correlation_id=correlation_id,
            idempotency_key=key,
        )
        self._metrics.increment("processing_jobs_queued")
        return self._repository.save_job(job)

    def cancel(self, job_id: str) -> ProcessingJob:
        job = self._repository.get_job(job_id)
        if job.status != JobState.QUEUED:
            raise JobStateError(f"only queued jobs can be cancelled: {job.status.value}")
        cancelled = job.model_copy(update={"status": JobState.CANCELLED, "completed_at": utc_now()})
        return self._repository.save_job(cancelled)

    def run(self, job_id: str, handler: JobHandler) -> ProcessingJob:
        with self._claim_lock:
            job = self._repository.get_job(job_id)
            if job.status in {JobState.RUNNING, JobState.SUCCEEDED}:
                return job
            if job.status != JobState.QUEUED:
                raise JobStateError(f"job cannot run from state {job.status.value}")
            started_at = utc_now()
            running = job.model_copy(update={"status": JobState.RUNNING, "started_at": started_at})
            self._repository.save_job(running)
        started = perf_counter()
        retries = 0
        while True:
            try:
                result_id = handler(running)
                succeeded = running.model_copy(
                    update={
                        "status": JobState.SUCCEEDED,
                        "completed_at": utc_now(),
                        "output_result_id": result_id,
                        "retry_count": retries,
                    }
                )
                self._metrics.increment("processing_jobs_succeeded")
                self._metrics.observe("processing_job_latency_seconds", perf_counter() - started)
                return self._repository.save_job(succeeded)
            except _TRANSIENT_ERRORS as exc:
                if retries + 1 < self._retry_policy.maximum_attempts:
                    delay = self._retry_policy.base_delay_seconds * (2**retries)
                    retries += 1
                    self._metrics.increment("processing_job_retries")
                    self._sleeper(delay)
                    continue
                error: Exception = exc
            except Exception as exc:  # process boundary converts to a safe failure record
                error = exc
            code, message = _failure(error)
            failed = running.model_copy(
                update={
                    "status": JobState.FAILED,
                    "completed_at": utc_now(),
                    "failure_code": code,
                    "safe_failure_message": message,
                    "retry_count": retries,
                }
            )
            self._metrics.increment("processing_jobs_failed")
            self._metrics.increment(f"processing_failure_{code.casefold()}")
            self._metrics.observe("processing_job_latency_seconds", perf_counter() - started)
            return self._repository.save_job(failed)

    @staticmethod
    def ready() -> bool:
        return True
