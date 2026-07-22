"""Thread-safe integration repository used by API and the in-process worker."""

from collections.abc import Callable
from threading import RLock

from nirman_netra.application.contracts import (
    AssetRecord,
    ChangeResultRecord,
    ImagePairRecord,
    MunicipalContext,
    ProcessingJob,
    ProcessPairRequest,
)
from nirman_netra.cases.contracts import InspectionCase
from nirman_netra.exceptions import RecordNotFoundError


class InMemoryIntegrationRepository:
    def __init__(self) -> None:
        self._lock = RLock()
        self.assets: dict[str, AssetRecord] = {}
        self.pairs: dict[str, ImagePairRecord] = {}
        self.jobs: dict[str, ProcessingJob] = {}
        self.job_keys: dict[str, str] = {}
        self.job_requests: dict[str, ProcessPairRequest] = {}
        self.results: dict[str, ChangeResultRecord] = {}
        self.cases: dict[str, InspectionCase] = {}
        self.contexts: dict[str, MunicipalContext] = {}

    def save_asset(self, record: AssetRecord) -> AssetRecord:
        with self._lock:
            existing = self.assets.get(record.asset_id)
            if existing is not None and existing != record:
                raise ValueError(f"asset already exists with different metadata: {record.asset_id}")
            self.assets[record.asset_id] = record
            return record

    def get_asset(self, asset_id: str) -> AssetRecord:
        try:
            return self.assets[asset_id]
        except KeyError as exc:
            raise RecordNotFoundError(f"asset not found: {asset_id}") from exc

    def save_pair(self, record: ImagePairRecord) -> ImagePairRecord:
        with self._lock:
            existing = self.pairs.get(record.pair_id)
            if existing is not None and existing != record:
                raise ValueError(f"image pair already exists: {record.pair_id}")
            self.pairs[record.pair_id] = record
            return record

    def get_pair(self, pair_id: str) -> ImagePairRecord:
        try:
            return self.pairs[pair_id]
        except KeyError as exc:
            raise RecordNotFoundError(f"image pair not found: {pair_id}") from exc

    def find_job_by_key(self, key: str) -> ProcessingJob | None:
        with self._lock:
            job_id = self.job_keys.get(key)
            return self.jobs.get(job_id) if job_id is not None else None

    def save_job(self, job: ProcessingJob) -> ProcessingJob:
        with self._lock:
            existing_id = self.job_keys.get(job.idempotency_key)
            if existing_id is not None and existing_id != job.job_id:
                return self.jobs[existing_id]
            self.jobs[job.job_id] = job
            self.job_keys[job.idempotency_key] = job.job_id
            return job

    def get_job(self, job_id: str) -> ProcessingJob:
        try:
            return self.jobs[job_id]
        except KeyError as exc:
            raise RecordNotFoundError(f"job not found: {job_id}") from exc

    def save_job_request(self, job_id: str, request: ProcessPairRequest) -> None:
        with self._lock:
            existing = self.job_requests.get(job_id)
            if existing is not None and existing != request:
                raise ValueError(f"job request already exists: {job_id}")
            self.job_requests[job_id] = request

    def get_job_request(self, job_id: str) -> ProcessPairRequest:
        try:
            return self.job_requests[job_id]
        except KeyError as exc:
            raise RecordNotFoundError(f"job request not found: {job_id}") from exc

    def commit_processing_bundle(
        self,
        result: ChangeResultRecord,
        case: InspectionCase,
    ) -> tuple[ChangeResultRecord, InspectionCase]:
        """Atomically publish result and case after every computation succeeds."""

        with self._lock:
            existing_result = self.results.get(result.result_id)
            existing_case = self.cases.get(case.case_id)
            if existing_result is not None or existing_case is not None:
                if existing_result is None or existing_case is None:
                    raise ValueError("partial processing bundle already exists")
                return existing_result, existing_case
            self.results[result.result_id] = result
            self.cases[case.case_id] = case
            return result, case

    def get_result(self, result_id: str) -> ChangeResultRecord:
        try:
            return self.results[result_id]
        except KeyError as exc:
            raise RecordNotFoundError(f"change result not found: {result_id}") from exc

    def get_case(self, case_id: str) -> InspectionCase:
        try:
            return self.cases[case_id]
        except KeyError as exc:
            raise RecordNotFoundError(f"case not found: {case_id}") from exc

    def update_case(self, case: InspectionCase) -> InspectionCase:
        with self._lock:
            if case.case_id not in self.cases:
                raise RecordNotFoundError(f"case not found: {case.case_id}")
            self.cases[case.case_id] = case
            return case

    def list_cases(self, predicate: Callable[[InspectionCase], bool]) -> tuple[InspectionCase, ...]:
        with self._lock:
            return tuple(
                sorted(
                    (case for case in self.cases.values() if predicate(case)),
                    key=lambda case: (-case.risk_score, case.case_id),
                )
            )

    def register_context(self, context: MunicipalContext) -> None:
        with self._lock:
            self.contexts[context.municipality_id] = context

    def get_context(self, municipality_id: str) -> MunicipalContext:
        try:
            return self.contexts[municipality_id]
        except KeyError as exc:
            raise RecordNotFoundError(f"municipal context not found: {municipality_id}") from exc
