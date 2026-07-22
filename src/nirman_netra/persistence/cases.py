"""Canonical machine-readable inspection case export."""

import json
from pathlib import Path

from nirman_netra.cases.contracts import CaseExportArtifact, InspectionCase
from nirman_netra.cases.evidence import verify_chain_of_custody
from nirman_netra.exceptions import StorageError
from nirman_netra.utils import file_content_hash

CASE_PACKAGE_FILENAME = "case-package.json"
CASE_PACKAGE_SCHEMA_VERSION = "1.0.0"


def export_case_package(case: InspectionCase, output_directory: Path) -> CaseExportArtifact:
    """Export the same immutable case snapshot to identical canonical JSON bytes."""

    integrity = verify_chain_of_custody(case.evidence_assets)
    package = {
        "schema_version": CASE_PACKAGE_SCHEMA_VERSION,
        "case": case.model_dump(mode="json"),
        "evidence_integrity": integrity.model_dump(mode="json"),
    }
    try:
        output_directory.mkdir(parents=True, exist_ok=True)
        path = (output_directory / CASE_PACKAGE_FILENAME).resolve()
        path.write_text(
            json.dumps(package, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        return CaseExportArtifact(
            case_id=case.case_id,
            schema_version=CASE_PACKAGE_SCHEMA_VERSION,
            storage_uri=path.as_uri(),
            checksum=file_content_hash(path),
        )
    except OSError as exc:
        raise StorageError(f"failed to export inspection case: {output_directory}") from exc
