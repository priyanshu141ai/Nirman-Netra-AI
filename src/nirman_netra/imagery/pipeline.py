"""Application service orchestrating read-only registration."""

import json
from pathlib import Path

from nirman_netra.exceptions import (
    InsufficientOverlapError,
    RasterReadError,
    RegistrationQualityError,
)
from nirman_netra.imagery.alignment import align_pair
from nirman_netra.imagery.config import ImageryPipelineConfig
from nirman_netra.imagery.contracts import QualityStatus, RegistrationArtifact
from nirman_netra.imagery.quality import assess_pair, assess_raster
from nirman_netra.imagery.raster import LocalRasterIngestor, RasterIngestor
from nirman_netra.imagery.registration import (
    accept_geospatial_alignment,
    register_aligned_pair,
)
from nirman_netra.persistence.registration import persist_registration
from nirman_netra.utils import content_hash, deterministic_id, file_content_hash


class RegistrationPipeline:
    """Register local before/after rasters while preserving source bytes."""

    def __init__(
        self,
        config: ImageryPipelineConfig | None = None,
        ingestor: RasterIngestor | None = None,
    ) -> None:
        self._config = config or ImageryPipelineConfig()
        self._ingestor = ingestor or LocalRasterIngestor()

    def run(
        self,
        before_path: Path,
        after_path: Path,
        output_directory: Path,
        *,
        before_asset_id: str | None = None,
        after_asset_id: str | None = None,
    ) -> RegistrationArtifact:
        before = self._ingestor.read(before_path, asset_id=before_asset_id)
        after = self._ingestor.read(after_path, asset_id=after_asset_id)
        before_quality = assess_raster(before, self._config)
        after_quality = assess_raster(after, self._config)
        rejected_inputs = [
            report
            for report in (before_quality, after_quality)
            if report.status == QualityStatus.REJECTED
        ]
        if rejected_inputs:
            codes = sorted({issue.code for report in rejected_inputs for issue in report.issues})
            raise RegistrationQualityError(f"raster quality rejected: {','.join(codes)}")

        pair_quality = assess_pair(before, after, self._config)
        if any(issue.code == "INSUFFICIENT_GEOGRAPHIC_OVERLAP" for issue in pair_quality.issues):
            raise InsufficientOverlapError(
                f"geographic overlap {pair_quality.geographic_overlap_ratio:.3f} is insufficient"
            )
        if pair_quality.status == QualityStatus.REJECTED:
            codes = sorted(issue.code for issue in pair_quality.issues)
            raise RegistrationQualityError(f"raster pair rejected: {','.join(codes)}")

        aligned = align_pair(before, after, self._config)
        computation = (
            register_aligned_pair(aligned, self._config)
            if self._config.enable_visual_refinement
            else accept_geospatial_alignment(aligned, self._config)
        )
        warning_codes = tuple(
            sorted(
                {
                    issue.code
                    for report in (before_quality, after_quality)
                    for issue in report.issues
                    if issue.severity != "rejected"
                }
                | {issue.code for issue in pair_quality.issues}
                | set(computation.warnings)
            )
        )
        if (
            computation.status == QualityStatus.REQUIRES_MANUAL_ALIGNMENT
            or pair_quality.status == QualityStatus.REQUIRES_MANUAL_ALIGNMENT
        ):
            status = QualityStatus.REQUIRES_MANUAL_ALIGNMENT
        elif warning_codes:
            status = QualityStatus.PASS_WITH_WARNING
        else:
            status = QualityStatus.PASS

        config_payload = json.dumps(
            self._config.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        pair_id = deterministic_id(
            "registration-pair",
            before.metadata.asset_id,
            after.metadata.asset_id,
            before.metadata.content_sha256,
            after.metadata.content_sha256,
            content_hash(config_payload.encode()),
        )
        if (
            file_content_hash(before_path) != before.metadata.content_sha256
            or file_content_hash(after_path) != after.metadata.content_sha256
        ):
            raise RasterReadError("source raster changed during registration")
        artifact = persist_registration(
            computation,
            aligned,
            pair_id,
            status,
            warning_codes,
            output_directory,
            (before_path, after_path),
        )
        if (
            file_content_hash(before_path) != before.metadata.content_sha256
            or file_content_hash(after_path) != after.metadata.content_sha256
        ):
            raise RasterReadError("source raster changed while persisting registration")
        return artifact
