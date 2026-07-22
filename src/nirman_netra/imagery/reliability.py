"""Failure-safe registration reliability lab facade."""

from pathlib import Path
from typing import Literal

from nirman_netra.exceptions import (
    CRSMismatchError,
    InsufficientOverlapError,
    RasterMetadataError,
    RasterReadError,
    RegistrationError,
    RegistrationQualityError,
)
from nirman_netra.imagery.config import ImageryPipelineConfig
from nirman_netra.imagery.contracts import (
    QualityStatus,
    RegistrationReliabilityReport,
)
from nirman_netra.imagery.pipeline import RegistrationPipeline
from nirman_netra.imagery.raster import RasterIngestor

RegistrationStage = Literal["ingestion", "geospatial_alignment", "visual_refinement"]


class RegistrationReliabilityLab:
    """Return explicit reliability outcomes instead of interpreting failures as no change."""

    def __init__(
        self,
        config: ImageryPipelineConfig | None = None,
        ingestor: RasterIngestor | None = None,
    ) -> None:
        self._pipeline = RegistrationPipeline(config, ingestor)

    def evaluate(
        self,
        before_path: Path,
        after_path: Path,
        output_directory: Path,
        *,
        before_asset_id: str | None = None,
        after_asset_id: str | None = None,
    ) -> RegistrationReliabilityReport:
        try:
            artifact = self._pipeline.run(
                before_path,
                after_path,
                output_directory,
                before_asset_id=before_asset_id,
                after_asset_id=after_asset_id,
            )
        except RasterReadError:
            return self._failure(
                QualityStatus.REJECTED,
                "ingestion",
                "UNREADABLE_RASTER",
                before_asset_id,
                after_asset_id,
            )
        except RasterMetadataError:
            return self._failure(
                QualityStatus.REJECTED,
                "ingestion",
                "INVALID_RASTER_METADATA",
                before_asset_id,
                after_asset_id,
            )
        except CRSMismatchError as exc:
            missing = "missing" in str(exc).casefold()
            return self._failure(
                QualityStatus.REJECTED,
                "ingestion" if missing else "geospatial_alignment",
                "MISSING_CRS" if missing else "CRS_MISMATCH",
                before_asset_id,
                after_asset_id,
            )
        except InsufficientOverlapError:
            return self._failure(
                QualityStatus.REJECTED,
                "geospatial_alignment",
                "INSUFFICIENT_GEOGRAPHIC_OVERLAP",
                before_asset_id,
                after_asset_id,
            )
        except RegistrationQualityError:
            return self._failure(
                QualityStatus.REJECTED,
                "geospatial_alignment",
                "REGISTRATION_INPUT_REJECTED",
                before_asset_id,
                after_asset_id,
            )
        except RegistrationError:
            return self._failure(
                QualityStatus.REQUIRES_MANUAL_ALIGNMENT,
                "visual_refinement",
                "VISUAL_REGISTRATION_FAILED",
                before_asset_id,
                after_asset_id,
            )

        reliable = artifact.status in {QualityStatus.PASS, QualityStatus.PASS_WITH_WARNING}
        reasons = artifact.warnings
        if not reliable and "MANUAL_ALIGNMENT_REQUIRED" not in reasons:
            reasons = (*reasons, "MANUAL_ALIGNMENT_REQUIRED")
        failure_stage: RegistrationStage | None = None
        if not reliable:
            failure_stage = (
                "geospatial_alignment"
                if "LARGE_RESOLUTION_MISMATCH" in reasons
                or not artifact.visual_refinement_applied
                else "visual_refinement"
            )
        return RegistrationReliabilityReport(
            pair_id=artifact.pair_id,
            before_asset_id=artifact.before_asset_id,
            after_asset_id=artifact.after_asset_id,
            status=artifact.status,
            reliable_for_change_detection=reliable,
            failure_stage=failure_stage,
            reason_codes=tuple(sorted(set(reasons))),
            artifact=artifact,
        )

    @staticmethod
    def _failure(
        status: QualityStatus,
        stage: RegistrationStage,
        reason: str,
        before_asset_id: str | None,
        after_asset_id: str | None,
    ) -> RegistrationReliabilityReport:
        return RegistrationReliabilityReport(
            before_asset_id=before_asset_id,
            after_asset_id=after_asset_id,
            status=status,
            reliable_for_change_detection=False,
            failure_stage=stage,
            reason_codes=(reason,),
        )
