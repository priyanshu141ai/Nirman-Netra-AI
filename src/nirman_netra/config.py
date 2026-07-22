"""Central application settings."""

import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, PostgresDsn, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pyproj import CRS
from pyproj.exceptions import CRSError

from nirman_netra.exceptions import ConfigurationError
from nirman_netra.imagery.config import ImageryPipelineConfig


class Settings(BaseSettings):
    """Validated settings loaded from environment variables or a local .env file."""

    model_config = SettingsConfigDict(
        env_file=".env", env_nested_delimiter="__", extra="ignore", case_sensitive=False
    )

    app_env: Literal["development", "test", "staging", "production"] = "development"
    database_url: PostgresDsn = PostgresDsn("postgresql://localhost/nirman_netra")
    object_storage_original_path: Path = Path("data/objects")
    object_storage_derived_path: Path = Path("data/derived")
    log_level: str = "INFO"
    max_upload_bytes: int = Field(default=50 * 1024 * 1024, gt=0)
    default_processing_crs: str | None = None
    model_artifact_path: Path | None = None
    model_requirement_path: Path | None = None
    municipal_context_path: Path | None = None
    use_database_persistence: bool = False
    database_geometry_srid: int = Field(default=32643, gt=0)
    enforce_runtime_readiness: bool = False
    maximum_raster_pixels: int = Field(default=100_000_000, gt=0)
    job_max_attempts: int = Field(default=3, ge=1, le=10)
    job_backoff_seconds: float = Field(default=0.1, ge=0, le=60)
    default_page_size: int = Field(default=25, ge=1, le=100)
    dashboard_api_url: str = "http://api:8000"
    imagery: ImageryPipelineConfig = Field(default_factory=ImageryPipelineConfig)

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        normalized = value.upper()
        if normalized not in logging.getLevelNamesMapping():
            raise ValueError(f"unsupported log level: {value}")
        return normalized

    @field_validator(
        "model_artifact_path",
        "model_requirement_path",
        "municipal_context_path",
        mode="before",
    )
    @classmethod
    def blank_optional_path(cls, value: Any) -> Any:
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("default_processing_crs")
    @classmethod
    def validate_processing_crs(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        try:
            return CRS.from_user_input(value).to_string()
        except CRSError as exc:
            raise ValueError(f"invalid processing CRS: {value}") from exc


def load_settings(**overrides: Any) -> Settings:
    """Load settings and expose validation failures as a domain exception."""

    try:
        return Settings(**overrides)
    except ValidationError as exc:
        raise ConfigurationError(str(exc)) from exc
