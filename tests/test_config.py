import pytest

from nirman_netra.config import load_settings
from nirman_netra.exceptions import ConfigurationError


def test_valid_settings() -> None:
    settings = load_settings(
        app_env="test",
        database_url="postgresql://localhost/test_db",
        max_upload_bytes=1024,
        default_processing_crs="EPSG:32643",
    )

    assert settings.app_env == "test"
    assert settings.max_upload_bytes == 1024
    assert settings.default_processing_crs == "EPSG:32643"


@pytest.mark.parametrize(
    ("key", "value"),
    [("database_url", "not-a-url"), ("max_upload_bytes", 0), ("default_processing_crs", "bad")],
)
def test_invalid_settings(key: str, value: object) -> None:
    with pytest.raises(ConfigurationError):
        load_settings(**{key: value})
