from fastapi.testclient import TestClient

from nirman_netra.api.main import create_app
from nirman_netra.config import load_settings


def test_liveness() -> None:
    client = TestClient(create_app(load_settings(app_env="test")))

    response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "live"}
    assert response.headers["x-correlation-id"]


def test_readiness() -> None:
    client = TestClient(create_app(load_settings(app_env="test")))

    response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready", "environment": "test"}
