"""Read-only dashboard API client; no domain decisions live here."""

import json
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class DashboardApiClient:
    def __init__(self, base_url: str, timeout_seconds: float = 10) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds

    def _get(self, path: str, query: dict[str, str] | None = None) -> Any:
        suffix = f"?{urlencode(query)}" if query else ""
        request = Request(f"{self._base_url}{path}{suffix}", headers={"Accept": "application/json"})
        with urlopen(request, timeout=self._timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def cases(
        self, *, risk_level: str | None = None, assigned_inspector: str | None = None
    ) -> dict[str, Any]:
        query = {
            key: value
            for key, value in {
                "risk_level": risk_level,
                "assigned_inspector": assigned_inspector,
            }.items()
            if value is not None
        }
        result = self._get("/api/v1/cases", query)
        return dict(result)

    def case(self, case_id: str) -> dict[str, Any]:
        return dict(self._get(f"/api/v1/cases/{case_id}"))

    def result(self, result_id: str) -> dict[str, Any]:
        return dict(self._get(f"/api/v1/change-results/{result_id}"))

    def parcel(self, parcel_id: str) -> dict[str, Any]:
        return dict(self._get(f"/api/v1/parcels/{parcel_id}"))

    def quality(self) -> dict[str, Any]:
        return dict(self._get("/api/v1/data-quality/latest"))
