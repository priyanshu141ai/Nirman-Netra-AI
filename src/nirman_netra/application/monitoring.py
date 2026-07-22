"""Small in-process structured metric registry."""

from collections import Counter, defaultdict
from threading import Lock


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = Lock()
        self._counters: Counter[str] = Counter()
        self._durations: dict[str, list[float]] = defaultdict(list)
        self._gauges: dict[str, float] = {}

    def increment(self, name: str, value: int = 1) -> None:
        with self._lock:
            self._counters[name] += value

    def observe(self, name: str, seconds: float) -> None:
        with self._lock:
            self._durations[name].append(max(0.0, seconds))

    def gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = value

    def value(self, name: str) -> int:
        with self._lock:
            return self._counters[name]

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "counters": dict(sorted(self._counters.items())),
                "durations": {
                    name: {
                        "count": len(values),
                        "total_seconds": round(sum(values), 6),
                    }
                    for name, values in sorted(self._durations.items())
                },
                "gauges": dict(sorted(self._gauges.items())),
            }
