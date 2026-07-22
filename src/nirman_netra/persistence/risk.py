"""Deterministic persistence for manual-review queue output."""

import json
from pathlib import Path

from nirman_netra.exceptions import StorageError
from nirman_netra.risk.contracts import ManualReviewQueue, QueueArtifact
from nirman_netra.utils import file_content_hash

QUEUE_FILENAME = "manual-review-queue.json"


def persist_manual_review_queue(
    queue: ManualReviewQueue,
    output_directory: Path,
) -> QueueArtifact:
    try:
        output_directory.mkdir(parents=True, exist_ok=True)
        path = (output_directory / QUEUE_FILENAME).resolve()
        path.write_text(
            json.dumps(queue.model_dump(mode="json"), sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        return QueueArtifact(
            storage_uri=path.as_uri(),
            checksum=file_content_hash(path),
            item_count=len(queue.items),
        )
    except OSError as exc:
        raise StorageError(f"failed to persist manual-review queue: {output_directory}") from exc
