"""Minimal immutable local object-storage adapter."""

import os
import shutil
from pathlib import Path

from nirman_netra.exceptions import StorageError, UploadValidationError
from nirman_netra.utils import file_content_hash


class LocalObjectStorage:
    def __init__(self, original_root: Path, derived_root: Path) -> None:
        self.original_root = original_root.resolve()
        self.derived_root = derived_root.resolve()

    @staticmethod
    def _resolve(root: Path, object_key: str) -> Path:
        if not object_key or "\\" in object_key:
            raise UploadValidationError("object key is empty or uses unsupported separators")
        candidate = (root / object_key).resolve()
        if not candidate.is_relative_to(root):
            raise UploadValidationError("object key escapes the configured storage root")
        return candidate

    def original_path(self, object_key: str) -> Path:
        path = self._resolve(self.original_root, object_key)
        if not path.is_file():
            raise StorageError("requested original object is unavailable")
        return path

    def derived_directory(self, object_key: str) -> Path:
        path = self._resolve(self.derived_root, object_key)
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise StorageError("derived object storage is unavailable") from exc
        return path

    def put_original_immutable(
        self,
        source: Path,
        object_key: str,
        expected_sha256: str | None = None,
    ) -> str:
        if not source.is_file():
            raise UploadValidationError("source upload object does not exist")
        destination = self._resolve(self.original_root, object_key)
        source_hash = file_content_hash(source)
        if expected_sha256 is not None and source_hash != expected_sha256:
            raise UploadValidationError("uploaded object checksum does not match metadata")
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if file_content_hash(destination) != source_hash:
                    raise StorageError(
                        "immutable original object already exists with other content"
                    )
            else:
                shutil.copyfile(source, destination)
        except OSError as exc:
            raise StorageError("failed to persist immutable original object") from exc
        return self.original_uri(object_key)

    def original_uri(self, object_key: str) -> str:
        self._resolve(self.original_root, object_key)
        return f"object://original/{object_key}"

    def derived_uri(self, object_key: str) -> str:
        self._resolve(self.derived_root, object_key)
        return f"object://derived/{object_key}"

    def read(self, storage_uri: str) -> bytes:
        prefixes = (
            ("object://original/", self.original_root),
            ("object://derived/", self.derived_root),
        )
        for prefix, root in prefixes:
            if storage_uri.startswith(prefix):
                path = self._resolve(root, storage_uri.removeprefix(prefix))
                try:
                    return path.read_bytes()
                except OSError as exc:
                    raise StorageError("evidence object is unavailable") from exc
        raise StorageError("unsupported evidence storage URI")

    def ready(self) -> bool:
        try:
            self.original_root.mkdir(parents=True, exist_ok=True)
            self.derived_root.mkdir(parents=True, exist_ok=True)
        except OSError:
            return False
        return all(
            os.access(path, os.R_OK | os.W_OK) for path in (self.original_root, self.derived_root)
        )
