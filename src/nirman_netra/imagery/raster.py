"""Read-only local raster ingestion and metadata extraction."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

import numpy as np
import rasterio
from numpy.typing import NDArray
from pydantic import ValidationError
from rasterio.errors import RasterioIOError

from nirman_netra.domain import BoundingBox
from nirman_netra.exceptions import CRSMismatchError, RasterMetadataError, RasterReadError
from nirman_netra.geospatial import validate_crs
from nirman_netra.imagery.contracts import RasterMetadata
from nirman_netra.utils import deterministic_id, file_content_hash


@dataclass(frozen=True)
class IngestedRaster:
    metadata: RasterMetadata
    pixels: NDArray[np.generic]
    valid_mask: NDArray[np.bool_]


class RasterIngestor(Protocol):
    def read(
        self,
        path: Path,
        *,
        asset_id: str | None = None,
        captured_at: datetime | None = None,
    ) -> IngestedRaster: ...


def _capture_timestamp(tags: dict[str, str]) -> tuple[datetime | None, str | None]:
    for key in ("capture_time", "ACQUISITIONDATETIME", "datetime", "TIFFTAG_DATETIME"):
        raw = tags.get(key)
        if not raw:
            continue
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None, raw
        if parsed.tzinfo is not None and parsed.utcoffset() is not None:
            return parsed, raw
        return None, raw
    return None, None


class LocalRasterIngestor:
    """Ingest a local raster without opening it in write mode."""

    def read(
        self,
        path: Path,
        *,
        asset_id: str | None = None,
        captured_at: datetime | None = None,
    ) -> IngestedRaster:
        if not path.is_file():
            raise RasterReadError(f"raster file does not exist: {path}")
        try:
            checksum = file_content_hash(path)
            with rasterio.open(path, "r") as source:
                if source.crs is None:
                    raise CRSMismatchError(f"raster is missing CRS: {path}")
                transform_values = (
                    source.transform.a,
                    source.transform.b,
                    source.transform.c,
                    source.transform.d,
                    source.transform.e,
                    source.transform.f,
                )
                determinant = (
                    source.transform.a * source.transform.e
                    - source.transform.b * source.transform.d
                )
                if not np.isfinite(transform_values).all() or determinant == 0:
                    raise RasterMetadataError(f"raster has invalid transform: {path}")
                if len(set(source.dtypes)) != 1:
                    raise RasterMetadataError(f"raster bands use incompatible dtypes: {path}")

                pixels = source.read()
                valid_mask = np.all(source.read_masks() > 0, axis=0)
                parsed_capture, raw_capture = _capture_timestamp(source.tags())
                crs = validate_crs(source.crs.to_string())
                metadata = RasterMetadata(
                    asset_id=asset_id or deterministic_id("local-raster", checksum),
                    source_uri=path.resolve().as_uri(),
                    width=source.width,
                    height=source.height,
                    band_count=source.count,
                    dtype=source.dtypes[0],
                    channel_order=tuple(item.name for item in source.colorinterp),
                    crs=crs,
                    epsg_code=source.crs.to_epsg(),
                    transform=transform_values,
                    bounds=BoundingBox(
                        min_x=source.bounds.left,
                        min_y=source.bounds.bottom,
                        max_x=source.bounds.right,
                        max_y=source.bounds.top,
                        crs=crs,
                    ),
                    resolution=(abs(source.res[0]), abs(source.res[1])),
                    nodata=source.nodata,
                    captured_at=captured_at or parsed_capture,
                    capture_timestamp_raw=raw_capture,
                    content_sha256=checksum,
                )
        except RasterioIOError as exc:
            raise RasterReadError(f"unable to read raster: {path}") from exc
        except OSError as exc:
            raise RasterReadError(f"unable to access raster: {path}") from exc
        except ValidationError as exc:
            raise RasterMetadataError(f"invalid raster metadata: {path}") from exc

        pixels.setflags(write=False)
        valid_mask.setflags(write=False)
        return IngestedRaster(metadata=metadata, pixels=pixels, valid_mask=valid_mask)
