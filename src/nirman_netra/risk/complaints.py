"""Complaint duplicate grouping that preserves every source record."""

from collections.abc import Sequence
from difflib import SequenceMatcher
from hashlib import sha256

from pyproj import CRS, Transformer
from pyproj.exceptions import ProjError
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform

from nirman_netra.data.contracts import Complaint
from nirman_netra.exceptions import CRSMismatchError, DatasetValidationError
from nirman_netra.geospatial import validate_geometry
from nirman_netra.risk.contracts import (
    ComplaintDeduplicationConfig,
    ComplaintDeduplicationResult,
    ComplaintGroup,
)


def _normalise_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _metre_factor(crs_value: str) -> float:
    parsed = CRS.from_user_input(crs_value)
    if not parsed.is_projected or not parsed.axis_info:
        raise CRSMismatchError("complaint distance requires a projected CRS")
    factor = parsed.axis_info[0].unit_conversion_factor
    if factor is None:
        raise CRSMismatchError("complaint CRS linear units are unavailable")
    return float(factor)


def _convert(
    geometry: BaseGeometry,
    source_crs: str,
    target_crs: str,
) -> BaseGeometry:
    if source_crs == target_crs:
        return geometry
    try:
        transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
        return transform(transformer.transform, geometry)
    except ProjError as exc:
        raise CRSMismatchError(f"cannot transform complaint CRS {source_crs}") from exc


def _perceptual_hash_distance(left: str, right: str) -> int | None:
    if len(left) != len(right):
        return None
    return (int(left, 16) ^ int(right, 16)).bit_count()


def _duplicate_signals(
    left: Complaint,
    right: Complaint,
    config: ComplaintDeduplicationConfig,
) -> tuple[str, ...]:
    if left.parcel_id != right.parcel_id:
        return ()
    if abs((left.received_at - right.received_at).total_seconds()) > (
        config.time_window_days * 86_400
    ):
        return ()
    left_point = validate_geometry(left.coordinate)
    right_point = _convert(validate_geometry(right.coordinate), right.crs.value, left.crs.value)
    distance_metres = left_point.distance(right_point) * _metre_factor(left.crs.value)
    if distance_metres > config.spatial_distance_metres:
        return ()
    content_signals: list[str] = []
    left_text, right_text = _normalise_text(left.description), _normalise_text(right.description)
    if (
        left_text
        and right_text
        and SequenceMatcher(None, left_text, right_text).ratio() >= config.text_similarity_threshold
    ):
        content_signals.append("TEXT_SIMILARITY")
    if left.media_sha256 is not None and left.media_sha256 == right.media_sha256:
        content_signals.append("MEDIA_HASH")
    if left.perceptual_hash is not None and right.perceptual_hash is not None:
        distance = _perceptual_hash_distance(left.perceptual_hash, right.perceptual_hash)
        if distance is not None and distance <= config.perceptual_hash_max_distance:
            content_signals.append("PERCEPTUAL_HASH")
    if not content_signals:
        return ()
    return (
        "SAME_PARCEL",
        "SPATIAL_PROXIMITY",
        "TIME_WINDOW",
        *content_signals,
    )


def group_duplicate_complaints(
    complaints: Sequence[Complaint],
    config: ComplaintDeduplicationConfig,
) -> ComplaintDeduplicationResult:
    """Group probable duplicates while returning all original complaints unchanged."""

    if len({item.complaint_id for item in complaints}) != len(complaints):
        raise DatasetValidationError("complaint identifiers must be unique")
    parents = list(range(len(complaints)))
    pair_signals: dict[tuple[int, int], tuple[str, ...]] = {}

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left_index, left in enumerate(complaints):
        for right_index in range(left_index + 1, len(complaints)):
            signals = _duplicate_signals(left, complaints[right_index], config)
            if signals:
                pair_signals[(left_index, right_index)] = signals
                union(left_index, right_index)

    components: dict[int, list[int]] = {}
    for index in range(len(complaints)):
        components.setdefault(find(index), []).append(index)
    groups: list[ComplaintGroup] = []
    for indices in components.values():
        ordered = sorted(
            (complaints[index] for index in indices),
            key=lambda item: (item.received_at, item.complaint_id),
        )
        ids = tuple(item.complaint_id for item in ordered)
        signals = tuple(
            sorted(
                {
                    signal
                    for (left, right), values in pair_signals.items()
                    if left in indices and right in indices
                    for signal in values
                }
            )
        )
        digest = sha256("|".join(ids).encode()).hexdigest()[:16]
        groups.append(
            ComplaintGroup(
                group_id=f"complaint-group-{digest}",
                representative_complaint_id=ids[0],
                complaint_ids=ids,
                duplicate_signals=signals,
            )
        )
    groups.sort(key=lambda item: item.group_id)
    return ComplaintDeduplicationResult(
        original_complaints=tuple(complaints),
        groups=tuple(groups),
    )
