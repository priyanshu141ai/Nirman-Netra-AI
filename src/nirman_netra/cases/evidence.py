"""Append-only evidence timeline and chain-of-custody verification."""

from collections.abc import Mapping, Sequence
from datetime import datetime

from pydantic import JsonValue

from nirman_netra.cases.contracts import (
    ActorIdentity,
    EvidenceAsset,
    EvidenceIntegrityReport,
    EvidenceTimeline,
    TimelineEvent,
    TimelineEventType,
    timeline_event_hash,
)
from nirman_netra.exceptions import AuditIntegrityError, EvidenceIntegrityError
from nirman_netra.utils import content_hash, deterministic_id


def append_timeline_event(
    timeline: EvidenceTimeline,
    *,
    case_id: str,
    event_type: TimelineEventType,
    actor: ActorIdentity,
    timestamp: datetime,
    evidence_asset_id: str | None = None,
    details: dict[str, JsonValue] | None = None,
) -> EvidenceTimeline:
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise AuditIntegrityError("timeline timestamp must be timezone-aware")
    safe_details = details or {}
    event_id = deterministic_id(
        "case-timeline",
        case_id,
        event_type.value,
        actor.actor_id,
        timestamp.isoformat(),
        timeline.head_hash or "root",
    )
    unsigned = TimelineEvent(
        event_id=event_id,
        case_id=case_id,
        event_type=event_type,
        actor_id=actor.actor_id,
        timestamp=timestamp,
        evidence_asset_id=evidence_asset_id,
        details=safe_details,
        previous_event_hash=timeline.head_hash,
        event_hash="0" * 64,
    )
    event = unsigned.model_copy(update={"event_hash": timeline_event_hash(unsigned)})
    return EvidenceTimeline(events=(*timeline.events, event), head_hash=event.event_hash)


def verify_evidence_content(asset: EvidenceAsset, content: bytes) -> None:
    actual = content_hash(content)
    if actual != asset.content_sha256:
        raise EvidenceIntegrityError(
            f"evidence hash mismatch for {asset.asset_id}: expected {asset.content_sha256}"
        )


def verify_chain_of_custody(
    assets: Sequence[EvidenceAsset],
    content_by_asset_id: Mapping[str, bytes] | None = None,
) -> EvidenceIntegrityReport:
    asset_ids: set[str] = set()
    content_hashes: dict[str, str] = {}
    indexed: dict[str, EvidenceAsset] = {}
    for asset in assets:
        if asset.asset_id in asset_ids:
            raise EvidenceIntegrityError(f"duplicate evidence asset ID: {asset.asset_id}")
        asset_ids.add(asset.asset_id)
        previous_asset_id = content_hashes.get(asset.content_sha256)
        if previous_asset_id is not None:
            raise EvidenceIntegrityError(
                f"duplicate evidence content: {previous_asset_id}, {asset.asset_id}"
            )
        content_hashes[asset.content_sha256] = asset.asset_id
        indexed[asset.asset_id] = asset
    source_links: list[tuple[str, str]] = []
    for asset in assets:
        if asset.source_asset_id is not None:
            source = indexed.get(asset.source_asset_id)
            if source is None:
                raise EvidenceIntegrityError(
                    f"missing source asset {asset.source_asset_id} for {asset.asset_id}"
                )
            if asset.original_asset_sha256 != source.content_sha256:
                raise EvidenceIntegrityError(
                    f"source hash linkage mismatch for derived evidence {asset.asset_id}"
                )
            source_links.append((source.asset_id, asset.asset_id))
        if content_by_asset_id is not None:
            content = content_by_asset_id.get(asset.asset_id)
            if content is None:
                raise EvidenceIntegrityError(f"evidence content unavailable: {asset.asset_id}")
            verify_evidence_content(asset, content)
    return EvidenceIntegrityReport(
        verified_asset_ids=tuple(asset.asset_id for asset in assets),
        verified_source_links=tuple(source_links),
    )
