"""Time-aware permit and municipal rule-set selection."""

from collections.abc import Sequence
from datetime import datetime

from nirman_netra.data.contracts import Permit
from nirman_netra.exceptions import ConfigurationError
from nirman_netra.risk.contracts import (
    MunicipalRuleSet,
    PermitMatch,
    PermitMatchStatus,
)


def _require_aware(value: datetime, field: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ConfigurationError(f"{field} must be timezone-aware")


def match_permit_at_capture(
    permits: Sequence[Permit],
    *,
    parcel_id: str,
    capture_time: datetime,
    approved_plan_version: int,
) -> PermitMatch:
    """Use validity at capture time, never today's permit status alone."""

    _require_aware(capture_time, "capture_time")
    capture_date = capture_time.date()
    candidates = tuple(
        sorted(
            (item for item in permits if item.parcel_id == parcel_id),
            key=lambda item: item.permit_id,
        )
    )
    candidate_ids = tuple(item.permit_id for item in candidates)
    if not candidates:
        status = PermitMatchStatus.NO_MATCH
    else:
        valid_at_capture = tuple(
            item for item in candidates if item.valid_from <= capture_date <= item.valid_until
        )
        matching_plan = tuple(
            item for item in valid_at_capture if item.plan_version == approved_plan_version
        )
        if matching_plan:
            selected = sorted(
                matching_plan,
                key=lambda item: (item.valid_from, item.plan_version, item.permit_id),
                reverse=True,
            )[0]
            return PermitMatch(
                parcel_id=parcel_id,
                capture_date=capture_date,
                requested_plan_version=approved_plan_version,
                status=PermitMatchStatus.VALID_MATCH,
                matched_permit=selected,
                candidate_permit_ids=candidate_ids,
            )
        if valid_at_capture:
            status = PermitMatchStatus.WRONG_PLAN_VERSION
        elif any(item.valid_until < capture_date for item in candidates):
            status = PermitMatchStatus.EXPIRED_AT_CAPTURE
        else:
            status = PermitMatchStatus.NOT_YET_VALID
    return PermitMatch(
        parcel_id=parcel_id,
        capture_date=capture_date,
        requested_plan_version=approved_plan_version,
        status=status,
        matched_permit=None,
        candidate_permit_ids=candidate_ids,
    )


def select_effective_rule_set(
    rule_sets: Sequence[MunicipalRuleSet],
    *,
    municipality_id: str,
    zone: str,
    capture_time: datetime,
) -> MunicipalRuleSet:
    """Select the municipality/zone rule version effective for historical evidence."""

    _require_aware(capture_time, "capture_time")
    capture_date = capture_time.date()
    matching = [
        rule
        for rule in rule_sets
        if rule.municipality_id == municipality_id
        and rule.zone == zone
        and rule.effective_from <= capture_date
        and (rule.effective_until is None or capture_date <= rule.effective_until)
    ]
    if not matching:
        raise ConfigurationError(
            f"no effective municipal rule set for {municipality_id}/{zone} at {capture_date}"
        )
    latest_date = max(rule.effective_from for rule in matching)
    latest = sorted(
        (rule for rule in matching if rule.effective_from == latest_date),
        key=lambda rule: rule.rule_set_id,
    )
    if len(latest) > 1:
        raise ConfigurationError(
            f"ambiguous municipal rule sets for {municipality_id}/{zone} at {capture_date}"
        )
    return latest[0]
