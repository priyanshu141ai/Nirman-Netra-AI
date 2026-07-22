"""Strict versioned API request and error contracts."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from nirman_netra.application.contracts import ModelRequirement, RuleSetRequirement
from nirman_netra.cases.contracts import ActorRole


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AssetCreate(ApiModel):
    object_key: str = Field(min_length=1, max_length=500)
    municipality_id: str = Field(min_length=1, max_length=96)
    captured_at: datetime | None = None


class ImagePairCreate(ApiModel):
    old_asset_id: str = Field(min_length=1)
    new_asset_id: str = Field(min_length=1)


class PairProcess(ApiModel):
    model_requirement: ModelRequirement
    rule_requirement: RuleSetRequirement
    processing_configuration_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class HumanActor(ApiModel):
    actor_id: str = Field(min_length=1, max_length=96)
    role: ActorRole


class CaseAssign(ApiModel):
    inspector_id: str = Field(min_length=1)
    actor: HumanActor
    reason: str = Field(min_length=1, max_length=1_000)


class CaseReview(ApiModel):
    action: Literal["start", "accept", "dismiss", "evidence_insufficient"]
    actor: HumanActor
    reason: str = Field(min_length=1, max_length=1_000)


class CaseReinspection(ApiModel):
    actor: HumanActor
    reason: str = Field(min_length=1, max_length=1_000)


class ApiErrorBody(ApiModel):
    code: str
    message: str
    request_id: str
    correlation_id: str


class ApiError(ApiModel):
    error: ApiErrorBody
