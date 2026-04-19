"""Pydantic schemas for FFE rule JSON validation.

ISO 27034 : input validation.
ISO 25012 : data quality structural schema.

Document ID: ALICE-FFE-SCHEMAS
Version: 1.0.0
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class RuleModel(BaseModel):
    """One FFE rule as stored in chess-app JSON."""

    uuid: str = Field(..., min_length=1)
    uuid_rfc4122: str
    id: str
    source_ref: str
    document: str
    version: str
    article: str
    texte: str
    conditions: dict[str, Any]
    effet: Literal[
        "restrict_team_composition",
        "restrict_arbitrage",
        "restrict_player_eligibility",
    ]
    exceptions: list[Any] = Field(default_factory=list)
    priority: int = Field(ge=1)
    date_version: str


class MetadataModel(BaseModel):
    """JSON document metadata."""

    championship: str | None = None
    description: str
    version: str
    source: str


class RulesDocument(BaseModel):
    """A full FFE rules JSON document (one per competition)."""

    metadata: MetadataModel
    rules: list[RuleModel] = Field(default_factory=list)
