"""Retrieved evidence schema independent of underlying RAG architecture."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from rag_benchmark.schemas.strict_json import validate_strict_json_dict


class EvidenceType(StrEnum):
    """Types of evidence that different RAG architectures may retrieve."""

    CHUNK = "chunk"
    DOCUMENT = "document"
    ENTITY = "entity"
    RELATIONSHIP = "relationship"
    COMMUNITY_REPORT = "community_report"
    SUMMARY_NODE = "summary_node"
    TREE_NODE = "tree_node"
    CYPHER_RESULT = "cypher_result"


class RetrievedEvidence(BaseModel):
    """A single unit of retrieved evidence from any RAG architecture."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_assignment=True)

    document_id: str | None = Field(
        default=None,
        description="ID of the original source document, preserving provenance.",
    )
    chunk_id: str | None = Field(
        default=None,
        description="ID of the retrieved chunk, entity, or node if applicable.",
    )
    text: str = Field(
        ...,
        description="Retrieved Persian or multilingual textual evidence content.",
    )
    score: float | None = Field(
        default=None,
        description="Relevance or similarity score assigned by the retriever.",
    )
    rank: int | None = Field(
        default=None,
        ge=1,
        description="1-based rank position of this item in the retrieval list.",
    )
    evidence_type: EvidenceType = Field(
        default=EvidenceType.CHUNK,
        description="Structural category of the retrieved evidence.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Strict JSON-native auxiliary metadata (e.g. node labels, cypher queries).",
    )

    @field_validator("score")
    @classmethod
    def validate_score(cls, v: float | None) -> float | None:
        if v is not None and (math.isnan(v) or math.isinf(v)):
            raise ValueError("Score cannot be NaN, Infinity, or -Infinity")
        return v

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v: dict[str, Any]) -> dict[str, Any]:
        return validate_strict_json_dict(v, "metadata")
