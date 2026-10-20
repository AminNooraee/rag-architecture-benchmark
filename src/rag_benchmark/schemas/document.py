"""Canonical Persian document schema."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from rag_benchmark.schemas.strict_json import validate_strict_json_dict


class DocumentSection(BaseModel):
    """A distinct structural subsection of a document (e.g., chapter, section, paragraph)."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_assignment=True)

    section_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for the section within the document.",
    )
    heading: str = Field(
        default="",
        description="Section heading or title (e.g. Persian section title).",
    )
    text: str = Field(
        default="",
        description="Text content of the section.",
    )
    order: int = Field(
        default=0,
        ge=0,
        description="0-indexed sequence position of this section within the document.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Strict JSON-native metadata specific to this section.",
    )

    @field_validator("section_id")
    @classmethod
    def validate_section_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("section_id cannot be empty or whitespace only")
        return v

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v: dict[str, Any]) -> dict[str, Any]:
        return validate_strict_json_dict(v, "metadata")


class BenchmarkDocument(BaseModel):
    """Canonical representation of a document in the Persian RAG benchmark."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_assignment=True)

    document_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for the document.",
    )
    title: str = Field(
        ...,
        min_length=1,
        description="Document title, preserved in full Persian Unicode.",
    )
    text: str = Field(
        default="",
        description="Full text content of the document.",
    )
    language: str = Field(
        default="fa",
        description="BCP-47 language code, defaults to 'fa' (Persian).",
    )
    source: str = Field(
        default="",
        description="Origin or dataset source name.",
    )
    url: str | None = Field(
        default=None,
        description="Optional origin URL or canonical URI.",
    )
    sections: list[DocumentSection] = Field(
        default_factory=list,
        description="Ordered structural sections contained within this document.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Strict JSON-native metadata associated with the document.",
    )

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("document_id cannot be empty or whitespace only")
        return v

    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("title cannot be empty or whitespace only")
        return v

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v: dict[str, Any]) -> dict[str, Any]:
        return validate_strict_json_dict(v, "metadata")
