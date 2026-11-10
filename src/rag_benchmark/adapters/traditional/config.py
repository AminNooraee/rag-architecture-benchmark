"""Traditional RAG adapter-specific validated configuration."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TraditionalRAGConfig(BaseModel):
    """Validated Traditional RAG (dense exact vector search) specific options.

    Carried inside ``BenchmarkConfig.extra["traditional_rag"]`` so that
    system-specific knobs stay validated while the root configuration schema
    remains shared across all benchmark architectures.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    chunk_size: int = Field(
        default=512,
        ge=1,
        description="Maximum chunk length in characters (baseline character-based chunking).",
    )
    overlap: int = Field(
        default=0,
        ge=0,
        description="Trailing characters of a chunk repeated at the start of the next chunk.",
    )

    @model_validator(mode="after")
    def validate_overlap_against_chunk_size(self) -> Self:
        if self.overlap >= self.chunk_size:
            message = (
                f"overlap ({self.overlap}) must be strictly smaller "
                f"than chunk_size ({self.chunk_size})"
            )
            raise ValueError(message)
        return self
