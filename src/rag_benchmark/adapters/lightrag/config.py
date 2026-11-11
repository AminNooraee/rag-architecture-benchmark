"""Official HKUDS LightRAG adapter-specific validated configuration."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class LightRAGConfig(BaseModel):
    """Validated HKUDS LightRAG specific options.

    Carried inside ``BenchmarkConfig.extra["lightrag"]`` so that system-specific
    knobs stay validated while the root configuration schema remains shared.

    Chunk sizing note: the benchmark injects a deterministic offline character
    tokenizer by default, so ``chunk_token_size`` / ``chunk_overlap_token_size``
    act as character windows rather than BPE token counts. The defaults mirror
    the upstream LightRAG canonical values (1200 / 100) so the system under test
    runs with its intended chunking shape, without any network BPE download.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    query_mode: Literal["hybrid"] = Field(
        default="hybrid",
        description=(
            "LightRAG QueryParam mode. Phase 3D pins the 'hybrid_no_rerank' "
            "variant to 'hybrid' (combined local entity-level and global "
            "relationship-level KG retrieval); the variant label must match "
            "the run, so every other mode is rejected at validation time."
        ),
    )
    chunk_token_size: int = Field(
        default=1200,
        ge=1,
        description=(
            "Maximum chunk length window for LightRAG F chunking. With the "
            "default offline character tokenizer this is a character count."
        ),
    )
    chunk_overlap_token_size: int = Field(
        default=100,
        ge=0,
        description=(
            "Overlap window repeated between consecutive chunks. With the "
            "default offline character tokenizer this is a character count."
        ),
    )
    top_k: int = Field(
        default=40,
        ge=1,
        description="LightRAG QueryParam top_k (max entities/relations considered).",
    )
    chunk_top_k: int = Field(
        default=20,
        ge=1,
        description="LightRAG QueryParam chunk_top_k (max text chunks considered).",
    )
    max_entity_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Optional per-entity token cap; None lets LightRAG choose.",
    )
    max_relation_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Optional per-relation token cap; None lets LightRAG choose.",
    )
    max_total_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Optional total context token budget; None lets LightRAG choose.",
    )
    embedding_batch_num: int = Field(
        default=10,
        ge=1,
        description="LightRAG embedding batch size for the underlying embedding func.",
    )
    llm_model_max_async: int = Field(
        default=4,
        ge=1,
        description="Maximum concurrent LLM calls allowed by LightRAG.",
    )
    llm_timeout_seconds: float = Field(
        default=240.0,
        gt=0.0,
        description="Per-request timeout for the OpenAI-compatible LLM endpoint.",
    )
    embedding_timeout_seconds: float = Field(
        default=240.0,
        gt=0.0,
        description="Per-request timeout for the OpenAI-compatible embedding endpoint.",
    )
    query_timeout_seconds: float = Field(
        default=180.0,
        gt=0.0,
        description="Wall-clock budget for a single query before TIMEOUT.",
    )

    @model_validator(mode="after")
    def validate_overlap_against_chunk_size(self) -> Self:
        if self.chunk_overlap_token_size >= self.chunk_token_size:
            message = (
                "chunk_overlap_token_size "
                f"({self.chunk_overlap_token_size}) must be strictly smaller than "
                f"chunk_token_size ({self.chunk_token_size})"
            )
            raise ValueError(message)
        return self
