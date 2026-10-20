"""Validated shared configuration models for the benchmark harness."""

from __future__ import annotations

import copy
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from rag_benchmark.schemas.strict_json import validate_strict_json_dict
from rag_benchmark.security.sanitizer import (
    is_sensitive_key,
    sanitize_dict_for_manifest,
)


class GenerationConfig(BaseModel):
    """Configuration for LLM generation stage."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1)
    seed: int = Field(default=42)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("extra")
    @classmethod
    def validate_extra(cls, v: dict[str, Any]) -> dict[str, Any]:
        return validate_strict_json_dict(v, "extra")


class RetrievalConfig(BaseModel):
    """Configuration for retrieval stage."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    top_k: int = Field(default=5, ge=1)
    score_threshold: float | None = Field(default=None, ge=0.0)
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("extra")
    @classmethod
    def validate_extra(cls, v: dict[str, Any]) -> dict[str, Any]:
        return validate_strict_json_dict(v, "extra")


class EmbeddingConfig(BaseModel):
    """Configuration for embedding models."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    model: str = Field(default="", description="Embedding model name or identifier.")
    dimension: int | None = Field(default=None, ge=1, description="Embedding vector dimension.")
    endpoint: str | None = Field(default=None, description="Optional embedding endpoint URL.")
    batch_size: int = Field(default=32, ge=1)
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("extra")
    @classmethod
    def validate_extra(cls, v: dict[str, Any]) -> dict[str, Any]:
        return validate_strict_json_dict(v, "extra")


class LLMConfig(BaseModel):
    """Configuration for LLM inference (credentials MUST remain in environment variables)."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    model: str = Field(default="", description="LLM model name.")
    endpoint: str = Field(default="", description="LLM endpoint or base URL.")
    timeout_seconds: float = Field(default=60.0, ge=1.0)
    max_retries: int = Field(default=3, ge=0)
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("extra")
    @classmethod
    def validate_extra(cls, v: dict[str, Any]) -> dict[str, Any]:
        return validate_strict_json_dict(v, "extra")


class BenchmarkMetaConfig(BaseModel):
    """Metadata regarding dataset, versions, and execution parameters."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    language: str = Field(default="fa", description="Target benchmark language.")
    seed: int = Field(default=42, description="Global random seed.")
    dataset_version: str = Field(default="", description="Dataset version tag or hash.")
    question_set_version: str = Field(default="", description="Question set version tag or hash.")
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("extra")
    @classmethod
    def validate_extra(cls, v: dict[str, Any]) -> dict[str, Any]:
        return validate_strict_json_dict(v, "extra")


class BenchmarkConfig(BaseModel):
    """Root configuration model for benchmark runs.

    Guarantees that credentials remain outside serializable configuration.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    system: str = Field(default="", description="Name of the RAG architecture under test.")
    variant: str = Field(default="baseline", description="Variant name.")
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    benchmark: BenchmarkMetaConfig = Field(default_factory=BenchmarkMetaConfig)
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("extra")
    @classmethod
    def validate_no_secrets_in_extra(cls, v: dict[str, Any]) -> dict[str, Any]:
        validated = validate_strict_json_dict(v, "extra")
        for k in validated:
            if is_sensitive_key(k):
                raise ValueError(
                    f"Configuration key '{k}' appears to contain sensitive credential information. "
                    "API keys, tokens, and passwords must be passed via environment variables only."
                )
        return validated

    def to_safe_dict(self) -> dict[str, Any]:
        """Convert configuration to a safe dictionary suitable for run manifests.

        Ensures no secrets or credentials can be leaked into manifests.
        """
        raw = self.model_dump(mode="json")
        return sanitize_dict_for_manifest(copy.deepcopy(raw))
