"""Validated shared configuration models for the benchmark harness."""

from __future__ import annotations

import copy
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

SENSITIVE_KEY_PATTERNS = {
    "api_key",
    "apikey",
    "secret",
    "password",
    "token",
    "auth",
    "authorization",
    "credential",
    "private_key",
}


def sanitize_dict_for_manifest(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively scrub any keys that resemble credentials or secrets."""
    clean: dict[str, Any] = {}
    for k, v in data.items():
        key_lower = k.lower().replace("-", "_")
        if any(pattern in key_lower for pattern in SENSITIVE_KEY_PATTERNS):
            continue  # Exclude sensitive keys entirely from serialized manifests
        if isinstance(v, dict):
            clean[k] = sanitize_dict_for_manifest(v)
        elif isinstance(v, list):
            clean[k] = [
                sanitize_dict_for_manifest(item) if isinstance(item, dict) else item for item in v
            ]
        else:
            clean[k] = v
    return clean


class GenerationConfig(BaseModel):
    """Configuration for LLM generation stage."""

    model_config = ConfigDict(extra="forbid")

    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1)
    seed: int = Field(default=42)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    extra: dict[str, Any] = Field(default_factory=dict)


class RetrievalConfig(BaseModel):
    """Configuration for retrieval stage."""

    model_config = ConfigDict(extra="forbid")

    top_k: int = Field(default=5, ge=1)
    score_threshold: float | None = Field(default=None, ge=0.0)
    extra: dict[str, Any] = Field(default_factory=dict)


class EmbeddingConfig(BaseModel):
    """Configuration for embedding models."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(default="", description="Embedding model name or identifier.")
    dimension: int | None = Field(default=None, ge=1, description="Embedding vector dimension.")
    endpoint: str | None = Field(default=None, description="Optional embedding endpoint URL.")
    batch_size: int = Field(default=32, ge=1)
    extra: dict[str, Any] = Field(default_factory=dict)


class LLMConfig(BaseModel):
    """Configuration for LLM inference (credentials MUST remain in environment variables)."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(default="", description="LLM model name.")
    endpoint: str = Field(default="", description="LLM endpoint or base URL.")
    timeout_seconds: float = Field(default=60.0, ge=1.0)
    max_retries: int = Field(default=3, ge=0)
    extra: dict[str, Any] = Field(default_factory=dict)


class BenchmarkMetaConfig(BaseModel):
    """Metadata regarding dataset, versions, and execution parameters."""

    model_config = ConfigDict(extra="forbid")

    language: str = Field(default="fa", description="Target benchmark language.")
    seed: int = Field(default=42, description="Global random seed.")
    dataset_version: str = Field(default="", description="Dataset version tag or hash.")
    question_set_version: str = Field(default="", description="Question set version tag or hash.")
    extra: dict[str, Any] = Field(default_factory=dict)


class BenchmarkConfig(BaseModel):
    """Root configuration model for benchmark runs.

    Guarantees that credentials remain outside serializable configuration.
    """

    model_config = ConfigDict(extra="forbid")

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
        for k in v:
            key_lower = k.lower().replace("-", "_")
            if any(pattern in key_lower for pattern in SENSITIVE_KEY_PATTERNS):
                raise ValueError(
                    f"Configuration key '{k}' appears to contain sensitive information. "
                    "API keys, tokens, and passwords must be passed via environment variables only."
                )
        return v

    def to_safe_dict(self) -> dict[str, Any]:
        """Convert configuration to a safe dictionary suitable for run manifests.

        Ensures no secrets or credentials can be leaked into manifests.
        """
        raw = self.model_dump(mode="json")
        return sanitize_dict_for_manifest(copy.deepcopy(raw))
