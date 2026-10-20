"""Structured error models and exception types for benchmark runs."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class BenchmarkErrorType(StrEnum):
    """Categorized error types encountered during benchmark indexing, retrieval, or generation."""

    TIMEOUT = "timeout"
    CONNECTION_ERROR = "connection_error"
    INVALID_STRUCTURED_OUTPUT = "invalid_structured_output"
    RETRIEVAL_ERROR = "retrieval_error"
    GENERATION_ERROR = "generation_error"
    INDEXING_ERROR = "indexing_error"
    CONFIGURATION_ERROR = "configuration_error"
    UNKNOWN = "unknown"


class StructuredError(BaseModel):
    """Structured representation of an error occurring during benchmark execution."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    question_id: str | None = Field(
        default=None,
        description="ID of the question being processed when the error occurred, if applicable.",
    )
    system: str = Field(
        default="",
        description="Name of the RAG system under test.",
    )
    variant: str = Field(
        default="",
        description="Variant of the RAG system under test.",
    )
    error_type: BenchmarkErrorType = Field(
        default=BenchmarkErrorType.UNKNOWN,
        description="Standardized error category.",
    )
    message: str = Field(
        ...,
        min_length=1,
        description="Human-readable error message or exception traceback summary.",
    )
    retry_count: int = Field(
        default=0,
        ge=0,
        description="Number of retries attempted before failing.",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timezone-aware UTC timestamp when the error occurred.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional serializable debugging metadata.",
    )


class BenchmarkException(Exception):
    """Base exception for benchmark harness operations."""

    def __init__(
        self,
        message: str,
        error_type: BenchmarkErrorType = BenchmarkErrorType.UNKNOWN,
        question_id: str | None = None,
        system: str = "",
        variant: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.question_id = question_id
        self.system = system
        self.variant = variant
        self.metadata = metadata or {}

    def to_structured_error(self, retry_count: int = 0) -> StructuredError:
        """Convert this exception into a serializable StructuredError."""
        return StructuredError(
            question_id=self.question_id,
            system=self.system,
            variant=self.variant,
            error_type=self.error_type,
            message=self.message,
            retry_count=retry_count,
            metadata=self.metadata,
        )


class ConfigurationError(BenchmarkException):
    """Raised when configuration validation or loading fails."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, error_type=BenchmarkErrorType.CONFIGURATION_ERROR, **kwargs)


class UnsupportedOperationError(BenchmarkException):
    """Raised when an adapter is asked to perform an operation it does not support."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, error_type=BenchmarkErrorType.UNKNOWN, **kwargs)


class ArtifactCollisionError(BenchmarkException):
    """Raised when attempting to overwrite an existing benchmark artifact without permission."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, error_type=BenchmarkErrorType.CONFIGURATION_ERROR, **kwargs)
