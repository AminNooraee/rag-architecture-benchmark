"""Structured error models and exception types for benchmark runs."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from rag_benchmark.schemas.strict_json import validate_strict_json_dict
from rag_benchmark.security.sanitizer import (
    sanitize_dict_for_manifest,
    sanitize_error_message,
)


def normalize_datetime_to_utc(v: Any) -> datetime:
    """Validate that datetime is timezone-aware and convert to UTC."""
    if isinstance(v, str):
        try:
            parsed = datetime.fromisoformat(v)
            v = parsed
        except Exception as e:
            raise ValueError(f"Invalid ISO datetime string '{v}': {e}") from e

    if not isinstance(v, datetime):
        raise ValueError(f"Expected datetime object or ISO string, got {type(v).__name__}")

    if v.tzinfo is None or v.tzinfo.utcoffset(v) is None:
        raise ValueError("Naive datetimes are rejected; timestamp must be timezone-aware.")

    return v.astimezone(UTC)


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

    model_config = ConfigDict(frozen=True, extra="forbid", validate_assignment=True)

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
        description="Sanitized error message with credentials and secrets redacted.",
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
        description="Debugging metadata (strictly JSON-native and sanitized).",
    )

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        return sanitize_error_message(v)

    @field_validator("timestamp", mode="before")
    @classmethod
    def validate_timestamp(cls, v: Any) -> datetime:
        return normalize_datetime_to_utc(v)

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v: dict[str, Any]) -> dict[str, Any]:
        strict = validate_strict_json_dict(v, "metadata")
        return sanitize_dict_for_manifest(strict)


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
        sanitized_msg = sanitize_error_message(message)
        super().__init__(sanitized_msg)
        self.message = sanitized_msg
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


class InvalidRunIdError(BenchmarkException):
    """Raised when a run ID contains path traversal or invalid filesystem characters."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, error_type=BenchmarkErrorType.CONFIGURATION_ERROR, **kwargs)


class WriterConcurrencyError(ArtifactCollisionError):
    """Raised when another writer currently holds exclusive ownership of the run directory."""


class InvalidLifecycleError(BenchmarkException):
    """Raised when an illegal state transition occurs in manifest or writer lifecycle."""


class ArtifactConsistencyError(InvalidLifecycleError):
    """Raised when durable artifacts are malformed, incomplete, or inconsistent on resume."""


class IdentityMismatchError(BenchmarkException):
    """Raised when a query result, error, or manifest does not match the bound run identity."""


class DuplicateResultError(BenchmarkException):
    """Raised when a duplicate question_id result is submitted to a single run."""


class GroundTruthLeakageError(BenchmarkException):
    """Raised when ground-truth fields are detected in query-time adapter inputs."""


class InvalidCapabilityDeclarationError(BenchmarkException):
    """Raised when an adapter declares a capability but omits implementation."""
