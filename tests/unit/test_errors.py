"""Tests for structured error models and exception hierarchies."""

from __future__ import annotations

import json

from rag_benchmark.errors.models import (
    ArtifactCollisionError,
    BenchmarkErrorType,
    BenchmarkException,
    ConfigurationError,
    StructuredError,
    UnsupportedOperationError,
)


def test_all_benchmark_error_types_present() -> None:
    """Verify all 8 mandatory error types are available."""
    expected = {
        "timeout",
        "connection_error",
        "invalid_structured_output",
        "retrieval_error",
        "generation_error",
        "indexing_error",
        "configuration_error",
        "unknown",
    }
    actual = {et.value for et in BenchmarkErrorType}
    assert expected == actual


def test_structured_error_creation_and_timezone() -> None:
    """StructuredError must record timezone-aware UTC timestamp."""
    err = StructuredError(
        question_id="q-err-01",
        system="traditional_rag",
        variant="baseline",
        error_type=BenchmarkErrorType.CONNECTION_ERROR,
        message="Failed to connect to vector database at 127.0.0.1:8000",
        retry_count=1,
    )

    assert err.timestamp.tzinfo is not None
    assert err.error_type == BenchmarkErrorType.CONNECTION_ERROR
    assert err.retry_count == 1

    # Round trip
    json_str = json.dumps(err.model_dump(mode="json"), ensure_ascii=False)
    deserialized = StructuredError.model_validate(json.loads(json_str))
    assert deserialized.timestamp.tzinfo is not None
    assert deserialized.message == err.message


def test_benchmark_exception_conversion() -> None:
    """BenchmarkException should convert cleanly to StructuredError."""
    exc = BenchmarkException(
        message="خطای زمان پاسخ مدل زبانی",
        error_type=BenchmarkErrorType.TIMEOUT,
        question_id="q-exc-1",
        system="raptor",
        variant="tree_depth_3",
        metadata={"timeout_limit": 30},
    )

    structured = exc.to_structured_error(retry_count=2)
    assert structured.error_type == BenchmarkErrorType.TIMEOUT
    assert structured.question_id == "q-exc-1"
    assert structured.system == "raptor"
    assert structured.message == "خطای زمان پاسخ مدل زبانی"
    assert structured.retry_count == 2
    assert structured.metadata["timeout_limit"] == 30


def test_specific_subclass_exceptions() -> None:
    """Test ConfigurationError, UnsupportedOperationError, ArtifactCollisionError."""
    cfg_err = ConfigurationError("Missing generation section")
    assert cfg_err.error_type == BenchmarkErrorType.CONFIGURATION_ERROR

    unsupported_err = UnsupportedOperationError("Delete not supported")
    assert isinstance(unsupported_err, BenchmarkException)

    collision_err = ArtifactCollisionError("Cannot overwrite run directory")
    assert isinstance(collision_err, BenchmarkException)
