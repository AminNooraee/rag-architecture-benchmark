"""Tests for canonical query result schema, failure handling, and timing/usage validation."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from rag_benchmark.errors.models import BenchmarkErrorType
from rag_benchmark.schemas.evidence import EvidenceType, RetrievedEvidence
from rag_benchmark.schemas.result import (
    QueryResult,
    QueryStatus,
    QueryTiming,
    QueryUsage,
)


def test_successful_query_result() -> None:
    """Test standard successful query result serialization."""
    evidence = RetrievedEvidence(
        document_id="doc-1",
        chunk_id="chunk-1",
        text="میدان آزادی در غرب تهران قرار دارد.",
        score=0.95,
        rank=1,
        evidence_type=EvidenceType.CHUNK,
    )
    result = QueryResult.create_success(
        question_id="q-azadi",
        system="traditional_rag",
        variant="dense_top5",
        answer="میدان آزادی در بخش غربی پایتخت ایران (تهران) واقع شده است.",
        retrieved_evidence=[evidence],
        usage=QueryUsage(llm_calls=1, embedding_calls=1, input_tokens=250, output_tokens=45),
        timing=QueryTiming(retrieval_ms=18.5, generation_ms=310.2, total_ms=328.7),
        metadata={"model": "persian-llama"},
    )

    assert result.status == QueryStatus.SUCCESS
    assert result.answer is not None
    assert "میدان آزادی" in result.answer
    assert result.error is None
    assert result.usage.input_tokens == 250
    assert result.timing.total_ms == 328.7


def test_failed_query_result_is_valid() -> None:
    """CRITICAL: A failed query must still produce a valid canonical benchmark result."""
    result = QueryResult.create_failure(
        question_id="q-timeout-1",
        system="lightrag",
        variant="hybrid",
        error_message="Gateway Timeout after 60s connecting to inference backend",
        error_type=BenchmarkErrorType.TIMEOUT,
        status=QueryStatus.TIMEOUT,
        usage=QueryUsage(llm_calls=1, embedding_calls=1, input_tokens=100, output_tokens=0),
        timing=QueryTiming(retrieval_ms=45.0, generation_ms=60000.0, total_ms=60045.0),
        retry_count=2,
    )

    assert result.status == QueryStatus.TIMEOUT
    assert result.answer is None
    assert result.error is not None
    assert result.error.error_type == BenchmarkErrorType.TIMEOUT
    assert "Gateway Timeout" in result.error.message
    assert result.error.retry_count == 2

    # Verify JSON serialization of failed query
    json_str = json.dumps(result.model_dump(mode="json"), ensure_ascii=False)
    assert "Gateway Timeout" in json_str
    deserialized = QueryResult.model_validate(json.loads(json_str))
    assert deserialized.status == QueryStatus.TIMEOUT
    assert deserialized.error is not None
    assert deserialized.error.error_type == BenchmarkErrorType.TIMEOUT


def test_timing_rejection_of_nan_and_infinity() -> None:
    """Strict JSON: timing values must not be NaN, Infinity, or -Infinity."""
    with pytest.raises(ValidationError):
        QueryTiming(retrieval_ms=float("nan"))

    with pytest.raises(ValidationError):
        QueryTiming(generation_ms=float("inf"))

    with pytest.raises(ValidationError):
        QueryTiming(total_ms=float("-inf"))


def test_timing_rejection_of_negative_values() -> None:
    """Timing values cannot be negative."""
    with pytest.raises(ValidationError):
        QueryTiming(retrieval_ms=-5.0)

    with pytest.raises(ValidationError):
        QueryTiming(generation_ms=-1.0)


def test_usage_rejection_of_negative_values() -> None:
    """Usage values cannot be negative."""
    with pytest.raises(ValidationError):
        QueryUsage(llm_calls=-1)

    with pytest.raises(ValidationError):
        QueryUsage(input_tokens=-10)


def test_query_result_persian_unicode_roundtrip() -> None:
    """Persian answer and metadata in QueryResult survive roundtrip."""
    result = QueryResult.create_success(
        question_id="q-uni",
        system="neo4j_graphrag",
        answer="پاسخ کامل همراه با نیم‌فاصله: این یک دستاورد بزرگ است.",
        metadata={"یادداشت": "پاسخ مدل بهینه است"},
    )

    json_str = json.dumps(result.model_dump(mode="json"), ensure_ascii=False)
    assert "نیم‌فاصله" in json_str
    assert "یادداشت" in json_str

    deserialized = QueryResult.model_validate(json.loads(json_str))
    assert deserialized.answer == result.answer
    assert deserialized.metadata["یادداشت"] == "پاسخ مدل بهینه است"
