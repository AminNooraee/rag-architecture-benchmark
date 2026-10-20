"""Tests for RunArtifactWriter, directory layout, JSONL appending, and overwrite protection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_benchmark.errors.models import ArtifactCollisionError, BenchmarkErrorType
from rag_benchmark.runs.manifest import create_manifest
from rag_benchmark.runs.writer import RunArtifactWriter
from rag_benchmark.schemas.evidence import RetrievedEvidence
from rag_benchmark.schemas.manifest import RunStatus
from rag_benchmark.schemas.result import (
    QueryResult,
    QueryStatus,
    QueryTiming,
    QueryUsage,
)


def test_run_directory_creation_and_manifest_write(tmp_path: Path) -> None:
    """Artifact writer creates directory and writes manifest."""
    writer = RunArtifactWriter(base_dir=tmp_path, run_id="run-init-01")
    manifest = create_manifest(run_id="run-init-01", system="traditional_rag")

    writer.initialize(manifest)

    assert writer.paths.run_dir.exists()
    assert writer.paths.manifest_path.exists()


def test_overwrite_protection_prevents_accidental_clobbering(tmp_path: Path) -> None:
    """Attempting to re-initialize an existing finalized run raises ArtifactCollisionError."""
    writer = RunArtifactWriter(base_dir=tmp_path, run_id="run-clobber-test", allow_overwrite=False)
    manifest = create_manifest(run_id="run-clobber-test", system="traditional_rag")
    writer.initialize(manifest)
    writer.finalize(status=RunStatus.COMPLETED)

    # Attempting to re-initialize without allow_overwrite
    writer2 = RunArtifactWriter(base_dir=tmp_path, run_id="run-clobber-test", allow_overwrite=False)
    with pytest.raises(ArtifactCollisionError, match="already exists"):
        writer2.initialize(manifest)

    # With allow_overwrite=True, it succeeds
    writer_allowed = RunArtifactWriter(
        base_dir=tmp_path, run_id="run-clobber-test", allow_overwrite=True
    )
    writer_allowed.initialize(manifest)
    assert writer_allowed.paths.manifest_path.exists()


def test_jsonl_append_and_persian_unicode(tmp_path: Path) -> None:
    """Verify appending results to raw_results.jsonl and errors.jsonl with Persian Unicode."""
    writer = RunArtifactWriter(base_dir=tmp_path, run_id="run-results-test")
    manifest = create_manifest(run_id="run-results-test", system="traditional_rag")
    writer.initialize(manifest)

    evidence1 = RetrievedEvidence(
        document_id="doc-1",
        chunk_id="ch-1",
        text="تکه متن فارسی با نیم‌فاصله",
        score=0.9,
        rank=1,
    )
    res_success = QueryResult.create_success(
        question_id="q-1",
        system="traditional_rag",
        answer="پاسخ فارسی موفقیت‌آمیز",
        retrieved_evidence=[evidence1],
        usage=QueryUsage(input_tokens=100, output_tokens=20),
        timing=QueryTiming(retrieval_ms=12.0, generation_ms=250.0, total_ms=262.0),
    )
    res_failed = QueryResult.create_failure(
        question_id="q-2",
        system="traditional_rag",
        error_message="خطای اتصال به مدل زبانی",
        error_type=BenchmarkErrorType.CONNECTION_ERROR,
        status=QueryStatus.FAILED,
        timing=QueryTiming(retrieval_ms=10.0, generation_ms=0.0, total_ms=10.0),
    )

    writer.append_result(res_success)
    writer.append_result(res_failed)

    # Check raw_results.jsonl
    assert writer.paths.raw_results_path.exists()
    with open(writer.paths.raw_results_path, encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 2

    # Check Persian Unicode in file text directly (ensure_ascii=False)
    raw_content = writer.paths.raw_results_path.read_text(encoding="utf-8")
    assert "پاسخ فارسی موفقیت‌آمیز" in raw_content
    assert "نیم‌فاصله" in raw_content
    assert "خطای اتصال به مدل زبانی" in raw_content

    # Check errors.jsonl was written for the failed query
    assert writer.paths.errors_path.exists()
    with open(writer.paths.errors_path, encoding="utf-8") as f:
        err_lines = f.readlines()
    assert len(err_lines) == 1
    err_obj = json.loads(err_lines[0])
    assert err_obj["question_id"] == "q-2"
    assert err_obj["error_type"] == "connection_error"

    # Finalize and verify manifest counters
    finalized = writer.finalize(status=RunStatus.COMPLETED)
    assert finalized.total_queries == 2
    assert finalized.successful_queries == 1
    assert finalized.failed_queries == 1


def test_read_results_and_errors(tmp_path: Path) -> None:
    """Verify reading back query results and errors."""
    writer = RunArtifactWriter(base_dir=tmp_path, run_id="run-read-test")
    manifest = create_manifest(run_id="run-read-test", system="test_sys")
    writer.initialize(manifest)

    res1 = QueryResult.create_success(
        question_id="q-read-1",
        system="test_sys",
        answer="پاسخ ۱",
    )
    res2 = QueryResult.create_failure(
        question_id="q-read-2",
        system="test_sys",
        error_message="خطا در بازیابی",
        error_type=BenchmarkErrorType.RETRIEVAL_ERROR,
    )

    writer.append_result(res1)
    writer.append_result(res2)

    results = writer.read_results()
    assert len(results) == 2
    assert results[0].question_id == "q-read-1"
    assert results[0].status == QueryStatus.SUCCESS
    assert results[1].question_id == "q-read-2"
    assert results[1].status == QueryStatus.FAILED

    errors = writer.read_errors()
    assert len(errors) == 1
    assert errors[0].question_id == "q-read-2"
    assert errors[0].error_type == BenchmarkErrorType.RETRIEVAL_ERROR
