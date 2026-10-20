"""Tests for reproducible run manifests, lifecycle transitions, and safety."""

from __future__ import annotations

from pathlib import Path

from rag_benchmark.config.models import BenchmarkConfig, LLMConfig
from rag_benchmark.runs.manifest import create_manifest, load_manifest, save_manifest
from rag_benchmark.schemas.manifest import RunStatus


def test_manifest_start_lifecycle() -> None:
    """Initial manifest starts with RUNNING status and timezone-aware started_at."""
    cfg = BenchmarkConfig(
        system="traditional_rag",
        variant="baseline",
        llm=LLMConfig(model="gpt-4o"),
    )
    manifest = create_manifest(
        run_id="run-20261020-01",
        system="traditional_rag",
        config=cfg,
        variant="baseline",
    )

    assert manifest.run_id == "run-20261020-01"
    assert manifest.system == "traditional_rag"
    assert manifest.status == RunStatus.RUNNING
    assert manifest.started_at.tzinfo is not None
    assert manifest.finished_at is None
    assert manifest.runtime_metadata.python_version != ""
    assert manifest.runtime_metadata.os_name != ""


def test_manifest_finalize_lifecycle() -> None:
    """Finalizing updates status, sets finished_at, and records query metrics."""
    manifest = create_manifest(
        run_id="run-finalize-test",
        system="raptor",
    )

    finalized = manifest.finalize(
        status=RunStatus.COMPLETED,
        total_queries=100,
        successful_queries=95,
        failed_queries=5,
    )

    assert finalized.status == RunStatus.COMPLETED
    assert finalized.finished_at is not None
    assert finalized.finished_at.tzinfo is not None
    assert finalized.finished_at >= finalized.started_at
    assert finalized.total_queries == 100
    assert finalized.successful_queries == 95
    assert finalized.failed_queries == 5


def test_manifest_safe_config_no_secrets() -> None:
    """Manifest benchmark_config must be sanitized with secrets redacted."""
    cfg = BenchmarkConfig(
        system="neo4j_graphrag",
        llm=LLMConfig(model="qwen-72b", endpoint="https://user:password@llm.example.com"),
    )
    manifest = create_manifest(
        run_id="run-safe-test",
        system="neo4j_graphrag",
        config=cfg,
    )

    assert "password" not in manifest.benchmark_config["llm"]["endpoint"]
    assert "[REDACTED]" in manifest.benchmark_config["llm"]["endpoint"]


def test_manifest_file_save_and_load_roundtrip(tmp_path: Path) -> None:
    """Save and load manifest file preserving Persian metadata."""
    manifest_file = tmp_path / "test_run" / "manifest.json"

    manifest = create_manifest(
        run_id="run-persian-01",
        system="traditional_rag",
        metadata={"توضیحات": "اجرای آزمایشی بنچمارک روی دادگان فارسی"},
    )
    finalized = manifest.finalize(
        status=RunStatus.COMPLETED,
        total_queries=10,
        successful_queries=10,
        failed_queries=0,
    )

    save_manifest(finalized, manifest_file)
    assert manifest_file.exists()

    loaded = load_manifest(manifest_file)
    assert loaded.run_id == finalized.run_id
    assert loaded.status == RunStatus.COMPLETED
    assert loaded.metadata["توضیحات"] == "اجرای آزمایشی بنچمارک روی دادگان فارسی"
    assert loaded.started_at.tzinfo is not None
    assert loaded.finished_at is not None
