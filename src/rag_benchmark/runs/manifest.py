"""Run manifest management, Git SHA retrieval, and lifecycle helpers."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from rag_benchmark.config.models import BenchmarkConfig
from rag_benchmark.schemas.manifest import RunManifest, RunStatus, RuntimeMetadata


def get_git_commit(cwd: Path | None = None) -> str | None:
    """Safely obtain the current Git HEAD commit hash without throwing."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        sha = result.stdout.strip()
        return sha if sha else None
    except Exception:
        return None


def create_manifest(
    run_id: str,
    system: str,
    config: BenchmarkConfig | None = None,
    variant: str = "baseline",
    system_version: str = "0.1.0",
    dataset_version: str = "",
    question_set_version: str = "",
    llm_model: str = "",
    embedding_model: str = "",
    metadata: dict[str, Any] | None = None,
    repo_root: Path | None = None,
) -> RunManifest:
    """Create a new running benchmark manifest with environment details captured."""
    git_commit = get_git_commit(repo_root)
    safe_config = config.to_safe_dict() if config else {}

    # Extract default model names from config if not explicitly provided
    if config:
        if not llm_model and config.llm.model:
            llm_model = config.llm.model
        if not embedding_model and config.embedding.model:
            embedding_model = config.embedding.model
        if not dataset_version and config.benchmark.dataset_version:
            dataset_version = config.benchmark.dataset_version
        if not question_set_version and config.benchmark.question_set_version:
            question_set_version = config.benchmark.question_set_version

    return RunManifest(
        run_id=run_id,
        system=system,
        variant=variant,
        system_version=system_version,
        git_commit=git_commit,
        dataset_version=dataset_version,
        question_set_version=question_set_version,
        llm_model=llm_model,
        embedding_model=embedding_model,
        benchmark_config=safe_config,
        runtime_metadata=RuntimeMetadata(),
        status=RunStatus.RUNNING,
        metadata=metadata or {},
    )


def save_manifest(manifest: RunManifest, path: Path) -> None:
    """Persist manifest to JSON ensuring Persian Unicode and deterministic formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest_dict = manifest.model_dump(mode="json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest_dict, f, ensure_ascii=False, indent=2)


def load_manifest(path: Path) -> RunManifest:
    """Load and validate manifest from JSON file."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return RunManifest.model_validate(data)
