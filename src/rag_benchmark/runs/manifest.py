"""Run manifest management, Git SHA retrieval, atomic writing, and identity helpers."""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from pathlib import Path
from typing import Any

from rag_benchmark.config.models import BenchmarkConfig
from rag_benchmark.errors.models import ConfigurationError
from rag_benchmark.schemas.manifest import RunManifest, RunStatus, RuntimeMetadata
from rag_benchmark.schemas.strict_json import strict_json_dumps


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


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Atomically write JSON data to target path using a temp file and fsync.

    Guarantees that destination file is never corrupted or partially written if
    the operation is interrupted.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = strict_json_dumps(data, indent=2)

    temp_path = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(serialized)
            f.flush()
            os.fsync(f.fileno())

        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def create_manifest(
    run_id: str,
    system: str | None = None,
    config: BenchmarkConfig | None = None,
    variant: str | None = None,
    system_version: str = "0.1.0",
    dataset_version: str | None = None,
    question_set_version: str | None = None,
    llm_model: str | None = None,
    embedding_model: str | None = None,
    metadata: dict[str, Any] | None = None,
    repo_root: Path | None = None,
) -> RunManifest:
    """Create a new running benchmark manifest reconciling identity with BenchmarkConfig.

    Reconciles explicit arguments with BenchmarkConfig:
      - Defaults missing fields from config.
      - Rejects conflicting explicit arguments with ConfigurationError.
    """
    final_system = system
    final_variant = variant or "baseline"
    final_dataset_ver = dataset_version or ""
    final_qset_ver = question_set_version or ""
    final_llm = llm_model or ""
    final_embedding = embedding_model or ""

    if config is not None:
        # Reconcile system
        if config.system:
            if system and system != config.system:
                raise ConfigurationError(
                    f"Conflicting system specification: explicit system '{system}' != "
                    f"config.system '{config.system}'"
                )
            final_system = config.system
        elif not final_system:
            raise ConfigurationError(
                "System name must be specified either explicitly or via BenchmarkConfig.system"
            )

        # Reconcile variant
        if variant and config.variant and variant != config.variant:
            raise ConfigurationError(
                f"Conflicting variant specification: explicit variant '{variant}' != "
                f"config.variant '{config.variant}'"
            )
        if not variant and config.variant:
            final_variant = config.variant

        # Reconcile dataset_version
        if (
            dataset_version
            and config.benchmark.dataset_version
            and dataset_version != config.benchmark.dataset_version
        ):
            raise ConfigurationError(
                f"Conflicting dataset_version: '{dataset_version}' != "
                f"'{config.benchmark.dataset_version}'"
            )
        if not dataset_version and config.benchmark.dataset_version:
            final_dataset_ver = config.benchmark.dataset_version

        # Reconcile question_set_version
        if (
            question_set_version
            and config.benchmark.question_set_version
            and question_set_version != config.benchmark.question_set_version
        ):
            raise ConfigurationError(
                f"Conflicting question_set_version: '{question_set_version}' != "
                f"'{config.benchmark.question_set_version}'"
            )
        if not question_set_version and config.benchmark.question_set_version:
            final_qset_ver = config.benchmark.question_set_version

        # Reconcile llm_model
        if llm_model and config.llm.model and llm_model != config.llm.model:
            raise ConfigurationError(
                f"Conflicting llm_model: '{llm_model}' != '{config.llm.model}'"
            )
        if not llm_model and config.llm.model:
            final_llm = config.llm.model

        # Reconcile embedding_model
        if embedding_model and config.embedding.model and embedding_model != config.embedding.model:
            raise ConfigurationError(
                f"Conflicting embedding_model: '{embedding_model}' != '{config.embedding.model}'"
            )
        if not embedding_model and config.embedding.model:
            final_embedding = config.embedding.model

    if not final_system:
        raise ConfigurationError(
            "System name must be specified either explicitly or via BenchmarkConfig.system"
        )

    git_commit = get_git_commit(repo_root)
    safe_config = config.to_safe_dict() if config else {}

    return RunManifest(
        run_id=run_id,
        system=final_system,
        variant=final_variant,
        system_version=system_version,
        git_commit=git_commit,
        dataset_version=final_dataset_ver,
        question_set_version=final_qset_ver,
        llm_model=final_llm,
        embedding_model=final_embedding,
        benchmark_config=safe_config,
        runtime_metadata=RuntimeMetadata(),
        status=RunStatus.RUNNING,
        metadata=metadata or {},
    )


def save_manifest(manifest: RunManifest, path: Path) -> None:
    """Persist manifest to JSON using atomic replacement and fsync.

    Revalidates the complete manifest object before writing.
    """
    revalidated = RunManifest.model_validate(manifest.model_dump())
    manifest_dict = revalidated.model_dump(mode="json")
    atomic_write_json(path, manifest_dict)


def load_manifest(path: Path) -> RunManifest:
    """Load and validate manifest from JSON file."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return RunManifest.model_validate(data)
