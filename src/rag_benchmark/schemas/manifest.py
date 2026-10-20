"""Run manifest schema for benchmark reproducibility."""

from __future__ import annotations

import platform
import sys
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from rag_benchmark.config.models import sanitize_dict_for_manifest
from rag_benchmark.schemas.document import _check_json_serializable


class RunStatus(StrEnum):
    """Execution status of the entire benchmark run."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class RuntimeMetadata(BaseModel):
    """Environment and platform details for benchmark reproducibility."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    python_version: str = Field(default_factory=lambda: sys.version.split()[0])
    os_name: str = Field(default_factory=platform.system)
    os_version: str = Field(default_factory=platform.version)
    architecture: str = Field(default_factory=lambda: platform.machine())
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("extra")
    @classmethod
    def validate_extra(cls, v: dict[str, Any]) -> dict[str, Any]:
        return _check_json_serializable(v, "extra")


class RunManifest(BaseModel):
    """Reproducible record of a benchmark run execution."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(
        ...,
        min_length=1,
        description="Globally unique identifier for this benchmark execution run.",
    )
    system: str = Field(
        ...,
        min_length=1,
        description="Name of the RAG system under benchmark.",
    )
    variant: str = Field(
        default="baseline",
        description="Name of the specific architecture variant.",
    )
    system_version: str = Field(
        default="0.1.0",
        description="Version of the RAG system implementation or benchmark harness.",
    )
    git_commit: str | None = Field(
        default=None,
        description="Git commit SHA of the repository at run execution time.",
    )
    dataset_version: str = Field(
        default="",
        description="Version or checksum of the Persian document corpus.",
    )
    question_set_version: str = Field(
        default="",
        description="Version or checksum of the benchmark question set.",
    )
    llm_model: str = Field(
        default="",
        description="Identifier of the LLM evaluated.",
    )
    embedding_model: str = Field(
        default="",
        description="Identifier of the embedding model evaluated.",
    )
    benchmark_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Sanitized configuration dictionary (secrets strictly excluded).",
    )
    runtime_metadata: RuntimeMetadata = Field(
        default_factory=RuntimeMetadata,
        description="Hardware and Python runtime environment metadata.",
    )
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timezone-aware UTC timestamp when run began.",
    )
    finished_at: datetime | None = Field(
        default=None,
        description="Timezone-aware UTC timestamp when run finished.",
    )
    status: RunStatus = Field(
        default=RunStatus.RUNNING,
        description="Current lifecycle status of the run.",
    )
    total_queries: int = Field(default=0, ge=0)
    successful_queries: int = Field(default=0, ge=0)
    failed_queries: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional serializable run metadata.",
    )

    @field_validator("run_id", "system")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty or whitespace only")
        return v

    @field_validator("benchmark_config")
    @classmethod
    def validate_sanitized_config(cls, v: dict[str, Any]) -> dict[str, Any]:
        _check_json_serializable(v, "benchmark_config")
        return sanitize_dict_for_manifest(v)

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v: dict[str, Any]) -> dict[str, Any]:
        return _check_json_serializable(v, "metadata")

    def finalize(
        self,
        status: RunStatus = RunStatus.COMPLETED,
        total_queries: int | None = None,
        successful_queries: int | None = None,
        failed_queries: int | None = None,
    ) -> RunManifest:
        """Create a finalized copy of this manifest with completed/failed status."""
        updates: dict[str, Any] = {
            "status": status,
            "finished_at": datetime.now(UTC),
        }
        if total_queries is not None:
            updates["total_queries"] = total_queries
        if successful_queries is not None:
            updates["successful_queries"] = successful_queries
        if failed_queries is not None:
            updates["failed_queries"] = failed_queries

        return self.model_copy(update=updates)
