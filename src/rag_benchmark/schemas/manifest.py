"""Run manifest schema for benchmark reproducibility and lifecycle tracking."""

from __future__ import annotations

import platform
import sys
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rag_benchmark.errors.models import InvalidLifecycleError, normalize_datetime_to_utc
from rag_benchmark.schemas.strict_json import validate_strict_json_dict
from rag_benchmark.security.sanitizer import sanitize_dict_for_manifest


class RunStatus(StrEnum):
    """Execution status of the entire benchmark run."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class RuntimeMetadata(BaseModel):
    """Environment and platform details for benchmark reproducibility."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_assignment=True)

    python_version: str = Field(default_factory=lambda: sys.version.split()[0])
    os_name: str = Field(default_factory=platform.system)
    os_version: str = Field(default_factory=platform.version)
    architecture: str = Field(default_factory=lambda: platform.machine())
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("extra")
    @classmethod
    def validate_extra(cls, v: dict[str, Any]) -> dict[str, Any]:
        strict = validate_strict_json_dict(v, "extra")
        return sanitize_dict_for_manifest(strict)


class RunManifest(BaseModel):
    """Reproducible record of a benchmark run execution."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

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

    @field_validator("started_at", mode="before")
    @classmethod
    def validate_started_at(cls, v: Any) -> datetime:
        return normalize_datetime_to_utc(v)

    @field_validator("finished_at", mode="before")
    @classmethod
    def validate_finished_at(cls, v: Any) -> datetime | None:
        if v is None:
            return None
        return normalize_datetime_to_utc(v)

    @field_validator("benchmark_config")
    @classmethod
    def validate_sanitized_config(cls, v: dict[str, Any]) -> dict[str, Any]:
        strict = validate_strict_json_dict(v, "benchmark_config")
        return sanitize_dict_for_manifest(strict)

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v: dict[str, Any]) -> dict[str, Any]:
        strict = validate_strict_json_dict(v, "metadata")
        return sanitize_dict_for_manifest(strict)

    @model_validator(mode="after")
    def validate_lifecycle_invariants(self) -> Self:
        """Enforce strict lifecycle and counter invariants across fields."""
        if self.status == RunStatus.RUNNING:
            if self.finished_at is not None:
                raise ValueError("finished_at must be None while manifest status is RUNNING")
        else:  # COMPLETED or FAILED
            if self.finished_at is None:
                raise ValueError(
                    f"finished_at must be set when status is terminal '{self.status.value}'"
                )
            if self.finished_at < self.started_at:
                raise ValueError(
                    f"finished_at ({self.finished_at.isoformat()}) earlier than "
                    f"started_at ({self.started_at.isoformat()})"
                )
            if self.total_queries != self.successful_queries + self.failed_queries:
                raise ValueError(
                    f"Terminal count mismatch: total ({self.total_queries}) != "
                    f"success ({self.successful_queries}) + failed ({self.failed_queries})"
                )
        return self

    def finalize(
        self,
        status: RunStatus = RunStatus.COMPLETED,
        total_queries: int | None = None,
        successful_queries: int | None = None,
        failed_queries: int | None = None,
    ) -> RunManifest:
        """Create a finalized copy of this manifest with completed/failed status.

        Validates all transitions: only RUNNING -> COMPLETED or RUNNING -> FAILED are allowed.
        """
        if self.status != RunStatus.RUNNING:
            raise InvalidLifecycleError(
                f"Cannot finalize: current status is already terminal '{self.status.value}'"
            )

        if status == RunStatus.RUNNING:
            raise InvalidLifecycleError("Cannot finalize manifest to RUNNING status")

        raw_data = self.model_dump()
        raw_data["status"] = status
        raw_data["finished_at"] = datetime.now(UTC)

        if total_queries is not None:
            raw_data["total_queries"] = total_queries
        if successful_queries is not None:
            raw_data["successful_queries"] = successful_queries
        if failed_queries is not None:
            raw_data["failed_queries"] = failed_queries

        # Instantiate fresh instance to run full validator suite
        return RunManifest.model_validate(raw_data)
