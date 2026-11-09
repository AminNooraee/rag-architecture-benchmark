"""Dataset‑level manifest model.

Contains reproducibility metadata required by the benchmark harness.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SourceInfo(BaseModel):
    source_name: str
    source_version: str
    source_checksum: str
    document_count: int
    question_count: int

    model_config = ConfigDict(extra="forbid")


class BenchmarkDatasetManifest(BaseModel):
    dataset_name: str
    dataset_version: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    builder_version: str
    sources: list[SourceInfo]
    document_count: int
    question_count: int
    content_checksum: str
    # Optional aggregates (can be empty dicts)
    question_type_counts: dict[str, int] = Field(default_factory=dict)
    scope_counts: dict[str, int] = Field(default_factory=dict)
    answerable_counts: dict[str, int] = Field(default_factory=dict)
    split_counts: dict[str, int] = Field(default_factory=dict)
    statistics: Any  # populated with DatasetStatistics instance

    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("created_at", mode="before")
    @classmethod
    def ensure_utc(cls, v: datetime | str) -> datetime:
        if isinstance(v, str):
            dt = datetime.fromisoformat(v)
        else:
            dt = v
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt
