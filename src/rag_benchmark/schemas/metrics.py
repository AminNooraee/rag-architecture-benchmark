"""Metric schemas for future evaluation phases.

Note: No fake metrics are generated in Phase 3A. This module provides
data contracts for downstream metric evaluators (retrieval, generation, efficiency).
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from rag_benchmark.schemas.strict_json import validate_strict_json_dict


class MetricScore(BaseModel):
    """A single evaluation metric score."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_assignment=True)

    name: str = Field(
        ..., min_length=1, description="Metric name (e.g. hit_rate@5, mrr, exact_match)."
    )
    value: float = Field(..., description="Numeric value of the metric.")
    description: str = Field(default="", description="Explanation or definition of the metric.")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Metric-specific calculation details."
    )

    @field_validator("value")
    @classmethod
    def validate_finite_value(cls, v: float) -> float:
        if math.isnan(v) or math.isinf(v):
            raise ValueError("Metric value must be finite (no NaN or Infinity)")
        return v

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v: dict[str, Any]) -> dict[str, Any]:
        return validate_strict_json_dict(v, "metadata")
