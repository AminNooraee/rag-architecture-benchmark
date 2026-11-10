"""Unit tests for the official LightRAG adapter-specific configuration."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag_benchmark.adapters.lightrag.config import LightRAGConfig


def test_benchmark_defaults() -> None:
    config = LightRAGConfig()
    assert config.query_mode == "hybrid"
    assert config.chunk_token_size == 1200
    assert config.chunk_overlap_token_size == 100
    assert config.top_k == 40
    assert config.chunk_top_k == 20
    assert config.max_entity_tokens is None
    assert config.max_relation_tokens is None
    assert config.max_total_tokens is None
    assert config.embedding_batch_num == 10
    assert config.llm_model_max_async == 4
    assert config.llm_timeout_seconds == 240.0
    assert config.embedding_timeout_seconds == 240.0
    assert config.query_timeout_seconds == 180.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("chunk_token_size", 0),
        ("chunk_token_size", -5),
        ("top_k", 0),
        ("chunk_top_k", 0),
        ("embedding_batch_num", 0),
        ("llm_model_max_async", 0),
        ("max_entity_tokens", 0),
        ("llm_timeout_seconds", 0.0),
        ("embedding_timeout_seconds", -1.0),
        ("query_timeout_seconds", 0),
    ],
)
def test_rejects_non_positive_bounds(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        LightRAGConfig(**{field: value})  # type: ignore[arg-type]


def test_overlap_must_be_strictly_smaller_than_chunk_size() -> None:
    with pytest.raises(ValidationError):
        LightRAGConfig(chunk_token_size=100, chunk_overlap_token_size=100)
    with pytest.raises(ValidationError):
        LightRAGConfig(chunk_token_size=100, chunk_overlap_token_size=101)
    # Boundary just under the chunk size is valid.
    config = LightRAGConfig(chunk_token_size=100, chunk_overlap_token_size=99)
    assert config.chunk_overlap_token_size == 99


def test_unknown_keys_are_forbidden() -> None:
    with pytest.raises(ValidationError):
        LightRAGConfig(chunk_size=100)  # type: ignore[call-arg]


def test_invalid_query_mode_rejected() -> None:
    with pytest.raises(ValidationError):
        LightRAGConfig(query_mode="local_global")  # type: ignore[arg-type]


def test_query_mode_accepts_all_lightrag_modes() -> None:
    for mode in ("local", "global", "hybrid", "naive", "mix", "bypass"):
        assert LightRAGConfig(query_mode=mode).query_mode == mode


def test_assignment_validation_enforced() -> None:
    config = LightRAGConfig()
    with pytest.raises(ValidationError):
        config.chunk_overlap_token_size = config.chunk_token_size
