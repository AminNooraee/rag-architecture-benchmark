"""Tests for the exact deterministic cosine vector index."""

import math
from collections.abc import Sequence
from typing import cast

import pytest

from rag_benchmark.adapters.traditional.chunking import TextChunk, make_chunk_id
from rag_benchmark.adapters.traditional.vector_index import (
    ExactCosineIndex,
    cosine_similarity,
    validate_vector,
)


def _chunk(document_id: str, ordinal: int, text: str = "متن") -> TextChunk:
    return TextChunk(
        document_id=document_id,
        chunk_id=make_chunk_id(document_id, ordinal),
        text=text,
        ordinal=ordinal,
        source="test-source",
        char_start=0,
        char_end=len(text),
    )


def test_cosine_similarity_values() -> None:
    assert cosine_similarity((1.0, 0.0), (0.0, 1.0)) == pytest.approx(0.0)
    assert cosine_similarity((1.0, 1.0), (1.0, 1.0)) == pytest.approx(1.0)
    assert cosine_similarity((1.0, 0.0), (-1.0, 0.0)) == pytest.approx(-1.0)
    assert cosine_similarity((0.6, 0.8), (1.0, 0.0)) == pytest.approx(0.6)


def test_cosine_similarity_rejects_mismatch_and_zero() -> None:
    with pytest.raises(ValueError):
        cosine_similarity((1.0, 0.0), (1.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        cosine_similarity((0.0, 0.0), (1.0, 0.0))
    with pytest.raises(ValueError):
        cosine_similarity((1.0, 0.0), (0.0, 0.0))


def test_validate_vector_rejects_invalid_input() -> None:
    with pytest.raises(ValueError):
        validate_vector([])
    with pytest.raises(ValueError):
        validate_vector(cast("Sequence[float]", "abc"))
    with pytest.raises(ValueError):
        validate_vector(b"\x00\x01")
    with pytest.raises(ValueError):
        validate_vector([1.0, cast(float, "x")])
    with pytest.raises(ValueError):
        validate_vector([True, False])
    with pytest.raises(ValueError):
        validate_vector([1.0, float("nan")])
    with pytest.raises(ValueError):
        validate_vector([1.0, float("inf")])
    assert validate_vector([1, 2]) == (1.0, 2.0)


def test_ranking_by_cosine_similarity() -> None:
    index = ExactCosineIndex()
    index.add(_chunk("d1", 0, "A"), (1.0, 0.0))
    index.add(_chunk("d1", 1, "B"), (0.0, 1.0))
    index.add(_chunk("d1", 2, "C"), (0.6, 0.8))
    results = index.search((1.0, 0.0), top_k=3)
    assert [entry.chunk.chunk_id for score, entry in results] == [
        make_chunk_id("d1", 0),
        make_chunk_id("d1", 2),
        make_chunk_id("d1", 1),
    ]
    assert [score for score, _ in results] == pytest.approx([1.0, 0.6, 0.0])


def test_tie_broken_by_chunk_id() -> None:
    index = ExactCosineIndex()
    index.add(_chunk("d1", 1, "B"), (1.0, 1.0))
    index.add(_chunk("d1", 0, "A"), (1.0, 1.0))
    results = index.search((1.0, 1.0), top_k=2)
    assert [entry.chunk.chunk_id for _, entry in results] == [
        make_chunk_id("d1", 0),
        make_chunk_id("d1", 1),
    ]
    assert all(score == pytest.approx(1.0) for score, _ in results)


def test_top_k_limits_results() -> None:
    index = ExactCosineIndex()
    for ordinal in range(4):
        index.add(_chunk("d1", ordinal), (1.0, ordinal / 3.0))
    results = index.search((1.0, 0.0), top_k=2)
    assert len(results) == 2


def test_score_threshold_filters_results() -> None:
    index = ExactCosineIndex()
    index.add(_chunk("d1", 0), (1.0, 0.0))
    index.add(_chunk("d1", 1), (0.6, 0.8))
    index.add(_chunk("d1", 2), (0.0, 1.0))
    assert len(index.search((1.0, 0.0), top_k=5, score_threshold=0.7)) == 1
    assert len(index.search((1.0, 0.0), top_k=5, score_threshold=0.6)) == 2
    assert len(index.search((1.0, 0.0), top_k=5, score_threshold=1.0)) == 1
    assert index.search((1.0, 0.0), top_k=5, score_threshold=1.1) == ()


def test_dimension_mismatch_rejected() -> None:
    index = ExactCosineIndex()
    index.add(_chunk("d1", 0), (1.0, 0.0))
    with pytest.raises(ValueError):
        index.add(_chunk("d1", 1), (1.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        index.search((1.0, 0.0, 0.0), top_k=3)
    assert index.dimension == 2


def test_zero_norm_vector_rejected_on_add() -> None:
    index = ExactCosineIndex()
    with pytest.raises(ValueError):
        index.add(_chunk("d1", 0), (0.0, 0.0))
    assert len(index) == 0


def test_zero_norm_query_yields_no_results() -> None:
    index = ExactCosineIndex()
    index.add(_chunk("d1", 0), (1.0, 0.0))
    assert index.search((0.0, 0.0), top_k=5) == ()


def test_query_vector_rejects_nan() -> None:
    index = ExactCosineIndex()
    index.add(_chunk("d1", 0), (1.0, 0.0))
    with pytest.raises(ValueError):
        index.search((float("nan"), 0.0), top_k=5)


def test_caller_cannot_mutate_stored_vectors() -> None:
    index = ExactCosineIndex()
    mutable = [1.0, 0.0]
    index.add(_chunk("d1", 0), mutable)
    mutable[0] = 99.0
    mutable[1] = 42.0
    results = index.search((1.0, 0.0), top_k=1)
    assert len(results) == 1
    score, entry = results[0]
    assert score == pytest.approx(1.0)
    assert entry.vector == (1.0, 0.0)


def test_duplicate_chunk_id_rejected() -> None:
    index = ExactCosineIndex()
    index.add(_chunk("d1", 0), (1.0, 0.0))
    with pytest.raises(ValueError):
        index.add(_chunk("d1", 0), (0.0, 1.0))


def test_empty_index_search_returns_nothing() -> None:
    index = ExactCosineIndex()
    assert index.search((1.0, 0.0), top_k=5) == ()
    assert index.dimension is None


def test_clear_resets_index() -> None:
    index = ExactCosineIndex()
    index.add(_chunk("d1", 0), (1.0, 0.0))
    index.clear()
    assert len(index) == 0
    assert index.dimension is None
    index.add(_chunk("d2", 0), (1.0, 0.0, 0.0))
    assert index.dimension == 3


def test_all_scores_finite() -> None:
    index = ExactCosineIndex()
    for ordinal in range(5):
        index.add(_chunk("d1", ordinal), (math.cos(ordinal), math.sin(ordinal) or 1.0))
    results = index.search((1.0, 0.0), top_k=5)
    assert results
    assert all(math.isfinite(score) for score, _ in results)
