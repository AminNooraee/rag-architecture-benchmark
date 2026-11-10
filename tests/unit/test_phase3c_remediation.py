"""Regression tests for the confirmed Phase 3C review findings (F2-F9).

Each test pins a contract-level behavior the review confirmed broken in the
original baseline. F1 (provider response validation) lives in
``test_traditional_providers.py``; F5 (chunk ID identity) in
``test_traditional_chunking.py``; F7 (stable cosine) in
``test_traditional_vector_index.py``.
"""

from __future__ import annotations

import types
from collections.abc import Sequence

import pytest

from rag_benchmark.adapters.traditional import (
    LLMGenerationResult,
    TraditionalVectorRAGAdapter,
)
from rag_benchmark.adapters.traditional.config import TraditionalRAGConfig
from rag_benchmark.adapters.traditional.providers import EmbeddingProvider
from rag_benchmark.config.models import BenchmarkConfig
from rag_benchmark.errors.models import BenchmarkErrorType, BenchmarkException
from rag_benchmark.schemas.document import BenchmarkDocument
from rag_benchmark.schemas.question import AdapterQueryInput
from rag_benchmark.schemas.result import QueryResult, QueryStatus
from tests.unit.fakes import FakeEmbeddingProvider, FakeLLMProvider

QUESTION = "سؤال آزمایشی"
DOC_TEXT = "متن آزمایشی فارسی"


def _doc(doc_id: str, text: str = DOC_TEXT) -> BenchmarkDocument:
    return BenchmarkDocument(document_id=doc_id, title=f"عنوان {doc_id}", text=text)


def _adapter(
    embedding: EmbeddingProvider | None = None,
    llm: FakeLLMProvider | None = None,
    chunking_config: TraditionalRAGConfig | None = None,
    config: dict[str, object] | None = None,
) -> TraditionalVectorRAGAdapter:
    adapter = TraditionalVectorRAGAdapter(
        embedding_provider=embedding or FakeEmbeddingProvider(dimension=4),
        llm_provider=llm or FakeLLMProvider(),
        chunking_config=chunking_config,
    )
    adapter.prepare(BenchmarkConfig.model_validate(config or {"system": "traditional_rag"}))
    return adapter


def _indexed_adapter(
    embedding: EmbeddingProvider | None = None,
    llm: FakeLLMProvider | None = None,
) -> TraditionalVectorRAGAdapter:
    adapter = _adapter(embedding=embedding, llm=llm)
    adapter.index([_doc("doc-1"), _doc("doc-2")])
    return adapter


def _query(adapter: TraditionalVectorRAGAdapter, question_id: str) -> QueryResult:
    return adapter.query(AdapterQueryInput(question_id=question_id, question=QUESTION))


# --- F2: per-batch embedding count alignment ---------------------------------


class _MisalignedBatchEmbeddingProvider:
    """Returns a caller-scripted number of vectors per embed_documents call."""

    def __init__(self, counts: list[int]) -> None:
        self._counts = list(counts)
        self._next = 0

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        count = self._counts[self._next]
        self._next += 1
        return [[0.1, 0.2] for _ in range(count)]

    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2]


def test_index_rejects_per_batch_mismatch_even_if_aggregate_matches() -> None:
    # 4 chunks, batch_size=2: first batch returns 1 vector, second returns 3.
    # Aggregate 1 + 3 == 4 would pass an aggregate-only check, so the failure
    # must come from the per-batch alignment check.
    provider = _MisalignedBatchEmbeddingProvider([1, 3])
    adapter = _adapter(
        embedding=provider, config={"system": "traditional_rag", "embedding": {"batch_size": 2}}
    )
    with pytest.raises(BenchmarkException) as excinfo:
        adapter.index([_doc(f"doc-{i}") for i in range(4)])
    assert excinfo.value.error_type is BenchmarkErrorType.INDEXING_ERROR
    assert "batch 1" in str(excinfo.value)
    assert "1 vectors for 2 texts" in str(excinfo.value)


def test_index_rejects_per_batch_overcount_even_if_aggregate_matches() -> None:
    provider = _MisalignedBatchEmbeddingProvider([3, 1])
    adapter = _adapter(
        embedding=provider, config={"system": "traditional_rag", "embedding": {"batch_size": 2}}
    )
    with pytest.raises(BenchmarkException) as excinfo:
        adapter.index([_doc(f"doc-{i}") for i in range(4)])
    assert excinfo.value.error_type is BenchmarkErrorType.INDEXING_ERROR
    assert "3 vectors for 2 texts" in str(excinfo.value)


# --- F3: retrieval_ms must not include generation latency --------------------


def test_generation_delay_does_not_inflate_retrieval_ms() -> None:
    adapter = _indexed_adapter(
        embedding=FakeEmbeddingProvider(dimension=4, delay_seconds=0.03),
        llm=FakeLLMProvider(delay_seconds=0.12),
    )
    result = _query(adapter, "q-timing")
    assert result.status == QueryStatus.SUCCESS
    timing = result.timing
    # Retrieval stage is ~30ms of fake embedding delay; the 120ms generation
    # delay must not leak into retrieval_ms.
    assert timing.retrieval_ms < 100.0
    assert timing.generation_ms >= 100.0
    assert timing.total_ms >= 150.0
    # No double counting: retrieval + generation fits inside total.
    assert timing.retrieval_ms + timing.generation_ms <= timing.total_ms + 1.0


def test_retrieval_failure_timing_excludes_generation() -> None:
    adapter = _indexed_adapter(
        embedding=FakeEmbeddingProvider(
            dimension=4, delay_seconds=0.03, fail_query=RuntimeError("query-down")
        ),
        llm=FakeLLMProvider(delay_seconds=0.01),
    )
    result = _query(adapter, "q-timing-fail")
    assert result.status == QueryStatus.FAILED
    assert result.timing.retrieval_ms < 100.0
    assert result.timing.generation_ms == 0.0
    assert result.timing.total_ms >= result.timing.retrieval_ms - 1.0


# --- F4: a failed embedding attempt is counted as one call --------------------


def test_failed_embedding_query_is_counted_as_one_call() -> None:
    adapter = _indexed_adapter(
        embedding=FakeEmbeddingProvider(dimension=4, fail_query=RuntimeError("query-down"))
    )
    result = _query(adapter, "q-embed-fail")
    assert result.status == QueryStatus.FAILED
    assert result.error is not None
    assert result.error.error_type is BenchmarkErrorType.RETRIEVAL_ERROR
    assert result.usage.embedding_calls == 1
    assert result.usage.llm_calls == 0


# --- F6: malformed LLM usage cannot escape canonical failure handling --------


def _assert_canonical_generation_failure(result: QueryResult) -> None:
    assert result.status == QueryStatus.FAILED
    assert result.answer is None
    assert result.error is not None
    assert result.error.error_type is BenchmarkErrorType.GENERATION_ERROR
    assert len(result.retrieved_evidence) == 2
    assert result.timing.retrieval_ms >= 0.0
    assert result.usage.embedding_calls == 1
    assert result.usage.llm_calls == 1


def test_negative_input_tokens_yield_canonical_generation_failure() -> None:
    llm = FakeLLMProvider(
        malformed_result=LLMGenerationResult(answer="پاسخ", input_tokens=-1, output_tokens=3)
    )
    _assert_canonical_generation_failure(_query(_indexed_adapter(llm=llm), "q-mal-1"))


def test_negative_output_tokens_yield_canonical_generation_failure() -> None:
    llm = FakeLLMProvider(
        malformed_result=LLMGenerationResult(answer="پاسخ", input_tokens=3, output_tokens=-5)
    )
    _assert_canonical_generation_failure(_query(_indexed_adapter(llm=llm), "q-mal-2"))


def test_boolean_token_counts_yield_canonical_generation_failure() -> None:
    llm = FakeLLMProvider(
        malformed_result=LLMGenerationResult(answer="پاسخ", input_tokens=True, output_tokens=False)
    )
    _assert_canonical_generation_failure(_query(_indexed_adapter(llm=llm), "q-mal-3"))


def test_non_string_answer_yields_canonical_generation_failure() -> None:
    llm = FakeLLMProvider(
        malformed_result=types.SimpleNamespace(answer=42, input_tokens=None, output_tokens=None)
    )
    _assert_canonical_generation_failure(_query(_indexed_adapter(llm=llm), "q-mal-4"))


def test_arbitrary_provider_result_yields_canonical_generation_failure() -> None:
    llm = FakeLLMProvider(malformed_result=object())
    _assert_canonical_generation_failure(_query(_indexed_adapter(llm=llm), "q-mal-5"))


def test_valid_none_usage_still_records_zero_tokens() -> None:
    adapter = _indexed_adapter()
    result = _query(adapter, "q-valid-usage")
    assert result.status == QueryStatus.SUCCESS
    assert result.usage.input_tokens == 0
    assert result.usage.output_tokens == 0


# --- F8: constructor chunking config survives prepare() ------------------------


def _ctor_adapter() -> TraditionalVectorRAGAdapter:
    return TraditionalVectorRAGAdapter(
        embedding_provider=FakeEmbeddingProvider(dimension=4),
        llm_provider=FakeLLMProvider(),
        chunking_config=TraditionalRAGConfig(chunk_size=3, overlap=1),
    )


def test_prepare_without_extra_keeps_constructor_chunking_config() -> None:
    adapter = _ctor_adapter()
    adapter.prepare(BenchmarkConfig.model_validate({"system": "traditional_rag"}))
    result = adapter.index([_doc("doc-1", "aaaa bbbb cccc")])
    assert result.metadata["chunk_size"] == 3
    assert result.metadata["overlap"] == 1


def test_prepare_with_explicit_key_overrides_constructor_config() -> None:
    adapter = _ctor_adapter()
    adapter.prepare(
        BenchmarkConfig.model_validate(
            {
                "system": "traditional_rag",
                "extra": {"traditional_rag": {"chunk_size": 8, "overlap": 2}},
            }
        )
    )
    result = adapter.index([_doc("doc-1", "متن کوتاه")])
    assert result.metadata["chunk_size"] == 8
    assert result.metadata["overlap"] == 2


def test_prepare_without_extra_restores_constructor_config_after_override() -> None:
    adapter = _ctor_adapter()
    adapter.prepare(
        BenchmarkConfig.model_validate(
            {
                "system": "traditional_rag",
                "extra": {"traditional_rag": {"chunk_size": 8, "overlap": 2}},
            }
        )
    )
    adapter.index([_doc("doc-1", "متن کوتاه")])
    adapter.prepare(BenchmarkConfig.model_validate({"system": "traditional_rag"}))
    result = adapter.index([_doc("doc-1", "متن کوتاه")])
    assert result.metadata["chunk_size"] == 3
    assert result.metadata["overlap"] == 1


# --- F9: indexing failures use the INDEXING_ERROR taxonomy ---------------------


class _FixedVectorEmbeddingProvider:
    """Returns the same (possibly invalid) vector for every text."""

    def __init__(self, vector: Sequence[float]) -> None:
        self._vector = vector

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [list(self._vector) for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return list(self._vector)


def test_indexing_provider_failure_uses_indexing_error_taxonomy() -> None:
    adapter = _adapter(
        embedding=FakeEmbeddingProvider(dimension=4, fail_documents=RuntimeError("boom"))
    )
    with pytest.raises(BenchmarkException) as excinfo:
        adapter.index([_doc("doc-1")])
    assert excinfo.value.error_type is BenchmarkErrorType.INDEXING_ERROR
    assert "Embedding provider failed during indexing" in str(excinfo.value)


def test_indexing_provider_failure_does_not_leak_credentials() -> None:
    secret = "sk-super-secret-credential-123"
    adapter = _adapter(
        embedding=FakeEmbeddingProvider(
            dimension=4, fail_documents=RuntimeError(f"auth failed: {secret}")
        )
    )
    with pytest.raises(BenchmarkException) as excinfo:
        adapter.index([_doc("doc-1")])
    assert excinfo.value.error_type is BenchmarkErrorType.INDEXING_ERROR
    assert secret not in str(excinfo.value)


@pytest.mark.parametrize(
    "vector",
    [
        pytest.param([float("nan"), 0.2], id="nan-component"),
        pytest.param([float("inf"), 0.2], id="inf-component"),
        pytest.param([0.0, 0.0], id="zero-norm"),
        pytest.param([True, False], id="bool-components"),
        pytest.param(["a", "b"], id="string-components"),
    ],
)
def test_indexing_rejects_invalid_vectors_with_indexing_error(vector: list[float]) -> None:
    adapter = _adapter(embedding=_FixedVectorEmbeddingProvider(vector))
    with pytest.raises(BenchmarkException) as excinfo:
        adapter.index([_doc("doc-1")])
    assert excinfo.value.error_type is BenchmarkErrorType.INDEXING_ERROR


class _MixedDimensionEmbeddingProvider:
    """Returns 2-dim vectors for the first batch and 3-dim for later batches."""

    def __init__(self) -> None:
        self._call = 0

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self._call += 1
        if self._call == 1:
            return [[0.1, 0.2] for _ in texts]
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2]


def test_indexing_rejects_mixed_dimensions_with_indexing_error() -> None:
    adapter = _adapter(
        embedding=_MixedDimensionEmbeddingProvider(),
        config={"system": "traditional_rag", "embedding": {"batch_size": 2}},
    )
    with pytest.raises(BenchmarkException) as excinfo:
        adapter.index([_doc(f"doc-{i}") for i in range(4)])
    assert excinfo.value.error_type is BenchmarkErrorType.INDEXING_ERROR
    assert "dimension" in str(excinfo.value).lower()
