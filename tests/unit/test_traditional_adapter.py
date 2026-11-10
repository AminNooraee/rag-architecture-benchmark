"""Tests for TraditionalVectorRAGAdapter lifecycle, configuration, and errors."""

import pytest

from rag_benchmark.adapters.base import AdapterCapabilities
from rag_benchmark.adapters.traditional import (
    TraditionalRAGConfig,
    TraditionalVectorRAGAdapter,
)
from rag_benchmark.config.models import BenchmarkConfig, RetrievalConfig
from rag_benchmark.errors.models import ConfigurationError, UnsupportedOperationError
from rag_benchmark.schemas.document import BenchmarkDocument
from rag_benchmark.schemas.question import AdapterQueryInput
from rag_benchmark.schemas.result import QueryStatus
from tests.unit.fakes import FakeEmbeddingProvider, FakeLLMProvider


def _config(**overrides: object) -> BenchmarkConfig:
    data: dict[str, object] = {"system": "traditional_rag"}
    data.update(overrides)
    return BenchmarkConfig.model_validate(data)


def _doc(doc_id: str, text: str) -> BenchmarkDocument:
    return BenchmarkDocument(document_id=doc_id, title=f"عنوان {doc_id}", text=text)


def _adapter(
    embedding: FakeEmbeddingProvider | None = None, llm: FakeLLMProvider | None = None
) -> TraditionalVectorRAGAdapter:
    return TraditionalVectorRAGAdapter(
        embedding_provider=embedding or FakeEmbeddingProvider(),
        llm_provider=llm or FakeLLMProvider(),
    )


def _query(question: str = "سؤال آزمایشی", question_id: str = "q-1") -> AdapterQueryInput:
    return AdapterQueryInput(question_id=question_id, question=question)


def test_identity_and_capabilities() -> None:
    adapter = _adapter()
    assert adapter.system_name == "traditional_rag"
    assert adapter.variant_name == "dense_exact"
    caps = adapter.capabilities
    assert isinstance(caps, AdapterCapabilities)
    assert not caps.supports_incremental_insert
    assert not caps.supports_update
    assert not caps.supports_delete
    assert not caps.supports_persistence
    assert not caps.supports_streaming
    assert not caps.supports_native_citations
    adapter.validate_capabilities()


def test_health_before_prepare_is_unhealthy() -> None:
    health = _adapter().health()
    assert health.healthy is False
    assert health.system == "traditional_rag"


def test_activate_prepares_adapter() -> None:
    adapter = _adapter()
    adapter.activate(_config())
    health = adapter.health()
    assert health.healthy is True
    assert health.details["prepared"] is True
    assert health.details["indexed"] is False
    assert health.details["embedding_provider"] == "FakeEmbeddingProvider"
    assert health.details["llm_provider"] == "FakeLLMProvider"


def test_prepare_rejects_wrong_system() -> None:
    with pytest.raises(ConfigurationError):
        _adapter().prepare(BenchmarkConfig(system="lightrag"))


def test_prepare_rejects_contradictory_variant() -> None:
    with pytest.raises(ConfigurationError):
        _adapter().prepare(_config(variant="graph_hybrid"))


def test_prepare_accepts_default_and_baseline_variant() -> None:
    _adapter().prepare(_config())
    _adapter().prepare(_config(variant="baseline"))
    _adapter().prepare(_config(variant="dense_exact"))


def test_prepare_applies_traditional_rag_extra() -> None:
    adapter = _adapter()
    adapter.prepare(_config(extra={"traditional_rag": {"chunk_size": 256, "overlap": 16}}))
    assert adapter._chunking_config == TraditionalRAGConfig(chunk_size=256, overlap=16)


def test_prepare_rejects_invalid_traditional_rag_extra() -> None:
    with pytest.raises(ConfigurationError):
        _adapter().prepare(_config(extra={"traditional_rag": {"chunk_size": 100, "overlap": 100}}))
    with pytest.raises(ConfigurationError):
        _adapter().prepare(_config(extra={"traditional_rag": {"chunk_size": -5}}))
    with pytest.raises(ConfigurationError):
        _adapter().prepare(_config(extra={"traditional_rag": "not-a-mapping"}))
    with pytest.raises(ConfigurationError):
        _adapter().prepare(_config(extra={"traditional_rag": {"chunk_size": 100, "bogus": 1}}))


def test_query_before_prepare_fails_with_configuration_error() -> None:
    result = _adapter().query(_query())
    assert result.status == QueryStatus.FAILED
    assert result.answer is None
    assert result.error is not None
    assert result.error.error_type.value == "configuration_error"


def test_query_before_index_fails_with_configuration_error() -> None:
    adapter = _adapter()
    adapter.prepare(_config())
    result = adapter.query(_query())
    assert result.status == QueryStatus.FAILED
    assert result.answer is None
    assert result.error is not None
    assert result.error.error_type.value == "configuration_error"


def test_index_before_prepare_raises() -> None:
    with pytest.raises(ConfigurationError):
        _adapter().index([_doc("d1", "متن")])


def test_index_reports_counts_and_metadata() -> None:
    adapter = _adapter(embedding=FakeEmbeddingProvider(dimension=8))
    adapter.prepare(_config())
    documents = [_doc("d1", "اولین متن"), _doc("d2", "دومین متن")]
    result = adapter.index(documents)
    assert result.indexed_documents == 2
    assert result.indexed_chunks == 2
    assert result.elapsed_ms >= 0.0
    assert result.metadata["chunk_size"] == 512
    assert result.metadata["overlap"] == 0
    assert result.metadata["embedding_dimension"] == 8
    assert result.metadata["embedding_batches"] == 1


def test_index_batches_by_embedding_batch_size() -> None:
    embedding = FakeEmbeddingProvider(dimension=4)
    adapter = _adapter(embedding=embedding)
    adapter.prepare(
        BenchmarkConfig.model_validate(
            {"system": "traditional_rag", "embedding": {"batch_size": 2}}
        )
    )
    documents = [_doc(f"d{i}", f"متن {i}") for i in range(5)]
    result = adapter.index(documents)
    assert result.indexed_chunks == 5
    assert result.metadata["embedding_batches"] == 3
    assert [len(batch) for batch in embedding.documents_batches] == [2, 2, 1]


def test_repeated_index_is_full_rebuild_without_duplicates() -> None:
    adapter = _adapter()
    adapter.prepare(_config())
    documents = [_doc("d1", "متن یک"), _doc("d2", "متن دو")]
    first = adapter.index(documents)
    second = adapter.index(documents)
    assert second.indexed_chunks == first.indexed_chunks == 2
    result = adapter.query(_query())
    assert result.status == QueryStatus.SUCCESS
    chunk_ids = [e.chunk_id for e in result.retrieved_evidence]
    assert len(chunk_ids) == len(set(chunk_ids)) == 2


def test_cleanup_clears_index_and_is_idempotent() -> None:
    adapter = _adapter()
    adapter.prepare(_config())
    adapter.index([_doc("d1", "متن")])
    adapter.cleanup()
    adapter.cleanup()
    health = adapter.health()
    assert health.healthy is True
    assert health.details["indexed"] is False
    assert health.details["indexed_documents"] == 0
    result = adapter.query(_query())
    assert result.status == QueryStatus.FAILED
    adapter.index([_doc("d2", "متن مجدد")])
    assert adapter.health().details["indexed"] is True


def test_optional_operations_unsupported() -> None:
    adapter = _adapter()
    adapter.prepare(_config())
    document = _doc("d1", "متن")
    with pytest.raises(UnsupportedOperationError):
        adapter.insert([document])
    with pytest.raises(UnsupportedOperationError):
        adapter.update([document])
    with pytest.raises(UnsupportedOperationError):
        adapter.delete(["d1"])


def test_retrieval_config_is_copied_not_shared() -> None:
    config = _config(retrieval=RetrievalConfig(top_k=7))
    adapter = _adapter()
    adapter.prepare(config)
    config.retrieval.top_k = 3
    result = adapter.query(_query())
    assert result.status == QueryStatus.FAILED  # not indexed
    assert adapter._retrieval is not None
    assert adapter._retrieval.top_k == 7
