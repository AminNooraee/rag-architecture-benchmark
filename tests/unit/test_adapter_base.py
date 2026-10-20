"""Tests for BaseRAGAdapter contract, capability reporting, and lifecycle."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from rag_benchmark.adapters.base import (
    AdapterCapabilities,
    BaseRAGAdapter,
    HealthCheckResult,
    IndexingResult,
)
from rag_benchmark.config.models import BenchmarkConfig
from rag_benchmark.errors.models import UnsupportedOperationError
from rag_benchmark.schemas.document import BenchmarkDocument
from rag_benchmark.schemas.evidence import EvidenceType, RetrievedEvidence
from rag_benchmark.schemas.question import AdapterQueryInput
from rag_benchmark.schemas.result import QueryResult, QueryStatus


def test_abstract_adapter_cannot_be_instantiated() -> None:
    """Instantiating BaseRAGAdapter directly must raise TypeError."""
    with pytest.raises(TypeError):
        BaseRAGAdapter()  # type: ignore[abstract]


class ConcreteTestAdapter(BaseRAGAdapter):
    """Minimal concrete adapter implementation satisfying contract for testing."""

    def __init__(self, capabilities: AdapterCapabilities | None = None) -> None:
        self._capabilities = capabilities or AdapterCapabilities()
        self.prepared = False
        self.cleaned_up = False
        self.indexed_docs: list[BenchmarkDocument] = []

    @property
    def system_name(self) -> str:
        return "mock_rag"

    @property
    def variant_name(self) -> str:
        return "test_variant"

    @property
    def capabilities(self) -> AdapterCapabilities:
        return self._capabilities

    def prepare(self, config: BenchmarkConfig) -> None:
        self.prepared = True

    def index(self, documents: Sequence[BenchmarkDocument]) -> IndexingResult:
        self.indexed_docs.extend(documents)
        return IndexingResult(
            indexed_documents=len(documents),
            indexed_chunks=len(documents) * 2,
            elapsed_ms=50.0,
        )

    def query(self, query_input: AdapterQueryInput) -> QueryResult:
        evidence = RetrievedEvidence(
            document_id="doc-1",
            text="پاسخ شواهد",
            evidence_type=EvidenceType.CHUNK,
        )
        return QueryResult.create_success(
            question_id=query_input.question_id,
            system=self.system_name,
            variant=self.variant_name,
            answer=f"پاسخ آزمایشی برای: {query_input.question}",
            retrieved_evidence=[evidence],
        )

    def health(self) -> HealthCheckResult:
        return HealthCheckResult(
            healthy=True,
            system=self.system_name,
            message="Mock system is operational",
        )

    def cleanup(self) -> None:
        self.cleaned_up = True


def test_concrete_adapter_lifecycle() -> None:
    """Verify full lifecycle of concrete adapter implementation."""
    adapter = ConcreteTestAdapter()

    assert adapter.system_name == "mock_rag"
    assert adapter.variant_name == "test_variant"

    # Health check
    health = adapter.health()
    assert health.healthy is True
    assert health.system == "mock_rag"

    # Prepare
    config = BenchmarkConfig(system="mock_rag")
    adapter.prepare(config)
    assert adapter.prepared is True

    # Index
    doc = BenchmarkDocument(
        document_id="doc-test-1",
        title="عنوان تستی",
        text="متن تستی فارسی",
    )
    idx_res = adapter.index([doc])
    assert idx_res.indexed_documents == 1
    assert idx_res.indexed_chunks == 2

    # Query with AdapterQueryInput
    query_in = AdapterQueryInput(
        question_id="q-1",
        question="سؤال آزمایشی فارسی؟",
    )
    res = adapter.query(query_in)
    assert res.status == QueryStatus.SUCCESS
    assert res.answer == "پاسخ آزمایشی برای: سؤال آزمایشی فارسی؟"

    # Cleanup
    adapter.cleanup()
    assert adapter.cleaned_up is True


def test_optional_operations_raise_unsupported_by_default() -> None:
    """Optional methods (insert, update, delete) raise UnsupportedOperationError if unsupported."""
    adapter = ConcreteTestAdapter(
        capabilities=AdapterCapabilities(
            supports_incremental_insert=False,
            supports_update=False,
            supports_delete=False,
        )
    )

    doc = BenchmarkDocument(document_id="d1", title="تستی")

    with pytest.raises(UnsupportedOperationError, match="does not support incremental insert"):
        adapter.insert([doc])

    with pytest.raises(UnsupportedOperationError, match="does not support document update"):
        adapter.update([doc])

    with pytest.raises(UnsupportedOperationError, match="does not support document deletion"):
        adapter.delete(["d1"])


def test_optional_operations_delegate_when_supported() -> None:
    """If capability flag is set True but subclass didn't override, raises NotImplementedError."""
    adapter = ConcreteTestAdapter(
        capabilities=AdapterCapabilities(
            supports_incremental_insert=True,
            supports_update=True,
            supports_delete=True,
        )
    )

    doc = BenchmarkDocument(document_id="d1", title="تستی")

    with pytest.raises(NotImplementedError, match="insert.*not implemented"):
        adapter.insert([doc])

    with pytest.raises(NotImplementedError, match="update.*not implemented"):
        adapter.update([doc])

    with pytest.raises(NotImplementedError, match="delete.*not implemented"):
        adapter.delete(["d1"])
