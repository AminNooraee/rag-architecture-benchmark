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
from rag_benchmark.errors.models import (
    InvalidCapabilityDeclarationError,
    UnsupportedOperationError,
)
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
        self.validate_capabilities()
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


class FullyImplementedAdapter(ConcreteTestAdapter):
    """Adapter that overrides optional methods when capabilities are True."""

    def insert(self, documents: Sequence[BenchmarkDocument]) -> IndexingResult:
        return IndexingResult(indexed_documents=len(documents), elapsed_ms=10.0)

    def update(self, documents: Sequence[BenchmarkDocument]) -> IndexingResult:
        return IndexingResult(indexed_documents=len(documents), elapsed_ms=10.0)

    def delete(self, document_ids: Sequence[str]) -> None:
        pass


def test_concrete_adapter_lifecycle() -> None:
    """Verify full lifecycle of concrete adapter implementation."""
    adapter = ConcreteTestAdapter()

    assert adapter.system_name == "mock_rag"
    assert adapter.variant_name == "test_variant"

    health = adapter.health()
    assert health.healthy is True
    assert health.system == "mock_rag"

    config = BenchmarkConfig(system="mock_rag")
    adapter.prepare(config)
    assert adapter.prepared is True

    doc = BenchmarkDocument(
        document_id="doc-test-1",
        title="عنوان تستی",
        text="متن تستی فارسی",
    )
    idx_res = adapter.index([doc])
    assert idx_res.indexed_documents == 1
    assert idx_res.indexed_chunks == 2

    query_in = AdapterQueryInput(
        question_id="q-1",
        question="سؤال آزمایشی فارسی؟",
    )
    res = adapter.query(query_in)
    assert res.status == QueryStatus.SUCCESS
    assert res.answer == "پاسخ آزمایشی برای: سؤال آزمایشی فارسی؟"

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


def test_unimplemented_capability_declaration_fails_early() -> None:
    """Declaring capability without overriding placeholder raises error."""
    adapter = ConcreteTestAdapter(
        capabilities=AdapterCapabilities(supports_incremental_insert=True)
    )
    with pytest.raises(InvalidCapabilityDeclarationError, match="does not override.*insert"):
        adapter.validate_capabilities()


def test_fully_implemented_capabilities_succeed() -> None:
    """Adapter overriding optional methods validates and executes successfully."""
    adapter = FullyImplementedAdapter(
        capabilities=AdapterCapabilities(
            supports_incremental_insert=True,
            supports_update=True,
            supports_delete=True,
        )
    )
    # Validation passes without error
    adapter.validate_capabilities()

    doc = BenchmarkDocument(document_id="d1", title="تستی")
    res_insert = adapter.insert([doc])
    assert res_insert.indexed_documents == 1

    res_update = adapter.update([doc])
    assert res_update.indexed_documents == 1

    adapter.delete(["d1"])
