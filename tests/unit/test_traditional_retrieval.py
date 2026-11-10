"""Tests for retrieval, evidence construction, and query accounting."""

import math
from typing import cast

import pytest

from rag_benchmark.adapters.traditional import TraditionalVectorRAGAdapter
from rag_benchmark.config.models import BenchmarkConfig, RetrievalConfig
from rag_benchmark.schemas.document import BenchmarkDocument
from rag_benchmark.schemas.evidence import EvidenceType
from rag_benchmark.schemas.question import AdapterQueryInput
from rag_benchmark.schemas.result import QueryStatus
from tests.unit.fakes import FakeEmbeddingProvider, FakeLLMProvider

ZWNJ = "\u200c"
DOC_A = f"آفتاب از مشرق طلوع می{ZWNJ}کند"
DOC_B = "ماه در آسمان شب می‌درخشد"
QUESTION = "سؤال آزمایشی فارسی"

A_VEC = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
B_VEC = (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
Q_VEC = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _doc(doc_id: str, text: str) -> BenchmarkDocument:
    return BenchmarkDocument(document_id=doc_id, title=f"عنوان {doc_id}", text=text)


def _prepared_adapter(
    retrieval: RetrievalConfig | None = None,
    llm: FakeLLMProvider | None = None,
    embedding: FakeEmbeddingProvider | None = None,
) -> TraditionalVectorRAGAdapter:
    config: dict[str, object] = {"system": "traditional_rag"}
    if retrieval is not None:
        config["retrieval"] = retrieval.model_dump()
    adapter = TraditionalVectorRAGAdapter(
        embedding_provider=embedding
        or FakeEmbeddingProvider(
            dimension=8, mapping={DOC_A: A_VEC, DOC_B: B_VEC, QUESTION: Q_VEC}
        ),
        llm_provider=llm or FakeLLMProvider(),
    )
    adapter.prepare(BenchmarkConfig.model_validate(config))
    adapter.index([_doc("doc-a", DOC_A), _doc("doc-b", DOC_B)])
    return adapter


def test_query_success_returns_ranked_evidence() -> None:
    adapter = _prepared_adapter()
    result = adapter.query(AdapterQueryInput(question_id="q-1", question=QUESTION))
    assert result.status == QueryStatus.SUCCESS
    assert result.answer is not None
    assert QUESTION in result.answer
    assert result.system == "traditional_rag"
    assert result.variant == "dense_exact"
    evidence = result.retrieved_evidence
    assert len(evidence) == 2
    top = evidence[0]
    assert top.document_id == "doc-a"
    assert top.chunk_id == "doc-a::chunk-000000"
    assert top.text == DOC_A
    assert top.score == pytest.approx(1.0)
    assert top.rank == 1
    assert top.evidence_type == EvidenceType.CHUNK
    bottom = evidence[1]
    assert bottom.document_id == "doc-b"
    assert bottom.score == pytest.approx(0.0)
    assert bottom.rank == 2


def test_persian_and_zwnj_preserved_in_evidence() -> None:
    result = _prepared_adapter().query(AdapterQueryInput(question_id="q-1", question=QUESTION))
    assert ZWNJ in result.retrieved_evidence[0].text
    assert result.retrieved_evidence[0].text == DOC_A


def test_only_question_is_embedded_at_query_time() -> None:
    adapter = _prepared_adapter()
    adapter.query(AdapterQueryInput(question_id="q-1", question=QUESTION))
    embedding = cast(FakeEmbeddingProvider, adapter._embedding_provider)
    assert embedding.query_texts == [QUESTION]


def test_top_k_comes_from_retrieval_config() -> None:
    adapter = _prepared_adapter(retrieval=RetrievalConfig(top_k=1))
    result = adapter.query(AdapterQueryInput(question_id="q-1", question=QUESTION))
    assert result.status == QueryStatus.SUCCESS
    assert len(result.retrieved_evidence) == 1
    assert result.retrieved_evidence[0].document_id == "doc-a"


def test_score_threshold_filters_low_relevance() -> None:
    adapter = _prepared_adapter(retrieval=RetrievalConfig(top_k=5, score_threshold=0.5))
    result = adapter.query(AdapterQueryInput(question_id="q-1", question=QUESTION))
    assert result.status == QueryStatus.SUCCESS
    assert len(result.retrieved_evidence) == 1
    assert result.retrieved_evidence[0].document_id == "doc-a"


def test_no_evidence_yields_insufficient_context_answer() -> None:
    adapter = _prepared_adapter(retrieval=RetrievalConfig(top_k=5, score_threshold=10.0))
    llm = cast(FakeLLMProvider, adapter._llm_provider)
    result = adapter.query(AdapterQueryInput(question_id="q-1", question=QUESTION))
    assert result.status == QueryStatus.SUCCESS
    assert result.retrieved_evidence == []
    assert result.answer is not None
    assert "insufficient" in result.answer.lower()
    (question, context, _) = llm.calls[0]
    assert question == QUESTION
    assert context == ()


def test_usage_and_timing_are_reported() -> None:
    result = _prepared_adapter().query(AdapterQueryInput(question_id="q-1", question=QUESTION))
    assert result.usage.embedding_calls == 1
    assert result.usage.llm_calls == 1
    assert result.usage.input_tokens == 0
    assert result.usage.output_tokens == 0
    assert math.isfinite(result.timing.retrieval_ms)
    assert math.isfinite(result.timing.generation_ms)
    assert math.isfinite(result.timing.total_ms)
    assert result.timing.retrieval_ms >= 0.0
    assert result.timing.generation_ms >= 0.0
    assert result.timing.total_ms >= 0.0


def test_usage_tokens_forwarded_from_provider() -> None:
    adapter = _prepared_adapter(llm=FakeLLMProvider(report_usage=True))
    result = adapter.query(AdapterQueryInput(question_id="q-1", question=QUESTION))
    assert result.status == QueryStatus.SUCCESS
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 5
