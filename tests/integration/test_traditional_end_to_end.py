"""End-to-end and fairness regression tests for the Traditional Vector RAG baseline.

Covers:
- full pipeline: activate -> index -> query with deterministic fake providers;
- truthful usage/timing accounting;
- full-rebuild re-indexing without duplicate accumulation;
- generation-stage failure preserving retrieval evidence and timing;
- retrieval-stage failure producing no evidence;
- architectural fairness: query() accepts only AdapterQueryInput, and gold
  answers / supporting evidence from a BenchmarkQuestion never reach the
  adapter, the LLM context, or the produced answer.
"""

from __future__ import annotations

import inspect
from typing import get_type_hints

import pytest

from rag_benchmark.adapters.traditional import TraditionalVectorRAGAdapter
from rag_benchmark.config.models import BenchmarkConfig, RetrievalConfig
from rag_benchmark.schemas.document import BenchmarkDocument
from rag_benchmark.schemas.evidence import EvidenceType, RetrievedEvidence
from rag_benchmark.schemas.question import (
    AdapterQueryInput,
    BenchmarkQuestion,
    QuestionScope,
    QuestionType,
)
from rag_benchmark.schemas.result import QueryStatus
from tests.unit.fakes import FakeEmbeddingProvider, FakeLLMProvider

ZWNJ = "\u200c"

DOC1_TEXT = "ناصرالدین شاه قاجار دومین پادشاه ایران بود که در اواخر قرن نوزدهم حکومت کرد"
DOC2_TEXT = f"پارک ملی لار در جنوب ایران با جنگل{ZWNJ}های انبوه خود شناخته شده است"
DOC3_TEXT = f"دریاچه ارومیه بزرگ{ZWNJ}ترین دریاچه آب شیرین ایران است"
QUESTION = "دومین پادشاه ایران در اواخر قرن نوزدهم چه کسی بود؟"

VEC1 = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
VEC2 = (0.8, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
VEC3 = (0.0, 0.8, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0)
Q_VEC = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _doc(doc_id: str, text: str) -> BenchmarkDocument:
    return BenchmarkDocument(document_id=doc_id, title=f"عنوان {doc_id}", text=text)


def _documents() -> list[BenchmarkDocument]:
    return [
        _doc("doc1", DOC1_TEXT),
        _doc("doc2", DOC2_TEXT),
        _doc("doc3", DOC3_TEXT),
    ]


def _embedding_provider() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider(
        dimension=8,
        mapping={
            DOC1_TEXT: VEC1,
            DOC2_TEXT: VEC2,
            DOC3_TEXT: VEC3,
            QUESTION: Q_VEC,
        },
    )


def _prepared_adapter(
    llm: FakeLLMProvider | None = None,
    embedding: FakeEmbeddingProvider | None = None,
    retrieval: RetrievalConfig | None = None,
) -> TraditionalVectorRAGAdapter:
    config: dict[str, object] = {"system": "traditional_rag"}
    if retrieval is not None:
        config["retrieval"] = retrieval.model_dump()
    adapter = TraditionalVectorRAGAdapter(
        embedding_provider=embedding or _embedding_provider(),
        llm_provider=llm or FakeLLMProvider(),
    )
    adapter.prepare(BenchmarkConfig.model_validate(config))
    adapter.index(_documents())
    return adapter


def test_end_to_end_success() -> None:
    adapter = _prepared_adapter()
    result = adapter.query(AdapterQueryInput(question_id="q-e2e-1", question=QUESTION))
    assert result.status == QueryStatus.SUCCESS
    assert result.question_id == "q-e2e-1"
    assert result.system == "traditional_rag"
    assert result.variant == "dense_exact"
    assert result.answer is not None
    assert QUESTION in result.answer
    evidence = result.retrieved_evidence
    assert [e.document_id for e in evidence] == ["doc1", "doc2", "doc3"]
    assert [e.rank for e in evidence] == [1, 2, 3]
    assert [e.score for e in evidence] == pytest.approx([1.0, 0.8, 0.0])
    assert all(e.evidence_type == EvidenceType.CHUNK for e in evidence)
    assert evidence[0].chunk_id == "doc1::chunk-000000"
    assert evidence[0].text == DOC1_TEXT
    assert evidence[2].text == DOC3_TEXT
    assert ZWNJ in evidence[2].text


def test_indexing_result_counts_and_metadata() -> None:
    adapter = TraditionalVectorRAGAdapter(
        embedding_provider=_embedding_provider(), llm_provider=FakeLLMProvider()
    )
    adapter.prepare(BenchmarkConfig(system="traditional_rag"))
    result = adapter.index(_documents())
    assert result.indexed_documents == 3
    assert result.indexed_chunks == 3
    assert result.elapsed_ms >= 0.0
    assert result.metadata["embedding_dimension"] == 8
    assert result.metadata["embedding_batches"] == 1


def test_usage_and_timing_are_truthful() -> None:
    result = _prepared_adapter().query(AdapterQueryInput(question_id="q-e2e-2", question=QUESTION))
    assert result.status == QueryStatus.SUCCESS
    assert result.usage.llm_calls == 1
    assert result.usage.embedding_calls == 1
    assert result.usage.input_tokens == 0
    assert result.usage.output_tokens == 0
    assert result.timing.total_ms >= result.timing.retrieval_ms >= 0.0
    assert result.timing.total_ms >= result.timing.generation_ms >= 0.0


def test_usage_tokens_forwarded_when_provider_reports() -> None:
    adapter = _prepared_adapter(llm=FakeLLMProvider(report_usage=True))
    result = adapter.query(AdapterQueryInput(question_id="q-e2e-3", question=QUESTION))
    assert result.status == QueryStatus.SUCCESS
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 5


def test_repeated_indexing_is_full_rebuild() -> None:
    adapter = _prepared_adapter()
    adapter.index(_documents())
    result = adapter.query(AdapterQueryInput(question_id="q-e2e-4", question=QUESTION))
    assert result.status == QueryStatus.SUCCESS
    chunk_ids = [e.chunk_id for e in result.retrieved_evidence]
    assert len(chunk_ids) == len(set(chunk_ids)) == 3
    assert adapter.health().details["indexed_chunks"] == 3


def test_generation_failure_preserves_evidence_and_timing() -> None:
    llm = FakeLLMProvider(raise_error=RuntimeError("llm-service-down"))
    adapter = _prepared_adapter(llm=llm)
    result = adapter.query(AdapterQueryInput(question_id="q-e2e-5", question=QUESTION))
    assert result.status == QueryStatus.FAILED
    assert result.answer is None
    assert result.error is not None
    assert result.error.error_type.value == "generation_error"
    assert len(result.retrieved_evidence) == 3
    assert result.retrieved_evidence[0].document_id == "doc1"
    assert result.timing.retrieval_ms >= 0.0
    assert result.usage.embedding_calls == 1
    assert result.usage.llm_calls == 1
    assert llm.call_count == 1


def test_retrieval_failure_yields_no_evidence() -> None:
    embedding = _embedding_provider()
    embedding._fail_query = RuntimeError("embedding-service-down")
    adapter = _prepared_adapter(embedding=embedding)
    result = adapter.query(AdapterQueryInput(question_id="q-e2e-6", question=QUESTION))
    assert result.status == QueryStatus.FAILED
    assert result.answer is None
    assert result.error is not None
    assert result.error.error_type.value == "retrieval_error"
    assert result.retrieved_evidence == []
    assert result.usage.llm_calls == 0


def test_query_signature_accepts_only_adapter_query_input() -> None:
    signature = inspect.signature(TraditionalVectorRAGAdapter.query)
    parameters = list(signature.parameters)
    assert parameters == ["self", "query_input"]
    hints = get_type_hints(TraditionalVectorRAGAdapter.query)
    assert hints["query_input"] is AdapterQueryInput


def test_gold_answers_never_reach_generation() -> None:
    gold_answer = "پاسخ‌طلایی-مشخص"
    gold_evidence_text = "شواهد‌طلایی-مشخص"
    question = BenchmarkQuestion(
        question_id="q-gold-1",
        question=QUESTION,
        question_type=QuestionType.DIRECT_FACTUAL,
        scope=QuestionScope.LOCAL,
        gold_answer=gold_answer,
        acceptable_answers=[gold_answer],
        supporting_document_ids=["doc1"],
        supporting_evidence=[
            RetrievedEvidence(
                document_id="doc1",
                chunk_id="doc1::chunk-000000",
                text=gold_evidence_text,
                score=1.0,
                rank=1,
                evidence_type=EvidenceType.CHUNK,
            )
        ],
    )
    llm = FakeLLMProvider()
    adapter = _prepared_adapter(llm=llm)
    result = adapter.query(question.to_query_input())
    assert result.status == QueryStatus.SUCCESS
    assert gold_answer not in (result.answer or "")
    assert gold_evidence_text not in (result.answer or "")
    for evidence in result.retrieved_evidence:
        assert gold_answer not in evidence.text
        assert gold_evidence_text not in evidence.text
    assert gold_answer not in str(result.metadata)
    (context_question, context_passages, _) = llm.calls[0]
    assert context_question == QUESTION
    assert gold_answer not in context_question
    for passage in context_passages:
        assert gold_answer not in passage
        assert gold_evidence_text not in passage
