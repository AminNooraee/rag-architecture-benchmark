"""Tests for canonical question schema and adapter input separation."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from rag_benchmark.errors.models import GroundTruthLeakageError
from rag_benchmark.schemas.evidence import EvidenceType, RetrievedEvidence
from rag_benchmark.schemas.question import (
    AdapterQueryInput,
    BenchmarkQuestion,
    DifficultyLevel,
    QuestionScope,
    QuestionType,
)


def test_valid_persian_question_creation() -> None:
    """Test creating a valid benchmark question with Persian text."""
    evidence = RetrievedEvidence(
        document_id="doc-1",
        chunk_id="chunk-1",
        text="پایتخت ایران، شهر تهران است.",
        evidence_type=EvidenceType.CHUNK,
    )
    q = BenchmarkQuestion(
        question_id="q-001",
        question="پایتخت ایران کدام شهر است و چه ویژگی‌هایی دارد؟",
        question_type=QuestionType.DIRECT_FACTUAL,
        difficulty=DifficultyLevel.EASY,
        scope=QuestionScope.LOCAL,
        reasoning_hops=1,
        gold_answer="تهران",
        acceptable_answers=["شهر تهران", "تهران بزرگ"],
        supporting_document_ids=["doc-1"],
        supporting_evidence=[evidence],
        answerable=True,
        dataset_source="persian_bench_v1",
        query_metadata={"domain": "geography"},
        metadata={"annotator": "کاربر ۱"},
    )

    assert q.question_id == "q-001"
    assert q.question_type == QuestionType.DIRECT_FACTUAL
    assert q.scope == QuestionScope.LOCAL
    assert q.reasoning_hops == 1
    assert q.gold_answer == "تهران"
    assert len(q.supporting_evidence) == 1
    assert q.query_metadata["domain"] == "geography"


def test_all_question_types_supported() -> None:
    """Verify all 8 mandatory question types are valid enum members."""
    expected_types = {
        "direct_factual",
        "detailed_factual",
        "multi_hop",
        "relationship",
        "cross_document",
        "global_summary",
        "hierarchical",
        "unanswerable",
    }
    actual_types = {qt.value for qt in QuestionType}
    assert expected_types == actual_types


def test_all_question_scopes_supported() -> None:
    """Verify minimum required question scopes are supported."""
    expected_scopes = {"local", "multi_document", "global"}
    actual_scopes = {qs.value for qs in QuestionScope}
    assert expected_scopes.issubset(actual_scopes)


def test_reject_invalid_question_type() -> None:
    """Invalid question type must raise ValidationError."""
    with pytest.raises(ValidationError):
        BenchmarkQuestion(
            question_id="q-bad-type",
            question="یک سؤال تستی؟",
            question_type="invalid_type",  # type: ignore[arg-type]
            scope=QuestionScope.LOCAL,
        )


def test_reject_invalid_scope() -> None:
    """Invalid scope must raise ValidationError."""
    with pytest.raises(ValidationError):
        BenchmarkQuestion(
            question_id="q-bad-scope",
            question="یک سؤال تستی؟",
            question_type=QuestionType.DIRECT_FACTUAL,
            scope="infinite_universe",  # type: ignore[arg-type]
        )


def test_reject_negative_reasoning_hops() -> None:
    """Reasoning hops must be >= 0."""
    with pytest.raises(ValidationError):
        BenchmarkQuestion(
            question_id="q-neg-hops",
            question="سؤال؟",
            question_type=QuestionType.DIRECT_FACTUAL,
            scope=QuestionScope.LOCAL,
            reasoning_hops=-1,
        )


def test_reject_empty_question_and_id() -> None:
    """Empty question_id or question must be rejected."""
    with pytest.raises(ValidationError):
        BenchmarkQuestion(
            question_id="",
            question="سؤال؟",
            question_type=QuestionType.DIRECT_FACTUAL,
            scope=QuestionScope.LOCAL,
        )

    with pytest.raises(ValidationError):
        BenchmarkQuestion(
            question_id="q-1",
            question="   ",
            question_type=QuestionType.DIRECT_FACTUAL,
            scope=QuestionScope.LOCAL,
        )


def test_to_query_input_enforces_ground_truth_isolation() -> None:
    """CRITICAL: RAG adapter query() must NEVER receive gold answers or ground truth evidence.

    to_query_input() forwards query_metadata, strictly withholding evaluation metadata.
    """
    evidence = RetrievedEvidence(
        document_id="doc-99",
        text="شواهد مرجع",
    )
    question = BenchmarkQuestion(
        question_id="q-isolate-1",
        question="چه سالی دانشگاه تهران تأسیس شد؟",
        question_type=QuestionType.DIRECT_FACTUAL,
        scope=QuestionScope.LOCAL,
        reasoning_hops=1,
        gold_answer="۱۳۱۳ هجری شمسی",
        acceptable_answers=["1313", "۱۳۱۳"],
        supporting_document_ids=["doc-99"],
        supporting_evidence=[evidence],
        query_metadata={"category": "history"},
        metadata={"eval_secret": "do_not_leak"},
    )

    query_input = question.to_query_input()

    assert isinstance(query_input, AdapterQueryInput)
    assert query_input.question_id == "q-isolate-1"
    assert query_input.question == "چه سالی دانشگاه تهران تأسیس شد؟"
    assert query_input.metadata.get("category") == "history"
    assert "eval_secret" not in query_input.metadata

    # AdapterQueryInput does not possess ground-truth fields
    assert not hasattr(query_input, "gold_answer")
    assert not hasattr(query_input, "acceptable_answers")
    assert not hasattr(query_input, "supporting_document_ids")
    assert not hasattr(query_input, "supporting_evidence")


def test_query_metadata_rejects_ground_truth_leakage() -> None:
    """Attempting to embed ground truth keys into query_metadata raises GroundTruthLeakageError."""
    question = BenchmarkQuestion(
        question_id="q-leak",
        question="یک سؤال؟",
        question_type=QuestionType.DIRECT_FACTUAL,
        scope=QuestionScope.LOCAL,
        gold_answer="پاسخ",
        query_metadata={"gold_answer": "leaked_answer"},
    )
    with pytest.raises(GroundTruthLeakageError, match="Forbidden ground-truth key"):
        question.to_query_input()


def test_question_unicode_roundtrip() -> None:
    """Persian question Unicode roundtrip."""
    q = BenchmarkQuestion(
        question_id="q-fa-unicode",
        question="آیا سیستم‌های خبره می‌توانند جایگزین مدل‌های زبانی شوند؟",
        question_type=QuestionType.RELATIONSHIP,
        scope=QuestionScope.MULTI_DOCUMENT,
        reasoning_hops=2,
        gold_answer="پاسخ شامل نیم‌فاصله‌ها: خیر، مکمل یکدیگرند.",
        metadata={"برچسب": "هوش مصنوعی"},
    )

    json_str = json.dumps(q.model_dump(mode="json"), ensure_ascii=False)
    assert "می‌توانند" in json_str
    assert "هوش مصنوعی" in json_str

    deserialized = BenchmarkQuestion.model_validate(json.loads(json_str))
    assert deserialized.question == q.question
    assert deserialized.gold_answer == q.gold_answer
    assert deserialized.metadata["برچسب"] == "هوش مصنوعی"
