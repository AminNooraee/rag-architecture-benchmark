"""Deterministic statistics for a built dataset.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from rag_benchmark.schemas.document import BenchmarkDocument
from rag_benchmark.schemas.question import BenchmarkQuestion


class DatasetStatistics(BaseModel):
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    document_count: int
    section_count: int
    question_count: int
    answerable_question_count: int
    unanswerable_question_count: int
    average_document_length: float
    min_document_length: int
    max_document_length: int

    model_config = ConfigDict(extra="forbid", frozen=True)

    @classmethod
    def from_items(
        cls,
        documents: Iterable[BenchmarkDocument],
        questions: Iterable[BenchmarkQuestion],
    ) -> DatasetStatistics:
        docs = list(documents)
        qs = list(questions)
        doc_lengths = [len(d.text) for d in docs]
        avg_len = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0.0
        min_len = min(doc_lengths) if doc_lengths else 0
        max_len = max(doc_lengths) if doc_lengths else 0
        section_cnt = sum(len(d.sections) for d in docs)
        ans_cnt = sum(1 for q in qs if q.answerable)
        unans_cnt = len(qs) - ans_cnt
        return cls(
            document_count=len(docs),
            section_count=section_cnt,
            question_count=len(qs),
            answerable_question_count=ans_cnt,
            unanswerable_question_count=unans_cnt,
            average_document_length=avg_len,
            min_document_length=min_len,
            max_document_length=max_len,
        )
