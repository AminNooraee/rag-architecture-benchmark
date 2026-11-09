"""Builder validation tests."""

import pytest
from pathlib import Path

from rag_benchmark.datasets.builder import BenchmarkDatasetBuilder
from rag_benchmark.datasets.base import DatasetLoader
from rag_benchmark.schemas.document import BenchmarkDocument, DocumentSection
from rag_benchmark.schemas.question import (
    BenchmarkQuestion,
    DifficultyLevel,
    QuestionScope,
    QuestionType,
)


class DummyLoader(DatasetLoader):
    def __init__(self, docs, questions, name: str = "dummy", version: str = "") -> None:
        self._docs = docs
        self._qs = questions
        self._name = name
        self._ver = version

    @property
    def source_name(self) -> str:
        return self._name

    @property
    def source_version(self) -> str:
        return self._ver

    def load_documents(self):
        return self._docs

    def load_questions(self):
        return self._qs


def make_doc(doc_id: str) -> BenchmarkDocument:
    return BenchmarkDocument(
        document_id=doc_id,
        title="t",
        text="text",
        language="fa",
        source="src",
        url=None,
        sections=[DocumentSection(section_id="s1", heading="", text="text", order=0)],
        metadata={},
    )


def make_q(q_id: str, doc_ids) -> BenchmarkQuestion:
    return BenchmarkQuestion(
        question_id=q_id,
        question="q?",
        question_type=QuestionType.DIRECT_FACTUAL,
        difficulty=DifficultyLevel.MEDIUM,
        scope=QuestionScope.LOCAL,
        reasoning_hops=1,
        gold_answer="a",
        acceptable_answers=[],
        supporting_document_ids=doc_ids,
        supporting_evidence=[],
        answerable=True,
        dataset_source="src",
        query_metadata={},
        metadata={},
    )


def test_duplicate_document_detection(tmp_path: Path) -> None:
    d1 = make_doc("doc1")
    d2 = make_doc("doc1")  # duplicate ID
    loader = DummyLoader([d1, d2], [])
    builder = BenchmarkDatasetBuilder()
    builder.add_source(loader)
    with pytest.raises(ValueError, match="Duplicate document ID"):
        builder.build(tmp_path)


def test_missing_supporting_document(tmp_path: Path) -> None:
    d = make_doc("doc1")
    q = make_q("q1", ["nonexistent"])
    loader = DummyLoader([d], [q])
    builder = BenchmarkDatasetBuilder()
    builder.add_source(loader)
    with pytest.raises(ValueError, match="Dangling supporting document references"):
        builder.build(tmp_path)
