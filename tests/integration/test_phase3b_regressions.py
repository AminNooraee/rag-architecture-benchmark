"""Regression tests for Phase 3B fixes.

These tests verify that the builder, configuration loader and JSON writers
behave according to the specifications in Findings 1-7.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from rag_benchmark.datasets.base import DatasetLoader
from rag_benchmark.datasets.builder import BenchmarkDatasetBuilder
from rag_benchmark.datasets.config import load_config
from rag_benchmark.datasets.custom_wiki_questions import CustomWikiQuestionLoader
from rag_benchmark.datasets.io import write_json
from rag_benchmark.datasets.persian_mhqa import PersianMHQALoader
from rag_benchmark.datasets.pquad import PQuADLoader
from rag_benchmark.datasets.wikipedia import WikipediaLoader
from rag_benchmark.schemas.document import BenchmarkDocument, DocumentSection
from rag_benchmark.schemas.evidence import RetrievedEvidence
from rag_benchmark.schemas.question import (
    BenchmarkQuestion,
    DifficultyLevel,
    QuestionScope,
    QuestionType,
)


class DummyLoader(DatasetLoader):
    """Minimal typed loader for exercising the builder directly."""

    def __init__(
        self,
        docs: Iterable[BenchmarkDocument],
        questions: Iterable[BenchmarkQuestion],
        name: str = "dummy",
        version: str = "",
    ) -> None:
        self._docs = docs
        self._questions = questions
        self._name = name
        self._version = version

    @property
    def source_name(self) -> str:
        return self._name

    @property
    def source_version(self) -> str:
        return self._version

    def load_documents(self) -> Iterable[BenchmarkDocument]:
        return self._docs

    def load_questions(self) -> Iterable[BenchmarkQuestion]:
        return self._questions


def make_doc(doc_id: str, text: str = "text") -> BenchmarkDocument:
    return BenchmarkDocument(
        document_id=doc_id,
        title="t",
        text=text,
        language="fa",
        source="src",
        url=None,
        sections=[DocumentSection(section_id="s1", heading="", text="s", order=0)],
        metadata={},
    )


def make_q(q_id: str, doc_ids: list[str], question: str = "q?") -> BenchmarkQuestion:
    return BenchmarkQuestion(
        question_id=q_id,
        question=question,
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


def write_yaml(tmp_path: Path, data: dict[str, Any]) -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Findings 1-3: persisted split metadata, checksum stability, referential integrity
# ---------------------------------------------------------------------------


def test_split_metadata_and_counts(tmp_path: Path) -> None:
    d = make_doc("doc1")
    q1 = make_q("q1", ["doc1"])
    q2 = make_q("q2", ["doc1"])
    builder = BenchmarkDatasetBuilder("regression-dataset", "v1")
    builder.add_source(DummyLoader([d], [q1, q2]))
    manifest = builder.build(tmp_path)
    lines = (tmp_path / "questions.jsonl").read_text(encoding="utf-8").splitlines()
    metas = [json.loads(line)["metadata"] for line in lines]
    assert all("split" in m for m in metas)
    counts: dict[str, int] = {"train": 0, "dev": 0, "test": 0, "benchmark": 0}
    for m in metas:
        counts[m["split"]] += 1
    assert counts == manifest.split_counts


def test_checksum_consistent_across_output_dirs(tmp_path: Path) -> None:
    d = make_doc("doc1")
    q = make_q("q1", ["doc1"])
    b1 = BenchmarkDatasetBuilder("regression-dataset", "v1")
    b1.add_source(DummyLoader([d], [q]))
    manifest1 = b1.build(tmp_path / "out1")
    b2 = BenchmarkDatasetBuilder("regression-dataset", "v1")
    b2.add_source(DummyLoader([d], [q]))
    manifest2 = b2.build(tmp_path / "out2")
    assert manifest1.content_checksum == manifest2.content_checksum


def test_checksum_changes_when_document_changes(tmp_path: Path) -> None:
    q = make_q("q1", ["doc1"])
    b1 = BenchmarkDatasetBuilder("regression-dataset", "v1")
    b1.add_source(DummyLoader([make_doc("doc1", text="A")], [q]))
    m1 = b1.build(tmp_path / "a")
    b2 = BenchmarkDatasetBuilder("regression-dataset", "v1")
    b2.add_source(DummyLoader([make_doc("doc1", text="B")], [q]))
    m2 = b2.build(tmp_path / "b")
    assert m1.content_checksum != m2.content_checksum


def test_checksum_changes_when_question_changes(tmp_path: Path) -> None:
    d = make_doc("doc1")
    b1 = BenchmarkDatasetBuilder("regression-dataset", "v1")
    b1.add_source(DummyLoader([d], [make_q("q1", ["doc1"], question="What A?")]))
    m1 = b1.build(tmp_path / "a")
    b2 = BenchmarkDatasetBuilder("regression-dataset", "v1")
    b2.add_source(DummyLoader([d], [make_q("q1", ["doc1"], question="What B?")]))
    m2 = b2.build(tmp_path / "b")
    assert m1.content_checksum != m2.content_checksum


def test_checksum_changes_when_split_changes(tmp_path: Path) -> None:
    d = make_doc("doc1")
    b1 = BenchmarkDatasetBuilder("regression-dataset", "v1")
    b1.add_source(DummyLoader([d], [make_q("qA", ["doc1"])]))
    m1 = b1.build(tmp_path / "a")
    b2 = BenchmarkDatasetBuilder("regression-dataset", "v1")
    b2.add_source(DummyLoader([d], [make_q("qB", ["doc1"])]))
    m2 = b2.build(tmp_path / "b")
    assert m1.content_checksum != m2.content_checksum


def test_dangling_supporting_evidence_raises(tmp_path: Path) -> None:
    d = make_doc("doc1")
    evidence = RetrievedEvidence(document_id="missing", text="evidence", score=1.0)
    q = make_q("q1", ["doc1"]).model_copy(update={"supporting_evidence": [evidence]})
    builder = BenchmarkDatasetBuilder("regression-dataset", "v1")
    builder.add_source(DummyLoader([d], [q]))
    with pytest.raises(ValueError, match="Dangling supporting document references"):
        builder.build(tmp_path)


# ---------------------------------------------------------------------------
# Finding 5: typed configuration
# ---------------------------------------------------------------------------


def test_dataset_name_version_preserved(tmp_path: Path) -> None:
    cfg: dict[str, Any] = {
        "dataset": {"name": "myset", "version": "v2"},
        "output": {"directory": str(tmp_path / "out")},
        "sources": {},
    }
    root = load_config(write_yaml(tmp_path, cfg))
    assert root.dataset.name == "myset"
    assert root.dataset.version == "v2"


def test_output_dir_does_not_control_dataset_identity(tmp_path: Path) -> None:
    output: dict[str, Any] = {"directory": str(tmp_path / "out1")}
    cfg: dict[str, Any] = {
        "dataset": {"name": "myset", "version": "v2"},
        "output": output,
        "sources": {},
    }
    root1 = load_config(write_yaml(tmp_path, cfg))
    output["directory"] = str(tmp_path / "out2")
    root2 = load_config(write_yaml(tmp_path, cfg))
    assert root1.dataset == root2.dataset


def test_unknown_top_level_key_rejected(tmp_path: Path) -> None:
    cfg: dict[str, Any] = {
        "dataset": {"name": "a", "version": "b"},
        "output": {"directory": "out"},
        "unknown": {},
    }
    with pytest.raises(ValidationError):
        load_config(write_yaml(tmp_path, cfg))


def test_unknown_source_key_rejected(tmp_path: Path) -> None:
    cfg: dict[str, Any] = {
        "dataset": {"name": "a", "version": "b"},
        "output": {"directory": "out"},
        "sources": {"unknown_source": {"enabled": True, "path": "x"}},
    }
    with pytest.raises(ValidationError):
        load_config(write_yaml(tmp_path, cfg))


def test_enabled_string_false_rejected(tmp_path: Path) -> None:
    cfg: dict[str, Any] = {
        "dataset": {"name": "a", "version": "b"},
        "output": {"directory": "out"},
        "sources": {"pquad": {"enabled": "false", "path": "x"}},
    }
    with pytest.raises(ValidationError):
        load_config(write_yaml(tmp_path, cfg))


def test_enabled_source_without_path_rejected(tmp_path: Path) -> None:
    cfg: dict[str, Any] = {
        "dataset": {"name": "a", "version": "b"},
        "output": {"directory": "out"},
        "sources": {"pquad": {"enabled": True}},
    }
    with pytest.raises(ValidationError):
        load_config(write_yaml(tmp_path, cfg))


def test_duplicate_yaml_keys_rejected(tmp_path: Path) -> None:
    p = tmp_path / "dup.yaml"
    p.write_text("dataset:\n  name: first\n  name: second\n", encoding="utf-8")
    with pytest.raises(yaml.YAMLError):
        load_config(p)


def test_blank_dataset_name_rejected(tmp_path: Path) -> None:
    cfg: dict[str, Any] = {
        "dataset": {"name": "", "version": "v1"},
        "output": {"directory": "out"},
    }
    with pytest.raises(ValidationError):
        load_config(write_yaml(tmp_path, cfg))


def test_whitespace_dataset_name_rejected(tmp_path: Path) -> None:
    cfg: dict[str, Any] = {
        "dataset": {"name": "   ", "version": "v1"},
        "output": {"directory": "out"},
    }
    with pytest.raises(ValidationError):
        load_config(write_yaml(tmp_path, cfg))


def test_blank_dataset_version_rejected(tmp_path: Path) -> None:
    cfg: dict[str, Any] = {
        "dataset": {"name": "a", "version": ""},
        "output": {"directory": "out"},
    }
    with pytest.raises(ValidationError):
        load_config(write_yaml(tmp_path, cfg))


def test_whitespace_dataset_version_rejected(tmp_path: Path) -> None:
    cfg: dict[str, Any] = {
        "dataset": {"name": "a", "version": "   "},
        "output": {"directory": "out"},
    }
    with pytest.raises(ValidationError):
        load_config(write_yaml(tmp_path, cfg))


# ---------------------------------------------------------------------------
# Finding 6: malformed source records fail with context
# ---------------------------------------------------------------------------


def test_malformed_custom_wiki_question_missing_id(tmp_path: Path) -> None:
    record = json.dumps({"question": "What?", "question_type": "direct_factual"})
    p = tmp_path / "questions.jsonl"
    p.write_text(record, encoding="utf-8")
    loader = CustomWikiQuestionLoader(p)
    with pytest.raises(ValueError, match="Missing 'question_id'"):
        list(loader.load_questions())


def test_custom_wiki_object_supporting_document_rejected(tmp_path: Path) -> None:
    record = json.dumps(
        {
            "question_id": "q1",
            "question": "Sual?",
            "supporting_document_ids": [{"id": "doc1"}],
        }
    )
    p = tmp_path / "questions.jsonl"
    p.write_text(record, encoding="utf-8")
    loader = CustomWikiQuestionLoader(p)
    with pytest.raises(ValueError, match="must be a non-empty string"):
        list(loader.load_questions())


def test_custom_wiki_string_reasoning_hops_rejected(tmp_path: Path) -> None:
    record = json.dumps({"question_id": "q1", "question": "Sual?", "reasoning_hops": "2"})
    p = tmp_path / "questions.jsonl"
    p.write_text(record, encoding="utf-8")
    loader = CustomWikiQuestionLoader(p)
    with pytest.raises(ValueError, match="'reasoning_hops' must be an integer"):
        list(loader.load_questions())


def test_custom_wiki_malformed_reasoning_hops_reports_context(tmp_path: Path) -> None:
    line1 = json.dumps({"question_id": "ok", "question": "Sual?", "reasoning_hops": 1})
    line2 = json.dumps({"question_id": "bad-q", "question": "Sual?", "reasoning_hops": "abc"})
    p = tmp_path / "questions.jsonl"
    p.write_text(line1 + "\n" + line2 + "\n", encoding="utf-8")
    loader = CustomWikiQuestionLoader(p)
    with pytest.raises(ValueError) as exc_info:
        list(loader.load_questions())
    message = str(exc_info.value)
    assert "bad-q" in message
    assert "line 2" in message
    assert "reasoning_hops" in message


def test_custom_wiki_boolean_reasoning_hops_rejected(tmp_path: Path) -> None:
    record = json.dumps({"question_id": "q1", "question": "Sual?", "reasoning_hops": True})
    p = tmp_path / "questions.jsonl"
    p.write_text(record, encoding="utf-8")
    loader = CustomWikiQuestionLoader(p)
    with pytest.raises(ValueError, match="'reasoning_hops' must be an integer"):
        list(loader.load_questions())


def test_malformed_pquad_missing_question_id(tmp_path: Path) -> None:
    pquad: dict[str, Any] = {
        "data": [{"title": "t", "paragraphs": [{"qas": [{"question": "q?"}]}]}]
    }
    p = tmp_path / "pquad.json"
    p.write_text(json.dumps(pquad), encoding="utf-8")
    loader = PQuADLoader(p)
    with pytest.raises(ValueError, match="Missing question id"):
        list(loader.load_questions())


def test_pquad_missing_data_rejected(tmp_path: Path) -> None:
    p = tmp_path / "pquad.json"
    p.write_text("{}", encoding="utf-8")
    loader = PQuADLoader(p)
    with pytest.raises(ValueError, match="missing the required 'data'"):
        list(loader.load_questions())


def test_pquad_non_list_data_rejected(tmp_path: Path) -> None:
    p = tmp_path / "pquad.json"
    p.write_text(json.dumps({"data": {}}), encoding="utf-8")
    loader = PQuADLoader(p)
    with pytest.raises(ValueError, match="'data' must be a list"):
        list(loader.load_questions())


def test_persian_mhqa_missing_questions_rejected(tmp_path: Path) -> None:
    p = tmp_path / "mhqa.json"
    p.write_text("{}", encoding="utf-8")
    loader = PersianMHQALoader(p)
    with pytest.raises(ValueError, match="missing the required 'questions'"):
        list(loader.load_questions())


def test_persian_mhqa_non_list_questions_rejected(tmp_path: Path) -> None:
    p = tmp_path / "mhqa.json"
    p.write_text(json.dumps({"questions": {}}), encoding="utf-8")
    loader = PersianMHQALoader(p)
    with pytest.raises(ValueError, match="'questions' must be a list"):
        list(loader.load_questions())


def test_persian_mhqa_object_supporting_document_rejected(tmp_path: Path) -> None:
    data: dict[str, Any] = {
        "questions": [
            {
                "question_id": "q1",
                "question": "Sual?",
                "type": "direct_factual",
                "reasoning_hops": 1,
                "answer": "a",
                "supporting_documents": [{"id": "doc1"}],
            }
        ]
    }
    p = tmp_path / "mhqa.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    loader = PersianMHQALoader(p)
    with pytest.raises(ValueError, match="must be a non-empty string"):
        list(loader.load_questions())


def test_missing_required_collection_rejected(tmp_path: Path) -> None:
    p = tmp_path / "wiki.jsonl"
    p.write_text("{}\n", encoding="utf-8")
    loader = WikipediaLoader(p)
    with pytest.raises(ValueError, match="Missing 'page_id'"):
        list(loader.load_documents())


# ---------------------------------------------------------------------------
# Finding 7: strict JSON
# ---------------------------------------------------------------------------


def test_strict_json_nan_rejected(tmp_path: Path) -> None:
    obj: dict[str, Any] = {"bad": float("nan")}
    with pytest.raises(ValueError):
        write_json(tmp_path / "nan.json", obj)


def test_strict_json_inf_rejected(tmp_path: Path) -> None:
    obj: dict[str, Any] = {"bad": float("inf")}
    with pytest.raises(ValueError):
        write_json(tmp_path / "inf.json", obj)


def test_strict_json_ninf_rejected(tmp_path: Path) -> None:
    obj: dict[str, Any] = {"bad": float("-inf")}
    with pytest.raises(ValueError):
        write_json(tmp_path / "ninf.json", obj)
