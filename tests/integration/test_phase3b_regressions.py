"""Regression tests for Phase 3B fixes.

These tests verify that the builder, configuration loader and JSON writers behave
according to the specifications introduced in Findings 5‑7.
from __future__ import annotations

import json
import pathlib
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from rag_benchmark.datasets.builder import BenchmarkDatasetBuilder
from rag_benchmark.datasets.config import load_config, RootConfig
from rag_benchmark.datasets.custom_wiki_questions import CustomWikiQuestionLoader
from rag_benchmark.datasets.pquad import PQuADLoader
from rag_benchmark.datasets.persian_mhqa import PersianMHQALoader
from rag_benchmark.datasets.wikipedia import WikipediaLoader
from rag_benchmark.schemas.document import BenchmarkDocument, DocumentSection
from rag_benchmark.schemas.question import (
    BenchmarkQuestion,
    DifficultyLevel,
    QuestionScope,
    QuestionType,
)
from rag_benchmark.schemas.evidence import RetrievedEvidence

# Helper dummy loader used across many tests
class DummyLoader:
    def __init__(self, docs, questions, name="dummy", version=""):
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

# ---------------------------------------------------------------------------
# 1‑7: Builder split metadata, checksum, split‑change effects
# ---------------------------------------------------------------------------

def test_split_metadata_and_counts(tmp_path: Path) -> None:
    d = make_doc("doc1")
    q1 = make_q("q1", ["doc1"])
    q2 = make_q("q2", ["doc1"])
    loader = DummyLoader([d], [q1, q2])
    builder = BenchmarkDatasetBuilder()
    builder.add_source(loader)
    manifest = builder.build(tmp_path)
    # Load persisted questions
    q_path = tmp_path / "questions.jsonl"
    lines = q_path.read_text(encoding="utf-8").splitlines()
    metas = [json.loads(l)["metadata"] for l in lines]
    assert all("split" in m for m in metas)
    # Count splits from metadata and compare to manifest
    counts = {s: 0 for s in ["train", "dev", "test", "benchmark"]}
    for m in metas:
        counts[m["split"]] += 1
    assert counts == manifest.split_counts

def test_checksum_consistency_across_output_dirs(tmp_path: Path) -> None:
    d = make_doc("doc1")
    q = make_q("q1", ["doc1"])
    loader = DummyLoader([d], [q])
    builder = BenchmarkDatasetBuilder()
    builder.add_source(loader)
    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"
    m1 = builder.build(out1)
    # Need a fresh builder instance for second build (stateful)
    builder2 = BenchmarkDatasetBuilder()
    builder2.add_source(loader)
    m2 = builder2.build(out2)
    assert m1.content_checksum == m2.content_checksum

def test_checksum_changes_when_document_changes(tmp_path: Path) -> None:
    d1 = make_doc("doc1", text="A")
    d2 = make_doc("doc1", text="B")
    q = make_q("q1", ["doc1"])
    # First build
    b1 = BenchmarkDatasetBuilder()
    b1.add_source(DummyLoader([d1], [q]))
    m1 = b1.build(tmp_path / "a")
    # Second build with different document content
    b2 = BenchmarkDatasetBuilder()
    b2.add_source(DummyLoader([d2], [q]))
    m2 = b2.build(tmp_path / "b")
    assert m1.content_checksum != m2.content_checksum

def test_checksum_changes_when_question_changes(tmp_path: Path) -> None:
    d = make_doc("doc1")
    q1 = make_q("q1", ["doc1"], question="What A?")
    q2 = make_q("q1", ["doc1"], question="What B?")
    b1 = BenchmarkDatasetBuilder()
    b1.add_source(DummyLoader([d], [q1]))
    m1 = b1.build(tmp_path / "a")
    b2 = BenchmarkDatasetBuilder()
    b2.add_source(DummyLoader([d], [q2]))
    m2 = b2.build(tmp_path / "b")
    assert m1.content_checksum != m2.content_checksum

def test_checksum_changes_when_split_changes(tmp_path: Path) -> None:
    d = make_doc("doc1")
    # Two different question IDs will hash to different splits with high probability
    q1 = make_q("qA", ["doc1"])
    q2 = make_q("qB", ["doc1"])
    b1 = BenchmarkDatasetBuilder()
    b1.add_source(DummyLoader([d], [q1]))
    m1 = b1.build(tmp_path / "a")
    b2 = BenchmarkDatasetBuilder()
    b2.add_source(DummyLoader([d], [q2]))
    m2 = b2.build(tmp_path / "b")
    assert m1.content_checksum != m2.content_checksum

# ---------------------------------------------------------------------------
# 8: dangling supporting evidence
# ---------------------------------------------------------------------------
def test_dangling_supporting_evidence_raises(tmp_path: Path) -> None:
    d = make_doc("doc1")
    evidence = RetrievedEvidence(document_id="missing", text="evidence", score=1.0)
    q = make_q("q1", ["doc1"]).model_copy(update={"supporting_evidence": [evidence]})
    loader = DummyLoader([d], [q])
    builder = BenchmarkDatasetBuilder()
    builder.add_source(loader)
    with pytest.raises(ValueError, match="Dangling supporting references"):
        builder.build(tmp_path)

# ---------------------------------------------------------------------------
# 9‑15: Configuration validation (Findings 5)
# ---------------------------------------------------------------------------
def write_yaml(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return p

def test_dataset_name_version_preserved(tmp_path: Path) -> None:
    cfg = {
        "dataset": {"name": "myset", "version": "v2"},
        "output": {"directory": str(tmp_path / "out")},
        "sources": {},
    }
    cfg_path = write_yaml(tmp_path, cfg)
    root = load_config(Path(cfg_path))
    assert root.dataset.name == "myset"
    assert root.dataset.version == "v2"

def test_output_dir_does_not_control_dataset_identity(tmp_path: Path) -> None:
    cfg = {
        "dataset": {"name": "myset", "version": "v2"},
        "output": {"directory": str(tmp_path / "out1")},
        "sources": {},
    }
    cfg_path = write_yaml(tmp_path, cfg)
    root1 = load_config(Path(cfg_path))
    # change only output dir
    cfg["output"]["directory"] = str(tmp_path / "out2")
    cfg_path2 = write_yaml(tmp_path, cfg)
    root2 = load_config(Path(cfg_path2))
    assert root1.dataset == root2.dataset

def test_unknown_top_level_key_rejected(tmp_path: Path) -> None:
    cfg = {"dataset": {"name": "a", "version": "b"}, "unknown": {}}
    cfg_path = write_yaml(tmp_path, cfg)
    with pytest.raises(ValidationError):
        load_config(Path(cfg_path))

def test_unknown_source_key_rejected(tmp_path: Path) -> None:
    cfg = {
        "dataset": {"name": "a", "version": "b"},
        "output": {"directory": "out"},
        "sources": {"unknown_source": {"enabled": True, "path": "x"}},
    }
    cfg_path = write_yaml(tmp_path, cfg)
    with pytest.raises(ValidationError):
        load_config(Path(cfg_path))

def test_enabled_string_false_rejected(tmp_path: Path) -> None:
    cfg = {
        "dataset": {"name": "a", "version": "b"},
        "output": {"directory": "out"},
        "sources": {"pquad": {"enabled": "false", "path": "x"}},
    }
    cfg_path = write_yaml(tmp_path, cfg)
    with pytest.raises(ValidationError):
        load_config(Path(cfg_path))

def test_enabled_source_without_path_rejected(tmp_path: Path) -> None:
    cfg = {
        "dataset": {"name": "a", "version": "b"},
        "output": {"directory": "out"},
        "sources": {"pquad": {"enabled": True}},
    }
    cfg_path = write_yaml(tmp_path, cfg)
    with pytest.raises(ValidationError):
        load_config(Path(cfg_path))

def test_duplicate_yaml_keys_rejected(tmp_path: Path) -> None:
    yaml_text = "dataset:\n  name: first\n  name: second\n"
    p = tmp_path / "dup.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    with pytest.raises(yaml.YAMLError):
        load_config(Path(p))

# ---------------------------------------------------------------------------
# 16‑18: Source record validation (Findings 6)
# ---------------------------------------------------------------------------
def test_malformed_custom_wiki_question_missing_id(tmp_path: Path) -> None:
    data = [{"question": "What?", "question_type": "direct_factual"}]
    p = tmp_path / "questions.jsonl"
    p.write_text("\n".join([json.dumps(r) for r in data]), encoding="utf-8")
    loader = CustomWikiQuestionLoader(p)
    with pytest.raises(ValueError, match="Missing 'question_id'"):
        list(loader.load_questions())

def test_malformed_pquad_missing_question_id(tmp_path: Path) -> None:
    pquad = {"data": [{"title": "t", "paragraphs": [{"qas": [{"question": "q?"}]}]}]}
    p = tmp_path / "pquad.json"
    p.write_text(json.dumps(pquad), encoding="utf-8")
    loader = PQuADLoader(p)
    with pytest.raises(ValueError, match="Missing question id"):
        list(loader.load_questions())

def test_missing_required_collection_rejected(tmp_path: Path) -> None:
    # Wikipedia loader expects a mapping per line; an empty file triggers missing page_id later
    p = tmp_path / "wiki.jsonl"
    p.write_text("{}\n", encoding="utf-8")
    loader = WikipediaLoader(p)
    with pytest.raises(ValueError, match="Missing 'page_id'"):
        list(loader.load_documents())

# ---------------------------------------------------------------------------
# 19‑21: Strict JSON rejects NaN/Infinity
# ---------------------------------------------------------------------------
def test_strict_json_nan_rejected(tmp_path: Path) -> None:
    obj = {"bad": float("nan")}
    from rag_benchmark.datasets.io import write_json
    with pytest.raises(ValueError):
        write_json(tmp_path / "nan.json", obj)

def test_strict_json_inf_rejected(tmp_path: Path) -> None:
    obj = {"bad": float("inf")}
    from rag_benchmark.datasets.io import write_json
    with pytest.raises(ValueError):
        write_json(tmp_path / "inf.json", obj)

def test_strict_json_ninf_rejected(tmp_path: Path) -> None:
    obj = {"bad": float("-inf")}
    from rag_benchmark.datasets.io import write_json
    with pytest.raises(ValueError):
        write_json(tmp_path / "ninf.json", obj)
"""
