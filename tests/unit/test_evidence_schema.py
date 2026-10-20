"""Tests for retrieved evidence representation across diverse RAG architectures."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from rag_benchmark.schemas.evidence import EvidenceType, RetrievedEvidence


def test_all_evidence_types_supported() -> None:
    """Verify all required evidence types are present."""
    expected_types = {
        "chunk",
        "document",
        "entity",
        "relationship",
        "community_report",
        "summary_node",
        "tree_node",
        "cypher_result",
    }
    actual_types = {et.value for et in EvidenceType}
    assert expected_types == actual_types


def test_chunk_evidence() -> None:
    """Standard chunk evidence with provenance."""
    evidence = RetrievedEvidence(
        document_id="doc-123",
        chunk_id="chunk-456",
        text="متن تکه بازیابی شده از سند.",
        score=0.895,
        rank=1,
        evidence_type=EvidenceType.CHUNK,
        metadata={"start_char": 0, "end_char": 150},
    )
    assert evidence.document_id == "doc-123"
    assert evidence.chunk_id == "chunk-456"
    assert evidence.evidence_type == EvidenceType.CHUNK
    assert evidence.score == 0.895
    assert evidence.rank == 1


def test_graph_and_hierarchical_evidence_types() -> None:
    """Non-chunk representations: entities, relations, community reports, tree nodes."""
    entity_evidence = RetrievedEvidence(
        document_id="doc-biography",
        text="ابوعلی سینا (فیلسوف و پزشک ایرانی)",
        evidence_type=EvidenceType.ENTITY,
        metadata={"entity_type": "Person", "neo4j_id": 42},
    )
    assert entity_evidence.evidence_type == EvidenceType.ENTITY

    relation_evidence = RetrievedEvidence(
        text="(ابوعلی سینا)-[:نویسنده_است]->(کتاب قانون در طب)",
        evidence_type=EvidenceType.RELATIONSHIP,
        metadata={"source": "ابوعلی سینا", "target": "کتاب قانون در طب"},
    )
    assert relation_evidence.evidence_type == EvidenceType.RELATIONSHIP

    tree_evidence = RetrievedEvidence(
        document_id="doc-tree-root",
        text="خلاصه سطح ۲ درخت ساختاری مقالات علمی",
        evidence_type=EvidenceType.TREE_NODE,
        metadata={"layer": 2, "cluster_id": "c_12"},
    )
    assert tree_evidence.evidence_type == EvidenceType.TREE_NODE

    cypher_evidence = RetrievedEvidence(
        text="MATCH (n:Person) RETURN n LIMIT 5",
        evidence_type=EvidenceType.CYPHER_RESULT,
        metadata={"rows": 5},
    )
    assert cypher_evidence.evidence_type == EvidenceType.CYPHER_RESULT


def test_reject_nan_or_infinite_score() -> None:
    """Strict JSON: score cannot be NaN, Infinity, or -Infinity."""
    with pytest.raises(ValidationError):
        RetrievedEvidence(
            text="متن",
            score=float("nan"),
        )

    with pytest.raises(ValidationError):
        RetrievedEvidence(
            text="متن",
            score=float("inf"),
        )

    with pytest.raises(ValidationError):
        RetrievedEvidence(
            text="متن",
            score=float("-inf"),
        )


def test_reject_invalid_rank() -> None:
    """Rank must be >= 1 (1-based ranking)."""
    with pytest.raises(ValidationError):
        RetrievedEvidence(
            text="متن",
            rank=0,
        )

    with pytest.raises(ValidationError):
        RetrievedEvidence(
            text="متن",
            rank=-2,
        )


def test_evidence_unicode_roundtrip() -> None:
    """Persian evidence Unicode roundtrip."""
    evidence = RetrievedEvidence(
        document_id="سند-فارسی-۱",
        chunk_id="بخش-۲",
        text="داده‌های گراف دانش شامل ارتباطات بین‌مفاهیمی: پژوهشگاه دانش‌های بنیادی.",
        score=0.92,
        rank=1,
        evidence_type=EvidenceType.COMMUNITY_REPORT,
    )
    json_str = json.dumps(evidence.model_dump(mode="json"), ensure_ascii=False)
    assert "سند-فارسی-۱" in json_str
    assert "بین‌مفاهیمی" in json_str

    deserialized = RetrievedEvidence.model_validate(json.loads(json_str))
    assert deserialized.document_id == evidence.document_id
    assert deserialized.text == evidence.text
