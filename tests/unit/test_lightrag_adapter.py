"""Lifecycle and contract tests for the official LightRAG adapter (Phase 3D).

All tests run offline: a fake embedding provider and a scripted async LLM
replace the OpenAI-compatible endpoints. LightRAG itself (installed optional
extra) is exercised for real: chunking, doc-status bookkeeping, storage
initialization, hybrid retrieval, and prompt construction.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("lightrag")

from rag_benchmark.adapters.base import IndexingResult  # noqa: E402
from rag_benchmark.adapters.lightrag import LightRAGAdapter  # noqa: E402
from rag_benchmark.config.models import BenchmarkConfig  # noqa: E402
from rag_benchmark.errors.models import (  # noqa: E402
    BenchmarkErrorType,
    BenchmarkException,
    ConfigurationError,
)
from rag_benchmark.schemas.document import BenchmarkDocument  # noqa: E402
from rag_benchmark.schemas.evidence import EvidenceType, RetrievedEvidence  # noqa: E402
from rag_benchmark.schemas.question import AdapterQueryInput  # noqa: E402
from rag_benchmark.schemas.result import QueryStatus  # noqa: E402
from tests.unit.fakes import FakeEmbeddingProvider  # noqa: E402

DOC_ONE_TEXT = "متن اول تکراری برای تکه‌بندی. " * 8
DOC_TWO_TEXT = "متن دوم با موضوع دیگر کاملاً. " * 8
LIGHTRAG_EXTRA = {"chunk_token_size": 80, "chunk_overlap_token_size": 10}
ANSWER = "پاسخ آزمایشی"


def make_doc(doc_id: str, text: str | None = None) -> BenchmarkDocument:
    return BenchmarkDocument(
        document_id=doc_id,
        title=f"عنوان {doc_id}",
        text=text if text is not None else DOC_ONE_TEXT,
    )


def make_config(
    extra: dict[str, Any] | None = None,
    *,
    embedding_dimension: int | None = None,
) -> BenchmarkConfig:
    payload: dict[str, Any] = {"system": "lightrag"}
    if extra is not None:
        payload["extra"] = {"lightrag": extra}
    if embedding_dimension is not None:
        payload["embedding"] = {"dimension": embedding_dimension}
    return BenchmarkConfig.model_validate(payload)


class UniformEmbeddingProvider:
    """Provider returning an identical vector for every text.

    Every chunk vector is cosine-identical to the query vector (similarity
    1.0), so LightRAG's 0.2 cosine floor always passes and retrieval is
    deterministic regardless of chunker slicing or text content.
    """

    def __init__(self, dimension: int = 8) -> None:
        self._vector = [1.0] + [0.0] * (dimension - 1)
        self.documents_batches: list[list[str]] = []

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.documents_batches.append(list(texts))
        return [list(self._vector) for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return list(self._vector)


class ScriptedLLM:
    """Deterministic async LLM fake with scripted latency and failure.

    Emulates the LightRAG pipeline prompts so the real LightRAG indexing and
    query pipeline runs to completion offline:

    - entity/relation extraction prompts receive well-formed extraction rows
      (two entities derived from the chunk's first tokens plus one relation),
    - gleaning rounds are answered with the completion delimiter only,
    - keyword extraction prompts receive a JSON keyword object derived from
      the query,
    - every other prompt (final answer, description merge) receives the
      scripted answer.
    """

    TUPLE_DELIMITER = "<|#|>"
    COMPLETION_DELIMITER = "<|COMPLETE|>"

    def __init__(
        self,
        answer: str = ANSWER,
        *,
        fail_after: int | None = None,
        sleep_seconds: float = 0.0,
    ) -> None:
        self.answer = answer
        self.fail_after = fail_after
        self.sleep_seconds = sleep_seconds
        self.call_count = 0
        self.prompts: list[str] = []

    async def __call__(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history_messages: Any = None,
        **kwargs: Any,
    ) -> str:
        self.call_count += 1
        self.prompts.append(prompt)
        if self.sleep_seconds:
            await asyncio.sleep(self.sleep_seconds)
        if self.fail_after is not None and self.call_count > self.fail_after:
            raise RuntimeError("scripted LLM failure")
        combined = f"{system_prompt or ''}\n{prompt}"
        if "Knowledge Graph Specialist" in combined:
            return self._extraction_response(prompt)
        if "expert keyword extractor" in combined:
            return self._keyword_response(prompt)
        return self.answer

    def _extraction_response(self, prompt: str) -> str:
        if "last extraction task" in prompt:
            return self.COMPLETION_DELIMITER
        section = (
            prompt[prompt.find("---Input Text---") :] if "---Input Text---" in prompt else prompt
        )
        fence_open = section.find("```")
        fence_close = section.find("```", fence_open + 3) if fence_open != -1 else -1
        body = (
            section[fence_open + 3 : fence_close].strip() if fence_close > fence_open else section
        )
        tokens = [token for token in body.split() if token]
        if not tokens:
            return self.COMPLETION_DELIMITER
        first = tokens[0]
        second = tokens[1] if len(tokens) > 1 else first
        d = self.TUPLE_DELIMITER
        return "\n".join(
            [
                f"entity{d}{first}{d}Other{d}Deterministic test entity {first}",
                f"entity{d}{second}{d}Other{d}Deterministic test entity {second}",
                f"relation{d}{first}{d}{second}{d}mention{d}Deterministic test relation",
                self.COMPLETION_DELIMITER,
            ]
        )

    @staticmethod
    def _keyword_response(prompt: str) -> str:
        query = ""
        for line in prompt.splitlines():
            if line.startswith("User Query:"):
                query = line.removeprefix("User Query:").strip()
                break
        return json.dumps(
            {"high_level_keywords": [query], "low_level_keywords": [query]}, ensure_ascii=False
        )


def make_adapter(
    tmp_path: Path,
    *,
    llm: ScriptedLLM | None = None,
    extra: dict[str, Any] | None = None,
    embedding: FakeEmbeddingProvider | UniformEmbeddingProvider | None = None,
    working_dir: Path | None = None,
    embedding_dimension: int | None = None,
) -> tuple[LightRAGAdapter, FakeEmbeddingProvider | UniformEmbeddingProvider, ScriptedLLM]:
    llm = llm or ScriptedLLM()
    embedding = embedding or FakeEmbeddingProvider(dimension=8)
    adapter = LightRAGAdapter(
        embedding_provider=embedding,
        llm_func=llm,
        working_dir=working_dir if working_dir is not None else tmp_path / "storage",
    )
    adapter.activate(make_config(extra, embedding_dimension=embedding_dimension))
    return adapter, embedding, llm


def _index_two_docs(adapter: LightRAGAdapter) -> IndexingResult:
    return adapter.index([make_doc("doc-1"), make_doc("doc-2", text=DOC_TWO_TEXT)])


# ---------------------------------------------------------------------------
# Identity, capabilities, configuration
# ---------------------------------------------------------------------------


def test_system_variant_and_capabilities() -> None:
    adapter = LightRAGAdapter(
        embedding_provider=FakeEmbeddingProvider(dimension=8), llm_func=ScriptedLLM()
    )
    assert adapter.system_name == "lightrag"
    assert adapter.variant_name == "hybrid_no_rerank"
    capabilities = adapter.capabilities
    assert capabilities.supports_incremental_insert is True
    assert capabilities.supports_persistence is True
    adapter.validate_capabilities()


def test_prepare_rejects_mismatched_system(tmp_path: Path) -> None:
    adapter, _, _ = make_adapter(tmp_path)
    with pytest.raises(ConfigurationError, match="does not match"):
        adapter.prepare(BenchmarkConfig.model_validate({"system": "traditional_rag"}))


def test_prepare_rejects_unsupported_variant(tmp_path: Path) -> None:
    adapter, _, _ = make_adapter(tmp_path)
    with pytest.raises(ConfigurationError, match="not compatible"):
        adapter.prepare(
            BenchmarkConfig.model_validate({"system": "lightrag", "variant": "chunk_512"})
        )


def test_prepare_parses_lightrag_extra_and_restores_on_absence(tmp_path: Path) -> None:
    adapter, _, _ = make_adapter(tmp_path)
    adapter.prepare(make_config({"top_k": 7, "chunk_top_k": 3}))
    assert adapter._lightrag_config.top_k == 7
    assert adapter._lightrag_config.chunk_top_k == 3
    # Re-prepare without the extra key restores the constructor config.
    adapter.prepare(make_config(None))
    assert adapter._lightrag_config.top_k == 40
    assert adapter._lightrag_config.chunk_top_k == 20
    with pytest.raises(ConfigurationError, match="Invalid lightrag configuration"):
        adapter.prepare(make_config({"chunk_token_size": 100, "chunk_overlap_token_size": 100}))
    with pytest.raises(ConfigurationError, match="must be a mapping"):
        adapter.prepare(
            BenchmarkConfig.model_validate({"system": "lightrag", "extra": {"lightrag": [1, 2]}})
        )


def test_index_requires_prepare() -> None:
    adapter = LightRAGAdapter(
        embedding_provider=FakeEmbeddingProvider(dimension=8), llm_func=ScriptedLLM()
    )
    with pytest.raises(ConfigurationError, match="before index"):
        adapter.index([make_doc("doc-1")])


def test_index_duplicate_ids_rejected(tmp_path: Path) -> None:
    adapter, _, _ = make_adapter(tmp_path, extra=LIGHTRAG_EXTRA)
    with pytest.raises(ConfigurationError, match="duplicate document_id"):
        adapter.index([make_doc("doc-1"), make_doc("doc-1")])


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


def test_index_reports_documents_and_chunks(tmp_path: Path) -> None:
    adapter, _, _ = make_adapter(tmp_path, extra=LIGHTRAG_EXTRA)
    result = _index_two_docs(adapter)
    assert result.indexed_documents == 2
    assert result.indexed_chunks == 8
    assert result.elapsed_ms >= 0.0
    assert result.metadata["chunk_token_size"] == 80
    assert result.metadata["chunk_overlap_token_size"] == 10
    assert result.metadata["embedding_dimension"] == 8


def test_index_embedding_failure_raises_indexing_error(tmp_path: Path) -> None:
    # embedding_dimension set => no probe; the provider fails during the real
    # insert, LightRAG marks the document failed, and the status check raises.
    adapter, _, _ = make_adapter(
        tmp_path,
        extra=LIGHTRAG_EXTRA,
        embedding=FakeEmbeddingProvider(dimension=8, fail_documents=RuntimeError("embed down")),
        embedding_dimension=8,
    )
    with pytest.raises(BenchmarkException) as exc_info:
        adapter.index([make_doc("doc-1")])
    assert exc_info.value.error_type is BenchmarkErrorType.INDEXING_ERROR
    assert "document processing failed" in str(exc_info.value)


def test_index_embedding_probe_failure_raises_indexing_error(tmp_path: Path) -> None:
    # dimension unset => the adapter probes the provider, which fails.
    adapter, _, _ = make_adapter(
        tmp_path,
        extra=LIGHTRAG_EXTRA,
        embedding=FakeEmbeddingProvider(dimension=8, fail_documents=RuntimeError("embed down")),
    )
    with pytest.raises(BenchmarkException) as exc_info:
        adapter.index([make_doc("doc-1")])
    assert exc_info.value.error_type is BenchmarkErrorType.INDEXING_ERROR
    assert "probing the vector dimension" in str(exc_info.value)


def test_repeated_index_is_full_rebuild(tmp_path: Path) -> None:
    adapter, _, _ = make_adapter(
        tmp_path, extra=LIGHTRAG_EXTRA, embedding=UniformEmbeddingProvider()
    )
    _index_two_docs(adapter)
    # Explicit full rebuild with different documents: the old documents must be gone.
    adapter.index([make_doc("doc-3"), make_doc("doc-4", text=DOC_TWO_TEXT)])
    health = adapter.health()
    assert health.details["indexed_documents"] == 2
    assert health.details["indexed_chunks"] == 8
    result = adapter.query(AdapterQueryInput(question_id="q-1", question="پرسش"))
    assert result.status is QueryStatus.SUCCESS
    assert result.retrieved_evidence
    for evidence in result.retrieved_evidence:
        assert evidence.document_id in {"doc-3", "doc-4"}


def test_embedding_dimension_probe_when_unconfigured(tmp_path: Path) -> None:
    adapter, _, _ = make_adapter(tmp_path, extra=LIGHTRAG_EXTRA)  # dimension defaults to None
    result = adapter.index([make_doc("doc-1")])
    assert result.indexed_documents == 1
    assert result.metadata["embedding_dimension"] == 8


def test_run_async_rejects_running_event_loop(tmp_path: Path) -> None:
    adapter, _, _ = make_adapter(tmp_path, extra=LIGHTRAG_EXTRA)

    async def driver() -> None:
        with pytest.raises(ConfigurationError, match="running asyncio"):
            adapter.index([make_doc("doc-1")])

    asyncio.run(driver())
    assert adapter._rag is None  # no half-initialized instance left behind


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


def test_query_unprepared_returns_configuration_failure() -> None:
    adapter = LightRAGAdapter(
        embedding_provider=FakeEmbeddingProvider(dimension=8), llm_func=ScriptedLLM()
    )
    result = adapter.query(AdapterQueryInput(question_id="q-1", question="پرسش"))
    assert result.status is QueryStatus.FAILED
    assert result.answer is None
    assert result.error is not None
    assert result.error.error_type is BenchmarkErrorType.CONFIGURATION_ERROR


def test_query_before_index_returns_configuration_failure(tmp_path: Path) -> None:
    adapter, _, _ = make_adapter(tmp_path, extra=LIGHTRAG_EXTRA)
    result = adapter.query(AdapterQueryInput(question_id="q-1", question="پرسش"))
    assert result.status is QueryStatus.FAILED
    assert result.answer is None
    assert result.error is not None
    assert result.error.error_type is BenchmarkErrorType.CONFIGURATION_ERROR


def test_query_success_provenance_usage_and_timing(tmp_path: Path) -> None:
    adapter, embedding, llm = make_adapter(
        tmp_path, extra=LIGHTRAG_EXTRA, embedding=UniformEmbeddingProvider()
    )
    _index_two_docs(adapter)

    result = adapter.query(AdapterQueryInput(question_id="q-1", question="پرسش آزمایشی"))

    assert result.status is QueryStatus.SUCCESS
    assert result.answer == ANSWER
    assert result.system == "lightrag"
    assert result.variant == "hybrid_no_rerank"
    assert result.error is None

    chunk_evidence = [e for e in result.retrieved_evidence if e.evidence_type is EvidenceType.CHUNK]
    assert chunk_evidence
    for evidence in chunk_evidence:
        assert evidence.document_id in {"doc-1", "doc-2"}
        assert evidence.chunk_id is not None
        assert evidence.text
        assert evidence.rank is not None and evidence.rank >= 1
    ranks = [e.rank for e in result.retrieved_evidence if e.rank is not None]
    assert ranks == list(range(1, len(ranks) + 1))

    assert result.metadata["query_mode"] == "hybrid"
    assert result.metadata["retrieved_chunks"] == len(chunk_evidence)
    assert result.usage.llm_calls >= 2  # keyword extraction + final answer
    assert result.usage.embedding_calls >= 1
    assert embedding.documents_batches  # provider was exercised during indexing
    assert llm.call_count >= 2
    assert result.timing.total_ms >= 0.0
    assert result.timing.retrieval_ms + result.timing.generation_ms <= result.timing.total_ms + 1e-6


def test_prepare_rejects_non_hybrid_query_mode(tmp_path: Path) -> None:
    adapter, _, _ = make_adapter(tmp_path)
    with pytest.raises(ConfigurationError, match="Invalid lightrag configuration"):
        adapter.prepare(make_config({"query_mode": "naive"}))
    # The adapter keeps its pinned hybrid configuration.
    assert adapter._lightrag_config.query_mode == "hybrid"


def test_query_param_always_receives_hybrid_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, _, _ = make_adapter(
        tmp_path, extra=LIGHTRAG_EXTRA, embedding=UniformEmbeddingProvider()
    )
    _index_two_docs(adapter)

    captured: dict[str, Any] = {}

    async def fake_aquery_llm(question: str, param: Any = None, **kwargs: Any) -> dict[str, Any]:
        captured["param"] = param
        return {
            "status": "success",
            "message": "ok",
            "llm_response": {"content": ANSWER},
            "data": None,
        }

    monkeypatch.setattr(adapter._rag, "aquery_llm", fake_aquery_llm)
    result = adapter.query(AdapterQueryInput(question_id="q-1", question="پرسش"))

    assert result.status is QueryStatus.SUCCESS
    assert captured["param"].mode == "hybrid"
    assert result.metadata["query_mode"] == "hybrid"


def test_query_llm_failure_returns_failed_result(tmp_path: Path) -> None:
    adapter, _, llm = make_adapter(tmp_path, extra=LIGHTRAG_EXTRA)
    _index_two_docs(adapter)
    llm.fail_after = llm.call_count  # the next call (first of the query) fails

    result = adapter.query(AdapterQueryInput(question_id="q-1", question="پرسش"))

    assert result.status is not QueryStatus.SUCCESS
    assert result.answer is None
    assert result.error is not None
    assert "scripted LLM failure" in result.error.message


def test_query_timeout_returns_timeout_result(tmp_path: Path) -> None:
    adapter, _, llm = make_adapter(
        tmp_path, extra={**LIGHTRAG_EXTRA, "query_timeout_seconds": 0.05}
    )
    _index_two_docs(adapter)
    llm.sleep_seconds = 0.5
    try:
        result = adapter.query(AdapterQueryInput(question_id="q-1", question="پرسش"))
    finally:
        adapter.cleanup()

    assert result.status is QueryStatus.TIMEOUT
    assert result.answer is None
    assert result.error is not None
    assert result.error.error_type is BenchmarkErrorType.TIMEOUT


# ---------------------------------------------------------------------------
# Incremental insert / cleanup / health
# ---------------------------------------------------------------------------


def test_incremental_insert_appends_to_index(tmp_path: Path) -> None:
    adapter, _, _ = make_adapter(tmp_path, extra=LIGHTRAG_EXTRA)
    first = adapter.index([make_doc("doc-1")])
    assert first.indexed_documents == 1
    assert first.indexed_chunks == 4

    second = adapter.insert([make_doc("doc-2", text=DOC_TWO_TEXT)])
    assert second.indexed_documents == 1
    assert second.indexed_chunks == 4
    assert second.metadata["incremental"] is True

    health = adapter.health()
    assert health.details["indexed_documents"] == 2
    assert health.details["indexed_chunks"] == 8


def test_insert_without_index_rejected(tmp_path: Path) -> None:
    adapter, _, _ = make_adapter(tmp_path, extra=LIGHTRAG_EXTRA)
    with pytest.raises(ConfigurationError, match="call index\\(\\) before insert"):
        adapter.insert([make_doc("doc-1")])


def test_cleanup_removes_owned_working_dir_and_allows_reindex(tmp_path: Path) -> None:
    # working_dir=None => the adapter creates and owns a temp working directory.
    adapter = LightRAGAdapter(
        embedding_provider=FakeEmbeddingProvider(dimension=8),
        llm_func=ScriptedLLM(),
        working_dir=None,
    )
    adapter.activate(make_config(LIGHTRAG_EXTRA))
    adapter.index([make_doc("doc-1")])
    working_dir = adapter._working_dir
    assert working_dir is not None
    assert working_dir.exists()

    adapter.cleanup()
    assert not working_dir.exists()
    result = adapter.query(AdapterQueryInput(question_id="q-1", question="پرسش"))
    assert result.status is QueryStatus.FAILED
    assert result.error is not None
    assert result.error.error_type is BenchmarkErrorType.CONFIGURATION_ERROR

    # The adapter stays re-indexable after cleanup.
    reindex = adapter.index([make_doc("doc-1")])
    assert reindex.indexed_documents == 1
    adapter.cleanup()


def test_cleanup_keeps_caller_owned_working_dir(tmp_path: Path) -> None:
    working_dir = tmp_path / "caller-owned"
    adapter, _, _ = make_adapter(tmp_path, extra=LIGHTRAG_EXTRA, working_dir=working_dir)
    adapter.index([make_doc("doc-1")])
    assert working_dir.exists()

    adapter.cleanup()
    assert working_dir.exists()  # caller-owned directories are never deleted


def test_health_reports_index_state(tmp_path: Path) -> None:
    adapter, _, _ = make_adapter(tmp_path, extra=LIGHTRAG_EXTRA)
    assert adapter.health().healthy is True
    details = adapter.health().details
    assert details["dependency_available"] is True
    assert details["initialized"] is False
    assert details["indexed"] is False
    assert details["indexed_documents"] == 0
    assert details["query_mode"] == "hybrid"

    _index_two_docs(adapter)
    details = adapter.health().details
    assert details["initialized"] is True
    assert details["indexed"] is True
    assert details["indexed_documents"] == 2
    assert details["indexed_chunks"] == 8
    assert details["embedding_dimension"] == 8
    assert details["llm_func"] == "injected"
    assert isinstance(details["lightrag_version"], str)
    assert details["lightrag_version"] != "unknown"


def test_health_unhealthy_before_prepare() -> None:
    adapter = LightRAGAdapter(
        embedding_provider=FakeEmbeddingProvider(dimension=8), llm_func=ScriptedLLM()
    )
    health = adapter.health()
    assert health.healthy is False
    assert health.system == "lightrag"


# ---------------------------------------------------------------------------
# Evidence mapping (pure)
# ---------------------------------------------------------------------------


def test_evidence_mapping_from_raw_data() -> None:
    raw = {
        "chunks": [
            {"file_path": "doc-1", "chunk_id": "doc-1-chunk-001", "content": "متن یک"},
            {"file_path": "doc-2", "chunk_id": "doc-2-chunk-000", "content": ""},  # dropped
        ],
        "entities": [
            {
                "entity_name": "تهران",
                "entity_type": "location",
                "description": "شرح شهر",
                "source_id": "doc-1-chunk-001",
                "file_path": "doc-1",
            },
            {"description": ""},  # dropped
        ],
        "relationships": [
            {
                "src_id": "تهران",
                "tgt_id": "ایران",
                "keywords": "مختصات",
                "weight": 42.5,
                "description": "شرح رابطه",
                "source_id": "doc-2-chunk-000",
                "file_path": "doc-2",
            }
        ],
    }
    evidence: list[RetrievedEvidence] = []
    assert LightRAGAdapter._evidence_from_raw_data(raw, evidence) is None
    assert [e.evidence_type for e in evidence] == [
        EvidenceType.CHUNK,
        EvidenceType.ENTITY,
        EvidenceType.RELATIONSHIP,
    ]
    assert [e.rank for e in evidence] == [1, 2, 3]

    chunk = evidence[0]
    assert chunk.document_id == "doc-1"
    assert chunk.chunk_id == "doc-1-chunk-001"
    assert chunk.text == "متن یک"

    entity = evidence[1]
    assert entity.document_id == "doc-1"
    assert entity.chunk_id == "doc-1-chunk-001"
    assert entity.metadata["entity_name"] == "تهران"
    assert entity.metadata["entity_type"] == "location"

    relation = evidence[2]
    assert relation.document_id == "doc-2"
    assert relation.metadata["src_id"] == "تهران"
    assert relation.metadata["tgt_id"] == "ایران"
    assert relation.metadata["weight"] == 42.5


def test_evidence_mapping_handles_unknown_source() -> None:
    raw = {"chunks": [{"file_path": "unknown_source", "chunk_id": "c", "content": "x"}]}
    evidence: list[RetrievedEvidence] = []
    assert LightRAGAdapter._evidence_from_raw_data(raw, evidence) is None
    assert len(evidence) == 1
    assert evidence[0].document_id is None


def test_evidence_mapping_rejects_non_dict_containers() -> None:
    # data=None is a legitimate empty payload; a bare non-dict is malformed.
    assert LightRAGAdapter._evidence_from_raw_data(None, []) is None
    assert LightRAGAdapter._evidence_from_raw_data("not-a-dict", []) is not None
    assert LightRAGAdapter._evidence_from_raw_data([1, 2], []) is not None
    # A chunk list that is not a list of dicts is malformed.
    assert LightRAGAdapter._evidence_from_raw_data({"chunks": "garbage"}, []) is not None
    assert LightRAGAdapter._evidence_from_raw_data({"chunks": ["not-a-dict"]}, []) is not None


def test_evidence_mapping_splits_merged_file_paths() -> None:
    raw = {
        "entities": [
            {
                "entity_name": "تهران",
                "entity_type": "location",
                "description": "توضیح",
                "source_id": "doc-3-chunk-000",
                "file_path": "doc-3<SEP>doc-4",
            }
        ],
        "relationships": [
            {
                "src_id": "a",
                "tgt_id": "b",
                "keywords": "k",
                "weight": 1.5,
                "description": "رابطه",
                "source_id": "doc-3-chunk-000",
                "file_path": "doc-3<SEP>doc-4",
            }
        ],
    }
    evidence: list[RetrievedEvidence] = []
    assert LightRAGAdapter._evidence_from_raw_data(raw, evidence) is None
    entity, relation = evidence
    assert entity.document_id == "doc-3"
    assert entity.metadata["source_file_paths"] == ["doc-3", "doc-4"]
    assert relation.document_id == "doc-3"
    assert relation.metadata["source_file_paths"] == ["doc-3", "doc-4"]


def test_evidence_mapping_ignores_empty_segments_of_merged_paths() -> None:
    # Repeated/trailing delimiters must not create bogus blank document IDs.
    raw = {
        "entities": [
            {
                "entity_name": "تهران",
                "entity_type": "location",
                "description": "توضیح",
                "source_id": "doc-3-chunk-000",
                "file_path": "doc-3<SEP><SEP>doc-4",
            }
        ],
        "relationships": [
            {
                "src_id": "a",
                "tgt_id": "b",
                "keywords": "k",
                "weight": 1.5,
                "description": "رابطه",
                "source_id": "doc-3-chunk-000",
                "file_path": "doc-3<SEP>",
            }
        ],
    }
    evidence: list[RetrievedEvidence] = []
    assert LightRAGAdapter._evidence_from_raw_data(raw, evidence) is None
    entity, relation = evidence
    assert entity.document_id == "doc-3"
    assert entity.metadata["source_file_paths"] == ["doc-3", "doc-4"]
    assert relation.document_id == "doc-3"
    assert "source_file_paths" not in relation.metadata
