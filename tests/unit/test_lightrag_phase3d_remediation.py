"""Phase 3D LightRAG remediation regression tests.

Covers the confirmed Phase 3D findings:
  1. variant/mode pinning (query_mode must always be hybrid),
  2. malformed successful raw_data must not crash the harness,
  3. internal LightRAG ImportError must not be mislabelled as missing package,
  4. partially initialized LightRAG resources must be finalized,
  5. '<SEP>' provenance delimiter must not corrupt document IDs,
  6. usage accounting counts failures and is query-local,
  7. health() must detect an unavailable optional dependency.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("lightrag")

from rag_benchmark.adapters.lightrag import LightRAGAdapter  # noqa: E402
from rag_benchmark.adapters.lightrag import adapter as adapter_module  # noqa: E402
from rag_benchmark.adapters.lightrag.config import LightRAGConfig  # noqa: E402
from rag_benchmark.adapters.lightrag.llm import (  # noqa: E402
    QueryUsageContext,
    bind_query_usage_context,
    build_default_llm_func,
    reset_query_usage_context,
)
from rag_benchmark.config.models import GenerationConfig, LLMConfig  # noqa: E402
from rag_benchmark.errors.models import (  # noqa: E402
    BenchmarkErrorType,
    BenchmarkException,
    ConfigurationError,
)
from rag_benchmark.schemas.evidence import EvidenceType, RetrievedEvidence  # noqa: E402
from rag_benchmark.schemas.question import AdapterQueryInput  # noqa: E402
from rag_benchmark.schemas.result import QueryResult, QueryStatus  # noqa: E402
from tests.unit.fakes import FakeEmbeddingProvider  # noqa: E402
from tests.unit.test_lightrag_adapter import (  # noqa: E402
    ANSWER,
    DOC_TWO_TEXT,
    LIGHTRAG_EXTRA,
    ScriptedLLM,
    UniformEmbeddingProvider,
    _index_two_docs,
    make_adapter,
    make_config,
    make_doc,
)


def _query_with_fake_pipeline(
    adapter: LightRAGAdapter,
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, Any],
) -> QueryResult:
    async def fake_aquery_llm(question: str, param: Any = None, **kwargs: Any) -> dict[str, Any]:
        return payload

    monkeypatch.setattr(adapter._rag, "aquery_llm", fake_aquery_llm)
    return adapter.query(AdapterQueryInput(question_id="q-1", question="پرسش"))


def _successful_pipeline_payload(data: Any) -> dict[str, Any]:
    return {
        "status": "success",
        "message": "ok",
        "llm_response": {"content": ANSWER},
        "data": data,
    }


# ---------------------------------------------------------------------------
# Finding 1: variant must always match retrieval mode
# ---------------------------------------------------------------------------


def test_default_query_mode_is_hybrid() -> None:
    assert LightRAGConfig().query_mode == "hybrid"


@pytest.mark.parametrize("mode", ["local", "global", "naive", "mix", "bypass"])
def test_prepare_rejects_non_hybrid_query_mode(tmp_path: Path, mode: str) -> None:
    adapter, _, _ = make_adapter(tmp_path)
    with pytest.raises(ConfigurationError, match="Invalid lightrag configuration"):
        adapter.prepare(make_config({"query_mode": mode}))
    # The pinned configuration is unchanged by a rejected prepare().
    assert adapter._lightrag_config.query_mode == "hybrid"


def test_prepare_accepts_explicit_hybrid_query_mode(tmp_path: Path) -> None:
    adapter, _, _ = make_adapter(tmp_path)
    adapter.prepare(make_config({"query_mode": "hybrid"}))
    assert adapter._lightrag_config.query_mode == "hybrid"


# ---------------------------------------------------------------------------
# Finding 2: malformed successful raw_data must not crash the harness
# ---------------------------------------------------------------------------


def test_nan_relationship_weight_returns_canonical_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, _, _ = make_adapter(
        tmp_path, extra=LIGHTRAG_EXTRA, embedding=UniformEmbeddingProvider()
    )
    _index_two_docs(adapter)
    payload = _successful_pipeline_payload(
        {
            "relationships": [
                {
                    "src_id": "a",
                    "tgt_id": "b",
                    "keywords": "k",
                    "weight": float("nan"),
                    "description": "شرح رابطه",
                    "file_path": "doc-1",
                }
            ]
        }
    )
    result = _query_with_fake_pipeline(adapter, monkeypatch, payload)
    assert result.status is QueryStatus.FAILED
    assert result.answer is None
    assert result.error is not None
    assert result.error.error_type is BenchmarkErrorType.RETRIEVAL_ERROR
    assert "malformed evidence data" in result.error.message


def test_infinite_relationship_weight_returns_canonical_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, _, _ = make_adapter(
        tmp_path, extra=LIGHTRAG_EXTRA, embedding=UniformEmbeddingProvider()
    )
    _index_two_docs(adapter)
    for weight in (float("inf"), float("-inf")):
        payload = _successful_pipeline_payload(
            {
                "relationships": [
                    {
                        "src_id": "a",
                        "tgt_id": "b",
                        "keywords": "k",
                        "weight": weight,
                        "description": "شرح رابطه",
                        "file_path": "doc-1",
                    }
                ]
            }
        )
        result = _query_with_fake_pipeline(adapter, monkeypatch, payload)
        assert result.status is QueryStatus.FAILED
        assert result.answer is None
        assert result.error is not None
        assert result.error.error_type is BenchmarkErrorType.RETRIEVAL_ERROR


def test_entity_non_finite_float_returns_canonical_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, _, _ = make_adapter(
        tmp_path, extra=LIGHTRAG_EXTRA, embedding=UniformEmbeddingProvider()
    )
    _index_two_docs(adapter)
    payload = _successful_pipeline_payload(
        {
            "entities": [
                {
                    "entity_name": "تهران",
                    "entity_type": float("inf"),
                    "description": "شرح",
                    "source_id": "doc-1-chunk-000",
                    "file_path": "doc-1",
                }
            ]
        }
    )
    result = _query_with_fake_pipeline(adapter, monkeypatch, payload)
    assert result.status is QueryStatus.FAILED
    assert result.answer is None
    assert result.error is not None
    assert result.error.error_type is BenchmarkErrorType.RETRIEVAL_ERROR


def test_chunk_non_finite_float_returns_canonical_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, _, _ = make_adapter(
        tmp_path, extra=LIGHTRAG_EXTRA, embedding=UniformEmbeddingProvider()
    )
    _index_two_docs(adapter)
    payload = _successful_pipeline_payload(
        {"chunks": [{"file_path": "doc-1", "chunk_id": float("nan"), "content": "متن"}]}
    )
    result = _query_with_fake_pipeline(adapter, monkeypatch, payload)
    assert result.status is QueryStatus.FAILED
    assert result.answer is None
    assert result.error is not None
    assert result.error.error_type is BenchmarkErrorType.RETRIEVAL_ERROR


def test_malformed_raw_data_container_types_return_canonical_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, _, _ = make_adapter(
        tmp_path, extra=LIGHTRAG_EXTRA, embedding=UniformEmbeddingProvider()
    )
    _index_two_docs(adapter)
    for data in (["not", "a", "dict"], {"chunks": {"not": "a list"}}, {"entities": 42}):
        result = _query_with_fake_pipeline(adapter, monkeypatch, _successful_pipeline_payload(data))
        assert result.status is QueryStatus.FAILED
        assert result.answer is None
        assert result.error is not None
        assert result.error.error_type is BenchmarkErrorType.RETRIEVAL_ERROR


def test_normalization_failure_preserves_prior_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, _, _ = make_adapter(
        tmp_path, extra=LIGHTRAG_EXTRA, embedding=UniformEmbeddingProvider()
    )
    _index_two_docs(adapter)
    payload = _successful_pipeline_payload(
        {
            "chunks": [
                {"file_path": "doc-1", "chunk_id": "doc-1-chunk-000", "content": "متن سالم"}
            ],
            "relationships": [
                {
                    "src_id": "a",
                    "tgt_id": "b",
                    "keywords": "k",
                    "weight": float("nan"),
                    "description": "شرح رابطه",
                    "file_path": "doc-1",
                }
            ],
        }
    )
    result = _query_with_fake_pipeline(adapter, monkeypatch, payload)
    assert result.status is QueryStatus.FAILED
    assert result.answer is None
    # The chunk normalized before the bad relationship must be preserved.
    assert len(result.retrieved_evidence) == 1
    assert result.retrieved_evidence[0].evidence_type is EvidenceType.CHUNK
    assert result.retrieved_evidence[0].document_id == "doc-1"


def test_raw_data_normalization_rejects_non_finite_explicitly() -> None:
    # Unit-level: non-finite numerics are rejected, never stringified.
    evidence: list[RetrievedEvidence] = []
    error = LightRAGAdapter._evidence_from_raw_data(
        {"relationships": [{"weight": float("nan"), "description": "x"}]}, evidence
    )
    assert error is not None
    assert "non-finite" in error
    assert evidence == []

    evidence = []
    error = LightRAGAdapter._evidence_from_raw_data(
        {"relationships": [{"weight": float("-inf"), "description": "x"}]}, evidence
    )
    assert error is not None
    assert "non-finite" in error

    # Finite weights still pass through unchanged.
    evidence = []
    assert (
        LightRAGAdapter._evidence_from_raw_data(
            {
                "relationships": [
                    {
                        "src_id": "a",
                        "tgt_id": "b",
                        "keywords": "k",
                        "weight": 3.5,
                        "description": "x",
                        "file_path": "doc-1",
                    }
                ]
            },
            evidence,
        )
        is None
    )
    assert evidence[0].metadata["weight"] == 3.5


def _assert_canonical_failure(result: QueryResult) -> None:
    assert result.status is QueryStatus.FAILED
    assert result.answer is None
    assert result.error is not None
    assert result.error.error_type is BenchmarkErrorType.RETRIEVAL_ERROR


@pytest.mark.parametrize("bad_value", [{"bad": "object"}, ["unexpected", "list"], True, 42])
def test_entity_name_wrong_type_returns_canonical_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_value: Any
) -> None:
    adapter, _, _ = make_adapter(
        tmp_path, extra=LIGHTRAG_EXTRA, embedding=UniformEmbeddingProvider()
    )
    _index_two_docs(adapter)
    payload = _successful_pipeline_payload(
        {
            "entities": [
                {
                    "entity_name": bad_value,
                    "entity_type": "Other",
                    "description": "شرح",
                    "source_id": "doc-1-chunk-000",
                    "file_path": "doc-1",
                }
            ]
        }
    )
    _assert_canonical_failure(_query_with_fake_pipeline(adapter, monkeypatch, payload))


@pytest.mark.parametrize("bad_value", [["unexpected", "list"], {"bad": "object"}])
def test_chunk_id_wrong_type_returns_canonical_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_value: Any
) -> None:
    adapter, _, _ = make_adapter(
        tmp_path, extra=LIGHTRAG_EXTRA, embedding=UniformEmbeddingProvider()
    )
    _index_two_docs(adapter)
    payload = _successful_pipeline_payload(
        {"chunks": [{"file_path": "doc-1", "chunk_id": bad_value, "content": "متن"}]}
    )
    _assert_canonical_failure(_query_with_fake_pipeline(adapter, monkeypatch, payload))


def test_chunk_content_mapping_returns_canonical_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, _, _ = make_adapter(
        tmp_path, extra=LIGHTRAG_EXTRA, embedding=UniformEmbeddingProvider()
    )
    _index_two_docs(adapter)
    payload = _successful_pipeline_payload(
        {"chunks": [{"file_path": "doc-1", "chunk_id": "c-1", "content": {"bad": "value"}}]}
    )
    _assert_canonical_failure(_query_with_fake_pipeline(adapter, monkeypatch, payload))


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [("src_id", {"bad": 1}), ("src_id", ["x"]), ("tgt_id", True), ("tgt_id", 42)],
)
def test_relationship_endpoint_wrong_type_returns_canonical_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, bad_value: Any
) -> None:
    adapter, _, _ = make_adapter(
        tmp_path, extra=LIGHTRAG_EXTRA, embedding=UniformEmbeddingProvider()
    )
    _index_two_docs(adapter)
    record: dict[str, Any] = {
        "src_id": "a",
        "tgt_id": "b",
        "keywords": "k",
        "weight": 1.5,
        "description": "شرح رابطه",
        "file_path": "doc-1",
    }
    record[field] = bad_value
    payload = _successful_pipeline_payload({"relationships": [record]})
    _assert_canonical_failure(_query_with_fake_pipeline(adapter, monkeypatch, payload))


@pytest.mark.parametrize("bad_value", [True, "1.5"])
def test_relationship_weight_wrong_type_returns_canonical_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_value: Any
) -> None:
    adapter, _, _ = make_adapter(
        tmp_path, extra=LIGHTRAG_EXTRA, embedding=UniformEmbeddingProvider()
    )
    _index_two_docs(adapter)
    payload = _successful_pipeline_payload(
        {
            "relationships": [
                {
                    "src_id": "a",
                    "tgt_id": "b",
                    "keywords": "k",
                    "weight": bad_value,
                    "description": "شرح رابطه",
                    "file_path": "doc-1",
                }
            ]
        }
    )
    _assert_canonical_failure(_query_with_fake_pipeline(adapter, monkeypatch, payload))


def test_entity_file_path_mapping_returns_canonical_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, _, _ = make_adapter(
        tmp_path, extra=LIGHTRAG_EXTRA, embedding=UniformEmbeddingProvider()
    )
    _index_two_docs(adapter)
    payload = _successful_pipeline_payload(
        {
            "entities": [
                {
                    "entity_name": "تهران",
                    "entity_type": "Other",
                    "description": "شرح",
                    "file_path": {"bad": "value"},
                }
            ]
        }
    )
    _assert_canonical_failure(_query_with_fake_pipeline(adapter, monkeypatch, payload))


def test_missing_optional_fields_still_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, _, _ = make_adapter(
        tmp_path, extra=LIGHTRAG_EXTRA, embedding=UniformEmbeddingProvider()
    )
    _index_two_docs(adapter)
    payload = _successful_pipeline_payload(
        {
            "chunks": [{"content": "متن بدون شناسه"}],
            "entities": [{"entity_name": "تهران", "description": "شرح بدون جزئیات"}],
            "relationships": [{"src_id": "a", "tgt_id": "b", "description": "شرح رابطه"}],
        }
    )
    result = _query_with_fake_pipeline(adapter, monkeypatch, payload)
    assert result.status is QueryStatus.SUCCESS
    assert len(result.retrieved_evidence) == 3
    by_type = {e.evidence_type: e for e in result.retrieved_evidence}
    chunk = by_type[EvidenceType.CHUNK]
    entity = by_type[EvidenceType.ENTITY]
    relationship = by_type[EvidenceType.RELATIONSHIP]
    assert chunk.document_id is None and chunk.chunk_id is None
    assert entity.document_id is None and entity.chunk_id is None
    assert entity.metadata["entity_type"] is None
    assert relationship.metadata["weight"] is None
    assert relationship.metadata["keywords"] is None


def test_explicit_none_optional_fields_still_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, _, _ = make_adapter(
        tmp_path, extra=LIGHTRAG_EXTRA, embedding=UniformEmbeddingProvider()
    )
    _index_two_docs(adapter)
    payload = _successful_pipeline_payload(
        {
            "entities": [
                {
                    "entity_name": "تهران",
                    "entity_type": None,
                    "description": "شرح",
                    "source_id": None,
                    "file_path": None,
                }
            ]
        }
    )
    result = _query_with_fake_pipeline(adapter, monkeypatch, payload)
    assert result.status is QueryStatus.SUCCESS
    entity = result.retrieved_evidence[0]
    assert entity.evidence_type is EvidenceType.ENTITY
    assert entity.document_id is None and entity.chunk_id is None
    assert entity.metadata["entity_type"] is None


def test_valid_full_evidence_still_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, _, _ = make_adapter(
        tmp_path, extra=LIGHTRAG_EXTRA, embedding=UniformEmbeddingProvider()
    )
    _index_two_docs(adapter)
    payload = _successful_pipeline_payload(
        {
            "chunks": [{"file_path": "doc-1", "chunk_id": "doc-1-chunk-000", "content": "متن تکه"}],
            "entities": [
                {
                    "entity_name": "تهران",
                    "entity_type": "City",
                    "description": "شرح شهر",
                    "source_id": "doc-1-chunk-000",
                    "file_path": "doc-1",
                }
            ],
            "relationships": [
                {
                    "src_id": "a",
                    "tgt_id": "b",
                    "keywords": "k",
                    "weight": 2.5,
                    "description": "شرح رابطه",
                    "source_id": "doc-1-chunk-000",
                    "file_path": "doc-2",
                }
            ],
        }
    )
    result = _query_with_fake_pipeline(adapter, monkeypatch, payload)
    assert result.status is QueryStatus.SUCCESS
    assert result.answer == ANSWER
    assert [e.rank for e in result.retrieved_evidence] == [1, 2, 3]
    chunk, entity, relationship = result.retrieved_evidence
    assert (chunk.evidence_type, chunk.document_id, chunk.chunk_id) == (
        EvidenceType.CHUNK,
        "doc-1",
        "doc-1-chunk-000",
    )
    assert (entity.evidence_type, entity.document_id, entity.chunk_id) == (
        EvidenceType.ENTITY,
        "doc-1",
        "doc-1-chunk-000",
    )
    assert (entity.metadata["entity_name"], entity.metadata["entity_type"]) == ("تهران", "City")
    assert (
        relationship.evidence_type,
        relationship.document_id,
        relationship.metadata["weight"],
    ) == (EvidenceType.RELATIONSHIP, "doc-2", 2.5)


def test_malformed_entity_after_valid_entity_preserves_prior_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, _, _ = make_adapter(
        tmp_path, extra=LIGHTRAG_EXTRA, embedding=UniformEmbeddingProvider()
    )
    _index_two_docs(adapter)
    payload = _successful_pipeline_payload(
        {
            "entities": [
                {
                    "entity_name": "سالم",
                    "entity_type": "Other",
                    "description": "شرح سالم",
                    "file_path": "doc-1",
                },
                {
                    "entity_name": ["bad"],
                    "entity_type": "Other",
                    "description": "شرح خراب",
                    "file_path": "doc-1",
                },
            ]
        }
    )
    result = _query_with_fake_pipeline(adapter, monkeypatch, payload)
    _assert_canonical_failure(result)
    assert len(result.retrieved_evidence) == 1
    assert result.retrieved_evidence[0].metadata["entity_name"] == "سالم"


# ---------------------------------------------------------------------------
# Finding 3: internal LightRAG ImportError must not be "package missing"
# ---------------------------------------------------------------------------


@pytest.fixture()
def fresh_lightrag_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(adapter_module, "_lightrag_modules", None)


def _patch_lightrag_import(monkeypatch: pytest.MonkeyPatch, error: BaseException) -> None:
    """Make the top-level ``import lightrag`` fail with a specific error.

    Patches ``builtins.__import__`` so the real
    :func:`_import_lightrag_modules` (including its exception handling) runs.
    """
    real_import = __import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "lightrag":
            raise error
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)


def test_missing_top_level_lightrag_raises_clear_dependency_error(
    monkeypatch: pytest.MonkeyPatch, fresh_lightrag_cache: None
) -> None:
    _patch_lightrag_import(
        monkeypatch, ModuleNotFoundError("No module named 'lightrag'", name="lightrag")
    )
    with pytest.raises(ConfigurationError, match="optional 'lightrag' dependency"):
        adapter_module._require_lightrag()
    assert adapter_module._lightrag_modules is None


def test_internal_module_not_found_is_preserved(
    monkeypatch: pytest.MonkeyPatch, fresh_lightrag_cache: None
) -> None:
    internal_error = ModuleNotFoundError("No module named 'nano_vectordb'", name="nano_vectordb")
    _patch_lightrag_import(monkeypatch, internal_error)
    with pytest.raises(ModuleNotFoundError) as exc_info:
        adapter_module._require_lightrag()
    assert exc_info.value is internal_error
    assert not isinstance(exc_info.value, ConfigurationError)


def test_internal_import_error_is_not_mislabelled(
    monkeypatch: pytest.MonkeyPatch, fresh_lightrag_cache: None
) -> None:
    internal_error = ImportError("broken internal shim")
    _patch_lightrag_import(monkeypatch, internal_error)
    with pytest.raises(ImportError) as exc_info:
        adapter_module._require_lightrag()
    assert exc_info.value is internal_error
    assert not isinstance(exc_info.value, ConfigurationError)


def test_module_not_found_with_none_name_is_preserved(
    monkeypatch: pytest.MonkeyPatch, fresh_lightrag_cache: None
) -> None:
    internal_error = ModuleNotFoundError("internal failure")
    assert internal_error.name is None
    _patch_lightrag_import(monkeypatch, internal_error)
    with pytest.raises(ModuleNotFoundError) as exc_info:
        adapter_module._require_lightrag()
    assert exc_info.value is internal_error
    assert not isinstance(exc_info.value, ConfigurationError)
    assert "not installed" not in str(exc_info.value)


def test_installed_lightrag_imports_normally(
    monkeypatch: pytest.MonkeyPatch, fresh_lightrag_cache: None
) -> None:
    modules = adapter_module._require_lightrag()
    assert modules.LightRAG is not None
    assert modules.QueryParam is not None
    assert modules.EmbeddingFunc is not None
    assert modules.always_get_an_event_loop is not None
    assert adapter_module._lightrag_modules is modules


# ---------------------------------------------------------------------------
# Finding 4: partially initialized LightRAG resources must be finalized
# ---------------------------------------------------------------------------


def _patch_failing_lightrag_init(
    monkeypatch: pytest.MonkeyPatch, finalize_calls: list[int]
) -> None:
    modules = adapter_module._require_lightrag()

    async def failing_init(self: Any) -> None:
        raise RuntimeError("init boom")

    async def recording_finalize(self: Any, *args: Any, **kwargs: Any) -> None:
        finalize_calls.append(1)

    monkeypatch.setattr(modules.LightRAG, "initialize_storages", failing_init)
    monkeypatch.setattr(modules.LightRAG, "finalize_storages", recording_finalize)


def test_init_failure_attempts_finalize_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    finalize_calls: list[int] = []
    _patch_failing_lightrag_init(monkeypatch, finalize_calls)
    adapter = LightRAGAdapter(
        embedding_provider=FakeEmbeddingProvider(dimension=8),
        llm_func=ScriptedLLM(),
        working_dir=None,  # adapter-owned working directory
    )
    adapter.activate(make_config(LIGHTRAG_EXTRA))
    with pytest.raises(ConfigurationError, match="Failed to initialize LightRAG"):
        adapter.index([make_doc("doc-1")])
    assert finalize_calls == [1]  # finalize attempted exactly once
    assert adapter._rag is None
    assert adapter._working_dir is None  # owned working dir removed


def test_init_failure_with_failing_finalize_keeps_original_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    modules = adapter_module._require_lightrag()

    async def failing_init(self: Any) -> None:
        raise RuntimeError("init boom")

    async def failing_finalize(self: Any, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("finalize boom")

    monkeypatch.setattr(modules.LightRAG, "initialize_storages", failing_init)
    monkeypatch.setattr(modules.LightRAG, "finalize_storages", failing_finalize)
    adapter, _, _ = make_adapter(tmp_path, extra=LIGHTRAG_EXTRA)

    with pytest.raises(ConfigurationError) as exc_info:
        adapter.index([make_doc("doc-1")])
    # The original initialization error remains primary.
    assert "init boom" in str(exc_info.value)
    assert exc_info.value.__cause__ is not None
    assert "init boom" in str(exc_info.value.__cause__)
    assert adapter._rag is None


def test_cleanup_is_safe_after_failed_init(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    finalize_calls: list[int] = []
    _patch_failing_lightrag_init(monkeypatch, finalize_calls)
    adapter, _, _ = make_adapter(tmp_path, extra=LIGHTRAG_EXTRA)
    with pytest.raises(ConfigurationError):
        adapter.index([make_doc("doc-1")])
    adapter.cleanup()  # must not raise after a failed initialization
    assert adapter._rag is None


def test_successful_init_finalizes_once_on_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    finalize_calls: list[int] = []
    modules = adapter_module._require_lightrag()
    real_finalize = modules.LightRAG.finalize_storages

    async def recording_finalize(self: Any, *args: Any, **kwargs: Any) -> None:
        finalize_calls.append(1)
        await real_finalize(self, *args, **kwargs)

    monkeypatch.setattr(modules.LightRAG, "finalize_storages", recording_finalize)
    adapter, _, _ = make_adapter(tmp_path, extra=LIGHTRAG_EXTRA)
    _index_two_docs(adapter)
    assert adapter._rag is not None
    adapter.cleanup()
    assert finalize_calls == [1]


# ---------------------------------------------------------------------------
# Finding 5: '<SEP>' provenance delimiter must not corrupt document IDs
# ---------------------------------------------------------------------------


def test_document_id_with_reserved_sep_rejected_before_insertion(tmp_path: Path) -> None:
    adapter, _, _ = make_adapter(tmp_path, extra=LIGHTRAG_EXTRA)
    with pytest.raises(BenchmarkException) as exc_info:
        adapter.index([make_doc("good-1"), make_doc("doc<SEP>part")])
    assert exc_info.value.error_type is BenchmarkErrorType.INDEXING_ERROR
    assert "reserved LightRAG provenance delimiter" in str(exc_info.value)
    # Nothing was constructed or inserted for the rejected batch.
    assert adapter._rag is None
    assert adapter._indexed_documents == 0


def test_insert_rejects_reserved_sep_without_touching_existing_index(tmp_path: Path) -> None:
    adapter, _, _ = make_adapter(tmp_path, extra=LIGHTRAG_EXTRA)
    first = adapter.index([make_doc("good-1")])
    assert first.indexed_documents == 1
    with pytest.raises(BenchmarkException) as exc_info:
        adapter.insert([make_doc("bad<SEP>id")])
    assert exc_info.value.error_type is BenchmarkErrorType.INDEXING_ERROR
    # The existing index is untouched.
    result = adapter.query(AdapterQueryInput(question_id="q-1", question="پرسش"))
    assert result.status is QueryStatus.SUCCESS
    assert adapter._indexed_documents == 1


def test_normal_document_ids_still_accepted(tmp_path: Path) -> None:
    # Plain IDs without the reserved delimiter are accepted (distinct texts so
    # LightRAG's content-hash de-duplication does not skip a document).
    adapter, _, _ = make_adapter(tmp_path, extra=LIGHTRAG_EXTRA)
    result = adapter.index(
        [make_doc("doc-001_plain"), make_doc("doc-002.dashed", text=DOC_TWO_TEXT)]
    )
    assert result.indexed_documents == 2


# ---------------------------------------------------------------------------
# Finding 6: usage accounting counts failures and is query-local
# ---------------------------------------------------------------------------


def test_failed_embedding_attempt_is_counted(tmp_path: Path) -> None:
    embedding = FakeEmbeddingProvider(dimension=8)
    adapter, _, _ = make_adapter(
        tmp_path, extra=LIGHTRAG_EXTRA, embedding=embedding, embedding_dimension=8
    )
    _index_two_docs(adapter)
    # Break the provider after indexing: the query's embedding attempt fails.
    embedding._fail_documents = RuntimeError("embed down")
    result = adapter.query(AdapterQueryInput(question_id="q-1", question="پرسش"))
    assert isinstance(result, QueryResult)
    if result.status is QueryStatus.FAILED:
        assert result.answer is None
    # The failed provider attempt was still counted.
    assert result.usage.embedding_calls >= 1


def test_failed_llm_attempt_is_counted(tmp_path: Path) -> None:
    adapter, _, llm = make_adapter(tmp_path, extra=LIGHTRAG_EXTRA)
    _index_two_docs(adapter)
    llm.fail_after = llm.call_count  # the next LLM call (first of the query) fails
    result = adapter.query(AdapterQueryInput(question_id="q-1", question="پرسش"))
    assert result.status is QueryStatus.FAILED
    assert result.error is not None
    assert "scripted LLM failure" in result.error.message
    # The failed LLM attempt was still recorded.
    assert result.usage.llm_calls >= 1


def test_sequential_queries_do_not_contaminate_usage(tmp_path: Path) -> None:
    adapter, _, llm = make_adapter(
        tmp_path, extra=LIGHTRAG_EXTRA, embedding=UniformEmbeddingProvider()
    )
    _index_two_docs(adapter)

    before_q1 = llm.call_count
    first = adapter.query(AdapterQueryInput(question_id="q-1", question="پرسش اول"))
    after_q1 = llm.call_count
    second = adapter.query(AdapterQueryInput(question_id="q-2", question="پرسش دوم"))
    after_q2 = llm.call_count

    assert first.status is QueryStatus.SUCCESS
    assert second.status is QueryStatus.SUCCESS
    assert first.usage.llm_calls == after_q1 - before_q1
    assert second.usage.llm_calls == after_q2 - after_q1


def test_indexing_usage_not_included_in_query(tmp_path: Path) -> None:
    adapter, _, llm = make_adapter(
        tmp_path, extra=LIGHTRAG_EXTRA, embedding=UniformEmbeddingProvider()
    )
    _index_two_docs(adapter)
    index_calls = llm.call_count
    assert index_calls >= 1  # entity extraction LLM calls happened during indexing

    before = llm.call_count
    result = adapter.query(AdapterQueryInput(question_id="q-1", question="پرسش"))
    after = llm.call_count

    assert result.status is QueryStatus.SUCCESS
    # Query accounting only sees the query's own LLM calls, not indexing.
    assert result.usage.llm_calls == after - before
    assert result.usage.llm_calls < index_calls


def test_concurrent_default_llm_contexts_are_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two threads driving LLM calls in parallel never share accounting.

    LightRAG's shared storage keeps a process-global asyncio lock, so two
    LightRAG instances cannot run on two event loops in one process; the
    isolation mechanism under test (per-query ContextVar binding) is exercised
    directly on the default LLM func instead.
    """
    from rag_benchmark.adapters.lightrag import llm as llm_module

    def fake_blocking_chat(*args: Any, **kwargs: Any) -> tuple[str, int, int]:
        time.sleep(0.005)
        return "جواب", 11, 4

    monkeypatch.setattr(llm_module, "_blocking_chat", fake_blocking_chat)
    llm_config = LLMConfig.model_validate(
        {"model": "m", "endpoint": "http://localhost:9999/v1", "timeout_seconds": 5.0}
    )
    generation = GenerationConfig()

    n_calls = 6
    contexts: dict[str, QueryUsageContext] = {}
    errors: list[BaseException] = []
    barrier = threading.Barrier(2)

    def worker(name: str) -> None:
        try:
            func = build_default_llm_func(llm_config=llm_config, generation=generation)
            context = QueryUsageContext()
            token = bind_query_usage_context(context)
            try:
                barrier.wait()

                async def driver() -> None:
                    for _ in range(n_calls):
                        await func("پرسش")

                asyncio.run(driver())
            finally:
                reset_query_usage_context(token)
            contexts[name] = context
        except BaseException as e:  # pragma: no cover - reported by the join asserts
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(f"t{i}",)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    for thread in threads:
        assert not thread.is_alive()
    assert not errors, errors

    assert set(contexts) == {"t0", "t1"}
    for context in contexts.values():
        # Each context sees exactly its own thread's calls and token totals.
        assert context.llm_calls == n_calls
        assert (context.input_tokens, context.output_tokens) == (n_calls * 11, n_calls * 4)


def test_concurrent_external_llm_work_does_not_contaminate_query_usage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real adapter query is immune to concurrent LLM work in another thread.

    The worker thread runs the default LLM func against a separate, thread-local
    accounting context while the main thread executes a real (fake-pipeline)
    LightRAG query. Neither context may observe the other's calls.
    """
    from rag_benchmark.adapters.lightrag import llm as llm_module

    external_calls = 0

    def fake_blocking_chat(*args: Any, **kwargs: Any) -> tuple[str, int, int]:
        nonlocal external_calls
        external_calls += 1
        return "جواب", 11, 4

    monkeypatch.setattr(llm_module, "_blocking_chat", fake_blocking_chat)
    adapter, _, llm = make_adapter(
        tmp_path, extra=LIGHTRAG_EXTRA, embedding=UniformEmbeddingProvider()
    )
    _index_two_docs(adapter)
    llm.sleep_seconds = 0.05  # widen the window so the work genuinely overlaps

    worker_calls = 6
    worker_context = QueryUsageContext()
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            func = build_default_llm_func(
                llm_config=LLMConfig.model_validate(
                    {
                        "model": "m",
                        "endpoint": "http://localhost:9999/v1",
                        "timeout_seconds": 5.0,
                    }
                ),
                generation=GenerationConfig(),
            )
            token = bind_query_usage_context(worker_context)
            try:
                for _ in range(worker_calls):
                    asyncio.run(func("پرسش"))
            finally:
                reset_query_usage_context(token)
        except BaseException as e:  # pragma: no cover - reported by the join asserts
            errors.append(e)

    thread = threading.Thread(target=worker)
    thread.start()
    try:
        before = llm.call_count
        result = adapter.query(AdapterQueryInput(question_id="q-1", question="پرسش"))
        after = llm.call_count
    finally:
        thread.join(timeout=120)
    assert not thread.is_alive()
    assert not errors, errors

    assert result.status is QueryStatus.SUCCESS
    # The query's usage sees exactly this query's own injected-LLM calls.
    assert result.usage.llm_calls == after - before
    # The concurrent external context sees exactly its own calls and tokens.
    assert worker_context.llm_calls == worker_calls
    assert (worker_context.input_tokens, worker_context.output_tokens) == (
        worker_calls * 11,
        worker_calls * 4,
    )
    assert external_calls == worker_calls


def test_token_totals_are_isolated_per_query(monkeypatch: pytest.MonkeyPatch) -> None:
    from rag_benchmark.adapters.lightrag import llm as llm_module

    def fake_blocking_chat(
        endpoint: str,
        model: str,
        api_key: str,
        timeout: float,
        messages: list[dict[str, str]],
        generation: GenerationConfig,
    ) -> tuple[str, int, int]:
        return "جواب", 11, 4

    monkeypatch.setattr(llm_module, "_blocking_chat", fake_blocking_chat)
    llm_config = LLMConfig.model_validate(
        {"model": "m", "endpoint": "http://localhost:9999/v1", "timeout_seconds": 5.0}
    )
    func = build_default_llm_func(llm_config=llm_config, generation=GenerationConfig())

    async def run_bound() -> QueryUsageContext:
        context = QueryUsageContext()
        token = bind_query_usage_context(context)
        try:
            await func("پرسش")
        finally:
            reset_query_usage_context(token)
        return context

    ctx_a = asyncio.run(run_bound())
    ctx_b = asyncio.run(run_bound())
    # Bound contexts are isolated from each other.
    assert (ctx_a.input_tokens, ctx_a.output_tokens) == (11, 4)
    assert (ctx_b.input_tokens, ctx_b.output_tokens) == (11, 4)
    assert ctx_a.llm_calls == 1 and ctx_b.llm_calls == 1
    # A context with no recorded calls stays zero (missing usage is zero,
    # never fabricated).
    empty = QueryUsageContext()
    assert (empty.input_tokens, empty.output_tokens) == (0, 0)
    assert empty.llm_calls == 0


# ---------------------------------------------------------------------------
# Finding 7: health() must detect an unavailable optional dependency
# ---------------------------------------------------------------------------


def test_health_unprepared_reports_not_prepared() -> None:
    adapter = LightRAGAdapter(
        embedding_provider=FakeEmbeddingProvider(dimension=8), llm_func=ScriptedLLM()
    )
    health = adapter.health()
    assert health.healthy is False
    assert "not prepared" in health.message
    assert health.details["prepared"] is False


def test_health_prepared_dependency_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(adapter_module, "_lightrag_dependency_available", lambda: False)
    adapter, _, _ = make_adapter(tmp_path)
    health = adapter.health()
    assert health.healthy is False
    assert "dependency" in health.message
    assert health.details["prepared"] is True
    assert health.details["dependency_available"] is False


def test_health_prepared_available_not_indexed(tmp_path: Path) -> None:
    adapter, _, _ = make_adapter(tmp_path)
    health = adapter.health()
    assert health.healthy is True
    details = health.details
    assert details["prepared"] is True
    assert details["dependency_available"] is True
    assert details["initialized"] is False
    assert details["indexed"] is False
    assert details["query_mode"] == "hybrid"


def test_health_indexed(tmp_path: Path) -> None:
    adapter, _, _ = make_adapter(tmp_path, extra=LIGHTRAG_EXTRA)
    _index_two_docs(adapter)
    health = adapter.health()
    assert health.healthy is True
    details = health.details
    assert details["initialized"] is True
    assert details["indexed"] is True
    assert details["indexed_documents"] == 2


def test_health_after_cleanup_reports_truthful_state(tmp_path: Path) -> None:
    adapter, _, _ = make_adapter(tmp_path, extra=LIGHTRAG_EXTRA)
    _index_two_docs(adapter)
    adapter.cleanup()
    health = adapter.health()
    assert health.healthy is True
    details = health.details
    assert details["initialized"] is False
    assert details["indexed"] is False
    assert details["indexed_documents"] == 0
