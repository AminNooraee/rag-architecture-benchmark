"""Official HKUDS LightRAG adapter (Phase 3D).

Wraps the upstream ``lightrag-hku`` package (1.5.7) behind the canonical
benchmark adapter contract:

    prepare(config) -> index(documents) -> query(AdapterQueryInput)

System under test stays canonical: LightRAG performs its own chunking (F
strategy), graph/knowledge construction, hybrid retrieval, and prompt
construction. The benchmark contributes only:

  - an offline deterministic character tokenizer (no network BPE download),
  - the configured OpenAI-compatible embedding/LLM endpoints,
  - validated LightRAG-specific options carried in
    ``BenchmarkConfig.extra["lightrag"]``.

No reranking, HyDE, query rewriting, multi-query expansion, or hidden
retrieval stages are added. LightRAG's internal LLM response cache is
disabled so every benchmark query performs real provider work.

Lifecycle and loop model
------------------------
LightRAG 1.5.x is asyncio-based and requires an explicit
``initialize_storages()`` before use. This adapter drives LightRAG from the
synchronous benchmark thread using LightRAG's own persistent per-thread
event loop (``always_get_an_event_loop``), so ``insert``/``query`` must be
called from a thread with no running asyncio loop.

Indexing semantics: the first ``index()`` builds and initializes the index.
Each subsequent ``index()`` performs an explicit full rebuild (clearing the
working directory and recreating the LightRAG instance) so repeated calls
never accumulate duplicate documents. ``insert()`` is the supported
incremental-append path on an existing index.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import logging
import math
import shutil
import tempfile
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from pydantic import ValidationError

from rag_benchmark.adapters.base import (
    AdapterCapabilities,
    BaseRAGAdapter,
    HealthCheckResult,
    IndexingResult,
)
from rag_benchmark.adapters.lightrag.config import LightRAGConfig
from rag_benchmark.adapters.lightrag.llm import (
    LightRAGLLMFunc,
    LLMCallRecord,
    QueryUsageContext,
    bind_query_usage_context,
    build_default_llm_func,
    record_embedding_call,
    record_llm_call,
    reset_query_usage_context,
)
from rag_benchmark.adapters.lightrag.tokenizer import build_offline_character_tokenizer
from rag_benchmark.adapters.traditional.providers import EmbeddingProvider
from rag_benchmark.config.models import (
    BenchmarkConfig,
    EmbeddingConfig,
    GenerationConfig,
    LLMConfig,
)
from rag_benchmark.errors.models import (
    BenchmarkErrorType,
    BenchmarkException,
    ConfigurationError,
)
from rag_benchmark.schemas.document import BenchmarkDocument
from rag_benchmark.schemas.evidence import EvidenceType, RetrievedEvidence
from rag_benchmark.schemas.question import AdapterQueryInput
from rag_benchmark.schemas.result import QueryResult, QueryStatus, QueryTiming, QueryUsage

_SYSTEM_NAME = "lightrag"
_VARIANT_NAME = "hybrid_no_rerank"
# "baseline" is the schema-wide default label meaning "no explicit variant requested".
_ACCEPTED_CONFIG_VARIANTS = ("", "baseline", _VARIANT_NAME)

_WORKING_DIR_PREFIX = "rag_benchmark_lightrag_"

_lightrag_modules: SimpleNamespace | None = None


def _import_lightrag_modules() -> SimpleNamespace:
    """Import the LightRAG symbols used by the adapter.

    Exception semantics:
      - positive evidence that the top-level ``lightrag`` package is missing
        (``ModuleNotFoundError`` with ``name`` exactly ``"lightrag"``) -> a
        clear optional-dependency :class:`ConfigurationError`;
      - anything else (a missing submodule or dependency, or an
        import-runtime failure with an unpopulated ``name``) -> the real
        import failure is re-raised untouched. ``name is None`` is never
        interpreted as the top-level package being absent.
    """
    try:
        from lightrag import LightRAG, QueryParam  # type: ignore[import-untyped]
        from lightrag.utils import (  # type: ignore[import-untyped]
            EmbeddingFunc,
            always_get_an_event_loop,
        )
    except ModuleNotFoundError as e:
        if e.name == "lightrag":
            raise ConfigurationError(
                "The optional 'lightrag' dependency is not installed. "
                "Install it with `uv sync --extra lightrag`."
            ) from e
        raise

    return SimpleNamespace(
        LightRAG=LightRAG,
        QueryParam=QueryParam,
        EmbeddingFunc=EmbeddingFunc,
        always_get_an_event_loop=always_get_an_event_loop,
    )


def _require_lightrag() -> SimpleNamespace:
    """Import the optional LightRAG dependency once, with a clear failure."""
    global _lightrag_modules
    if _lightrag_modules is not None:
        return _lightrag_modules
    modules = _import_lightrag_modules()
    # LightRAG logs very verbosely at INFO; the benchmark wants quiet runs.
    for logger_name in ("lightrag", "nano-vectordb"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    _lightrag_modules = modules
    return modules


def _lightrag_dependency_available() -> bool:
    """Side-effect-free availability probe for the optional LightRAG dependency.

    Never initializes LightRAG, never calls a provider, and never mutates
    adapter state; returns False for any import failure (missing package or
    broken internal dependency).
    """
    try:
        import lightrag  # noqa: F401
        from lightrag import LightRAG, QueryParam  # noqa: F401
        from lightrag.utils import (  # noqa: F401
            EmbeddingFunc,
            always_get_an_event_loop,
        )
    except Exception:
        return False
    return True


def _lightrag_version() -> str:
    try:
        return importlib.metadata.version("lightrag-hku")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


class LightRAGAdapter(BaseRAGAdapter):
    """Official HKUDS LightRAG benchmark adapter (variant: hybrid_no_rerank).

    Lifecycle: activate(config) -> index(documents) -> query(AdapterQueryInput).
    Repeated index() calls perform an explicit full rebuild. cleanup()
    finalizes storages, clears index state, and (for adapter-owned working
    directories) removes them; the adapter stays re-indexable.
    """

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        llm_func: LightRAGLLMFunc | None = None,
        working_dir: str | Path | None = None,
        lightrag_config: LightRAGConfig | None = None,
    ) -> None:
        self._embedding_provider = embedding_provider
        # Kept separately so a prepare() without a "lightrag" extra key never
        # inherits a previous explicit override.
        self._constructor_lightrag_config: LightRAGConfig = (
            lightrag_config if lightrag_config is not None else LightRAGConfig()
        )
        self._lightrag_config: LightRAGConfig = self._constructor_lightrag_config
        # Optional caller-supplied LightRAG llm_model_func (async callable);
        # when None the adapter wires the configured OpenAI-compatible
        # endpoint directly with LightRAG's own prompts.
        self._constructor_llm_func = llm_func
        self._working_dir = Path(working_dir) if working_dir is not None else None
        self._owns_working_dir = working_dir is None
        self._retrieval: Any = None
        self._generation: GenerationConfig | None = None
        self._embedding_config: EmbeddingConfig | None = None
        self._llm_config: LLMConfig | None = None
        self._prepared = False
        self._rag: Any = None
        self._llm_func: LightRAGLLMFunc | None = None
        self._embedding_dimension: int | None = None
        self._indexed_documents = 0
        self._indexed_chunks = 0

    @property
    def system_name(self) -> str:
        return _SYSTEM_NAME

    @property
    def variant_name(self) -> str:
        return _VARIANT_NAME

    @property
    def capabilities(self) -> AdapterCapabilities:
        # LightRAG appends documents to an existing graph/index and persists
        # its JsonKV/NetworkX/NanoVectorDB storages to the working directory.
        # In-place update and delete by benchmark document ID are not
        # declared: the benchmark has no canonical semantics for them.
        return AdapterCapabilities(
            supports_incremental_insert=True,
            supports_persistence=True,
            extra_capabilities={"graph_knowledge_base": True},
        )

    def prepare(self, config: BenchmarkConfig) -> None:
        if config.system != _SYSTEM_NAME:
            raise ConfigurationError(
                f"config.system '{config.system}' does not match adapter system '{_SYSTEM_NAME}'"
            )
        if config.variant not in _ACCEPTED_CONFIG_VARIANTS:
            raise ConfigurationError(
                f"config.variant '{config.variant}' is not compatible with adapter "
                f"variant '{_VARIANT_NAME}'"
            )
        if "lightrag" in config.extra:
            raw = config.extra["lightrag"]
            if not isinstance(raw, dict):
                raise ConfigurationError("config.extra['lightrag'] must be a mapping when present")
            try:
                self._lightrag_config = LightRAGConfig.model_validate(dict(raw))
            except ValidationError as e:
                raise ConfigurationError(f"Invalid lightrag configuration: {e}") from e
        else:
            self._lightrag_config = self._constructor_lightrag_config
        self._retrieval = config.retrieval.model_copy(deep=True)
        self._generation = config.generation.model_copy(deep=True)
        self._embedding_config = config.embedding.model_copy(deep=True)
        self._llm_config = config.llm.model_copy(deep=True)
        self._prepared = True

    def index(self, documents: Sequence[BenchmarkDocument]) -> IndexingResult:
        if not self._prepared:
            raise ConfigurationError(
                "LightRAGAdapter must be prepared (via activate/prepare) before index()"
            )
        docs = list(documents)
        started = time.perf_counter()
        rebuild = self._rag is not None
        if rebuild:
            # Explicit full rebuild: identical semantics to the traditional
            # baseline (no duplicate accumulation across index() calls).
            self._teardown_rag()
            if self._owns_working_dir and self._working_dir is not None:
                shutil.rmtree(self._working_dir, ignore_errors=True)
                self._working_dir = None
            self._indexed_documents = 0
            self._indexed_chunks = 0
        # Validate the batch (duplicate / reserved delimiter IDs) before any
        # LightRAG construction or insertion: a rejected batch must never
        # touch the index.
        ids, texts, seen = self._collect_batch(docs)
        self._ensure_rag(probe_text=self._first_nonempty_text(docs))
        assert self._rag is not None
        indexed_chunks = 0
        if texts:
            try:
                self._rag.insert(texts, ids=ids, file_paths=ids)
            except Exception as e:
                raise BenchmarkException(
                    f"LightRAG indexing failed: {e}",
                    error_type=BenchmarkErrorType.INDEXING_ERROR,
                ) from e
            failures, indexed_chunks = self._run_async(self._check_indexed_documents(ids))
            if failures:
                detail = "; ".join(f"{doc_id}: {reason}" for doc_id, reason in failures.items())
                raise BenchmarkException(
                    f"LightRAG document processing failed for {len(failures)} "
                    f"document(s): {detail}",
                    error_type=BenchmarkErrorType.INDEXING_ERROR,
                )
        self._indexed_documents += len(docs)
        self._indexed_chunks = indexed_chunks
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return IndexingResult(
            indexed_documents=len(docs),
            indexed_chunks=indexed_chunks,
            elapsed_ms=elapsed_ms,
            metadata={
                "chunk_token_size": self._lightrag_config.chunk_token_size,
                "chunk_overlap_token_size": self._lightrag_config.chunk_overlap_token_size,
                "embedding_dimension": self._embedding_dimension,
                "storage": "json_kv+networkx+nano_vector_db",
            },
        )

    def insert(self, documents: Sequence[BenchmarkDocument]) -> IndexingResult:
        """Incrementally append documents to the existing LightRAG index."""
        if not self._prepared:
            raise ConfigurationError(
                "LightRAGAdapter must be prepared (via activate/prepare) before insert()"
            )
        if self._rag is None:
            raise ConfigurationError(
                "LightRAGAdapter has no existing index; call index() before insert()"
            )
        docs = list(documents)
        started = time.perf_counter()
        ids, texts, seen = self._collect_batch(docs)
        indexed_chunks = 0
        if texts:
            try:
                self._rag.insert(texts, ids=ids, file_paths=ids)
            except Exception as e:
                raise BenchmarkException(
                    f"LightRAG incremental insert failed: {e}",
                    error_type=BenchmarkErrorType.INDEXING_ERROR,
                ) from e
            failures, indexed_chunks = self._run_async(self._check_indexed_documents(ids))
            if failures:
                detail = "; ".join(f"{doc_id}: {reason}" for doc_id, reason in failures.items())
                raise BenchmarkException(
                    f"LightRAG document processing failed for {len(failures)} "
                    f"document(s): {detail}",
                    error_type=BenchmarkErrorType.INDEXING_ERROR,
                )
        self._indexed_documents += len(docs)
        self._indexed_chunks += indexed_chunks
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return IndexingResult(
            indexed_documents=len(docs),
            indexed_chunks=indexed_chunks,
            elapsed_ms=elapsed_ms,
            metadata={
                "incremental": True,
                "embedding_dimension": self._embedding_dimension,
            },
        )

    # ------------------------------------------------------------------
    # LightRAG instance construction
    # ------------------------------------------------------------------

    def _collect_batch(
        self, docs: list[BenchmarkDocument]
    ) -> tuple[list[str], list[str], set[str]]:
        ids: list[str] = []
        texts: list[str] = []
        seen: set[str] = set()
        for doc in docs:
            if _GRAPH_FIELD_SEP in doc.document_id:
                raise BenchmarkException(
                    "Benchmark document_id contains reserved LightRAG provenance "
                    f"delimiter '<SEP>': '{doc.document_id}'",
                    error_type=BenchmarkErrorType.INDEXING_ERROR,
                )
            if doc.document_id in seen:
                raise ConfigurationError(
                    f"duplicate document_id '{doc.document_id}' in adapter batch"
                )
            seen.add(doc.document_id)
            ids.append(doc.document_id)
            texts.append(doc.text)
        return ids, texts, seen

    @staticmethod
    def _first_nonempty_text(docs: Sequence[BenchmarkDocument]) -> str:
        for doc in docs:
            if doc.text.strip():
                return doc.text
        return ""

    def _ensure_rag(self, probe_text: str) -> None:
        if self._rag is not None:
            return
        modules = _require_lightrag()
        embedding_config = self._embedding_config
        llm_config = self._llm_config
        generation = self._generation
        if embedding_config is None or llm_config is None or generation is None:
            raise ConfigurationError(
                "Adapter internal state is inconsistent: missing benchmark config"
            )
        if self._working_dir is None:
            self._working_dir = Path(tempfile.mkdtemp(prefix=_WORKING_DIR_PREFIX))
        else:
            self._working_dir.mkdir(parents=True, exist_ok=True)

        dimension = embedding_config.dimension
        if dimension is None:
            if not probe_text.strip():
                raise ConfigurationError(
                    "Cannot determine the embedding dimension: configure "
                    "embedding.dimension or index documents with non-empty text."
                )
            try:
                probe_vector = self._embedding_provider.embed_documents([probe_text])[0]
                dimension = self._validate_vector_shape(probe_vector, "probe")
            except Exception as e:
                raise BenchmarkException(
                    f"Embedding provider failed while probing the vector dimension: {e}",
                    error_type=BenchmarkErrorType.INDEXING_ERROR,
                ) from e

        provider = self._embedding_provider

        async def _embed_texts(texts: list[str]) -> Any:
            loop = asyncio.get_running_loop()
            # Count the embedding attempt BEFORE invoking the provider so a
            # failed provider call is still accounted (matches failed-LLM
            # attempt semantics). Outside a bound query context (indexing)
            # the call is not attributed to any query.
            record_embedding_call()
            vectors = await loop.run_in_executor(None, provider.embed_documents, list(texts))
            validated: list[list[float]] = []
            for position, vector in enumerate(vectors):
                validated.append(self._validate_vector(vector, f"texts[{position}]"))
            if not validated:
                raise ValueError("embedding provider returned an empty vector batch")
            reference_dim = len(validated[0])
            for position, vector in enumerate(validated[1:]):
                if len(vector) != reference_dim:
                    raise ValueError(
                        f"embedding dimension mismatch within batch: "
                        f"{len(vector)} != {reference_dim} at texts[{position + 1}]"
                    )
            import numpy as np  # local: keeps the module importable without the extra

            return np.asarray(validated, dtype=np.float32)

        embedding_func = modules.EmbeddingFunc(embedding_dim=dimension, func=_embed_texts)

        if self._constructor_llm_func is not None:
            self._llm_func = self._wrap_injected_llm(self._constructor_llm_func)
        else:
            self._llm_func = build_default_llm_func(llm_config=llm_config, generation=generation)

        ltr = self._lightrag_config
        rag_kwargs: dict[str, Any] = dict(
            working_dir=str(self._working_dir),
            workspace=f"rag-bench-{uuid.uuid4().hex[:16]}",
            tokenizer=build_offline_character_tokenizer(),
            chunk_token_size=ltr.chunk_token_size,
            chunk_overlap_token_size=ltr.chunk_overlap_token_size,
            top_k=ltr.top_k,
            chunk_top_k=ltr.chunk_top_k,
            embedding_func=embedding_func,
            embedding_batch_num=ltr.embedding_batch_num,
            default_embedding_timeout=int(ltr.embedding_timeout_seconds),
            llm_model_func=self._llm_func,
            llm_model_name=llm_config.model.strip() or "benchmark-llm",
            llm_model_max_async=ltr.llm_model_max_async,
            default_llm_timeout=int(ltr.llm_timeout_seconds),
            enable_llm_cache=False,
            enable_llm_cache_for_entity_extract=False,
            user_prompt_prefix="",
        )
        if ltr.max_entity_tokens is not None:
            rag_kwargs["max_entity_tokens"] = ltr.max_entity_tokens
        if ltr.max_relation_tokens is not None:
            rag_kwargs["max_relation_tokens"] = ltr.max_relation_tokens
        if ltr.max_total_tokens is not None:
            rag_kwargs["max_total_tokens"] = ltr.max_total_tokens
        rag: Any = None
        try:
            rag = modules.LightRAG(**rag_kwargs)
            self._run_async(rag.initialize_storages())
        except ConfigurationError:
            self._discard_failed_rag(rag)
            raise
        except Exception as e:
            self._discard_failed_rag(rag)
            raise ConfigurationError(f"Failed to initialize LightRAG instance: {e}") from e
        self._rag = rag
        self._embedding_dimension = dimension

    def _discard_failed_rag(self, rag: Any) -> None:
        """Best-effort finalization of a partially initialized LightRAG instance.

        Never hides the original initialization failure (finalize errors are
        swallowed), never assigns the instance to ``self._rag``, and removes
        an adapter-owned working directory so no resources are leaked.
        """
        self._best_effort_finalize(rag)
        if self._owns_working_dir and self._working_dir is not None:
            shutil.rmtree(self._working_dir, ignore_errors=True)
            self._working_dir = None

    def _best_effort_finalize(self, rag: Any) -> None:
        """Finalize storages of a constructed LightRAG instance, tolerating failure."""
        if rag is None:
            return
        try:
            self._run_async(rag.finalize_storages())
        except Exception:
            pass  # best effort; a finalize failure must not hide the original error

    def _wrap_injected_llm(self, base: LightRAGLLMFunc) -> LightRAGLLMFunc:
        async def _wrapped(
            prompt: str,
            system_prompt: str | None = None,
            history_messages: Any = None,
            **kwargs: Any,
        ) -> str:
            record = LLMCallRecord(start=time.perf_counter())
            record_llm_call(record)
            try:
                # LightRAG's llm_model_func contract returns the raw answer string.
                return cast(
                    str,
                    await base(
                        prompt,
                        system_prompt=system_prompt,
                        history_messages=history_messages,
                        **kwargs,
                    ),
                )
            finally:
                record.end = time.perf_counter()

        return _wrapped

    @staticmethod
    def _validate_vector(vector: Any, label: str) -> list[float]:
        if not isinstance(vector, (list, tuple)) or not vector:
            raise ValueError(f"embedding {label} must be a non-empty sequence of numbers")
        result: list[float] = []
        for position, value in enumerate(vector):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"embedding {label}[{position}] must be numeric")
            component = float(value)
            if not math.isfinite(component):
                raise ValueError(f"embedding {label}[{position}] must be finite")
            result.append(component)
        return result

    def _validate_vector_shape(self, vector: Any, label: str) -> int:
        return len(self._validate_vector(vector, label))

    # ------------------------------------------------------------------
    # Event-loop bridging
    # ------------------------------------------------------------------

    def _run_async(self, coro: Any) -> Any:
        """Drive a coroutine to completion on LightRAG's persistent thread loop.

        Reuses ``always_get_an_event_loop`` so the loop matches the one the
        storages were initialized on (cross-loop lock binding is fatal).
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            coro.close()  # avoid "coroutine was never awaited" warnings
            raise ConfigurationError(
                "LightRAGAdapter must be driven from a synchronous context; it is "
                "not safe to call from within a running asyncio event loop."
            )
        modules = _require_lightrag()
        return modules.always_get_an_event_loop().run_until_complete(coro)

    def _teardown_rag(self) -> None:
        rag = self._rag
        if rag is None:
            return
        try:
            self._run_async(rag.finalize_storages())
        except Exception:
            pass  # best effort; state is dropped below regardless
        self._rag = None
        self._llm_func = None

    async def _check_indexed_documents(self, ids: list[str]) -> tuple[dict[str, str], int]:
        """Verify every document reached PROCESSED status; sum chunk counts."""
        assert self._rag is not None
        failures: dict[str, str] = {}
        total_chunks = 0
        for doc_id in ids:
            record = await self._rag.doc_status.get_by_id(doc_id)
            if not isinstance(record, dict):
                failures[doc_id] = "missing from doc_status after insert"
                continue
            status = record.get("status")
            status_value = getattr(status, "value", status)
            if status_value != "processed":
                failures[doc_id] = f"status={status_value} error={record.get('error_msg') or 'n/a'}"
                continue
            chunks_count = record.get("chunks_count")
            if isinstance(chunks_count, bool) or not isinstance(chunks_count, int):
                failures[doc_id] = f"invalid chunks_count={chunks_count!r}"
                continue
            total_chunks += chunks_count
        return failures, total_chunks

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, query_input: AdapterQueryInput) -> QueryResult:
        question_id = query_input.question_id
        if not self._prepared:
            return QueryResult.create_failure(
                question_id=question_id,
                system=self.system_name,
                variant=self.variant_name,
                error_message="Adapter is not prepared; call activate(config) before query().",
                error_type=BenchmarkErrorType.CONFIGURATION_ERROR,
            )
        if self._rag is None or self._indexed_documents == 0:
            return QueryResult.create_failure(
                question_id=question_id,
                system=self.system_name,
                variant=self.variant_name,
                error_message="Adapter has no indexed documents; call index() before query().",
                error_type=BenchmarkErrorType.CONFIGURATION_ERROR,
            )
        ltr = self._lightrag_config
        modules = _require_lightrag()
        param_kwargs: dict[str, Any] = dict(
            mode=ltr.query_mode,
            top_k=ltr.top_k,
            chunk_top_k=ltr.chunk_top_k,
            enable_rerank=False,
            stream=False,
        )
        if ltr.max_entity_tokens is not None:
            param_kwargs["max_entity_tokens"] = ltr.max_entity_tokens
        if ltr.max_relation_tokens is not None:
            param_kwargs["max_relation_tokens"] = ltr.max_relation_tokens
        if ltr.max_total_tokens is not None:
            param_kwargs["max_total_tokens"] = ltr.max_total_tokens
        param = modules.QueryParam(**param_kwargs)

        total_start = time.perf_counter()
        # Query-local accounting: provider calls (including failed attempts)
        # are recorded into this context and only this context is read below,
        # so sequential or interleaved queries can never share accounting.
        usage = QueryUsageContext()
        token = bind_query_usage_context(usage)
        try:
            try:
                full = self._run_async(
                    asyncio.wait_for(
                        self._rag.aquery_llm(query_input.question, param=param),
                        timeout=ltr.query_timeout_seconds,
                    )
                )
            except TimeoutError:
                total_end = time.perf_counter()
                return QueryResult.create_failure(
                    question_id=question_id,
                    system=self.system_name,
                    variant=self.variant_name,
                    error_message=(
                        f"Query exceeded the wall-clock budget of "
                        f"{ltr.query_timeout_seconds} seconds."
                    ),
                    error_type=BenchmarkErrorType.TIMEOUT,
                    status=QueryStatus.TIMEOUT,
                    usage=self._usage_from_context(usage),
                    timing=self._query_timing(usage, total_start, total_end),
                    metadata={"stage": "query", "query_mode": ltr.query_mode},
                )
            except Exception as e:
                total_end = time.perf_counter()
                return QueryResult.create_failure(
                    question_id=question_id,
                    system=self.system_name,
                    variant=self.variant_name,
                    error_message=f"LightRAG query failed: {e}",
                    error_type=BenchmarkErrorType.RETRIEVAL_ERROR,
                    usage=self._usage_from_context(usage),
                    timing=self._query_timing(usage, total_start, total_end),
                    metadata={"stage": "query", "query_mode": ltr.query_mode},
                )

            if not isinstance(full, dict) or full.get("status") != "success":
                message = "LightRAG query pipeline reported a failure."
                if isinstance(full, dict) and isinstance(full.get("message"), str):
                    message = full["message"]
                total_end = time.perf_counter()
                return QueryResult.create_failure(
                    question_id=question_id,
                    system=self.system_name,
                    variant=self.variant_name,
                    error_message=f"LightRAG query pipeline failed: {message}",
                    error_type=BenchmarkErrorType.RETRIEVAL_ERROR,
                    usage=self._usage_from_context(usage),
                    timing=self._query_timing(usage, total_start, total_end),
                    metadata={"stage": "query", "query_mode": ltr.query_mode},
                )

            llm_response = full.get("llm_response")
            answer = llm_response.get("content") if isinstance(llm_response, dict) else None
            if not isinstance(answer, str) or not answer.strip():
                total_end = time.perf_counter()
                return QueryResult.create_failure(
                    question_id=question_id,
                    system=self.system_name,
                    variant=self.variant_name,
                    error_message="LightRAG returned an empty answer.",
                    error_type=BenchmarkErrorType.GENERATION_ERROR,
                    usage=self._usage_from_context(usage),
                    timing=self._query_timing(usage, total_start, total_end),
                    metadata={"stage": "generation", "query_mode": ltr.query_mode},
                )

            total_end = time.perf_counter()
            # Normalization of a successful LightRAG result is part of the
            # protected query execution: a malformed raw payload must become
            # a canonical failure, never an uncaught SDK/schema exception.
            evidence: list[RetrievedEvidence] = []
            normalization_error = self._evidence_from_raw_data(full.get("data"), evidence)
            if normalization_error is not None:
                return QueryResult.create_failure(
                    question_id=question_id,
                    system=self.system_name,
                    variant=self.variant_name,
                    error_message=(
                        f"LightRAG returned malformed evidence data: {normalization_error}"
                    ),
                    error_type=BenchmarkErrorType.RETRIEVAL_ERROR,
                    retrieved_evidence=evidence,
                    usage=self._usage_from_context(usage),
                    timing=self._query_timing(usage, total_start, total_end),
                    metadata={"stage": "normalization", "query_mode": ltr.query_mode},
                )
            return QueryResult.create_success(
                question_id=question_id,
                system=self.system_name,
                variant=self.variant_name,
                answer=answer,
                retrieved_evidence=evidence,
                usage=self._usage_from_context(usage),
                timing=self._query_timing(usage, total_start, total_end),
                metadata={
                    "query_mode": ltr.query_mode,
                    "retrieved_chunks": sum(
                        1 for e in evidence if e.evidence_type is EvidenceType.CHUNK
                    ),
                    "retrieved_entities": sum(
                        1 for e in evidence if e.evidence_type is EvidenceType.ENTITY
                    ),
                    "retrieved_relationships": sum(
                        1 for e in evidence if e.evidence_type is EvidenceType.RELATIONSHIP
                    ),
                },
            )
        finally:
            reset_query_usage_context(token)

    @staticmethod
    def _query_timing(
        usage: QueryUsageContext, total_start: float, total_end: float
    ) -> QueryTiming:
        """Split wall-clock time between the fused retrieval+generation stages.

        LightRAG interleaves retrieval and generation inside one pipeline
        call. generation_ms is attributed to the final answer LLM call (the
        last recorded LLM call of the query; earlier LLM calls belong to
        keyword extraction, which is part of LightRAG's retrieval phase).
        retrieval_ms is the remaining wall-clock time.
        """
        total_ms = max((total_end - total_start) * 1000.0, 0.0)
        generation_ms = usage.last_llm_elapsed_ms
        retrieval_ms = max(total_ms - generation_ms, 0.0)
        return QueryTiming(
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            total_ms=total_ms,
        )

    @staticmethod
    def _usage_from_context(usage: QueryUsageContext) -> QueryUsage:
        """Build the canonical QueryUsage from this query's own context only."""
        return QueryUsage(
            llm_calls=usage.llm_calls,
            embedding_calls=usage.embedding_calls,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        )

    @staticmethod
    def _evidence_from_raw_data(data: Any, evidence: list[RetrievedEvidence]) -> str | None:
        """Map LightRAG's raw query data to canonical RetrievedEvidence.

        Appends normalized items to ``evidence`` in retrieval order and
        returns ``None`` on success. If the raw payload is malformed (bad
        container types, present fields of the wrong type, non-finite
        numerics) a human-readable error message is returned instead; items
        already normalized before the failure remain in ``evidence`` so the
        caller can report a canonical failure with the partial provenance.
        """
        try:
            LightRAGAdapter._normalize_evidence(data, evidence)
        except Exception as e:
            return str(e) or type(e).__name__
        return None

    @staticmethod
    def _normalize_evidence(data: Any, evidence: list[RetrievedEvidence]) -> None:
        """Strict normalization; raises on any malformed raw_data structure.

        Chunks carry the benchmark document id in ``file_path`` (set at insert
        time), the LightRAG chunk id in ``chunk_id``, and the text in
        ``content``. Entities/relationships carry their source document in
        ``file_path``; their LLM-generated description is the evidence text.
        """
        if data is None:
            return
        if not isinstance(data, dict):
            raise ValueError(f"raw data container must be a dict, got {type(data).__name__}")
        for key in ("chunks", "entities", "relationships"):
            items = data.get(key)
            if items is None:
                continue
            if not isinstance(items, list):
                raise ValueError(f"raw data '{key}' must be a list, got {type(items).__name__}")
        rank = 1
        chunks = data.get("chunks")
        if isinstance(chunks, list):
            for chunk in chunks:
                if not isinstance(chunk, dict):
                    raise ValueError(f"raw chunk item must be a dict, got {type(chunk).__name__}")
                # Every consumed field is validated before the emit/skip
                # decision: a present, wrong-typed field is malformed
                # upstream raw_data even if the item would be skipped.
                content = _evidence_text(chunk.get("content"), "chunk.content")
                paths = _file_paths_from_raw(
                    _string_or_none(chunk.get("file_path"), "chunk.file_path")
                )
                chunk_id = _string_or_none(chunk.get("chunk_id"), "chunk.chunk_id")
                if content is None or not content.strip():
                    continue
                chunk_metadata: dict[str, Any] = {}
                if len(paths) > 1:
                    chunk_metadata["source_file_paths"] = paths
                evidence.append(
                    RetrievedEvidence(
                        document_id=paths[0] if paths else None,
                        chunk_id=chunk_id,
                        text=content,
                        rank=rank,
                        evidence_type=EvidenceType.CHUNK,
                        metadata=chunk_metadata,
                    )
                )
                rank += 1
        entities = data.get("entities")
        if isinstance(entities, list):
            for entity in entities:
                if not isinstance(entity, dict):
                    raise ValueError(f"raw entity item must be a dict, got {type(entity).__name__}")
                description = _evidence_text(entity.get("description"), "entity.description")
                paths = _file_paths_from_raw(
                    _string_or_none(entity.get("file_path"), "entity.file_path")
                )
                entity_name = _string_or_none(entity.get("entity_name"), "entity.entity_name")
                entity_type = _string_or_none(entity.get("entity_type"), "entity.entity_type")
                source_id = _string_or_none(entity.get("source_id"), "entity.source_id")
                if description is None or not description.strip():
                    continue
                entity_metadata: dict[str, Any] = {
                    "entity_name": entity_name,
                    "entity_type": entity_type,
                }
                if len(paths) > 1:
                    entity_metadata["source_file_paths"] = paths
                evidence.append(
                    RetrievedEvidence(
                        document_id=paths[0] if paths else None,
                        chunk_id=source_id,
                        text=description,
                        rank=rank,
                        evidence_type=EvidenceType.ENTITY,
                        metadata=entity_metadata,
                    )
                )
                rank += 1
        relationships = data.get("relationships")
        if isinstance(relationships, list):
            for relation in relationships:
                if not isinstance(relation, dict):
                    raise ValueError(
                        f"raw relationship item must be a dict, got {type(relation).__name__}"
                    )
                description = _evidence_text(
                    relation.get("description"), "relationship.description"
                )
                weight = _finite_evidence_number(relation.get("weight"), "relationship.weight")
                paths = _file_paths_from_raw(
                    _string_or_none(relation.get("file_path"), "relationship.file_path")
                )
                src_id = _string_or_none(relation.get("src_id"), "relationship.src_id")
                tgt_id = _string_or_none(relation.get("tgt_id"), "relationship.tgt_id")
                keywords = _string_or_none(relation.get("keywords"), "relationship.keywords")
                source_id = _string_or_none(relation.get("source_id"), "relationship.source_id")
                if description is None or not description.strip():
                    continue
                relation_metadata: dict[str, Any] = {
                    "src_id": src_id,
                    "tgt_id": tgt_id,
                    "keywords": keywords,
                    "weight": weight,
                }
                if len(paths) > 1:
                    relation_metadata["source_file_paths"] = paths
                evidence.append(
                    RetrievedEvidence(
                        document_id=paths[0] if paths else None,
                        chunk_id=source_id,
                        text=description,
                        rank=rank,
                        evidence_type=EvidenceType.RELATIONSHIP,
                        metadata=relation_metadata,
                    )
                )
                rank += 1

    # ------------------------------------------------------------------
    # Health / cleanup
    # ------------------------------------------------------------------

    def health(self) -> HealthCheckResult:
        if not self._prepared:
            return HealthCheckResult(
                healthy=False,
                system=self.system_name,
                message="Adapter is not prepared.",
                details={
                    "prepared": False,
                    "dependency_available": _lightrag_dependency_available(),
                },
            )
        dependency_available = _lightrag_dependency_available()
        initialized = self._rag is not None
        indexed = self._indexed_documents > 0
        details = {
            "prepared": True,
            "dependency_available": dependency_available,
            "initialized": initialized,
            "indexed": indexed,
            "indexed_documents": self._indexed_documents,
            "indexed_chunks": self._indexed_chunks,
            "embedding_dimension": self._embedding_dimension,
            "working_dir": str(self._working_dir) if self._working_dir is not None else None,
            "lightrag_version": _lightrag_version(),
            "query_mode": self._lightrag_config.query_mode,
            "embedding_provider": type(self._embedding_provider).__name__,
            "llm_func": (
                "injected" if self._constructor_llm_func is not None else "openai_compatible"
            ),
        }
        if not dependency_available:
            return HealthCheckResult(
                healthy=False,
                system=self.system_name,
                message=(
                    "LightRAG optional dependency is unavailable: the 'lightrag' "
                    "package is missing or fails to import."
                ),
                details=details,
            )
        return HealthCheckResult(
            healthy=True,
            system=self.system_name,
            message="OK",
            details=details,
        )

    def cleanup(self) -> None:
        # Defined non-indexed state: prepared flag remains so the adapter can
        # be reindexed; index state and storages are cleared. Idempotent.
        self._teardown_rag()
        self._indexed_documents = 0
        self._indexed_chunks = 0
        if self._owns_working_dir and self._working_dir is not None:
            shutil.rmtree(self._working_dir, ignore_errors=True)
            self._working_dir = None


def _string_or_none(value: Any, label: str) -> str | None:
    """Strict string field: missing or explicit None -> None; empty str -> None.

    A present, non-null value must be a string; any other type (dict, list,
    bool, number, arbitrary object) is malformed upstream raw_data and
    raises. Values are never stringified or silently degraded to None.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(
            f"raw evidence field '{label}' must be a string, got {type(value).__name__}"
        )
    return value or None


def _evidence_text(value: Any, label: str) -> str | None:
    """Strict text field: missing or explicit None -> None; else must be a string."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(
            f"raw evidence field '{label}' must be a string, got {type(value).__name__}"
        )
    return value


def _finite_evidence_number(value: Any, label: str) -> int | float | None:
    """Strict numeric field: missing or explicit None -> None.

    A present, non-null value must be a finite int or float (bool rejected);
    NaN/+Infinity/-Infinity raise so they can never reach strict JSON
    metadata.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"raw evidence field '{label}' must be numeric, got {type(value).__name__}"
        )
    if not math.isfinite(value):
        raise ValueError(f"non-finite numeric value {value!r} in raw evidence field '{label}'")
    return value


# LightRAG's GRAPH_FIELD_SEP: merged cross-document file paths are joined
# with this separator inside the knowledge graph.
_GRAPH_FIELD_SEP = "<SEP>"


def _file_paths_from_raw(file_path: str | None) -> list[str]:
    """The adapter inserts documents with file_paths equal to document IDs.

    LightRAG merges cross-document ``file_path`` values with ``<SEP>``; the
    primary benchmark document id is the first path in the merged list. The
    caller must have already validated the raw value as ``str | None``.
    """
    if not file_path or file_path == "unknown_source":
        return []
    return [part for part in (p.strip() for p in file_path.split(_GRAPH_FIELD_SEP)) if part]
