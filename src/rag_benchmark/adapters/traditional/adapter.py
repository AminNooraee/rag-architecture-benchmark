"""Traditional Vector RAG baseline adapter (Phase 3C).

Pipeline: BenchmarkDocument -> deterministic chunking -> embedding -> exact
cosine vector index -> AdapterQueryInput -> query embedding -> cosine top-k
retrieval -> top-k RetrievedEvidence -> LLM generation -> canonical
QueryResult.

Fairness contract: query() receives only AdapterQueryInput; the generation
context is built exclusively from retrieved evidence. No reranking, HyDE,
query rewriting, multi-query expansion, or hidden retrieval stages.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from pydantic import ValidationError

from rag_benchmark.adapters.base import (
    AdapterCapabilities,
    BaseRAGAdapter,
    HealthCheckResult,
    IndexingResult,
)
from rag_benchmark.adapters.traditional.chunking import TextChunk, chunk_document
from rag_benchmark.adapters.traditional.config import TraditionalRAGConfig
from rag_benchmark.adapters.traditional.providers import EmbeddingProvider, LLMProvider
from rag_benchmark.adapters.traditional.vector_index import ExactCosineIndex, validate_vector
from rag_benchmark.config.models import (
    BenchmarkConfig,
    EmbeddingConfig,
    GenerationConfig,
    LLMConfig,
    RetrievalConfig,
)
from rag_benchmark.errors.models import BenchmarkErrorType, BenchmarkException, ConfigurationError
from rag_benchmark.schemas.document import BenchmarkDocument
from rag_benchmark.schemas.evidence import EvidenceType, RetrievedEvidence
from rag_benchmark.schemas.question import AdapterQueryInput
from rag_benchmark.schemas.result import QueryResult, QueryTiming, QueryUsage

_SYSTEM_NAME = "traditional_rag"
_VARIANT_NAME = "dense_exact"
# "baseline" is the schema-wide default label meaning "no explicit variant requested".
_ACCEPTED_CONFIG_VARIANTS = ("", "baseline", _VARIANT_NAME)


class TraditionalVectorRAGAdapter(BaseRAGAdapter):
    """Dense exact-cosine vector RAG baseline (variant: dense_exact).

    Lifecycle: activate(config) -> index(documents) -> query(AdapterQueryInput).
    Repeated index() calls perform an explicit full rebuild (no duplicate
    accumulation). cleanup() clears index state and is idempotent.
    """

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        llm_provider: LLMProvider,
        chunking_config: TraditionalRAGConfig | None = None,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._llm_provider = llm_provider
        self._chunking_config: TraditionalRAGConfig = (
            chunking_config if chunking_config is not None else TraditionalRAGConfig()
        )
        self._retrieval: RetrievalConfig | None = None
        self._generation: GenerationConfig | None = None
        self._embedding_config: EmbeddingConfig | None = None
        self._llm_config: LLMConfig | None = None
        self._prepared = False
        self._index: ExactCosineIndex | None = None
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
        # Conservative baseline: no incremental insert/update/delete, no
        # persistence, no streaming, no native citations.
        return AdapterCapabilities()

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
        raw = config.extra.get("traditional_rag", {})
        if not isinstance(raw, dict):
            raise ConfigurationError(
                "config.extra['traditional_rag'] must be a mapping when present"
            )
        try:
            self._chunking_config = TraditionalRAGConfig.model_validate(dict(raw))
        except ValidationError as e:
            raise ConfigurationError(f"Invalid traditional_rag configuration: {e}") from e
        self._retrieval = config.retrieval.model_copy(deep=True)
        self._generation = config.generation.model_copy(deep=True)
        self._embedding_config = config.embedding.model_copy(deep=True)
        self._llm_config = config.llm.model_copy(deep=True)
        self._prepared = True

    def index(self, documents: Sequence[BenchmarkDocument]) -> IndexingResult:
        if not self._prepared:
            raise ConfigurationError(
                "TraditionalVectorRAGAdapter must be prepared (via activate/prepare) before index()"
            )
        chunking = self._chunking_config
        embedding_config = self._embedding_config
        if embedding_config is None:
            raise ConfigurationError(
                "Adapter internal state is inconsistent: missing embedding config"
            )

        started = time.perf_counter()
        all_chunks: list[TextChunk] = []
        for document in documents:
            all_chunks.extend(chunk_document(document, chunking.chunk_size, chunking.overlap))
        chunks = tuple(all_chunks)

        raw_vectors: list[list[float]] = []
        batches = 0
        batch_size = embedding_config.batch_size
        for offset in range(0, len(chunks), batch_size):
            batch_texts = [chunk.text for chunk in chunks[offset : offset + batch_size]]
            raw_vectors.extend(self._embedding_provider.embed_documents(batch_texts))
            batches += 1

        if len(raw_vectors) != len(chunks):
            raise BenchmarkException(
                f"Embedding provider returned {len(raw_vectors)} vectors for {len(chunks)} chunks",
                error_type=BenchmarkErrorType.INDEXING_ERROR,
            )

        vectors = [validate_vector(vector) for vector in raw_vectors]
        embedding_dimension: int | None = len(vectors[0]) if vectors else None
        for position, vector in enumerate(vectors):
            if embedding_dimension is not None and len(vector) != embedding_dimension:
                raise BenchmarkException(
                    f"Embedding dimension mismatch at chunk {position}: "
                    f"{len(vector)} != {embedding_dimension}",
                    error_type=BenchmarkErrorType.INDEXING_ERROR,
                )
        if (
            embedding_dimension is not None
            and embedding_config.dimension is not None
            and embedding_dimension != embedding_config.dimension
        ):
            raise BenchmarkException(
                f"Embedding dimension {embedding_dimension} does not match configured "
                f"dimension {embedding_config.dimension}",
                error_type=BenchmarkErrorType.INDEXING_ERROR,
            )

        index = ExactCosineIndex()
        for chunk, vector in zip(chunks, vectors, strict=True):
            index.add(chunk, vector)

        self._index = index
        self._indexed_documents = len(documents)
        self._indexed_chunks = len(chunks)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return IndexingResult(
            indexed_documents=len(documents),
            indexed_chunks=len(chunks),
            elapsed_ms=elapsed_ms,
            metadata={
                "chunk_size": chunking.chunk_size,
                "overlap": chunking.overlap,
                "embedding_dimension": embedding_dimension,
                "embedding_batches": batches,
            },
        )

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
        if self._index is None or len(self._index) == 0:
            return QueryResult.create_failure(
                question_id=question_id,
                system=self.system_name,
                variant=self.variant_name,
                error_message="Adapter has no indexed documents; call index() before query().",
                error_type=BenchmarkErrorType.CONFIGURATION_ERROR,
            )
        retrieval_config = self._retrieval
        generation_config = self._generation
        if retrieval_config is None or generation_config is None:
            return QueryResult.create_failure(
                question_id=question_id,
                system=self.system_name,
                variant=self.variant_name,
                error_message="Adapter internal state is inconsistent after prepare().",
                error_type=BenchmarkErrorType.CONFIGURATION_ERROR,
            )

        total_start = time.perf_counter()
        retrieval_start = total_start
        embedding_calls = 0
        try:
            query_vector = self._embedding_provider.embed_query(query_input.question)
            embedding_calls = 1
            vector = validate_vector(query_vector)
        except Exception as e:
            return QueryResult.create_failure(
                question_id=question_id,
                system=self.system_name,
                variant=self.variant_name,
                error_message=f"Query embedding failed: {e}",
                error_type=BenchmarkErrorType.RETRIEVAL_ERROR,
                usage=QueryUsage(embedding_calls=embedding_calls, llm_calls=0),
                timing=_elapsed(
                    retrieval_start=retrieval_start,
                    generation_start=None,
                    total_start=total_start,
                    generation_ms=0.0,
                ),
                metadata={"stage": "retrieval"},
            )
        index_dimension = self._index.dimension
        if index_dimension is None or len(vector) != index_dimension:
            return QueryResult.create_failure(
                question_id=question_id,
                system=self.system_name,
                variant=self.variant_name,
                error_message=(
                    f"Query vector dimension {len(vector)} does not match index dimension "
                    f"{index_dimension}"
                ),
                error_type=BenchmarkErrorType.RETRIEVAL_ERROR,
                usage=QueryUsage(embedding_calls=embedding_calls, llm_calls=0),
                timing=_elapsed(
                    retrieval_start=retrieval_start,
                    generation_start=None,
                    total_start=total_start,
                    generation_ms=0.0,
                ),
                metadata={"stage": "retrieval"},
            )

        results = self._index.search(
            vector,
            top_k=retrieval_config.top_k,
            score_threshold=retrieval_config.score_threshold,
        )
        evidence = [
            RetrievedEvidence(
                document_id=entry.chunk.document_id,
                chunk_id=entry.chunk.chunk_id,
                text=entry.chunk.text,
                score=score,
                rank=rank,
                evidence_type=EvidenceType.CHUNK,
            )
            for rank, (score, entry) in enumerate(results, start=1)
        ]
        context_texts = [item.text for item in evidence]

        generation_start = time.perf_counter()
        # The single generation call is attempted; count it even if the
        # provider fails, so usage reflects real provider traffic.
        llm_calls = 1
        try:
            llm_result = self._llm_provider.generate(
                query_input.question, context_texts, generation_config
            )
        except Exception as e:
            return QueryResult.create_failure(
                question_id=question_id,
                system=self.system_name,
                variant=self.variant_name,
                error_message=f"LLM generation failed: {e}",
                error_type=BenchmarkErrorType.GENERATION_ERROR,
                retrieved_evidence=evidence,
                usage=QueryUsage(embedding_calls=embedding_calls, llm_calls=llm_calls),
                timing=_elapsed(
                    retrieval_start=retrieval_start,
                    generation_start=generation_start,
                    total_start=total_start,
                    generation_ms=(time.perf_counter() - generation_start) * 1000.0,
                ),
                metadata={"stage": "generation"},
            )
        if not llm_result.answer.strip():
            return QueryResult.create_failure(
                question_id=question_id,
                system=self.system_name,
                variant=self.variant_name,
                error_message="LLM returned an empty answer.",
                error_type=BenchmarkErrorType.GENERATION_ERROR,
                retrieved_evidence=evidence,
                usage=QueryUsage(embedding_calls=embedding_calls, llm_calls=llm_calls),
                timing=_elapsed(
                    retrieval_start=retrieval_start,
                    generation_start=generation_start,
                    total_start=total_start,
                    generation_ms=(time.perf_counter() - generation_start) * 1000.0,
                ),
                metadata={"stage": "generation"},
            )

        return QueryResult.create_success(
            question_id=question_id,
            system=self.system_name,
            answer=llm_result.answer,
            variant=self.variant_name,
            retrieved_evidence=evidence,
            usage=QueryUsage(
                llm_calls=llm_calls,
                embedding_calls=embedding_calls,
                input_tokens=llm_result.input_tokens or 0,
                output_tokens=llm_result.output_tokens or 0,
            ),
            timing=_elapsed(
                retrieval_start=retrieval_start,
                generation_start=generation_start,
                total_start=total_start,
                generation_ms=(time.perf_counter() - generation_start) * 1000.0,
            ),
            metadata={"retrieved_chunks": len(evidence)},
        )

    def health(self) -> HealthCheckResult:
        if not self._prepared:
            return HealthCheckResult(
                healthy=False,
                system=self.system_name,
                message="Adapter is not prepared.",
            )
        index = self._index
        return HealthCheckResult(
            healthy=True,
            system=self.system_name,
            message="OK",
            details={
                "prepared": True,
                "indexed": index is not None and len(index) > 0,
                "indexed_documents": self._indexed_documents,
                "indexed_chunks": self._indexed_chunks,
                "embedding_dimension": index.dimension if index is not None else None,
                "embedding_provider": type(self._embedding_provider).__name__,
                "llm_provider": type(self._llm_provider).__name__,
            },
        )

    def cleanup(self) -> None:
        # Defined non-indexed state: prepared flag remains so the adapter can be
        # reindexed; index state is cleared. Safe to call repeatedly.
        self._index = None
        self._indexed_documents = 0
        self._indexed_chunks = 0


def _elapsed(
    retrieval_start: float,
    generation_start: float | None,
    total_start: float,
    generation_ms: float,
) -> QueryTiming:
    retrieval_ms = (time.perf_counter() - retrieval_start) * 1000.0
    if generation_start is not None:
        generation_ms = (time.perf_counter() - generation_start) * 1000.0
    total_ms = (time.perf_counter() - total_start) * 1000.0
    return QueryTiming(
        retrieval_ms=max(retrieval_ms, 0.0),
        generation_ms=max(generation_ms, 0.0),
        total_ms=max(total_ms, 0.0),
    )
